"""
End-to-end path: waveform → Whisper encoder → overlapping windows → StreamingAdapter → LLM text.

**Early-commit gate vs this module**

- ``L_gate`` in training (``training/adapter_asr_trainer.py``, ``training/adapter_task_trainer.py``)
  is the loss for :class:`adapter.early_commit_gate.EarlyCommitGate` (when to start generating),
  **not** for ``gate_scores`` from the adapter’s optional **rate controller**
  (:class:`adapter.rate_controller.AdaptiveRateController`).

- :class:`WhisperAdapterLLMPipeline` feeds windows through :meth:`StreamingAdapter.forward`, which
  **does not** run :class:`~adapter.early_commit_gate.EarlyCommitGate`. If your checkpoint or
  training loop includes a commit gate, use :class:`WhisperAdapterLLMCommitGatePipeline` for
  inference that mirrors the trainers (per-window ``forward_window`` + gate on the accumulated
  sequence). Full training with gradients should still follow the official trainer scripts.
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn

from adapter.early_commit_gate import EarlyCommitGate
from adapter.streaming_adapter import StreamingAdapter
from adapter.windowing import WhisperFrameWindowizer
from encoder.whisper_encoder import encode_waveform_to_hidden
from llm.config import LlmGenerationParams, build_hf_generation_config


def whisper_waveform_to_encoder_windows(
    waveform,
    *,
    whisper_processor: Any,
    whisper_model: nn.Module,
    windowizer: WhisperFrameWindowizer,
    device: str,
    torch_dtype: torch.dtype,
    sample_rate: int = 16000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Full-utterance Whisper encoder pass then ``WhisperFrameWindowizer``.

    Use this in training notebooks to mirror :meth:`WhisperAdapterLLMPipeline.encode_waveform`
    without constructing the full pipeline.

    ``whisper_model`` may be ``WhisperForConditionalGeneration`` (uses ``model.encoder``)
    or ``WhisperModel`` (uses ``encoder``).

    Args:
        waveform: 1D float tensor or array-like accepted by ``whisper_processor``.

    Returns:
        enc: ``(1, T, D)`` full encoder sequence.
        windows: ``(1, N, W, D)`` overlapping chunks for :meth:`StreamingAdapter.forward_window`.
    """
    enc = encode_waveform_to_hidden(
        waveform,
        whisper_processor=whisper_processor,
        whisper_model=whisper_model,
        device=device,
        torch_dtype=torch_dtype,
        sample_rate=sample_rate,
    )

    windows = windowizer(enc)
    return enc, windows


def windows_tensor_to_batch_list(windows: torch.Tensor) -> list[torch.Tensor]:
    """Split ``(1, N, W, D)`` into ``N`` tensors of shape ``(1, W, D)`` for streaming steps."""
    if windows.ndim != 4 or windows.shape[0] != 1:
        raise ValueError(f"Expected windows (1, N, W, D), got {tuple(windows.shape)}")
    n = windows.shape[1]
    return [windows[0, i].unsqueeze(0).contiguous() for i in range(n)]


class WhisperAdapterLLMPipeline:
    """
    Orchestrates: waveform → Whisper encoder → windowizer → adapter (queries) → causal LM.

    This path uses :meth:`StreamingAdapter.forward` (full window list in one call). It does
    **not** invoke :class:`~adapter.early_commit_gate.EarlyCommitGate`. Training losses
    ``L_sparse`` / ``L_rate`` refer to the optional **rate controller** inside the adapter;
    ``L_gate`` refers to the **early-commit gate** only. If you trained with a commit gate,
    switch to :class:`WhisperAdapterLLMCommitGatePipeline` for inference aligned with the
    stage-2/3 loop.

    Args:
        whisper_processor: HuggingFace WhisperProcessor (feature extraction + padding).
        whisper_model: WhisperForConditionalGeneration (only ``model.encoder`` is used).
        windowizer: Splits ``(B, T, D)`` encoder states into ``(B, N, W, D)`` windows.
        streaming_adapter: Maps each window to LLM-space token embeddings.
        llm_model: Causal LM (e.g. Qwen) with ``get_input_embeddings()`` and ``generate``.
        llm_tokenizer: Tokenizer for prompts and decoding.
        device: Torch device string for tensors moved in this class.
        torch_dtype: Model dtype (e.g. float16 on GPU).
        sample_rate: Passed to the Whisper processor (default 16 kHz).

    ``n_windows`` in :meth:`generate`:
        - Positive integer: use the first ``n_windows`` overlapping windows from the
          encoder output (each window is one adapter step; the adapter sees them in order).
        - ``-1``: use every window produced for that encoder output (1500 frames → many
          windows → long ``inputs_embeds``; use with care for memory).
    """

    def __init__(
        self,
        *,
        whisper_processor: Any,
        whisper_model: nn.Module,
        windowizer: WhisperFrameWindowizer,
        streaming_adapter: StreamingAdapter,
        llm_model: nn.Module,
        llm_tokenizer: Any,
        device: str,
        torch_dtype: torch.dtype,
        sample_rate: int = 16000,
    ) -> None:
        self.whisper_processor = whisper_processor
        self.whisper_model = whisper_model
        self.windowizer = windowizer
        self.streaming_adapter = streaming_adapter
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer
        self.device = device
        self.torch_dtype = torch_dtype
        self.sample_rate = sample_rate

    def encode_waveform(
        self,
        waveform: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        Run Whisper encoder and build overlapping windows.

        Returns:
            enc: (1, T, D) full encoder sequence (e.g. T=1500).
            windows: (1, N, W, D) overlapping chunks along time.
            encode_seconds: wall time for encoder + windowing.
        """
        t0 = time.time()
        enc, windows = whisper_waveform_to_encoder_windows(
            waveform,
            whisper_processor=self.whisper_processor,
            whisper_model=self.whisper_model,
            windowizer=self.windowizer,
            device=self.device,
            torch_dtype=self.torch_dtype,
            sample_rate=self.sample_rate,
        )
        return enc, windows, time.time() - t0

    @staticmethod
    def _select_windows(
        windows: torch.Tensor,
        n_windows: int,
    ) -> list[torch.Tensor]:
        """From ``windows`` (B, N, W, D), build list of (1, W, D) for the streaming adapter."""
        if windows.ndim != 4:
            raise ValueError(f"Expected windows (B, N, W, D), got {tuple(windows.shape)}")
        b, n, _w, _d = windows.shape
        if b != 1:
            raise ValueError(f"Batch size 1 expected for this pipeline, got B={b}")

        if n_windows == -1:
            take = n
        else:
            if n_windows <= 0:
                raise ValueError("n_windows must be positive or -1 for all windows")
            take = min(n_windows, n)

        return [windows[0, i].unsqueeze(0) for i in range(take)]

    def build_default_prompt(self, num_compressed_tokens: int) -> str:
        return (
            f"You are a helpful assistant. Below is a compressed audio encoding with "
            f"{num_compressed_tokens} token positions. "
            f"Please provide a concise and accurate summary of the audio content.\n\n"
            f"Focus on the main topics, speakers (if identifiable), and key information conveyed.\n\n"
            f"Summary:"
        )

    def _decode_from_compressed_tokens(
        self,
        *,
        tokens: torch.Tensor,
        enc: torch.Tensor,
        windows: torch.Tensor,
        encode_s: float,
        num_windows_used: int,
        adapter_out: dict[str, Any],
        prompt: str | None,
        generation: LlmGenerationParams | None,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run causal LM ``generate`` given compressed audio token embeddings."""
        num_compressed = int(tokens.shape[1])

        user_prompt = prompt if prompt is not None else self.build_default_prompt(num_compressed)

        if hasattr(self.llm_tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": user_prompt}]
            formatted = self.llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = self.llm_tokenizer(formatted, return_tensors="pt").input_ids.to(
                self.device
            )
        else:
            prompt_ids = self.llm_tokenizer(user_prompt, return_tensors="pt").input_ids.to(
                self.device
            )

        prompt_embs = self.llm_model.get_input_embeddings()(prompt_ids)
        input_embs = torch.cat([prompt_embs, tokens], dim=1).to(
            device=self.device, dtype=self.torch_dtype
        )

        attention_mask = torch.ones(
            input_embs.shape[0],
            input_embs.shape[1],
            dtype=torch.long,
            device=input_embs.device,
        )

        if generation is None:
            generation = LlmGenerationParams(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        gen_cfg = build_hf_generation_config(
            model=self.llm_model,
            tokenizer=self.llm_tokenizer,
            params=generation,
        )

        with torch.no_grad():
            out_ids = self.llm_model.generate(
                inputs_embeds=input_embs,
                attention_mask=attention_mask,
                generation_config=gen_cfg,
            )

        text = self.llm_tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

        out: dict[str, Any] = {
            "text": text,
            "encode_time_s": encode_s,
            "enc": enc,
            "windows": windows,
            "num_windows_used": num_windows_used,
            "adapter_out": adapter_out,
            "prompt_ids": prompt_ids,
            "generated_ids": out_ids,
        }
        if extra:
            out.update(extra)
        return out

    def generate(
        self,
        waveform: torch.Tensor,
        *,
        n_windows: int = 1,
        generation: LlmGenerationParams | None = None,
        max_new_tokens: int = 100,
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        Full inference: encode → adapter over selected windows → LLM generation.

        Args:
            waveform: 1D float tensor (mono) at ``sample_rate`` (e.g. from librosa).
            n_windows: Number of overlapping windows to feed the adapter for this call,
                or ``-1`` to use all windows from the encoder output.
            generation: Preferred API. If provided, overrides the legacy scalar args below.
            max_new_tokens/do_sample/temperature/top_p/top_k: Legacy args kept for backward
                compatibility. They are ignored when `generation` is provided.
            prompt: Optional override; if None, :meth:`build_default_prompt` is used.

        Returns:
            Dictionary with text, tensors, and timing metadata.
        """
        enc, windows, encode_s = self.encode_waveform(waveform)
        window_list = self._select_windows(windows, n_windows)

        with torch.no_grad():
            adapter_out = self.streaming_adapter(window_list)

        tokens = adapter_out["tokens"]
        return self._decode_from_compressed_tokens(
            tokens=tokens,
            enc=enc,
            windows=windows,
            encode_s=encode_s,
            num_windows_used=len(window_list),
            adapter_out=adapter_out,
            prompt=prompt,
            generation=generation,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )


class WhisperAdapterLLMCommitGatePipeline:
    """
    Same stack as :class:`WhisperAdapterLLMPipeline`, but runs **per-window**
    :meth:`StreamingAdapter.forward_window` and evaluates :class:`~adapter.early_commit_gate.EarlyCommitGate`
    on the accumulated token sequence—matching the structure used when optimizing ``L_gate``
    in stage 2/3 trainers.

    Use this when loading checkpoints that include ``gate_state_dict`` or when you care about
    commit probabilities / optional early truncation. For plain adapter-only inference, use
    :class:`WhisperAdapterLLMPipeline`.
    """

    def __init__(
        self,
        *,
        whisper_processor: Any,
        whisper_model: nn.Module,
        windowizer: WhisperFrameWindowizer,
        streaming_adapter: StreamingAdapter,
        early_commit_gate: EarlyCommitGate,
        llm_model: nn.Module,
        llm_tokenizer: Any,
        device: str,
        torch_dtype: torch.dtype,
        sample_rate: int = 16000,
    ) -> None:
        self._llm = WhisperAdapterLLMPipeline(
            whisper_processor=whisper_processor,
            whisper_model=whisper_model,
            windowizer=windowizer,
            streaming_adapter=streaming_adapter,
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer,
            device=device,
            torch_dtype=torch_dtype,
            sample_rate=sample_rate,
        )
        self.early_commit_gate = early_commit_gate.to(device=device)

    def encode_waveform(
        self, waveform: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        return self._llm.encode_waveform(waveform)

    def build_default_prompt(self, num_compressed_tokens: int) -> str:
        return self._llm.build_default_prompt(num_compressed_tokens)

    def generate(
        self,
        waveform: torch.Tensor,
        *,
        n_windows: int = 1,
        generation: LlmGenerationParams | None = None,
        max_new_tokens: int = 100,
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        prompt: str | None = None,
        use_early_commit_truncation: bool = False,
    ) -> dict[str, Any]:
        """
        Encode → per-window adapter + gate → LLM.

        Args:
            use_early_commit_truncation: If True, stop consuming further windows once
                ``should_commit`` is true (unless already at the last window). Default False
                uses the full selected window list for the LM (same total audio tokens as the
                non-gate pipeline), while still recording gate diagnostics.
        """
        enc, windows, encode_s = self._llm.encode_waveform(waveform)
        window_list = WhisperAdapterLLMPipeline._select_windows(windows, n_windows)

        adapter = self._llm.streaming_adapter
        gate = self.early_commit_gate
        adapter.reset_streaming_state()

        chunks: list[torch.Tensor] = []
        commit_probs: list[torch.Tensor] = []
        gate_losses: list[torch.Tensor] = []
        total_stab = torch.tensor(0.0, device=self._llm.device, dtype=self._llm.torch_dtype)
        total_sparse = torch.tensor(0.0, device=self._llm.device, dtype=self._llm.torch_dtype)
        total_rate = torch.tensor(0.0, device=self._llm.device, dtype=self._llm.torch_dtype)

        with torch.no_grad():
            for t, w in enumerate(window_list):
                wdev = w.to(device=self._llm.device, dtype=self._llm.torch_dtype)
                step = adapter.forward_window(wdev)
                chunks.append(step["tokens"])
                total_stab = total_stab + step["stability_loss"]
                if step["sparse_loss"] is not None:
                    total_sparse = total_sparse + step["sparse_loss"]
                if step["rate_loss"] is not None:
                    total_rate = total_rate + step["rate_loss"]

                accumulated = torch.cat(chunks, dim=1)
                gr = gate(accumulated, t, len(window_list))
                commit_probs.append(gr["commit_prob"])
                gate_losses.append(gr["gate_loss"])

                if (
                    use_early_commit_truncation
                    and t < len(window_list) - 1
                    and gr["should_commit"].item() > 0.5
                ):
                    break

            tokens = torch.cat(chunks, dim=1)

        adapter_out: dict[str, Any] = {
            "tokens": tokens,
            "stability_loss": total_stab,
            "gate_scores": None,
            "sparse_loss": total_sparse if adapter.rate_controller is not None else None,
            "rate_loss": total_rate if adapter.rate_controller is not None else None,
        }

        cp = torch.stack(commit_probs, dim=0) if commit_probs else torch.zeros(0)
        gl = torch.stack(gate_losses, dim=0) if gate_losses else torch.zeros(0)

        extra = {
            "early_commit_commit_probs": cp,
            "early_commit_gate_losses": gl,
            "early_commit_gate_loss_sum": float(gl.sum().item()) if gl.numel() else 0.0,
            "num_windows_committed": len(chunks),
        }

        return self._llm._decode_from_compressed_tokens(
            tokens=tokens,
            enc=enc,
            windows=windows,
            encode_s=encode_s,
            num_windows_used=len(chunks),
            adapter_out=adapter_out,
            prompt=prompt,
            generation=generation,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            extra=extra,
        )
