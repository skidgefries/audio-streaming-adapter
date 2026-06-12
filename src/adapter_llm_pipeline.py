"""
End-to-end path: waveform → Whisper encoder → overlapping windows → StreamingAdapter → LLM text.

**Turn-end commit gate (Component 3)**

- ``L_gate`` in training (``training/adapter_asr_trainer.py``, ``training/adapter_task_trainer.py``)
  is the loss for :class:`adapter.turn_end_commit_gate.TurnEndCommitGate` — the integrated
  turn-end / VAD replacement that fires ``should_commit`` when the user stops speaking.
  **Not** the adapter rate-controller ``gate_scores``
  (:class:`adapter.rate_controller.AdaptiveRateController`).

- :class:`WhisperAdapterLLMCommitGatePipeline` supports batch :meth:`generate` and incremental
  :meth:`generate_streaming` (per-window KV-cache + generate on ``should_commit``). See
  ``docs/AUDIO_STREAM.md`` §4 and §7.

# Previous mean-pool gate (reference only):
# from adapter.early_commit_gate import EarlyCommitGate  # deprecated alias → TurnEndCommitGate
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn as nn

from adapter.streaming_adapter import StreamingAdapter
from adapter.turn_end_commit_gate import TurnEndCommitGate
from adapter.windowing import AudioWaveformWindowizer, stack_encoder_windows
from encoder.whisper_encoder import encode_waveform_to_hidden
from llm.config import LlmGenerationParams, build_hf_generation_config
from llm.kv_cache import llm_input_device

# StreamingSession is imported lazily in generate_streaming / create_streaming_session.


def _tensor_shape(value: Any) -> str | int | float | bool | None:
    if isinstance(value, torch.Tensor):
        return str(tuple(value.shape))
    return value


@dataclass
class PipelineTraceStep:
    """One logged step in the end-to-end inference path."""

    index: int
    name: str
    elapsed_s: float = 0.0
    fields: dict[str, Any] = field(default_factory=dict)


class PipelineStepLogger:
    """
    Step-by-step console logger for pipeline debugging and smoke tests.

    Enable on a pipeline via ``step_logger=PipelineStepLogger(enabled=True)`` or
    ``verbose=True`` on :class:`WhisperAdapterLLMPipeline`.
    """

    def __init__(self, *, enabled: bool = False, prefix: str = "pipeline") -> None:
        self.enabled = enabled
        self.prefix = prefix
        self._steps: list[PipelineTraceStep] = []
        self._t0 = time.time()
        self._last = self._t0

    def reset(self) -> None:
        self._steps.clear()
        self._t0 = time.time()
        self._last = self._t0

    def log(self, name: str, **fields: Any) -> PipelineTraceStep:
        if not self.enabled:
            return PipelineTraceStep(index=len(self._steps) + 1, name=name, fields=fields)

        now = time.time()
        elapsed = now - self._last
        self._last = now
        safe_fields = {k: _tensor_shape(v) for k, v in fields.items()}
        step = PipelineTraceStep(
            index=len(self._steps) + 1,
            name=name,
            elapsed_s=elapsed,
            fields=safe_fields,
        )
        self._steps.append(step)
        detail = " ".join(f"{k}={v}" for k, v in safe_fields.items())
        msg = f"[{self.prefix}] step {step.index}: {name}"
        if detail:
            msg = f"{msg} | {detail}"
        msg = f"{msg} ({elapsed:.3f}s)"
        print(msg, flush=True)
        return step

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefix": self.prefix,
            "total_elapsed_s": time.time() - self._t0,
            "steps": [asdict(step) for step in self._steps],
        }


def _resolve_smoke_test_device(device: str) -> tuple[str, torch.dtype]:
    """Map CLI/env device string to (device, dtype). Honors explicit ``cpu`` / ``cuda``."""
    normalized = (device or "cpu").strip().lower()
    if normalized == "cpu":
        return "cpu", torch.float32
    if normalized.startswith("cuda"):
        return normalized, torch.bfloat16
    return normalized, torch.float32


def discover_stage2_checkpoints(checkpoints_dir: str) -> list[str]:
    """Return sorted ``adapter_stage2*.pt`` paths under ``checkpoints_dir``."""
    import glob
    import os

    pattern = os.path.join(checkpoints_dir, "adapter_stage2*.pt")
    return sorted(glob.glob(pattern))


def whisper_waveform_to_encoder_windows(
    waveform,
    *,
    whisper_processor: Any,
    whisper_model: nn.Module,
    windowizer: AudioWaveformWindowizer,
    device: str,
    torch_dtype: torch.dtype,
    sample_rate: int = 16000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Raw-audio windowing, then one Whisper encoder pass **per audio chunk**.

    Use this in training notebooks to mirror :meth:`WhisperAdapterLLMPipeline.encode_waveform`
    without constructing the full pipeline.

    Args:
        waveform: 1D float tensor or array-like accepted by ``whisper_processor``.
        windowizer: :class:`AudioWaveformWindowizer` (0.8s / 0.4s by default).

    Returns:
        enc: ``(1, T_total, D)`` concatenation of per-chunk encoder sequences (diagnostics).
        windows: ``(1, N, T_max, D)`` padded stack for :meth:`StreamingAdapter.forward_window`.
    """
    if isinstance(waveform, torch.Tensor):
        wave = waveform.detach().float().cpu().reshape(-1)
    else:
        wave = torch.tensor(waveform, dtype=torch.float32).reshape(-1)

    if wave.numel() == 0:
        empty = torch.zeros(1, 0, 0, device=device, dtype=torch_dtype)
        return empty, empty

    audio_chunks = windowizer(wave)
    enc_windows: list[torch.Tensor] = []
    for chunk in audio_chunks:
        enc_w = encode_waveform_to_hidden(
            chunk,
            whisper_processor=whisper_processor,
            whisper_model=whisper_model,
            device=device,
            torch_dtype=torch_dtype,
            sample_rate=sample_rate,
        )
        enc_windows.append(enc_w)

    enc = torch.cat(enc_windows, dim=1) if enc_windows else torch.zeros(1, 0, 0, device=device, dtype=torch_dtype)
    windows = stack_encoder_windows(enc_windows, device=device, dtype=torch_dtype)
    return enc, windows


def _truncate_asr_chat_tail(text: str) -> str:
    """
    Drop Qwen chat continuation after the transcript.

    Stage-2 greedy decode often emits the transcript first, then ``\\n\\nOkay, ...``
    assistant text. LibriSpeech references are single-line, so the first blank line
    is a safe boundary.
    """
    for marker in (
        "\n\nOkay",
        "\n\nSo ",
        "\n\nI ",
        "\n\nLet ",
        "\n\nWait",
        "\n\nThe user",
        "\n\nHmm",
        "\n\n",
    ):
        if marker in text:
            return text.split(marker, 1)[0].strip()
    return text.strip()


def windows_tensor_to_batch_list(windows: torch.Tensor) -> list[torch.Tensor]:
    """Split ``(1, N, W, D)`` into ``N`` tensors of shape ``(1, W, D)`` for streaming steps."""
    if windows.ndim != 4 or windows.shape[0] != 1:
        raise ValueError(f"Expected windows (1, N, W, D), got {tuple(windows.shape)}")
    n = windows.shape[1]
    return [windows[0, i].unsqueeze(0).contiguous() for i in range(n)]


class WhisperAdapterLLMPipeline:
    """
    Orchestrates: waveform → raw-audio windows → Whisper encode per chunk → adapter → causal LM.

    This path uses :meth:`StreamingAdapter.forward` (full window list in one call). It does
    **not** invoke :class:`~adapter.turn_end_commit_gate.TurnEndCommitGate`. Training losses
    ``L_sparse`` / ``L_rate`` refer to the optional **rate controller** inside the adapter;
    ``L_gate`` refers to the **turn-end commit gate** only. If you trained with a gate,
    switch to :class:`WhisperAdapterLLMCommitGatePipeline` for inference aligned with the
    stage-2/3 loop.

    Args:
        whisper_processor: HuggingFace WhisperProcessor (feature extraction + padding).
        whisper_model: WhisperForConditionalGeneration (only ``model.encoder`` is used).
        windowizer: Splits raw waveform into overlapping chunks, then Whisper-encodes each
            into ``(1, N, T_max, D)`` for the streaming adapter.
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
        windowizer: AudioWaveformWindowizer,
        streaming_adapter: StreamingAdapter,
        llm_model: nn.Module,
        llm_tokenizer: Any,
        device: str,
        torch_dtype: torch.dtype,
        sample_rate: int = 16000,
        step_logger: PipelineStepLogger | None = None,
        verbose: bool = False,
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
        self.step_logger = step_logger or PipelineStepLogger(
            enabled=verbose,
            prefix="WhisperAdapterLLMPipeline",
        )

    def _llm_tensor_device(self) -> torch.device:
        """Device for ``inputs_embeds`` / ``generate`` (matches sharded LMs)."""
        return llm_input_device(self.llm_model)

    @staticmethod
    def qwen_no_think_suffix_embeds(
        llm_model: Any,
        device: torch.device,
        torch_dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Qwen3 empty ``...`` block to disable chain-of-thought at decode time.

        Matches ``apply_chat_template(..., enable_thinking=False)`` suffix.
        """
        no_think_ids = torch.tensor(
            [[151667, 271, 151668, 271]], device=device, dtype=torch.long
        )
        return llm_model.get_input_embeddings()(no_think_ids).to(
            device=device, dtype=torch_dtype
        )

    @staticmethod
    def train_style_separator_token_id(tokenizer: Any) -> int:
        """
        Token after audio prefix for train-style ASR (matches ``adapter_asr_trainer``).

        Qwen3 has no ``bos_token_id``; training falls back to ``eos_token_id`` (``im_end``).
        """
        bos_id = getattr(tokenizer, "bos_token_id", None)
        if bos_id is not None:
            return int(bos_id)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:
            return int(eos_id)
        raise ValueError("Tokenizer has no bos_token_id or eos_token_id for train-style ASR")

    def _train_style_prefix_embeds(
        self,
        tokens: torch.Tensor,
        *,
        append_im_end: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``[audio tokens]`` or ``[audio tokens | im_end/BOS]`` for train-style inference."""
        llm_dev = self._llm_tensor_device()
        audio = tokens.to(device=llm_dev, dtype=self.torch_dtype)
        if not append_im_end:
            return audio, torch.zeros((1, 0), device=llm_dev, dtype=torch.long)
        sep_id = self.train_style_separator_token_id(self.llm_tokenizer)
        sep_ids = torch.tensor([[sep_id]], device=llm_dev, dtype=torch.long)
        sep_embed = self.llm_model.get_input_embeddings()(sep_ids)
        no_think_embed = self.qwen_no_think_suffix_embeds(
            self.llm_model, llm_dev, self.torch_dtype
        )
        input_embs = torch.cat([audio, sep_embed, no_think_embed], dim=1)
        return input_embs, sep_ids

    def encode_waveform(
        self,
        waveform: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        Run Whisper encoder on each raw-audio window and build padded encoder stack.

        Returns:
            enc: (1, T, D) concatenation of per-chunk encoder outputs (diagnostics).
            windows: (1, N, T_max, D) padded stack for the streaming adapter.
            encode_seconds: wall time for all chunk encodes + windowing.
        """
        wave = waveform.detach().float().reshape(-1) if isinstance(waveform, torch.Tensor) else waveform
        wave_samples = int(wave.numel()) if isinstance(wave, torch.Tensor) else len(wave)
        self.step_logger.log(
            "encode_waveform.start",
            waveform_samples=wave_samples,
            sample_rate=self.sample_rate,
        )
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
        encode_s = time.time() - t0
        self.step_logger.log(
            "encode_waveform.done",
            enc_shape=enc,
            windows_shape=windows,
            num_windows=int(windows.shape[1]) if windows.ndim == 4 else 0,
            encode_time_s=round(encode_s, 4),
        )
        return enc, windows, encode_s

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

    @staticmethod
    def build_asr_prompt() -> str:
        """Qwen chat user message for verbatim speech transcription."""
        from training.utils.asr_prompt import DEFAULT_ASR_PROMPT

        return DEFAULT_ASR_PROMPT

    @staticmethod
    def _resolve_asr_generation(
        *,
        prompt: str | None,
        train_style_asr: bool,
        prompt_asr: bool,
        asr_prompt: str | None,
    ) -> tuple[str | None, bool, bool]:
        """
        Resolve LM prefix for ASR inference.

        Returns:
            (prompt, train_style_asr, trim_asr_tail)
        """
        if prompt_asr:
            text = (
                prompt
                if prompt is not None
                else (asr_prompt if asr_prompt is not None else WhisperAdapterLLMPipeline.build_asr_prompt())
            )
            return text, False, True
        return prompt, train_style_asr, train_style_asr

    def _tokenize_asr_prompt(self, user_prompt: str) -> torch.Tensor:
        """Tokenize the ASR user prompt; disable Qwen3 thinking mode when supported."""
        messages = [{"role": "user", "content": user_prompt}]
        tokenizer = self.llm_tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            batch = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=False,
            )
            return batch.input_ids.to(self.device)

        return tokenizer(user_prompt, return_tensors="pt").input_ids.to(self.device)

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
        train_style_asr: bool = False,
        append_im_end: bool = True,
        trim_asr_tail: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run causal LM ``generate`` given compressed audio token embeddings."""
        self.step_logger.log(
            "llm_decode.start",
            compressed_tokens=tokens,
            num_windows_used=num_windows_used,
            train_style_asr=train_style_asr,
            append_im_end=append_im_end,
            trim_asr_tail=trim_asr_tail,
        )
        llm_dev = self._llm_tensor_device()
        if train_style_asr:
            input_embs, prompt_ids = self._train_style_prefix_embeds(
                tokens, append_im_end=append_im_end
            )
        else:
            num_compressed = int(tokens.shape[1])
            user_prompt = prompt if prompt is not None else self.build_default_prompt(num_compressed)
            prompt_ids = self._tokenize_asr_prompt(user_prompt)

            prompt_embs = self.llm_model.get_input_embeddings()(prompt_ids.to(llm_dev))
            input_embs = torch.cat([prompt_embs, tokens.to(llm_dev)], dim=1).to(
                device=llm_dev, dtype=self.torch_dtype
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
        if trim_asr_tail:
            text = _truncate_asr_chat_tail(text)

        self.step_logger.log(
            "llm_decode.done",
            input_embeds_shape=input_embs,
            generated_ids_shape=out_ids,
            text_chars=len(text),
            text_preview=text[:120],
        )

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
        if self.step_logger.enabled:
            out["pipeline_trace"] = self.step_logger.to_dict()
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
        train_style_asr: bool = False,
        prompt_asr: bool = False,
        asr_prompt: str | None = None,
        append_im_end: bool = True,
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
            prompt: Optional override; if None, :meth:`build_default_prompt` is used unless
                ``prompt_asr=True``.
            train_style_asr: If True, audio-only prefix (Stage 1–2 eval default). When
                ``append_im_end=True``, uses ``[audio | im_end/BOS] → generate`` (matches
                training). When ``append_im_end=False``, uses ``[audio] → generate``.
                Chat tails after a blank line are trimmed. If False,
                ``[prompt | audio_tokens]`` (Stage 3 target; chat template).
            prompt_asr: If True, use Qwen chat template with :meth:`build_asr_prompt`
                (or ``asr_prompt`` / ``prompt`` override): ``[asr_prompt | audio] → generate``.
            asr_prompt: Custom ASR instruction when ``prompt_asr=True``; defaults to
                :meth:`build_asr_prompt`.
            append_im_end: When ``train_style_asr=True``, append Qwen ``im_end``/BOS after
                audio tokens before ``generate`` (default True, matches training).

        Returns:
            Dictionary with text, tensors, and timing metadata.
        """
        prompt, train_style_asr, trim_asr_tail = self._resolve_asr_generation(
            prompt=prompt,
            train_style_asr=train_style_asr,
            prompt_asr=prompt_asr,
            asr_prompt=asr_prompt,
        )
        self.step_logger.reset()
        self.step_logger.log(
            "generate.start",
            n_windows=n_windows,
            train_style_asr=train_style_asr,
            prompt_asr=prompt_asr,
        )

        enc, windows, encode_s = self.encode_waveform(waveform)
        window_list = self._select_windows(windows, n_windows)
        self.step_logger.log(
            "adapter.select_windows",
            total_windows=int(windows.shape[1]) if windows.ndim == 4 else 0,
            windows_used=len(window_list),
        )

        with torch.no_grad():
            adapter_out = self.streaming_adapter(window_list)

        tokens = adapter_out["tokens"]
        self.step_logger.log(
            "adapter.forward.done",
            compressed_tokens=tokens,
            stability_loss=float(adapter_out.get("stability_loss", 0.0)),
            sparse_loss=(
                float(adapter_out["sparse_loss"])
                if adapter_out.get("sparse_loss") is not None
                else None
            ),
            rate_loss=(
                float(adapter_out["rate_loss"])
                if adapter_out.get("rate_loss") is not None
                else None
            ),
        )
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
            train_style_asr=train_style_asr,
            append_im_end=append_im_end,
            trim_asr_tail=trim_asr_tail,
        )


class WhisperAdapterLLMCommitGatePipeline:
    """
    Same stack as :class:`WhisperAdapterLLMPipeline`, but runs **per-window**
    :meth:`StreamingAdapter.forward_window` and evaluates :class:`~adapter.turn_end_commit_gate.TurnEndCommitGate`
    on the accumulated token sequence—matching the structure used when optimizing ``L_gate``
    in stage 2/3 trainers.

    When ``should_commit`` is true, the pipeline proceeds to LLM generation (turn-end detected).
    Use this when loading checkpoints that include ``gate_state_dict`` or when you care about
    commit probabilities / optional early truncation. For plain adapter-only inference, use
    :class:`WhisperAdapterLLMPipeline`.
    """

    def __init__(
        self,
        *,
        whisper_processor: Any,
        whisper_model: nn.Module,
        windowizer: AudioWaveformWindowizer,
        streaming_adapter: StreamingAdapter,
        early_commit_gate: TurnEndCommitGate,
        llm_model: nn.Module,
        llm_tokenizer: Any,
        device: str,
        torch_dtype: torch.dtype,
        sample_rate: int = 16000,
        step_logger: PipelineStepLogger | None = None,
        verbose: bool = False,
    ) -> None:
        self.step_logger = step_logger or PipelineStepLogger(
            enabled=verbose,
            prefix="WhisperAdapterLLMCommitGatePipeline",
        )
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
            step_logger=self.step_logger,
        )
        self.early_commit_gate = early_commit_gate.to(device=device)

    def encode_waveform(
        self, waveform: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        return self._llm.encode_waveform(waveform)

    def build_default_prompt(self, num_compressed_tokens: int) -> str:
        return self._llm.build_default_prompt(num_compressed_tokens)

    def build_asr_prompt(self) -> str:
        return self._llm.build_asr_prompt()

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
        train_style_asr: bool = False,
        prompt_asr: bool = False,
        asr_prompt: str | None = None,
        append_im_end: bool = True,
    ) -> dict[str, Any]:
        """
        Encode → per-window adapter + gate → LLM.

        Args:
            use_early_commit_truncation: If True, stop consuming further windows once
                ``should_commit`` is true (unless already at the last window). Default False
                uses the full selected window list for the LM (same total audio tokens as the
                non-gate pipeline), while still recording gate diagnostics.
            prompt_asr: If True, use :meth:`build_asr_prompt` via Qwen chat template before
                audio tokens (``[asr_prompt | audio] → generate``).
            asr_prompt: Override for the ASR instruction when ``prompt_asr=True``.
        """
        prompt, train_style_asr, trim_asr_tail = WhisperAdapterLLMPipeline._resolve_asr_generation(
            prompt=prompt,
            train_style_asr=train_style_asr,
            prompt_asr=prompt_asr,
            asr_prompt=asr_prompt,
        )
        self.step_logger.reset()
        self.step_logger.log(
            "generate.start",
            n_windows=n_windows,
            train_style_asr=train_style_asr,
            prompt_asr=prompt_asr,
            use_early_commit_truncation=use_early_commit_truncation,
        )

        enc, windows, encode_s = self._llm.encode_waveform(waveform)
        window_list = WhisperAdapterLLMPipeline._select_windows(windows, n_windows)
        self.step_logger.log(
            "adapter.select_windows",
            total_windows=int(windows.shape[1]) if windows.ndim == 4 else 0,
            windows_used=len(window_list),
        )

        adapter = self._llm.streaming_adapter
        gate = self.early_commit_gate
        adapter.reset_streaming_state()
        silence_tracker = (
            gate.make_silence_tracker() if gate.silence_mode in ("rule", "both") else None
        )
        learned_silence_tracker = (
            gate.make_learned_silence_tracker()
            if gate.silence_mode in ("learned", "both")
            else None
        )

        chunks: list[torch.Tensor] = []
        commit_probs: list[torch.Tensor] = []
        commit_probs_rule: list[torch.Tensor] = []
        commit_probs_learned: list[torch.Tensor] = []
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
                gr = gate(
                    accumulated,
                    t,
                    len(window_list),
                    endpoint_label=None,
                    silence_tracker=silence_tracker,
                    learned_silence_tracker=learned_silence_tracker,
                    window_tokens=step["tokens"],
                )
                commit_probs.append(gr["commit_prob"])
                if "commit_prob_rule" in gr:
                    commit_probs_rule.append(gr["commit_prob_rule"])
                if "commit_prob_learned" in gr:
                    commit_probs_learned.append(gr["commit_prob_learned"])
                gate_losses.append(gr["gate_loss"])
                self.step_logger.log(
                    "gate.window",
                    window_index=t,
                    commit_prob=float(gr["commit_prob"].item()),
                    should_commit=bool(gr["should_commit"].item() > 0.5),
                    accumulated_tokens=accumulated,
                )

                if (
                    use_early_commit_truncation
                    and t < len(window_list) - 1
                    and gr["should_commit"].item() > 0.5
                ):
                    self.step_logger.log(
                        "gate.early_commit_truncation",
                        stopped_at_window=t,
                    )
                    break

            tokens = torch.cat(chunks, dim=1)
            self.step_logger.log(
                "adapter.forward_window.done",
                compressed_tokens=tokens,
                windows_committed=len(chunks),
                stability_loss=float(total_stab.item()),
            )

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
        if commit_probs_rule:
            extra["early_commit_commit_probs_rule"] = torch.stack(commit_probs_rule, dim=0)
        if commit_probs_learned:
            extra["early_commit_commit_probs_learned"] = torch.stack(
                commit_probs_learned, dim=0
            )

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
            train_style_asr=train_style_asr,
            append_im_end=append_im_end,
            trim_asr_tail=trim_asr_tail,
            extra=extra,
        )

    def create_streaming_session(self) -> Any:
        """
        Build a :class:`~adapter_llm_streaming.WhisperAdapterStreamingSession` for live
        per-window KV-cache inference with Qwen3-8B (or the configured causal LM).
        """
        from adapter_llm_streaming import WhisperAdapterStreamingSession

        return WhisperAdapterStreamingSession(
            whisper_processor=self._llm.whisper_processor,
            whisper_model=self._llm.whisper_model,
            streaming_adapter=self._llm.streaming_adapter,
            early_commit_gate=self.early_commit_gate,
            llm_model=self._llm.llm_model,
            llm_tokenizer=self._llm.llm_tokenizer,
            device=self._llm.device,
            torch_dtype=self._llm.torch_dtype,
            sample_rate=self._llm.sample_rate,
        )

    def generate_streaming(
        self,
        waveform: torch.Tensor,
        *,
        n_windows: int = -1,
        generation: LlmGenerationParams | None = None,
        max_new_tokens: int = 100,
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        prompt: str | None = None,
        train_style_asr: bool = False,
        finalize_if_no_commit: bool = True,
    ) -> dict[str, Any]:
        """
        Incremental inference: append each window to the LLM KV-cache, generate on
        ``should_commit``, otherwise decode after the last window when
        ``finalize_if_no_commit=True``.

        Unlike :meth:`generate`, this does **not** batch all audio tokens before a single
        ``generate()`` call.
        """
        from adapter_llm_streaming import run_streaming_session_on_waveform

        return run_streaming_session_on_waveform(
            self,
            waveform,
            n_windows=n_windows,
            prompt=prompt,
            train_style_asr=train_style_asr,
            generation=generation,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            finalize_if_no_commit=finalize_if_no_commit,
        )


def _qwen_device_map_and_max_memory() -> tuple[str | None, dict[int, str] | None]:
    """LLM sharding for multi-GPU eval (matches ``evaluation.eval_stage1``)."""
    from training.utils.config import DeviceConfig
    from training.utils.devices import device_env_has_explicit_index, visible_gpu_count

    n = visible_gpu_count()
    max_memory = DeviceConfig.from_env().llm_max_memory
    if device_env_has_explicit_index():
        return None, None
    if n >= 2 and max_memory:
        return "sequential", max_memory
    if n >= 2:
        return "auto", None
    return None, None


def _build_stage2_asr_pipeline(
    *,
    checkpoint_path: str,
    device: str,
    torch_dtype: torch.dtype,
    stage2: Any,
    model_ids: Any,
    llm_device_map: str | None = None,
    llm_max_memory: dict[int, str] | None = None,
) -> tuple[WhisperAdapterLLMCommitGatePipeline, dict[str, Any]]:
    """
    Load a Stage-2 checkpoint into :class:`WhisperAdapterLLMCommitGatePipeline`.

    Includes the turn-end commit gate and optional rate controller (Stage 2 only).
    """
    from encoder import WhisperConfig, load_whisper_models
    from training.utils.checkpointing import adapt_adapter_state_dict_num_queries, load_gate_state_dict_safe
    from training.utils.loaders import load_frozen_qwen_causal_lm

    whisper = load_whisper_models(
        cfg=WhisperConfig(
            model_id=model_ids.whisper_model_id,
            device=device,
            torch_dtype=torch_dtype,
        )
    )
    qwen = load_frozen_qwen_causal_lm(
        model_id=model_ids.llm_model_id,
        device=device,
        torch_dtype=torch_dtype,
        device_map=llm_device_map,
        max_memory=llm_max_memory,
    )

    windowizer = AudioWaveformWindowizer(
        sample_rate=16000,
        window_seconds=0.8,
        stride_seconds=0.4,
    )

    adapter = StreamingAdapter(
        d_encoder=768,
        d_llm=4096,
        num_queries=2,
        num_layers=2,
        num_heads=4,
        d_ffn=2048,
        dropout=0.0,
        ema_alpha=0.8,
        learnable_ema=False,
        use_rate_controller=stage2.use_rate_controller,
        rate_threshold=0.5,
        target_rate=stage2.rate_target,
    ).to(device, dtype=torch_dtype)

    gate = TurnEndCommitGate(
        d_llm=4096,
        hidden_dim=256,
        threshold=0.5,
        latency_weight=0.1,
        min_silence_ms=200.0,
        require_silence_for_commit=True,
        token_activity_threshold=8.0,
    ).to(device, dtype=torch_dtype)

    ckpt = torch.load(checkpoint_path, map_location=device)
    adapter.load_state_dict(
        adapt_adapter_state_dict_num_queries(ckpt["adapter_state_dict"], adapter.num_queries)
    )
    if "gate_state_dict" not in ckpt:
        raise KeyError(
            f"Stage 2 checkpoint missing gate_state_dict: {checkpoint_path}. "
            "Gate weights are required for Stage-2 inference."
        )
    load_gate_state_dict_safe(gate, ckpt)
    adapter.eval()
    gate.eval()

    meta = {
        "checkpoint": checkpoint_path,
        "stage": 2,
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
        "use_rate_controller": stage2.use_rate_controller,
        "rate_target": stage2.rate_target,
    }

    pipeline = WhisperAdapterLLMCommitGatePipeline(
        whisper_processor=whisper.processor,
        whisper_model=whisper.model,
        windowizer=windowizer,
        streaming_adapter=adapter,
        early_commit_gate=gate,
        llm_model=qwen.causal_lm,
        llm_tokenizer=qwen.tokenizer,
        device=device,
        torch_dtype=torch_dtype,
    )
    return pipeline, meta


def run_stage2_checkpoint_smoke_test(
    *,
    checkpoint_path: str,
    audio_path: str,
    reference: str,
    device: str,
    torch_dtype: torch.dtype,
    n_windows: int = -1,
    max_new_tokens: int = 256,
    train_style_asr: bool = True,
    prompt_asr: bool = False,
    asr_prompt: str | None = None,
    append_im_end: bool = True,
    use_early_commit_truncation: bool = False,
) -> dict[str, Any]:
    """
    Load one Stage-2 checkpoint and run the full commit-gate pipeline on one utterance.

    Returns a summary dict with prediction, reference, checkpoint metadata, and pipeline trace.
    """
    from pathlib import Path

    from dataset import load_mono_waveform_16k
    from llm.config import LlmGenerationParams
    from training.utils.config import FrozenModelIdsConfig, Stage2Config

    stage2 = Stage2Config.from_env()
    model_ids = FrozenModelIdsConfig.from_env()
    if str(device).startswith("cpu"):
        llm_device_map, llm_max_memory = None, None
    else:
        llm_device_map, llm_max_memory = _qwen_device_map_and_max_memory()

    print(f"\n{'=' * 72}", flush=True)
    print(f"Stage-2 smoke test", flush=True)
    print(f"  device:     {device} ({torch_dtype})", flush=True)
    print(f"  checkpoint: {checkpoint_path}", flush=True)
    print(f"  audio:      {audio_path}", flush=True)
    print(f"  reference:  {reference[:120]}{'...' if len(reference) > 120 else ''}", flush=True)
    if prompt_asr:
        resolved_asr_prompt = (
            asr_prompt
            if asr_prompt is not None
            else WhisperAdapterLLMPipeline.build_asr_prompt()
        )
        print(f"  asr prompt: {resolved_asr_prompt}", flush=True)
    print(f"{'=' * 72}", flush=True)

    pipeline, ckpt_meta = _build_stage2_asr_pipeline(
        checkpoint_path=checkpoint_path,
        device=device,
        torch_dtype=torch_dtype,
        model_ids=model_ids,
        stage2=stage2,
        llm_device_map=llm_device_map,
        llm_max_memory=llm_max_memory,
    )
    pipeline.step_logger.enabled = True
    pipeline.step_logger.prefix = f"stage2:{Path(checkpoint_path).stem}"

    waveform = load_mono_waveform_16k(audio_path)
    generation = LlmGenerationParams(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )
    result = pipeline.generate(
        waveform,
        n_windows=n_windows,
        generation=generation,
        train_style_asr=train_style_asr,
        prompt_asr=prompt_asr,
        asr_prompt=asr_prompt,
        append_im_end=append_im_end,
        use_early_commit_truncation=use_early_commit_truncation,
    )

    summary = {
        "checkpoint": checkpoint_path,
        "checkpoint_meta": ckpt_meta,
        "audio_path": audio_path,
        "reference": reference,
        "prompt_asr": prompt_asr,
        "asr_prompt": (
            asr_prompt
            if asr_prompt is not None
            else (WhisperAdapterLLMPipeline.build_asr_prompt() if prompt_asr else None)
        ),
        "prediction": result["text"],
        "num_windows_used": result.get("num_windows_used"),
        "encode_time_s": result.get("encode_time_s"),
        "pipeline_trace": result.get("pipeline_trace"),
        "early_commit_commit_probs": (
            result["early_commit_commit_probs"].tolist()
            if "early_commit_commit_probs" in result
            and result["early_commit_commit_probs"].numel()
            else []
        ),
    }
    print(
        f"\nResult ({Path(checkpoint_path).stem}):\n"
        f"  REF: {reference}\n"
        f"  HYP: {result['text']}\n"
        f"  windows={result.get('num_windows_used')} encode_s={result.get('encode_time_s'):.2f}",
        flush=True,
    )
    return summary


def run_all_stage2_checkpoint_smoke_tests(
    *,
    checkpoints_dir: str,
    dataset_root: str,
    sample_index: int = 0,
    output_json: str | None = None,
    device: str = "cpu",
    torch_dtype: torch.dtype | None = None,
    n_windows: int = -1,
    max_new_tokens: int = 256,
    train_style_asr: bool = True,
    prompt_asr: bool = False,
    asr_prompt: str | None = None,
    append_im_end: bool = True,
    use_early_commit_truncation: bool = False,
) -> list[dict[str, Any]]:
    """
    Run the full Stage-2 pipeline on one LibriSpeech sample for every ``adapter_stage2*.pt``.
    """
    import json
    import os
    from pathlib import Path

    from dataset import LibriSpeechPairs

    resolved_device, resolved_dtype = _resolve_smoke_test_device(device)
    if torch_dtype is None:
        device, torch_dtype = resolved_device, resolved_dtype
    else:
        device = resolved_device
    print(f"Using device={device} dtype={torch_dtype}", flush=True)

    checkpoints = discover_stage2_checkpoints(checkpoints_dir)
    if not checkpoints:
        raise FileNotFoundError(
            f"No Stage-2 checkpoints found in {checkpoints_dir} (expected adapter_stage2*.pt)."
        )

    dataset = LibriSpeechPairs(dataset_root)
    if not dataset.pairs:
        raise FileNotFoundError(f"No LibriSpeech pairs found under {dataset_root}")
    if sample_index < 0 or sample_index >= len(dataset.pairs):
        raise IndexError(
            f"sample_index={sample_index} out of range for {len(dataset.pairs)} utterances"
        )

    audio_path, reference = dataset.pairs[sample_index]
    print(
        f"Dataset sample {sample_index}/{len(dataset.pairs) - 1}: "
        f"{os.path.basename(audio_path)}",
        flush=True,
    )
    print(f"Found {len(checkpoints)} Stage-2 checkpoint(s) in {checkpoints_dir}", flush=True)

    results: list[dict[str, Any]] = []
    for checkpoint_path in checkpoints:
        results.append(
            run_stage2_checkpoint_smoke_test(
                checkpoint_path=checkpoint_path,
                audio_path=audio_path,
                reference=reference,
                device=device,
                torch_dtype=torch_dtype,
                n_windows=n_windows,
                max_new_tokens=max_new_tokens,
                train_style_asr=train_style_asr,
                prompt_asr=prompt_asr,
                asr_prompt=asr_prompt,
                append_im_end=append_im_end,
                use_early_commit_truncation=use_early_commit_truncation,
            )
        )

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset_root": dataset_root,
            "sample_index": sample_index,
            "audio_path": audio_path,
            "reference": reference,
            "checkpoints_dir": checkpoints_dir,
            "results": results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nWrote smoke-test summary to {out_path}", flush=True)

    return results


def _main() -> None:
    import argparse
    import os
    import sys

    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(pkg_root, "src"))
    sys.path.insert(0, pkg_root)

    from dataset import LibriSpeechConfig
    from training.utils.asr_prompt import DEFAULT_ASR_PROMPT
    from training.utils.config import CheckpointConfig
    from training.utils.env import load_project_env

    load_project_env(pkg_root)
    ckpt_cfg = CheckpointConfig.from_env(pkg_root=pkg_root)
    training_dir = os.path.join(pkg_root, "training")
    default_dataset_root = LibriSpeechConfig.test_clean_root(training_dir)

    parser = argparse.ArgumentParser(
        description=(
            "Run step-by-step Stage-2 pipeline smoke tests on one LibriSpeech sample "
            "for all adapter_stage2*.pt checkpoints."
        )
    )
    parser.add_argument("--checkpoints-dir", default=ckpt_cfg.dir)
    parser.add_argument("--dataset-root", default=default_dataset_root)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--n-windows", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Compute device (default: cpu). Use cuda or cuda:0 for GPU.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--early-commit-truncation", action="store_true")
    parser.add_argument(
        "--prompt-asr",
        action="store_true",
        help=(
            "Use Qwen chat ASR prompt before audio tokens "
            "([asr_prompt | audio] → generate) instead of train-style audio-only prefix."
        ),
    )
    parser.add_argument(
        "--asr-prompt",
        type=str,
        default=DEFAULT_ASR_PROMPT,
        help="ASR instruction text when --prompt-asr is set.",
    )
    im_end = parser.add_mutually_exclusive_group()
    im_end.add_argument("--append-im-end", dest="append_im_end", action="store_true", default=True)
    im_end.add_argument("--no-append-im-end", dest="append_im_end", action="store_false")
    args = parser.parse_args()

    run_all_stage2_checkpoint_smoke_tests(
        checkpoints_dir=args.checkpoints_dir,
        dataset_root=args.dataset_root,
        sample_index=args.sample_index,
        output_json=args.output_json,
        device=args.device,
        n_windows=args.n_windows,
        max_new_tokens=args.max_new_tokens,
        train_style_asr=not args.prompt_asr,
        prompt_asr=args.prompt_asr,
        asr_prompt=args.asr_prompt if args.prompt_asr else None,
        append_im_end=args.append_im_end,
        use_early_commit_truncation=args.early_commit_truncation,
    )


if __name__ == "__main__":
    _main()
