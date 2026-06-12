"""
Live streaming inference: per-window KV-cache append + generate on turn-end commit.

Uses frozen Qwen3-8B (or any causal LM) with :class:`llm.kv_cache.LlmKvCacheSession`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from adapter.streaming_adapter import StreamingAdapter
from adapter.turn_end_commit_gate import TurnEndCommitGate
from adapter.windowing import AudioWaveformWindowizer
from llm.config import LlmGenerationParams
from llm.kv_cache import LlmKvCacheSession

from adapter_llm_pipeline import WhisperAdapterLLMPipeline, _truncate_asr_chat_tail


@dataclass
class StreamingWindowStep:
    """Result of processing one audio/encoder window."""

    window_index: int
    window_tokens: torch.Tensor
    commit_prob: float
    should_commit: bool
    committed: bool = False
    generated_ids: torch.Tensor | None = None
    generated_text: str | None = None


@dataclass
class StreamingSessionResult:
    """Aggregate result after a streaming session completes."""

    text: str
    generated_ids: torch.Tensor
    num_windows_processed: int
    num_audio_tokens: int
    committed_at_window: int | None
    window_steps: list[StreamingWindowStep] = field(default_factory=list)
    encode_time_s: float = 0.0
    first_token_time_s: float | None = None
    commit_probs: torch.Tensor | None = None


class WhisperAdapterStreamingSession:
    """
    Incremental streaming session aligned with stage 2/3 training.

    Per window:
        encoder features → adapter → append to LLM KV-cache → turn-end gate
        → on ``should_commit``: decode from cache (Qwen3-8B)

    Call :meth:`begin` once, then :meth:`push_encoder_window` or :meth:`push_waveform`
    for each 0.8s / 0.4s stride chunk.
    """

    def __init__(
        self,
        *,
        whisper_processor: Any,
        whisper_model: nn.Module,
        streaming_adapter: StreamingAdapter,
        early_commit_gate: TurnEndCommitGate,
        llm_model: nn.Module,
        llm_tokenizer: Any,
        device: str,
        torch_dtype: torch.dtype,
        sample_rate: int = 16000,
    ) -> None:
        self.whisper_processor = whisper_processor
        self.whisper_model = whisper_model
        self.streaming_adapter = streaming_adapter
        self.early_commit_gate = early_commit_gate.to(device=device)
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer
        self.device = device
        self.torch_dtype = torch_dtype
        self.sample_rate = sample_rate

        self._kv = LlmKvCacheSession(llm_model, dtype=torch_dtype)
        self._chunks: list[torch.Tensor] = []
        self._steps: list[StreamingWindowStep] = []
        self._commit_probs: list[torch.Tensor] = []
        self._total_windows_hint: int | None = None
        self._committed = False
        self._committed_at: int | None = None
        self._generated_ids: torch.Tensor | None = None
        self._generated_text: str | None = None
        self._first_token_time_s: float | None = None
        self._session_start: float | None = None
        self._silence_tracker = None
        self._learned_silence_tracker = None
        self._train_style_asr = False
        self._train_style_sep_appended = False

    def begin(
        self,
        *,
        prompt: str | None = None,
        train_style_asr: bool = False,
        total_windows_hint: int | None = None,
    ) -> None:
        """
        Reset adapter/gate/KV state and optionally seed the cache with a text prompt.

        Args:
            prompt: User prompt (chat template). Ignored when ``train_style_asr=True``.
            train_style_asr: ``[audio | im_end/BOS]`` prefix, matching stage 1–2 eval/training.
            total__hint: Expected window count for gate normalization; optional for live streams.
        """
        self.streaming_adapter.reset_streaming_state()
        self._kv.reset()
        self._chunks.clear()
        self._steps.clear()
        self._commit_probs.clear()
        self._total_windows_hint = total_windows_hint
        self._committed = False
        self._committed_at = None
        self._generated_ids = None
        self._generated_text = None
        self._first_token_time_s = None
        self._session_start = time.time()
        self._train_style_asr = train_style_asr
        self._train_style_sep_appended = False

        gate = self.early_commit_gate
        self._silence_tracker = (
            gate.make_silence_tracker() if gate.silence_mode in ("rule", "both") else None
        )
        self._learned_silence_tracker = (
            gate.make_learned_silence_tracker()
            if gate.silence_mode in ("learned", "both")
            else None
        )

        if not train_style_asr and prompt is not None:
            prompt_ids = self._tokenize_prompt(prompt)
            prompt_embeds = self.llm_model.get_input_embeddings()(prompt_ids).to(
                device=self._kv._input_device,
                dtype=self.torch_dtype,
            )
            self._kv.append_embeddings(prompt_embeds)

    def push_encoder_window(
        self,
        encoder_features: torch.Tensor,
        *,
        window_index: int | None = None,
        total_windows: int | None = None,
        generation: LlmGenerationParams | None = None,
        max_new_tokens: int = 100,
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        generate_on_commit: bool = True,
    ) -> StreamingWindowStep:
        """
        Adapter + KV append + gate for one encoder window ``(1, T, D_enc)``.
        """
        if self._committed:
            raise RuntimeError("Session already committed; call begin() to start a new turn")

        t = window_index if window_index is not None else len(self._steps)
        total = total_windows if total_windows is not None else self._total_windows_hint
        if total is None:
            total = max(t + 1, 1)

        enc = encoder_features.to(device=self.device, dtype=self.torch_dtype)
        with torch.no_grad():
            step = self.streaming_adapter.forward_window(enc)
            window_tokens = step["tokens"]
            self._kv.append_embeddings(window_tokens)
            self._chunks.append(window_tokens)

            accumulated = torch.cat(self._chunks, dim=1)
            gr = self.early_commit_gate(
                accumulated,
                t,
                total,
                endpoint_label=None,
                silence_tracker=self._silence_tracker,
                learned_silence_tracker=self._learned_silence_tracker,
                window_tokens=window_tokens,
            )

        commit_prob = float(gr["commit_prob"].item())
        should_commit = bool(gr["should_commit"].item() > 0.5)
        self._commit_probs.append(gr["commit_prob"].detach())

        result = StreamingWindowStep(
            window_index=t,
            window_tokens=window_tokens.detach(),
            commit_prob=commit_prob,
            should_commit=should_commit,
        )

        if generate_on_commit and should_commit:
            self._run_generation(
                result,
                generation=generation,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

        self._steps.append(result)
        return result

    def push_waveform(
        self,
        waveform_chunk: torch.Tensor,
        *,
        generation: LlmGenerationParams | None = None,
        max_new_tokens: int = 100,
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        generate_on_commit: bool = True,
    ) -> StreamingWindowStep:
        """Whisper-encode one raw-audio chunk, then :meth:`push_encoder_window`."""
        from encoder.whisper_encoder import encode_waveform_to_hidden

        enc = encode_waveform_to_hidden(
            waveform_chunk,
            whisper_processor=self.whisper_processor,
            whisper_model=self.whisper_model,
            device=self.device,
            torch_dtype=self._torch_dtype,
            sample_rate=self.sample_rate,
        )
        return self.push_encoder_window(
            enc,
            generation=generation,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            generate_on_commit=generate_on_commit,
        )

    def finalize(
        self,
        *,
        generation: LlmGenerationParams | None = None,
        max_new_tokens: int = 100,
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        force_generate: bool = True,
    ) -> StreamingSessionResult:
        """
        Finish the session. If no commit occurred and ``force_generate=True``, decode now.
        """
        if not self._committed and force_generate and self._chunks:
            dummy = StreamingWindowStep(
                window_index=len(self._steps),
                window_tokens=self._chunks[-1],
                commit_prob=0.0,
                should_commit=False,
            )
            self._run_generation(
                dummy,
                generation=generation,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            self._steps.append(dummy)

        text = self._generated_text or ""
        gen_ids = self._generated_ids if self._generated_ids is not None else torch.zeros(
            (1, 0), dtype=torch.long, device=self.device
        )
        num_audio = int(torch.cat(self._chunks, dim=1).shape[1]) if self._chunks else 0
        cp = torch.stack(self._commit_probs, dim=0) if self._commit_probs else None

        return StreamingSessionResult(
            text=text,
            generated_ids=gen_ids,
            num_windows_processed=len(self._steps),
            num_audio_tokens=num_audio,
            committed_at_window=self._committed_at,
            window_steps=list(self._steps),
            first_token_time_s=self._first_token_time_s,
            commit_probs=cp,
        )

    @property
    def committed(self) -> bool:
        return self._committed

    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        messages = [{"role": "user", "content": prompt}]
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
        return tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

    def _run_generation(
        self,
        step: StreamingWindowStep,
        *,
        generation: LlmGenerationParams | None,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
    ) -> None:
        if self._session_start is not None and self._first_token_time_s is None:
            self._first_token_time_s = time.time() - self._session_start

        if self._train_style_asr and not self._train_style_sep_appended:
            sep_id = WhisperAdapterLLMPipeline.train_style_separator_token_id(self.llm_tokenizer)
            sep_ids = torch.tensor([[sep_id]], device=self._kv._input_device, dtype=torch.long)
            sep_embed = self.llm_model.get_input_embeddings()(sep_ids).to(
                device=self._kv._input_device, dtype=self.torch_dtype
            )
            no_think_embed = WhisperAdapterLLMPipeline.qwen_no_think_suffix_embeds(
                self.llm_model, self._kv._input_device, self.torch_dtype
            )
            self._kv.append_embeddings(sep_embed)
            self._kv.append_embeddings(no_think_embed)
            self._train_style_sep_appended = True

        gen_ids = self._kv.generate(
            self.llm_tokenizer,
            generation=generation,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        text = self.llm_tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        if self._train_style_asr:
            text = _truncate_asr_chat_tail(text)

        step.committed = True
        step.generated_ids = gen_ids
        step.generated_text = text
        self._committed = True
        self._committed_at = step.window_index
        self._generated_ids = gen_ids
        self._generated_text = text


def run_streaming_session_on_waveform(
    pipeline: Any,
    waveform: torch.Tensor,
    *,
    n_windows: int = -1,
    prompt: str | None = None,
    train_style_asr: bool = False,
    generation: LlmGenerationParams | None = None,
    max_new_tokens: int = 100,
    do_sample: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    finalize_if_no_commit: bool = True,
) -> dict[str, Any]:
    """
    Drive :class:`WhisperAdapterStreamingSession` over pre-windowed audio from a pipeline.

    ``pipeline`` must expose ``encode_waveform``, ``build_default_prompt``, and the
    model handles needed by :meth:`WhisperAdapterLLMCommitGatePipeline.create_streaming_session`.
    """
    t0 = time.time()
    enc, windows, encode_s = pipeline.encode_waveform(waveform)
    window_list = WhisperAdapterLLMPipeline._select_windows(windows, n_windows)

    session = pipeline.create_streaming_session()
    if train_style_asr:
        session.begin(train_style_asr=True, total_windows_hint=len(window_list))
    else:
        num_compressed = len(window_list) * pipeline._llm.streaming_adapter.num_queries
        user_prompt = prompt if prompt is not None else pipeline.build_default_prompt(num_compressed)
        session.begin(prompt=user_prompt, total_windows_hint=len(window_list))

    last_step: StreamingWindowStep | None = None
    with torch.no_grad():
        for t, w in enumerate(window_list):
            wdev = w.to(device=pipeline._llm.device, dtype=pipeline._llm.torch_dtype)
            last_step = session.push_encoder_window(
                wdev,
                window_index=t,
                total_windows=len(window_list),
                generation=generation,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            if session.committed:
                break

    result = session.finalize(
        generation=generation,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        force_generate=finalize_if_no_commit and not session.committed,
    )

    tokens = torch.cat(session._chunks, dim=1) if session._chunks else torch.zeros(
        1, 0, pipeline._llm.streaming_adapter.d_llm, device=pipeline._llm.device
    )

    out: dict[str, Any] = {
        "text": result.text,
        "generated_ids": result.generated_ids,
        "encode_time_s": encode_s,
        "stream_setup_time_s": time.time() - t0 - encode_s,
        "first_token_time_s": result.first_token_time_s,
        "enc": enc,
        "windows": windows,
        "tokens": tokens,
        "num_windows_used": result.num_windows_processed,
        "num_windows_committed": result.committed_at_window,
        "num_audio_tokens": result.num_audio_tokens,
        "streaming": True,
        "committed_on_gate": session.committed and result.committed_at_window is not None,
        "window_steps": result.window_steps,
    }
    if result.commit_probs is not None:
        out["early_commit_commit_probs"] = result.commit_probs
    if last_step is not None:
        out["last_commit_prob"] = last_step.commit_prob
    return out
