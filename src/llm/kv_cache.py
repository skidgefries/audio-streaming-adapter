"""
Incremental ``inputs_embeds`` → KV-cache helpers for frozen causal LMs (Qwen3-8B).

Used by :class:`adapter_llm_streaming.WhisperAdapterStreamingSession` to append
compressed audio tokens per window without re-encoding the prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import GenerationConfig

from .config import LlmGenerationParams, build_hf_generation_config


def llm_input_device(model: torch.nn.Module) -> torch.device:
    """Device for ``inputs_embeds`` / ``input_ids`` when the LM uses ``hf_device_map``."""
    hf_map = getattr(model, "hf_device_map", None)
    if hf_map:
        for key in ("model.embed_tokens", "embed_tokens", "transformer.wte"):
            if key in hf_map:
                dev = hf_map[key]
                if isinstance(dev, int):
                    return torch.device(f"cuda:{dev}")
                return torch.device(dev)
    return next(model.parameters()).device


@dataclass
class LlmKvCacheState:
    past_key_values: Any | None
    attention_mask: torch.Tensor | None
    seq_len: int
    last_logits: torch.Tensor | None


class LlmKvCacheSession:
    """
    Append embedding chunks to a frozen causal LM and decode from the accumulated cache.

    Typical flow::

        session = LlmKvCacheSession(model, dtype=torch.float16)
        session.append_embeddings(prompt_embeds)
        session.append_embeddings(window_tokens)  # per streaming window
        token_ids = session.generate(tokenizer, generation=LlmGenerationParams(...))
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.model = model
        self.dtype = dtype
        self._input_device = llm_input_device(model)
        self.reset()

    def reset(self) -> None:
        self._past_key_values: Any | None = None
        self._attention_mask: torch.Tensor | None = None
        self._seq_len = 0
        self._last_logits: torch.Tensor | None = None

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def has_cache(self) -> bool:
        return self._past_key_values is not None

    def state(self) -> LlmKvCacheState:
        return LlmKvCacheState(
            past_key_values=self._past_key_values,
            attention_mask=self._attention_mask,
            seq_len=self._seq_len,
            last_logits=self._last_logits,
        )

    def append_embeddings(self, embeds: torch.Tensor) -> torch.Tensor:
        """
        Run one LM forward on ``embeds`` ``(batch, seq, hidden)`` and extend the KV cache.

        Returns:
            Logits for this chunk, shape ``(batch, seq, vocab)``.
        """
        if embeds.ndim != 3:
            raise ValueError(f"Expected embeds (batch, seq, hidden), got {tuple(embeds.shape)}")
        if embeds.shape[0] != 1:
            raise ValueError("LlmKvCacheSession currently supports batch size 1")

        chunk = embeds.to(device=self._input_device, dtype=self.dtype)
        batch, seq_len, _ = chunk.shape
        chunk_mask = torch.ones(batch, seq_len, dtype=torch.long, device=chunk.device)

        if self._attention_mask is None:
            attention_mask = chunk_mask
        else:
            attention_mask = torch.cat([self._attention_mask, chunk_mask], dim=1)

        outputs = self.model(
            inputs_embeds=chunk,
            attention_mask=attention_mask,
            past_key_values=self._past_key_values,
            use_cache=True,
            return_dict=True,
        )

        self._past_key_values = outputs.past_key_values
        self._attention_mask = attention_mask
        self._seq_len += seq_len
        self._last_logits = outputs.logits
        return outputs.logits

    def generate(
        self,
        tokenizer: Any,
        *,
        generation: LlmGenerationParams | None = None,
        max_new_tokens: int = 100,
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Sample/decode ``max_new_tokens`` from the current cache.

        Returns:
            ``(1, num_generated)`` token ids (new tokens only, not including prefix).
        """
        if self._last_logits is None or self._past_key_values is None:
            raise RuntimeError("Cannot generate before append_embeddings has been called")

        if generation is None:
            generation = LlmGenerationParams(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        gen_cfg = build_hf_generation_config(
            model=self.model,
            tokenizer=tokenizer,
            params=generation,
        )

        eos_id = gen_cfg.eos_token_id
        generated: list[torch.Tensor] = []

        logits = self._last_logits[:, -1, :]
        next_token = _sample_token(logits, gen_cfg)
        generated.append(next_token)

        past = self._past_key_values
        attention_mask = self._attention_mask
        assert attention_mask is not None

        for _ in range(int(gen_cfg.max_new_tokens) - 1):
            if eos_id is not None and int(next_token.item()) == int(eos_id):
                break

            next_token = next_token.view(1, 1).to(self._input_device)
            attention_mask = torch.cat(
                [attention_mask, torch.ones(1, 1, dtype=torch.long, device=attention_mask.device)],
                dim=1,
            )
            outputs = self.model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            next_token = _sample_token(outputs.logits[:, -1, :], gen_cfg)
            generated.append(next_token)

        if not generated:
            return torch.zeros((1, 0), dtype=torch.long, device=self._input_device)

        return torch.stack(generated, dim=1)


def _sample_token(logits: torch.Tensor, gen_cfg: GenerationConfig) -> torch.Tensor:
    if gen_cfg.do_sample:
        temp = float(gen_cfg.temperature) if gen_cfg.temperature is not None else 1.0
        scaled = logits / max(temp, 1e-5)
        if gen_cfg.top_k is not None and int(gen_cfg.top_k) > 0:
            top_k = int(gen_cfg.top_k)
            values, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
            scaled = scaled.masked_fill(scaled < values[:, [-1]], float("-inf"))
        if gen_cfg.top_p is not None and float(gen_cfg.top_p) < 1.0:
            sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            remove = cumulative > float(gen_cfg.top_p)
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            scaled = sorted_logits.scatter(1, sorted_idx, sorted_logits)
        probs = F.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    return logits.argmax(dim=-1)
