"""Tests for incremental LLM KV-cache streaming."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import torch
import torch.nn as nn

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from adapter.streaming_adapter import StreamingAdapter
from adapter.turn_end_commit_gate import TurnEndCommitGate
from llm.kv_cache import LlmKvCacheSession
from llm.config import LlmGenerationParams


class _TinyCacheLM(nn.Module):
    """Minimal causal LM stub that supports inputs_embeds + past_key_values."""

    def __init__(self, hidden: int = 32, vocab: int = 64) -> None:
        super().__init__()
        self.hidden = hidden
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self._layer = nn.Linear(hidden, hidden)

    def forward(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool = False,
        return_dict: bool = True,
    ):
        del attention_mask
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed(input_ids)
        hidden = self._layer(inputs_embeds)
        logits = self.lm_head(hidden)
        past_len = 0
        if past_key_values is not None:
            past_len = int(past_key_values[0][0].shape[-2])
        new_past = (
            (torch.zeros(1, 1, past_len + hidden.shape[1], self.hidden),),
        )
        out = MagicMock()
        out.logits = logits
        out.past_key_values = new_past if use_cache else None
        return out


def test_llm_kv_cache_append_and_generate():
    model = _TinyCacheLM(hidden=32, vocab=64).eval()
    session = LlmKvCacheSession(model, dtype=torch.float32)

    chunk_a = torch.randn(1, 4, 32)
    chunk_b = torch.randn(1, 3, 32)
    session.append_embeddings(chunk_a)
    assert session.seq_len == 4
    session.append_embeddings(chunk_b)
    assert session.seq_len == 7

    tok = MagicMock()
    tok.eos_token_id = 1
    ids = session.generate(tok, generation=LlmGenerationParams(max_new_tokens=5, do_sample=False))
    assert ids.shape[0] == 1
    assert ids.shape[1] >= 1


def test_streaming_session_commits_and_generates():
    from adapter_llm_streaming import WhisperAdapterStreamingSession

    adapter = StreamingAdapter(
        d_encoder=768,
        d_llm=32,
        num_queries=2,
        num_layers=1,
        num_heads=2,
        d_ffn=64,
        use_rate_controller=False,
    ).eval()
    gate = TurnEndCommitGate(d_llm=32, require_silence_for_commit=False, threshold=0.0).eval()
    llm = _TinyCacheLM(hidden=32, vocab=64).eval()
    tok = MagicMock()
    tok.eos_token_id = 1
    tok.decode = lambda ids, skip_special_tokens=True: "hello"

    session = WhisperAdapterStreamingSession(
        whisper_processor=MagicMock(),
        whisper_model=MagicMock(),
        streaming_adapter=adapter,
        early_commit_gate=gate,
        llm_model=llm,
        llm_tokenizer=tok,
        device="cpu",
        torch_dtype=torch.float32,
    )
    session.begin(train_style_asr=True, total_windows_hint=2)

    enc = torch.randn(1, 8, 768)
    step = session.push_encoder_window(enc, window_index=0, total_windows=2, max_new_tokens=4)
    assert step.window_tokens.shape == (1, 2, 32)
    assert session.committed or not step.should_commit

    result = session.finalize(force_generate=True, max_new_tokens=4)
    assert result.num_windows_processed >= 1
    assert isinstance(result.text, str)


if __name__ == "__main__":
    test_llm_kv_cache_append_and_generate()
    print("[PASS] LlmKvCacheSession append + generate")
    test_streaming_session_commits_and_generates()
    print("[PASS] WhisperAdapterStreamingSession")
