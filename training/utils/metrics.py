from __future__ import annotations

import math
import re
from typing import Any


class RunningMean:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def mean(self) -> float:
        return self.total / max(1, self.count)

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


def _bleu_sentence_sacrebleu(reference: str, hypothesis: str) -> float:
    """Sentence BLEU in [0, 100] (sacrebleu convention)."""
    try:
        from sacrebleu.metrics import BLEU
    except ImportError as e:
        raise ImportError(
            "BLEU metrics require `sacrebleu`. Install with: pip install sacrebleu"
        ) from e

    ref = reference.strip()
    hyp = hypothesis.strip()
    if not ref or not hyp:
        return 0.0
    bleu = BLEU(effective_order=True)
    return float(bleu.sentence_score(hyp, [ref]).score)


def bleu_sentence_0_100(reference: str, hypothesis: str) -> float:
    """Single-reference sentence BLEU, score in ``[0, 100]``."""
    return _bleu_sentence_sacrebleu(reference, hypothesis)


def bleu_sentence_0_1(reference: str, hypothesis: str) -> float:
    """Same as :func:`bleu_sentence_0_100` scaled to ``[0, 1]``."""
    return _bleu_sentence_sacrebleu(reference, hypothesis) / 100.0


def metrics_transcription_vs_response(
    reference_transcription: str,
    model_response: str,
) -> dict[str, Any]:
    """
    Text metrics when the **reference** is a ground-truth (or teacher) **transcript**
    and **model_response** is decoded LM output (e.g. after text-only or mixed prompt).
    """
    b = bleu_sentence_0_100(reference_transcription, model_response)
    return {
        "bleu_transcript_vs_response_0_100": b,
        "bleu_transcript_vs_response_0_1": b / 100.0,
    }


def metrics_reference_vs_response_after_compressed_audio(
    reference_transcription: str,
    model_response: str,
) -> dict[str, Any]:
    """
    Same BLEU as :func:`metrics_transcription_vs_response`, named for the case where
    the LM was conditioned on **compressed audio tokens** (adapter output) plus a prompt;
    the reference remains a string transcript (or other textual reference).
    """
    out = metrics_transcription_vs_response(reference_transcription, model_response)
    return {
        **out,
        "bleu_compressed_audio_path_reference_vs_response_0_100": out[
            "bleu_transcript_vs_response_0_100"
        ],
        "bleu_compressed_audio_path_reference_vs_response_0_1": out[
            "bleu_transcript_vs_response_0_1"
        ],
    }


_WS_RE = re.compile(r"\s+")


def normalize_asr_text(text: str) -> str:
    return _WS_RE.sub(" ", text.strip())


def _tokenize_asr_words(text: str) -> list[str]:
    normalized = normalize_asr_text(text).lower()
    return normalized.split() if normalized else []


def _edit_distance_words(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, word_a in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, word_b in enumerate(b, start=1):
            cur = dp[j]
            cost = 0 if word_a == word_b else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word error rate in ``[0, 1+]`` (0 = perfect match)."""
    ref_words = _tokenize_asr_words(reference)
    hyp_words = _tokenize_asr_words(hypothesis)
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _edit_distance_words(ref_words, hyp_words) / float(len(ref_words))


def corpus_bleu4(references: list[str], hypotheses: list[str]) -> float:
    """Corpus BLEU-4 in ``[0, 1]`` (same formula as ``evaluation/eval_stage1.py``)."""
    clipped = [0, 0, 0, 0]
    total = [0, 0, 0, 0]
    ref_len = 0
    hyp_len = 0

    def count_ngrams(tokens: list[str], n: int) -> dict[tuple[str, ...], int]:
        out: dict[tuple[str, ...], int] = {}
        if n <= 0 or len(tokens) < n:
            return out
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            out[ngram] = out.get(ngram, 0) + 1
        return out

    for reference, hypothesis in zip(references, hypotheses):
        ref_tokens = _tokenize_asr_words(reference)
        hyp_tokens = _tokenize_asr_words(hypothesis)
        ref_len += len(ref_tokens)
        hyp_len += len(hyp_tokens)
        for n in range(1, 5):
            ref_ngrams = count_ngrams(ref_tokens, n)
            hyp_ngrams = count_ngrams(hyp_tokens, n)
            total[n - 1] += max(len(hyp_tokens) - n + 1, 0)
            for ngram, count in hyp_ngrams.items():
                clipped[n - 1] += min(count, ref_ngrams.get(ngram, 0))

    if hyp_len == 0:
        return 0.0
    brevity_penalty = (
        1.0 if hyp_len > ref_len else math.exp(1.0 - float(ref_len) / float(hyp_len))
    )
    log_precision = 0.0
    for n in range(4):
        precision = (clipped[n] + 1.0) / (total[n] + 1.0)
        log_precision += 0.25 * math.log(precision)
    return float(brevity_penalty * math.exp(log_precision))
