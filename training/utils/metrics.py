from __future__ import annotations

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
