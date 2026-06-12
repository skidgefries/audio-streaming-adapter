"""
Analyze ASR evaluation outputs (``asr_predictions.json``).

Computes summary statistics, word-alignment error rates (WER, MER, SR/DR/IR,
recall/precision/F1), transcript-containment / LLM-paraphrase detection,
correlations, and diagnostic plots. Writes results under ``asr_analysis/`` in the
experiment folder (parent of the predictions file).

Usage::

    uv run evaluation/analyze_asr_predictions.py outputs/experiments/adapter_stage2_step34500_100
    uv run evaluation/analyze_asr_predictions.py outputs/experiments --recursive
    uv run evaluation/analyze_asr_predictions.py path/to/asr_predictions.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None

_WS_RE = re.compile(r"\s+")

CATEGORY_EXACT = "exact_match"
CATEGORY_TRANSCRIPT_FOUND = "transcript_found_in_prediction"
CATEGORY_SOME_WORDS = "some_words_matched"

CATEGORY_ORDER = (CATEGORY_EXACT, CATEGORY_TRANSCRIPT_FOUND, CATEGORY_SOME_WORDS)

CATEGORY_LABELS = {
    CATEGORY_EXACT: "Exact match",
    CATEGORY_TRANSCRIPT_FOUND: "Transcript found in prediction",
    CATEGORY_SOME_WORDS: "Partial match (not full transcript)",
}

CATEGORY_COLORS = {
    CATEGORY_EXACT: "#54A24B",
    CATEGORY_TRANSCRIPT_FOUND: "#4C78A8",
    CATEGORY_SOME_WORDS: "#F58518",
}

CATEGORY_SHORT_LABELS = {
    CATEGORY_EXACT: "Exact",
    CATEGORY_TRANSCRIPT_FOUND: "Transcript",
    CATEGORY_SOME_WORDS: "Partial",
}

# Ordered-token recall at or above this threshold counts as the full transcript
# being present in the prediction (possibly with extra LLM text around it).
TRANSCRIPT_FOUND_RECALL_THRESHOLD = 0.85

# WER histogram covers [0, WER_HIST_MAX] in WER_BIN_WIDTH steps; samples above go to
# asr_analysis_wer_outliers.json and appear as an "others" bar after 5.0.
WER_HIST_MAX = 5.0
WER_BIN_WIDTH = 0.1

# Coarse WER bins for substitution / deletion / insertion rate breakdown.
WER_RATE_BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
WER_RATE_BIN_LABELS = tuple(
    f"{lo:g}-{hi:g}" for lo, hi in zip(WER_RATE_BIN_EDGES[:-1], WER_RATE_BIN_EDGES[1:])
)
WER_DISPLAY_MAX = 1.0
WER_OVER_BIN_LABEL = "> 1"


@dataclass(frozen=True)
class AlignmentCounts:
    n: int
    substitutions: int
    deletions: int
    insertions: int
    correct: int


@dataclass(frozen=True)
class NumericSummary:
    count: int
    mean: float | None
    std: float | None
    min: float | None
    p25: float | None
    median: float | None
    p75: float | None
    p90: float | None
    p95: float | None
    max: float | None
    weighted_mean: float | None = None


def _normalize_text(s: str) -> str:
    return _WS_RE.sub(" ", s.strip())


def _tokenize_words(s: str) -> list[str]:
    s = _normalize_text(s).lower()
    return s.split() if s else []


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, q))


def _summarize(values: list[float], weights: list[float] | None = None) -> NumericSummary:
    if not values:
        return NumericSummary(
            count=0,
            mean=None,
            std=None,
            min=None,
            p25=None,
            median=None,
            p75=None,
            p90=None,
            p95=None,
            max=None,
            weighted_mean=None,
        )
    wmean = None
    if weights and len(weights) == len(values) and sum(weights) > 0:
        wmean = float(sum(v * w for v, w in zip(values, weights)) / sum(weights))
    std = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return NumericSummary(
        count=len(values),
        mean=float(statistics.mean(values)),
        std=std,
        min=float(min(values)),
        p25=_percentile(values, 25),
        median=float(statistics.median(values)),
        p75=_percentile(values, 75),
        p90=_percentile(values, 90),
        p95=_percentile(values, 95),
        max=float(max(values)),
        weighted_mean=wmean,
    )


def _summary_to_dict(summary: NumericSummary) -> dict[str, Any]:
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(summary).items()}


def _word_alignment_counts(ref_tokens: list[str], hyp_tokens: list[str]) -> AlignmentCounts:
    """Levenshtein backtrace: S, D, I, C with N = len(ref)."""
    n_ref = len(ref_tokens)
    n_hyp = len(hyp_tokens)
    if n_ref == 0:
        return AlignmentCounts(
            n=0,
            substitutions=0,
            deletions=0,
            insertions=n_hyp,
            correct=0,
        )

    dp = [[0] * (n_hyp + 1) for _ in range(n_ref + 1)]
    for i in range(1, n_ref + 1):
        dp[i][0] = i
    for j in range(1, n_hyp + 1):
        dp[0][j] = j
    for i in range(1, n_ref + 1):
        for j in range(1, n_hyp + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + cost,
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )

    substitutions = deletions = insertions = correct = 0
    i, j = n_ref, n_hyp
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and ref_tokens[i - 1] == hyp_tokens[j - 1]
            and dp[i][j] == dp[i - 1][j - 1]
        ):
            correct += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1

    return AlignmentCounts(
        n=n_ref,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        correct=correct,
    )


def _error_metrics_from_alignment(counts: AlignmentCounts) -> dict[str, float | int]:
    n = counts.n
    s = counts.substitutions
    d = counts.deletions
    i = counts.insertions
    c = counts.correct
    mer_den = s + d + i + c
    prec_den = c + s + i

    wer = (s + d + i) / float(n) if n else (1.0 if i else 0.0)
    wer_without_insertions = (s + d) / float(n) if n else 0.0
    match_error_rate = (s + d + i) / float(mer_den) if mer_den else 0.0
    substitution_rate = s / float(n) if n else 0.0
    deletion_rate = d / float(n) if n else 0.0
    insertion_rate = i / float(n) if n else 0.0
    recall = c / float(n) if n else 1.0
    precision = c / float(prec_den) if prec_den else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "ref_word_count_align": n,
        "num_substitutions": s,
        "num_deletions": d,
        "num_insertions": i,
        "num_correct": c,
        "wer_recomputed": wer,
        "wer_without_insertions": wer_without_insertions,
        "match_error_rate": match_error_rate,
        "substitution_rate": substitution_rate,
        "deletion_rate": deletion_rate,
        "insertion_rate": insertion_rate,
        "asr_recall": recall,
        "asr_precision": precision,
        "asr_f1": f1,
    }


def _alignment_error_metrics(reference: str, prediction: str) -> dict[str, float | int]:
    ref_tokens = _tokenize_words(reference)
    hyp_tokens = _tokenize_words(prediction)
    counts = _word_alignment_counts(ref_tokens, hyp_tokens)
    return _error_metrics_from_alignment(counts)


def _corpus_alignment_totals(rows: list[dict[str, Any]]) -> AlignmentCounts:
    total = AlignmentCounts(n=0, substitutions=0, deletions=0, insertions=0, correct=0)
    for row in rows:
        ref_tokens = _tokenize_words(row.get("reference", ""))
        hyp_tokens = _tokenize_words(row.get("prediction", ""))
        counts = _word_alignment_counts(ref_tokens, hyp_tokens)
        total = AlignmentCounts(
            n=total.n + counts.n,
            substitutions=total.substitutions + counts.substitutions,
            deletions=total.deletions + counts.deletions,
            insertions=total.insertions + counts.insertions,
            correct=total.correct + counts.correct,
        )
    return total


def _wer_rate_bin_label(wer: float) -> str:
    if wer > WER_RATE_BIN_EDGES[-1]:
        return WER_OVER_BIN_LABEL
    for lo, hi in zip(WER_RATE_BIN_EDGES[:-1], WER_RATE_BIN_EDGES[1:]):
        if lo <= wer < hi or (hi == WER_RATE_BIN_EDGES[-1] and lo <= wer <= hi):
            return f"{lo:g}-{hi:g}"
    return WER_RATE_BIN_LABELS[-1]


def _set_wer_axis_ticks(ax: plt.Axes, *, axis: str = "x") -> None:
    """Use 0–1 ticks with the last tick labeled ``> 1`` for overflow values."""
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, WER_DISPLAY_MAX]
    labels = ["0", "0.2", "0.4", "0.6", "0.8", WER_OVER_BIN_LABEL]
    if axis == "x":
        ax.set_xlim(0, WER_DISPLAY_MAX + 0.05)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    else:
        ax.set_ylim(0, WER_DISPLAY_MAX + 0.05)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)


def _build_wer_rate_bin_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bins: dict[str, list[dict[str, Any]]] = {label: [] for label in WER_RATE_BIN_LABELS}
    bins[WER_OVER_BIN_LABEL] = []
    for row in rows:
        wer = row.get("wer")
        if wer is None:
            continue
        bins[_wer_rate_bin_label(float(wer))].append(row)

    total = max(len(rows), 1)
    out_bins: dict[str, Any] = {}
    plot_bins = list(WER_RATE_BIN_LABELS)
    if bins[WER_OVER_BIN_LABEL]:
        plot_bins.append(WER_OVER_BIN_LABEL)
    for label in (*WER_RATE_BIN_LABELS, WER_OVER_BIN_LABEL):
        bucket = bins[label]
        n = len(bucket)
        sr_vals = [float(r["substitution_rate"]) for r in bucket if r.get("substitution_rate") is not None]
        dr_vals = [float(r["deletion_rate"]) for r in bucket if r.get("deletion_rate") is not None]
        ir_vals = [float(r["insertion_rate"]) for r in bucket if r.get("insertion_rate") is not None]
        out_bins[label] = {
            "count": n,
            "rate": n / total,
            "mean_wer": (
                float(statistics.mean(float(r["wer"]) for r in bucket))
                if bucket
                else None
            ),
            "substitution_rate": _summary_to_dict(_summarize(sr_vals)),
            "deletion_rate": _summary_to_dict(_summarize(dr_vals)),
            "insertion_rate": _summary_to_dict(_summarize(ir_vals)),
        }

    return {
        "bin_edges": list(WER_RATE_BIN_EDGES),
        "bins": out_bins,
        "plot_bins": plot_bins,
        "overflow_bin_label": WER_OVER_BIN_LABEL,
    }


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i, ta in enumerate(a, start=1):
        cur = [0]
        for j, tb in enumerate(b, start=1):
            if ta == tb:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _ordered_token_recall(reference: str, prediction: str) -> float:
    ref = _tokenize_words(reference)
    if not ref:
        return 1.0
    hyp = _tokenize_words(prediction)
    return _lcs_length(ref, hyp) / float(len(ref))


def _corpus_bleu4(references: list[str], hypotheses: list[str]) -> float:
    clipped = [0, 0, 0, 0]
    total = [0, 0, 0, 0]
    ref_len = 0
    hyp_len = 0

    def count_ngrams(tokens: list[str], n: int) -> dict[tuple[str, ...], int]:
        out: dict[tuple[str, ...], int] = {}
        if n <= 0 or len(tokens) < n:
            return out
        for i in range(len(tokens) - n + 1):
            ng = tuple(tokens[i : i + n])
            out[ng] = out.get(ng, 0) + 1
        return out

    for ref, hyp in zip(references, hypotheses):
        r_tok = _tokenize_words(ref)
        h_tok = _tokenize_words(hyp)
        ref_len += len(r_tok)
        hyp_len += len(h_tok)
        for n in range(1, 5):
            ref_ng = count_ngrams(r_tok, n)
            hyp_ng = count_ngrams(h_tok, n)
            total[n - 1] += max(len(h_tok) - n + 1, 0)
            for ng, c in hyp_ng.items():
                clipped[n - 1] += min(c, ref_ng.get(ng, 0))

    if hyp_len == 0:
        return 0.0
    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - float(ref_len) / float(hyp_len))
    log_p = 0.0
    for n in range(4):
        p = (clipped[n] + 1.0) / (total[n] + 1.0)
        log_p += 0.25 * math.log(p)
    return float(bp * math.exp(log_p))


def _bag_token_recall(reference: str, prediction: str) -> float:
    ref = _tokenize_words(reference)
    if not ref:
        return 1.0
    hyp = _tokenize_words(prediction)
    matched, _, _ = _word_match_detail_from_tokens(ref, hyp)
    return matched / float(len(ref))


def _word_match_detail_from_tokens(
    ref_tokens: list[str],
    pred_tokens: list[str],
) -> tuple[int, dict[str, int], dict[str, int]]:
    """Return matched token count, per-word matched counts, and unmatched ref counts."""
    ref_counts = Counter(ref_tokens)
    hyp_counts = Counter(pred_tokens)
    matched_word_counts: dict[str, int] = {}
    unmatched_ref_counts: dict[str, int] = {}
    matched_total = 0
    for word, ref_n in ref_counts.items():
        overlap = min(ref_n, hyp_counts.get(word, 0))
        if overlap:
            matched_word_counts[word] = overlap
            matched_total += overlap
        if ref_n > overlap:
            unmatched_ref_counts[word] = ref_n - overlap
    return matched_total, matched_word_counts, unmatched_ref_counts


def _word_match_detail(reference: str, prediction: str) -> dict[str, Any]:
    ref_tokens = _tokenize_words(reference)
    pred_tokens = _tokenize_words(prediction)
    matched_total, matched_word_counts, unmatched_ref_counts = _word_match_detail_from_tokens(
        ref_tokens, pred_tokens
    )
    ref_word_count = len(ref_tokens)
    distinct_matched = len(matched_word_counts)
    return {
        "ref_word_count": ref_word_count,
        "pred_word_count": len(pred_tokens),
        "num_matched_words": matched_total,
        "num_distinct_matched_words": distinct_matched,
        "num_unmatched_ref_words": ref_word_count - matched_total,
        "match_rate": matched_total / float(ref_word_count) if ref_word_count else 0.0,
        "matched_words": sorted(matched_word_counts.keys()),
        "matched_word_counts": dict(sorted(matched_word_counts.items())),
        "unmatched_ref_words": sorted(unmatched_ref_counts.keys()),
        "unmatched_ref_word_counts": dict(sorted(unmatched_ref_counts.items())),
    }


def _reference_substring(reference: str, prediction: str) -> bool:
    ref = _normalize_text(reference).lower()
    pred = _normalize_text(prediction).lower()
    return bool(ref) and ref in pred


def _quoted_reference_in_prediction(reference: str, prediction: str) -> bool:
    ref = _normalize_text(reference)
    if not ref:
        return False
    pred = prediction
    # Check quoted spans (straight and curly quotes).
    for m in re.finditer(r'["“”\'](.+?)["“”\']', pred, flags=re.DOTALL):
        quote = _normalize_text(m.group(1))
        if not quote:
            continue
        if quote.lower() == ref.lower():
            return True
        if _ordered_token_recall(ref, quote) >= 0.85:
            return True
    return False


def classify_prediction(reference: str, prediction: str, wer: float | None) -> dict[str, Any]:
    """Assign one of three mutually exclusive categories (priority order).

    1. exact_match — normalized reference equals prediction, or WER is 0.
    2. transcript_found_in_prediction — full reference appears inside the prediction
       (substring), appears quoted, or >=85% of reference words appear in order.
    3. some_words_matched — not exact and the full transcript is not found; only
       some reference words appear in the prediction (or none at all).
    """
    ordered_recall = _ordered_token_recall(reference, prediction)
    bag_recall = _bag_token_recall(reference, prediction)
    substring = _reference_substring(reference, prediction)
    quoted = _quoted_reference_in_prediction(reference, prediction)
    ref_norm = _normalize_text(reference).lower()
    pred_norm = _normalize_text(prediction).lower()

    transcript_found = (
        substring
        or quoted
        or ordered_recall >= TRANSCRIPT_FOUND_RECALL_THRESHOLD
    )

    if (wer is not None and wer == 0.0) or (ref_norm and ref_norm == pred_norm):
        category = CATEGORY_EXACT
    elif transcript_found:
        category = CATEGORY_TRANSCRIPT_FOUND
    else:
        category = CATEGORY_SOME_WORDS

    return {
        "ordered_token_recall": ordered_recall,
        "bag_token_recall": bag_recall,
        "reference_substring": substring,
        "reference_quoted_in_prediction": quoted,
        "transcript_found_in_prediction": transcript_found,
        "prediction_category": category,
        "prediction_category_label": CATEGORY_LABELS[category],
    }


def _audio_duration_s(audio_path: str | None) -> float | None:
    if not audio_path or librosa is None:
        return None
    path = Path(audio_path)
    if not path.is_file():
        return None
    try:
        return float(librosa.get_duration(path=path))
    except Exception:
        return None


def _load_predictions(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "items" in payload:
        items = payload["items"]
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError(f"Unsupported predictions format in {path}")

    metrics_path = path.parent / "asr_metrics.json"
    meta = None
    if metrics_path.is_file():
        with open(metrics_path, encoding="utf-8") as f:
            meta = json.load(f)
    return items, meta


def _enrich_item(item: dict[str, Any]) -> dict[str, Any]:
    ref = item.get("reference", "")
    pred = item.get("prediction", "")
    ref_tokens = _tokenize_words(ref)
    pred_tokens = _tokenize_words(pred)
    audio_duration = _audio_duration_s(item.get("audio_path"))
    latency = item.get("latency_s")
    windows = item.get("num_windows_used")

    enriched = dict(item)
    enriched.update(
        {
            "ref_token_count": len(ref_tokens),
            "pred_token_count": len(pred_tokens),
            "token_length_ratio": (
                len(pred_tokens) / float(len(ref_tokens)) if ref_tokens else None
            ),
            "audio_duration_s": audio_duration,
            "latency_per_audio_s": (
                latency / audio_duration if latency is not None and audio_duration else None
            ),
            "latency_per_window_s": (
                latency / windows if latency is not None and windows else None
            ),
            "windows_per_audio_s": (
                windows / audio_duration if windows is not None and audio_duration else None
            ),
        }
    )
    enriched.update(classify_prediction(ref, pred, item.get("wer")))
    enriched.update(_alignment_error_metrics(ref, pred))
    return enriched


def _category_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(r["prediction_category"] for r in rows)
    return {cat: counts.get(cat, 0) for cat in CATEGORY_ORDER}


def _wer_histogram_bin_edges() -> np.ndarray:
    return np.arange(0.0, WER_HIST_MAX + WER_BIN_WIDTH, WER_BIN_WIDTH)


def _split_wer_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    in_range: list[dict[str, Any]] = []
    outliers: list[dict[str, Any]] = []
    for row in rows:
        wer = row.get("wer")
        if wer is None:
            in_range.append(row)
            continue
        if float(wer) > WER_HIST_MAX:
            outliers.append(row)
        else:
            in_range.append(row)
    outliers.sort(key=lambda r: float(r["wer"]), reverse=True)
    return in_range, outliers


def _build_partial_match_word_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    partial_rows = [r for r in rows if r.get("prediction_category") == CATEGORY_SOME_WORDS]
    items: list[dict[str, Any]] = []
    for row in partial_rows:
        word_detail = _word_match_detail(row.get("reference", ""), row.get("prediction", ""))
        items.append(
            {
                "utterance_id": row.get("utterance_id"),
                "audio_path": row.get("audio_path"),
                "wer": row.get("wer"),
                "bleu4": row.get("bleu4"),
                "reference": row.get("reference"),
                "prediction": row.get("prediction"),
                **word_detail,
            }
        )
    items.sort(key=lambda x: (x["num_matched_words"], x["match_rate"]), reverse=True)

    matched_counts = [int(i["num_matched_words"]) for i in items]
    distinct_counts = [int(i["num_distinct_matched_words"]) for i in items]
    match_rates = [float(i["match_rate"]) for i in items]
    ref_counts = [int(i["ref_word_count"]) for i in items]
    return {
        "category": CATEGORY_SOME_WORDS,
        "category_label": CATEGORY_LABELS[CATEGORY_SOME_WORDS],
        "count": len(items),
        "summary": {
            "num_matched_words": _summary_to_dict(_summarize([float(v) for v in matched_counts])),
            "num_distinct_matched_words": _summary_to_dict(
                _summarize([float(v) for v in distinct_counts])
            ),
            "match_rate": _summary_to_dict(_summarize(match_rates, ref_counts)),
            "with_zero_matches": sum(1 for i in items if i["num_matched_words"] == 0),
            "with_some_matches": sum(1 for i in items if i["num_matched_words"] > 0),
        },
        "items": items,
    }


def _build_wer_split_summary(
    in_range: list[dict[str, Any]],
    outliers: list[dict[str, Any]],
) -> dict[str, Any]:
    in_range_wers = [float(r["wer"]) for r in in_range if r.get("wer") is not None]
    outlier_wers = [float(r["wer"]) for r in outliers if r.get("wer") is not None]
    outlier_weights = [
        float(r["ref_token_count"])
        for r in outliers
        if r.get("wer") is not None and r.get("ref_token_count")
    ]
    in_range_weights = [
        float(r["ref_token_count"])
        for r in in_range
        if r.get("wer") is not None and r.get("ref_token_count")
    ]
    total = max(len(in_range) + len(outliers), 1)
    return {
        "histogram": {
            "min": 0.0,
            "max": WER_HIST_MAX,
            "bin_width": WER_BIN_WIDTH,
            "bin_edges": [round(float(x), 1) for x in _wer_histogram_bin_edges()],
        },
        "in_range": {
            "wer_max_inclusive": WER_HIST_MAX,
            "count": len(in_range),
            "rate": len(in_range) / total,
            "summary": _summary_to_dict(_summarize(in_range_wers, in_range_weights)),
        },
        "outliers": {
            "wer_threshold_exclusive": WER_HIST_MAX,
            "count": len(outliers),
            "rate": len(outliers) / total,
            "summary": _summary_to_dict(_summarize(outlier_wers, outlier_weights)),
            "analysis_file": "asr_analysis_wer_outliers.json",
        },
    }


def build_analysis(
    items: list[dict[str, Any]],
    *,
    source_predictions: str,
    meta: dict[str, Any] | None,
) -> dict[str, Any]:
    rows = [_enrich_item(item) for item in items]

    metric_keys = {
        "wer": "wer",
        "bleu4": "bleu4",
        "latency_s": "latency_s",
        "num_windows_used": "num_windows_used",
        "ref_token_count": "ref_token_count",
        "pred_token_count": "pred_token_count",
        "token_length_ratio": "token_length_ratio",
        "audio_duration_s": "audio_duration_s",
        "latency_per_audio_s": "latency_per_audio_s",
        "latency_per_window_s": "latency_per_window_s",
        "windows_per_audio_s": "windows_per_audio_s",
        "ordered_token_recall": "ordered_token_recall",
        "bag_token_recall": "bag_token_recall",
        "wer_without_insertions": "wer_without_insertions",
        "match_error_rate": "match_error_rate",
        "substitution_rate": "substitution_rate",
        "deletion_rate": "deletion_rate",
        "insertion_rate": "insertion_rate",
        "asr_recall": "asr_recall",
        "asr_precision": "asr_precision",
        "asr_f1": "asr_f1",
    }

    summaries: dict[str, Any] = {}
    for out_key, row_key in metric_keys.items():
        vals = [float(r[row_key]) for r in rows if r.get(row_key) is not None]
        weights = None
        if out_key in (
            "wer",
            "bleu4",
            "ordered_token_recall",
            "bag_token_recall",
            "wer_without_insertions",
            "match_error_rate",
            "substitution_rate",
            "deletion_rate",
            "insertion_rate",
            "asr_recall",
            "asr_precision",
            "asr_f1",
        ):
            weights = [
                float(r["ref_token_count"])
                for r in rows
                if r.get(row_key) is not None and r.get("ref_token_count")
            ]
            if len(weights) != len(vals):
                weights = None
        summaries[out_key] = _summary_to_dict(_summarize(vals, weights))

    categories = _category_counts(rows)
    n = max(len(rows), 1)
    wer_in_range, wer_outliers = _split_wer_rows(rows)
    wer_split = _build_wer_split_summary(wer_in_range, wer_outliers)

    reported = None
    if meta and "metrics" in meta:
        reported = meta["metrics"]

    refs = [r["reference"] for r in rows]
    hyps = [r["prediction"] for r in rows]
    corpus_bleu4 = _corpus_bleu4(refs, hyps)
    corpus_align = _corpus_alignment_totals(rows)
    corpus_error = _error_metrics_from_alignment(corpus_align)
    wer_rate_bins = _build_wer_rate_bin_analysis(rows)

    return {
        "source_predictions": source_predictions,
        "num_samples": len(rows),
        "reported_metrics": reported,
        "recomputed_metrics": {
            "avg_wer": summaries["wer"]["mean"],
            "weighted_avg_wer": summaries["wer"]["weighted_mean"],
            "avg_sentence_bleu4": summaries["bleu4"]["mean"],
            "weighted_avg_sentence_bleu4": summaries["bleu4"]["weighted_mean"],
            "corpus_bleu4": corpus_bleu4,
            "corpus_wer": corpus_error["wer_recomputed"],
            "corpus_wer_without_insertions": corpus_error["wer_without_insertions"],
            "corpus_match_error_rate": corpus_error["match_error_rate"],
            "corpus_substitution_rate": corpus_error["substitution_rate"],
            "corpus_deletion_rate": corpus_error["deletion_rate"],
            "corpus_insertion_rate": corpus_error["insertion_rate"],
            "corpus_asr_recall": corpus_error["asr_recall"],
            "corpus_asr_precision": corpus_error["asr_precision"],
            "corpus_asr_f1": corpus_error["asr_f1"],
            "alignment_totals": {
                "ref_words": corpus_align.n,
                "substitutions": corpus_align.substitutions,
                "deletions": corpus_align.deletions,
                "insertions": corpus_align.insertions,
                "correct": corpus_align.correct,
            },
        },
        "error_rate_formulas": {
            "wer": "(S + D + I) / N",
            "wer_without_insertions": "(S + D) / N",
            "match_error_rate": "(S + D + I) / (S + D + I + C)",
            "substitution_rate": "S / N",
            "deletion_rate": "D / N",
            "insertion_rate": "I / N",
            "asr_recall": "(N - S - D) / N = C / N",
            "asr_precision": "C / (C + S + I)",
            "asr_f1": "2 * P * R / (P + R)",
        },
        "wer_rate_bins": wer_rate_bins,
        "prediction_categories": {
            "counts": categories,
            "rates": {cat: categories[cat] / n for cat in CATEGORY_ORDER},
            "labels": CATEGORY_LABELS,
            "rules": {
                CATEGORY_EXACT: "WER is 0 or normalized reference equals prediction.",
                CATEGORY_TRANSCRIPT_FOUND: (
                    "Reference appears as a substring, appears quoted in the prediction, "
                    f"or ordered token recall >= {TRANSCRIPT_FOUND_RECALL_THRESHOLD:.0%}."
                ),
                CATEGORY_SOME_WORDS: (
                    "Not exact match and full transcript not found: only some reference "
                    "words appear in the prediction, or none."
                ),
            },
        },
        "summaries": summaries,
        "wer_split": wer_split,
        "meta": meta.get("meta") if meta else None,
        "items": rows,
        "wer_outliers": wer_outliers,
    }


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _plot_wer_hist(
    ax: plt.Axes,
    in_range_wers: list[float],
    outlier_count: int,
) -> None:
    if not in_range_wers and outlier_count == 0:
        ax.set_title("WER distribution (no data)")
        return

    edges = _wer_histogram_bin_edges()
    ax.hist(
        in_range_wers,
        bins=edges,
        color="#4C78A8",
        edgecolor="white",
        alpha=0.9,
        label=f"0–{WER_HIST_MAX:g}",
    )
    if in_range_wers:
        ax.axvline(
            statistics.mean(in_range_wers),
            color="#E45756",
            linestyle="--",
            linewidth=1.5,
            label="mean (in range)",
        )
        ax.axvline(
            statistics.median(in_range_wers),
            color="#54A24B",
            linestyle=":",
            linewidth=1.5,
            label="median (in range)",
        )

    if outlier_count > 0:
        others_x = WER_HIST_MAX + WER_BIN_WIDTH * 2
        ax.bar(
            others_x,
            outlier_count,
            width=WER_BIN_WIDTH * 1.5,
            color="#E45756",
            edgecolor="white",
            alpha=0.9,
            label=f"> {WER_HIST_MAX:g} (n={outlier_count})",
        )
        ax.set_xticks(list(np.arange(0, WER_HIST_MAX + WER_BIN_WIDTH, 0.5)) + [others_x])
        ax.set_xticklabels(
            [f"{x:g}" for x in np.arange(0, WER_HIST_MAX + WER_BIN_WIDTH, 0.5)]
            + [f">{WER_HIST_MAX:g}"]
        )
    else:
        ax.set_xticks(np.arange(0, WER_HIST_MAX + WER_BIN_WIDTH, 0.5))

    ax.set_xlim(0, WER_HIST_MAX + (WER_BIN_WIDTH * 4 if outlier_count else WER_BIN_WIDTH))
    ax.set_title(f"WER distribution (0–{WER_HIST_MAX:g}, bin={WER_BIN_WIDTH:g})")
    ax.set_xlabel("WER")
    ax.set_ylabel("count")
    ax.legend(loc="best", fontsize=8)


def _plot_hist(ax: plt.Axes, values: list[float], title: str, xlabel: str, bins: int = 30) -> None:
    if not values:
        ax.set_title(f"{title} (no data)")
        return
    ax.hist(values, bins=bins, color="#4C78A8", edgecolor="white", alpha=0.9)
    ax.axvline(statistics.mean(values), color="#E45756", linestyle="--", linewidth=1.5, label="mean")
    ax.axvline(statistics.median(values), color="#54A24B", linestyle=":", linewidth=1.5, label="median")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.legend(loc="best", fontsize=8)


def _plot_scatter(
    ax: plt.Axes,
    xs: list[float],
    ys: list[float],
    title: str,
    xlabel: str,
    ylabel: str,
    colors: list[str] | None = None,
) -> None:
    if not xs:
        ax.set_title(f"{title} (no data)")
        return
    if colors:
        ax.scatter(xs, ys, c=colors, alpha=0.75, s=28, edgecolors="none")
    else:
        ax.scatter(xs, ys, alpha=0.75, s=28, color="#4C78A8", edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _category_legend(ax: plt.Axes) -> None:
    handles = [
        Patch(facecolor=CATEGORY_COLORS[cat], label=CATEGORY_LABELS[cat])
        for cat in CATEGORY_ORDER
    ]
    ax.legend(handles=handles, loc="best", fontsize=8)


def _values_by_category(
    rows: list[dict[str, Any]],
    metric_key: str,
    *,
    wer_max: float | None = None,
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {cat: [] for cat in CATEGORY_ORDER}
    for row in rows:
        val = row.get(metric_key)
        cat = row.get("prediction_category")
        if val is None or cat not in out:
            continue
        fval = float(val)
        if wer_max is not None and fval > wer_max:
            continue
        out[cat].append(fval)
    return out


def _plot_metric_hist_by_category(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    metric_key: str,
    title: str,
    xlabel: str,
    *,
    bins: np.ndarray | int,
    wer_max: float | None = None,
) -> None:
    by_cat = _values_by_category(rows, metric_key, wer_max=wer_max)
    has_data = any(by_cat[cat] for cat in CATEGORY_ORDER)
    if not has_data:
        ax.set_title(f"{title} (no data)")
        return

    stat_lines: list[str] = []
    for cat in CATEGORY_ORDER:
        vals = by_cat[cat]
        if not vals:
            continue
        ax.hist(
            vals,
            bins=bins,
            alpha=0.55,
            color=CATEGORY_COLORS[cat],
            edgecolor="white",
            label=CATEGORY_LABELS[cat],
        )
        mean_v = statistics.mean(vals)
        med_v = statistics.median(vals)
        color = CATEGORY_COLORS[cat]
        ax.axvline(mean_v, color=color, linestyle="--", linewidth=1.3, alpha=0.95)
        ax.axvline(med_v, color=color, linestyle=":", linewidth=1.3, alpha=0.95)
        stat_lines.append(
            f"{CATEGORY_SHORT_LABELS[cat]}: mean={mean_v:.3f}, median={med_v:.3f}"
        )

    if stat_lines:
        ax.text(
            0.98,
            0.98,
            "\n".join(stat_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            family="monospace",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    style_handles = [
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.3, label="mean"),
        Line2D([0], [0], color="gray", linestyle=":", linewidth=1.3, label="median"),
    ]
    cat_handles, cat_labels = ax.get_legend_handles_labels()
    ax.legend(cat_handles + style_handles, cat_labels + ["mean", "median"], loc="best", fontsize=7)


def _plot_wer_bleu_histograms_by_category(
    ax_wer: plt.Axes,
    ax_bleu: plt.Axes,
    rows: list[dict[str, Any]],
) -> None:
    wer_bins = _wer_histogram_bin_edges()
    bleu_bins = np.arange(0.0, 1.0 + 0.05, 0.05)
    _plot_metric_hist_by_category(
        ax_wer,
        rows,
        "wer",
        f"WER by category (0–{WER_HIST_MAX:g}, bin={WER_BIN_WIDTH:g})",
        "WER",
        bins=wer_bins,
        wer_max=WER_HIST_MAX,
    )
    _plot_metric_hist_by_category(
        ax_bleu,
        rows,
        "bleu4",
        "BLEU-4 by category",
        "BLEU-4",
        bins=bleu_bins,
    )


def generate_plots(analysis: dict[str, Any], out_dir: Path) -> None:
    rows = analysis["items"]

    def _vals(key: str) -> list[float]:
        return [float(r[key]) for r in rows if r.get(key) is not None]

    wer_in_range, wer_outliers = _split_wer_rows(rows)
    in_range_wers = [float(r["wer"]) for r in wer_in_range if r.get("wer") is not None]

    # 1. Core metric histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    _plot_wer_hist(axes[0, 0], in_range_wers, len(wer_outliers))
    _plot_hist(axes[0, 1], _vals("bleu4"), "BLEU-4 distribution", "BLEU-4")
    _plot_hist(axes[1, 0], _vals("latency_s"), "Latency distribution", "latency (s)")
    _plot_hist(axes[1, 1], _vals("num_windows_used"), "Windows used distribution", "num_windows_used")
    fig.tight_layout()
    p = out_dir / "metrics_histograms.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)

    # 2. Token and audio length histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    _plot_hist(axes[0, 0], _vals("ref_token_count"), "Reference token length", "tokens")
    _plot_hist(axes[0, 1], _vals("pred_token_count"), "Prediction token length", "tokens")
    _plot_hist(axes[1, 0], _vals("token_length_ratio"), "Pred/ref token ratio", "ratio")
    _plot_hist(axes[1, 1], _vals("audio_duration_s"), "Audio duration", "seconds")
    fig.tight_layout()
    p = out_dir / "length_histograms.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)

    # 3. Latency relationships
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x_audio, y_lat = zip(
        *[
            (r["audio_duration_s"], r["latency_s"])
            for r in rows
            if r.get("audio_duration_s") is not None and r.get("latency_s") is not None
        ]
    ) or ([], [])
    _plot_scatter(axes[0], list(x_audio), list(y_lat), "Latency vs audio duration", "audio (s)", "latency (s)")
    x_win, y_lat2 = zip(
        *[
            (r["num_windows_used"], r["latency_s"])
            for r in rows
            if r.get("num_windows_used") is not None and r.get("latency_s") is not None
        ]
    ) or ([], [])
    _plot_scatter(axes[1], list(x_win), list(y_lat2), "Latency vs windows", "windows", "latency (s)")
    fig.tight_layout()
    p = out_dir / "latency_scatter.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)

    # 4. WER and BLEU histograms by prediction category
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_wer_bleu_histograms_by_category(axes[0], axes[1], rows)
    fig.tight_layout()
    p = out_dir / "wer_vs_bleu_by_category.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)

    # 5. Transcript containment (same 3 categories, count + rate)
    cats = analysis["prediction_categories"]["counts"]
    rates = analysis["prediction_categories"]["rates"]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CATEGORY_ORDER))
    bar_labels = [CATEGORY_LABELS[cat] for cat in CATEGORY_ORDER]
    bar_counts = [cats[cat] for cat in CATEGORY_ORDER]
    bar_colors = [CATEGORY_COLORS[cat] for cat in CATEGORY_ORDER]
    bars = ax1.bar(x, bar_counts, color=bar_colors)
    ax1.set_ylabel("count")
    ax1.set_title("Transcript containment by category")
    ax1.set_xticks(x)
    ax1.set_xticklabels(bar_labels, rotation=15, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        [100 * rates[cat] for cat in CATEGORY_ORDER],
        color="#E45756",
        marker="o",
        linewidth=1.5,
        label="rate (%)",
    )
    ax2.set_ylabel("rate (%)")
    ax2.set_ylim(0, 100)
    for bar, count in zip(bars, bar_counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    p = out_dir / "transcript_containment.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)

    # 6. Ordered token recall vs WER
    fig, ax = plt.subplots(figsize=(8, 6))
    paired = [
        (float(r["ordered_token_recall"]), float(r["wer"]))
        for r in rows
        if r.get("ordered_token_recall") is not None and r.get("wer") is not None
    ]
    if paired:
        xs2, ys2_raw = zip(*paired)
        ys2 = [min(y, WER_DISPLAY_MAX) for y in ys2_raw]
        overflow = [y > WER_DISPLAY_MAX for y in ys2_raw]
        if any(overflow):
            in_range = [i for i, o in enumerate(overflow) if not o]
            over = [i for i, o in enumerate(overflow) if o]
            if in_range:
                ax.scatter(
                    [xs2[i] for i in in_range],
                    [ys2[i] for i in in_range],
                    alpha=0.75,
                    s=28,
                    color="#4C78A8",
                    edgecolors="none",
                )
            ax.scatter(
                [xs2[i] for i in over],
                [ys2[i] for i in over],
                alpha=0.75,
                s=36,
                color="#E45756",
                edgecolors="black",
                linewidths=0.5,
                label=WER_OVER_BIN_LABEL,
            )
            ax.legend(loc="best", fontsize=8)
        else:
            _plot_scatter(ax, list(xs2), list(ys2), "Ordered token recall vs WER", "ordered recall", "WER")
        ax.set_title("Ordered token recall vs WER")
        ax.set_xlabel("ordered recall")
        ax.set_ylabel("WER")
        _set_wer_axis_ticks(ax, axis="y")
    fig.tight_layout()
    p = out_dir / "recall_vs_wer.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)

    # 7. WER vs WER without insertions
    fig, ax = plt.subplots(figsize=(8, 6))
    paired_wer = [
        (float(r["wer"]), float(r["wer_without_insertions"]))
        for r in rows
        if r.get("wer") is not None and r.get("wer_without_insertions") is not None
    ]
    if paired_wer:
        xs_raw, ys_w = zip(*paired_wer)
        xs_w = [min(x, WER_DISPLAY_MAX) for x in xs_raw]
        overflow = [x > WER_DISPLAY_MAX for x in xs_raw]
        if any(overflow):
            in_range = [i for i, o in enumerate(overflow) if not o]
            over = [i for i, o in enumerate(overflow) if o]
            if in_range:
                ax.scatter(
                    [xs_w[i] for i in in_range],
                    [ys_w[i] for i in in_range],
                    alpha=0.75,
                    s=28,
                    color="#4C78A8",
                    edgecolors="none",
                )
            ax.scatter(
                [xs_w[i] for i in over],
                [ys_w[i] for i in over],
                alpha=0.75,
                s=36,
                color="#E45756",
                edgecolors="black",
                linewidths=0.5,
                label=WER_OVER_BIN_LABEL,
            )
            ax.legend(loc="best", fontsize=8)
        else:
            ax.scatter(xs_w, ys_w, alpha=0.75, s=28, color="#4C78A8", edgecolors="none")
        ax.plot(
            [0, WER_DISPLAY_MAX],
            [0, WER_DISPLAY_MAX],
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )
        ax.set_title("WER vs WER without insertions")
        ax.set_xlabel("WER = (S+D+I)/N")
        ax.set_ylabel("WER without I = (S+D)/N")
        _set_wer_axis_ticks(ax, axis="x")
        ax.set_ylim(0, min(max(ys_w) * 1.05, WER_DISPLAY_MAX + 0.05))
    fig.tight_layout()
    p = out_dir / "wer_vs_wer_without_insertions.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)

    # 8. Substitution / deletion / insertion rates by coarse WER bin
    wer_rate_bins = analysis.get("wer_rate_bins", {})
    plot_bins: list[str] = wer_rate_bins.get("plot_bins", list(WER_RATE_BIN_LABELS))
    bin_data = wer_rate_bins.get("bins", {})
    sr_means = [
        (bin_data.get(label, {}).get("substitution_rate") or {}).get("mean")
        for label in plot_bins
    ]
    dr_means = [
        (bin_data.get(label, {}).get("deletion_rate") or {}).get("mean")
        for label in plot_bins
    ]
    ir_means = [
        (bin_data.get(label, {}).get("insertion_rate") or {}).get("mean")
        for label in plot_bins
    ]
    if any(v is not None for v in (*sr_means, *dr_means, *ir_means)):
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(plot_bins))
        width = 0.25
        ax.bar(x - width, [v or 0.0 for v in sr_means], width, label="SR = S/N", color="#E45756")
        ax.bar(x, [v or 0.0 for v in dr_means], width, label="DR = D/N", color="#F58518")
        ax.bar(x + width, [v or 0.0 for v in ir_means], width, label="IR = I/N", color="#4C78A8")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_bins, rotation=15, ha="right")
        ax.set_xlabel("WER bin")
        ax.set_ylabel("rate")
        ax.set_title("Substitution / deletion / insertion rates by WER bin")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        p = out_dir / "sr_dr_ir_by_wer_bin.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)


def analyze_predictions_file(predictions_path: Path, output_dir: Path | None = None) -> Path:
    predictions_path = predictions_path.resolve()
    if predictions_path.is_dir():
        predictions_path = predictions_path / "asr_predictions.json"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Missing predictions file: {predictions_path}")

    items, meta = _load_predictions(predictions_path)
    analysis = build_analysis(
        items,
        source_predictions=str(predictions_path),
        meta=meta,
    )

    out_dir = output_dir or (predictions_path.parent / "asr_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {k: v for k, v in analysis.items() if k not in ("items", "wer_outliers")}
    _save_json(out_dir / "asr_analysis_items.json", {"items": analysis["items"]})
    outliers = analysis["wer_outliers"]
    _save_json(
        out_dir / "asr_analysis_wer_outliers.json",
        {
            "wer_threshold_exclusive": WER_HIST_MAX,
            "count": len(outliers),
            "summary": summary["wer_split"]["outliers"]["summary"],
            "prediction_categories": _category_counts(outliers),
            "items": outliers,
        },
    )
    partial_match = _build_partial_match_word_analysis(analysis["items"])
    _save_json(out_dir / "asr_analysis_partial_match_words.json", partial_match)
    summary["partial_match_words"] = {
        "count": partial_match["count"],
        "analysis_file": "asr_analysis_partial_match_words.json",
        "with_zero_matches": partial_match["summary"]["with_zero_matches"],
        "with_some_matches": partial_match["summary"]["with_some_matches"],
    }
    _save_json(out_dir / "asr_analysis_summary.json", summary)
    generate_plots(analysis, out_dir)

    print(f"\nAnalyzed {len(items)} samples from {predictions_path}")
    rec = summary["recomputed_metrics"]
    print(
        f"  avg WER={rec['avg_wer']:.4f} "
        f"(weighted={rec['weighted_avg_wer']:.4f}, corpus={rec['corpus_wer']:.4f})"
    )
    print(
        f"  WER w/o I={rec['corpus_wer_without_insertions']:.4f}  "
        f"MER={rec['corpus_match_error_rate']:.4f}  "
        f"SR/DR/IR={rec['corpus_substitution_rate']:.4f}/"
        f"{rec['corpus_deletion_rate']:.4f}/{rec['corpus_insertion_rate']:.4f}"
    )
    print(
        f"  recall={rec['corpus_asr_recall']:.4f}  "
        f"precision={rec['corpus_asr_precision']:.4f}  "
        f"F1={rec['corpus_asr_f1']:.4f}"
    )
    print(f"  corpus BLEU-4={rec['corpus_bleu4']:.4f}")
    cats = summary["prediction_categories"]["counts"]
    for cat in CATEGORY_ORDER:
        print(
            f"  {CATEGORY_LABELS[cat]}: {cats[cat]}/{summary['num_samples']} "
            f"({100 * summary['prediction_categories']['rates'][cat]:.1f}%)"
        )
    wer_split = summary["wer_split"]
    print(
        f"  WER in 0–{WER_HIST_MAX:g}: {wer_split['in_range']['count']}/{summary['num_samples']} "
        f"({100 * wer_split['in_range']['rate']:.1f}%)"
    )
    print(
        f"  WER outliers (> {WER_HIST_MAX:g}): {wer_split['outliers']['count']}/{summary['num_samples']} "
        f"({100 * wer_split['outliers']['rate']:.1f}%) → asr_analysis_wer_outliers.json"
    )
    pm = summary["partial_match_words"]
    print(
        f"  Partial match word detail: {pm['count']} samples "
        f"({pm['with_some_matches']} with matches, {pm['with_zero_matches']} with none) "
        f"→ asr_analysis_partial_match_words.json"
    )
    print(f"  wrote analysis to {out_dir}")
    return out_dir


def discover_prediction_files(root: Path, recursive: bool) -> list[Path]:
    if root.is_file() and root.name == "asr_predictions.json":
        return [root]
    if (root / "asr_predictions.json").is_file():
        return [root / "asr_predictions.json"]
    if not recursive:
        raise FileNotFoundError(
            f"No asr_predictions.json in {root}; pass --recursive to search subfolders"
        )
    return sorted(root.rglob("asr_predictions.json"))


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Analyze asr_predictions.json: metrics, transcript containment, plots"
    )
    ap.add_argument(
        "path",
        type=str,
        help="Experiment folder, asr_predictions.json path, or parent folder with --recursive",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="Find all asr_predictions.json files under PATH",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: <experiment>/asr_analysis)",
    )
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    root = Path(args.path)
    files = discover_prediction_files(root, args.recursive)
    if not files:
        raise SystemExit(f"No asr_predictions.json files found under {root}")

    for pred_path in files:
        out = Path(args.output_dir) if args.output_dir else None
        analyze_predictions_file(pred_path, output_dir=out)


if __name__ == "__main__":
    main()
