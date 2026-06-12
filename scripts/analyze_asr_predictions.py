"""
Bin ASR predictions by WER / BLEU-4 and compute aggregate metrics.

Reads ``asr_predictions.json`` from an eval run directory and writes:

- ``analysis/wer_bins/{0.0-0.1,...}.json`` — utterances per WER bucket
- ``analysis/bleu_bins/{0.0-0.1,...}.json`` — utterances per BLEU bucket
- ``analysis/wer_bins_summary.json`` / ``bleu_bins_summary.json`` — counts per bucket
- ``analysis/aggregate_metrics.json`` — simple + corpus (length-weighted) averages
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_pkg_root, "src"))
sys.path.insert(0, _pkg_root)

from evaluation.eval_stage1 import (  # noqa: E402
    _corpus_bleu4,
    _edit_distance,
    _tokenize_words,
    _wer,
)

BIN_LABELS = [f"{i / 10:.1f}-{(i + 1) / 10:.1f}" for i in range(10)]


def _bin_label(value: float) -> str:
    """Map score in [0, +inf) to one of ten 0.1-wide bins (last bin catches >= 0.9)."""
    if value >= 1.0:
        return BIN_LABELS[9]
    idx = min(int(value * 10), 9)
    return BIN_LABELS[idx]


def _empty_bins() -> dict[str, list[dict]]:
    return {label: [] for label in BIN_LABELS}


def _bin_items(items: list[dict], *, key: str) -> dict[str, list[dict]]:
    bins = _empty_bins()
    for item in items:
        score = float(item[key])
        bins[_bin_label(score)].append(item)
    return bins


def _bins_summary(bins: dict[str, list[dict]], *, metric: str) -> dict[str, dict]:
    total = sum(len(v) for v in bins.values())
    out: dict[str, dict] = {}
    for label in BIN_LABELS:
        bucket = bins[label]
        n = len(bucket)
        out[label] = {
            "count": n,
            "fraction": (n / total) if total else 0.0,
            f"mean_{metric}": (
                sum(float(x[metric]) for x in bucket) / n if n else None
            ),
        }
    out["_total"] = {"count": total}
    return out


def _corpus_wer(items: list[dict]) -> float:
    """Length-weighted WER: total edit errors / total reference words."""
    total_ref = 0
    total_err = 0
    for item in items:
        ref_tok = _tokenize_words(item["reference"])
        hyp_tok = _tokenize_words(item["prediction"])
        total_ref += len(ref_tok)
        total_err += _edit_distance(ref_tok, hyp_tok)
    if total_ref == 0:
        return 0.0
    return total_err / float(total_ref)


def _aggregate_metrics(items: list[dict]) -> dict:
    n = len(items)
    refs = [x["reference"] for x in items]
    hyps = [x["prediction"] for x in items]

    mean_wer = sum(float(x["wer"]) for x in items) / n if n else 0.0
    mean_bleu = sum(float(x["bleu4"]) for x in items) / n if n else 0.0
    corpus_wer = _corpus_wer(items)
    corpus_bleu = _corpus_bleu4(refs, hyps)

    # Recompute per-utterance WER from text (sanity) — use stored wer for mean
    ref_lengths = [len(_tokenize_words(r)) for r in refs]
    total_ref_len = sum(ref_lengths)
    weighted_wer = (
        sum(float(x["wer"]) * lw for x, lw in zip(items, ref_lengths)) / total_ref_len
        if total_ref_len
        else 0.0
    )

    return {
        "num_samples": n,
        "mean_wer": mean_wer,
        "mean_bleu4": mean_bleu,
        "corpus_wer": corpus_wer,
        "corpus_bleu4": corpus_bleu,
        "length_weighted_mean_wer": weighted_wer,
        "notes": {
            "mean_wer": "Unweighted average of per-utterance WER (micro).",
            "mean_bleu4": "Unweighted average of per-utterance BLEU-4 (micro).",
            "corpus_wer": "Total edit distance / total reference words (macro, length-weighted).",
            "corpus_bleu4": "Corpus-level BLEU-4 over all n-grams (macro).",
            "length_weighted_mean_wer": "sum(wer_i * |ref_i|) / sum(|ref_i|); equals corpus_wer when WER is exact per utterance.",
        },
    }


def analyze_run_dir(run_dir: Path) -> Path:
    pred_path = run_dir / "asr_predictions.json"
    if not pred_path.is_file():
        raise FileNotFoundError(f"Missing {pred_path}")

    payload = json.loads(pred_path.read_text(encoding="utf-8"))
    items: list[dict] = payload.get("items", [])
    if not items:
        raise RuntimeError(f"No items in {pred_path}")

    analysis_dir = run_dir / "analysis"
    wer_bins_dir = analysis_dir / "wer_bins"
    bleu_bins_dir = analysis_dir / "bleu_bins"
    wer_bins_dir.mkdir(parents=True, exist_ok=True)
    bleu_bins_dir.mkdir(parents=True, exist_ok=True)

    wer_bins = _bin_items(items, key="wer")
    bleu_bins = _bin_items(items, key="bleu4")

    for label in BIN_LABELS:
        (wer_bins_dir / f"{label}.json").write_text(
            json.dumps({"bin": label, "metric": "wer", "items": wer_bins[label]}, indent=2),
            encoding="utf-8",
        )
        (bleu_bins_dir / f"{label}.json").write_text(
            json.dumps({"bin": label, "metric": "bleu4", "items": bleu_bins[label]}, indent=2),
            encoding="utf-8",
        )

    wer_summary = _bins_summary(wer_bins, metric="wer")
    bleu_summary = _bins_summary(bleu_bins, metric="bleu4")
    aggregates = _aggregate_metrics(items)

    (analysis_dir / "wer_bins_summary.json").write_text(
        json.dumps(wer_summary, indent=2), encoding="utf-8"
    )
    (analysis_dir / "bleu_bins_summary.json").write_text(
        json.dumps(bleu_summary, indent=2), encoding="utf-8"
    )
    (analysis_dir / "aggregate_metrics.json").write_text(
        json.dumps(aggregates, indent=2), encoding="utf-8"
    )

    print(f"\n=== {run_dir.name} ===")
    print(f"  samples: {aggregates['num_samples']}")
    print(f"  mean WER:              {aggregates['mean_wer']:.4f}")
    print(f"  corpus WER (weighted): {aggregates['corpus_wer']:.4f}")
    print(f"  mean BLEU-4:           {aggregates['mean_bleu4']:.4f}")
    print(f"  corpus BLEU-4:         {aggregates['corpus_bleu4']:.4f}")
    print(f"  analysis -> {analysis_dir}")

    return analysis_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Bin ASR predictions by WER and BLEU-4")
    ap.add_argument(
        "run_dirs",
        nargs="+",
        help="Eval output dirs containing asr_predictions.json",
    )
    args = ap.parse_args()

    for raw in args.run_dirs:
        analyze_run_dir(Path(raw))


if __name__ == "__main__":
    main()
