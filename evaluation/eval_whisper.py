"""
Whisper ASR baseline eval on LibriSpeech.

Transcribes audio with stock HuggingFace Whisper (no adapter / LLM) and reports
the same ASR metrics as ``evaluation/eval_stage2.py --metric asr``:

- average WER
- corpus BLEU-4
- per-utterance WER / BLEU-4 in ``asr_predictions.json``

Examples::

    uv run evaluation/eval_whisper.py
    uv run evaluation/eval_whisper.py --dataset-root datasets/librispeech_data/LibriSpeech/dev-clean
    uv run evaluation/eval_whisper.py --num-samples all --batch-size 16
    uv run evaluation/eval_whisper.py --whisper-model-id openai/whisper-small --language en
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_pkg_root, "src"))
sys.path.insert(0, _pkg_root)

from dataset import load_mono_waveform_16k
from encoder import WhisperConfig, load_whisper_models
from evaluation.eval_stage1 import (
    AsrMetrics,
    _bleu4,
    _corpus_bleu4,
    _normalize_text,
    _select_asr_pairs,
    _utterance_id,
    _wer,
    _write_asr_json,
    experiment_output_dir,
    release_cuda_memory,
    resolve_num_samples,
)
from src.dataset import LibriSpeechConfig
from training.utils.config import FrozenModelIdsConfig, Stage2Config
from training.utils.devices import init_eval_device
from training.utils.env import env_int, env_str, load_project_env


def _apply_device_override(device: str | None) -> None:
    if device != "cpu":
        return
    os.environ["DEVICE"] = "cpu"
    os.environ["GPU_LOCK"] = "off"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.pop("LLM_DEVICE", None)
    os.environ.pop("LLM_MAX_MEMORY", None)


def _build_parser() -> argparse.ArgumentParser:
    stage2 = Stage2Config.from_env()
    training_dir = os.path.join(_pkg_root, "training")
    default_test_root = LibriSpeechConfig.test_clean_root(training_dir)
    default_output_dir = os.path.join(_pkg_root, "outputs", "experiments")
    default_whisper = (
        env_str("WHISPER_MODEL_ID", "openai/whisper-small") or "openai/whisper-small"
    )

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=None,
        help="Force compute device; cpu skips GPU lock",
    )
    ap.add_argument("--dataset-root", type=str, default=default_test_root)
    ap.add_argument(
        "--num-samples",
        default=str(env_int("ASR_EVAL_NUM_SAMPLES", 100)),
        help='Number of utterances, or "all"',
    )
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default=default_output_dir)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--whisper-model-id",
        type=str,
        default=default_whisper,
        help="HF Whisper checkpoint (default: WHISPER_MODEL_ID or openai/whisper-small)",
    )
    ap.add_argument(
        "--language",
        type=str,
        default="en",
        help='Whisper language code forced for decoding (default: "en"; empty = autodetect)',
    )
    ap.add_argument(
        "--task",
        choices=("transcribe", "translate"),
        default="transcribe",
        help="Whisper generation task",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=env_int("VAL_BATCH_SIZE", 32),
        help="Utterance batch size for Whisper generate (default: VAL_BATCH_SIZE or 32)",
    )
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        # Whisper max_target_positions is typically 448; stage2's 496 is for Qwen and OOMs Whisper.
        default=env_int("WHISPER_MAX_NEW_TOKENS", 224),
        help="Max new tokens for Whisper decode (must leave room for decoder "
        "prompt tokens under model max_target_positions≈448; default 224)",
    )
    ap.add_argument("--log-every", type=int, default=stage2.val_log_every)
    return ap


def _clamp_whisper_max_new_tokens(
    model: Any,
    *,
    requested: int,
    decoder_prompt_len: int = 4,
) -> int:
    """Keep ``prompt_len + max_new_tokens`` under Whisper ``max_target_positions``."""
    max_pos = int(getattr(model.config, "max_target_positions", 448) or 448)
    budget = max(1, max_pos - max(0, int(decoder_prompt_len)))
    return max(1, min(int(requested), budget))


@torch.no_grad()
def _transcribe_batch(
    *,
    waveforms: list[torch.Tensor],
    processor: Any,
    model: Any,
    device: str,
    torch_dtype: torch.dtype,
    language: str | None,
    task: str,
    num_beams: int,
    max_new_tokens: int,
) -> list[str]:
    """Run Whisper ASR on a list of mono 16 kHz waveforms."""
    arrays = [w.detach().float().cpu().numpy().reshape(-1) for w in waveforms]
    sr = int(getattr(processor.feature_extractor, "sampling_rate", 16000))
    inputs = processor(
        arrays,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )
    input_features = inputs.input_features.to(device=device, dtype=torch_dtype)
    attention_mask = getattr(inputs, "attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device)

    prompt_len = 0
    gen_kwargs: dict[str, Any] = {
        "num_beams": max(1, int(num_beams)),
        "do_sample": False,
    }
    if language:
        try:
            prompt_ids = processor.get_decoder_prompt_ids(language=language, task=task)
            gen_kwargs["forced_decoder_ids"] = prompt_ids
            prompt_len = len(prompt_ids) if prompt_ids is not None else 0
        except Exception:
            prompt_len = 4

    # Whisper counts start/prompt tokens toward max_target_positions (usually 448).
    clamped = _clamp_whisper_max_new_tokens(
        model, requested=max_new_tokens, decoder_prompt_len=max(prompt_len, 4)
    )
    gen_kwargs["max_new_tokens"] = clamped

    if attention_mask is not None:
        out_ids = model.generate(
            input_features,
            attention_mask=attention_mask,
            **gen_kwargs,
        )
    else:
        out_ids = model.generate(input_features, **gen_kwargs)

    texts = processor.batch_decode(out_ids, skip_special_tokens=True)
    return [t.strip() for t in texts]


@torch.no_grad()
def run_whisper_asr(args: argparse.Namespace) -> dict[str, Any]:
    if not os.path.isdir(args.dataset_root):
        raise FileNotFoundError(f"dataset root not found: {args.dataset_root}")

    device_obj, torch_dtype, _ = init_eval_device()
    device = str(device_obj)
    batch_size = max(1, int(args.batch_size))
    language = (args.language or "").strip() or None

    run_name = args.run_name
    if not run_name:
        model_stem = Path(args.whisper_model_id).name.replace("/", "_")
        run_name = f"whisper_asr_{model_stem}"

    print(f"\n{'=' * 60}")
    print(f"Whisper ASR eval — {run_name}")
    print(f"  model: {args.whisper_model_id}")
    print(f"{'=' * 60}")

    resolved = resolve_num_samples(args.dataset_root, args.num_samples)
    pairs = _select_asr_pairs(args.dataset_root, resolved, args.seed)
    print(f"Evaluating {len(pairs)} utterances from {args.dataset_root}")
    print(f"Device: {device}  dtype={torch_dtype}  batch_size={batch_size}")
    print(f"Language={language or 'autodetect'}  task={args.task}  beams={args.num_beams}")
    max_new = int(args.max_new_tokens)
    print(f"max_new_tokens={max_new} (clamped to Whisper max_target_positions at generate)\n")

    wm = load_whisper_models(
        cfg=WhisperConfig(
            model_id=args.whisper_model_id,
            device=device,
            torch_dtype=torch_dtype,
        )
    )
    model = wm.model
    processor = wm.processor

    refs: list[str] = []
    hyps: list[str] = []
    items: list[dict[str, Any]] = []

    print(
        f"Transcribing {len(pairs)} utterances "
        f"(batch_size={batch_size}, beams={args.num_beams}, "
        f"max_new_tokens={args.max_new_tokens})...",
        flush=True,
    )

    for i0 in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i0 : i0 + batch_size]
        if i0 == 0:
            print(
                f"  Starting utterances {i0 + 1}-{i0 + len(batch_pairs)}/{len(pairs)}...",
                flush=True,
            )

        waves = [load_mono_waveform_16k(p) for p, _ in batch_pairs]
        t0 = time.time()
        preds = _transcribe_batch(
            waveforms=waves,
            processor=processor,
            model=model,
            device=device,
            torch_dtype=torch_dtype,
            language=language,
            task=args.task,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
        )
        elapsed = time.time() - t0
        per_utt = elapsed / max(len(batch_pairs), 1)

        for j, ((audio_path, reference), hyp_raw) in enumerate(
            zip(batch_pairs, preds, strict=True)
        ):
            ref = _normalize_text(reference)
            hyp = _normalize_text(hyp_raw)
            w = _wer(ref, hyp)
            b = _bleu4(ref, hyp)
            idx = i0 + j
            items.append(
                {
                    "utterance_id": _utterance_id(audio_path),
                    "audio_path": audio_path,
                    "reference": ref,
                    "prediction": hyp,
                    "wer": w,
                    "bleu4": b,
                    "latency_s": per_utt,
                }
            )
            refs.append(ref)
            hyps.append(hyp)

            if args.log_every > 0 and (
                idx == 0
                or (idx + 1) % args.log_every == 0
                or idx + 1 == len(pairs)
            ):
                print(
                    f"  [{idx + 1}/{len(pairs)}] WER={w:.3f} BLEU-4={b:.3f} "
                    f"(~{per_utt:.1f}s/utt, batch={len(batch_pairs)})",
                    flush=True,
                )
                print(f"    REF: {ref}", flush=True)
                print(f"    HYP: {hyp}", flush=True)

    avg_wer = sum(p["wer"] for p in items) / max(len(items), 1)
    bleu = _corpus_bleu4(refs, hyps)
    metrics = AsrMetrics(num_samples=len(items), avg_wer=float(avg_wer), bleu4=float(bleu))
    print(f"\n{run_name} results: avg_WER={metrics.avg_wer:.4f} BLEU-4={metrics.bleu4:.4f}")

    stage_dir = experiment_output_dir(args.output_dir, run_name)
    _write_asr_json(
        stage_dir / "asr_metrics.json",
        {
            "metrics": asdict(metrics),
            "meta": {
                "backend": "whisper",
                "whisper_model_id": args.whisper_model_id,
                "dataset_root": args.dataset_root,
                "num_samples": args.num_samples,
                "seed": args.seed,
                "language": language,
                "task": args.task,
                "batch_size": batch_size,
                "num_beams": args.num_beams,
                "max_new_tokens": args.max_new_tokens,
                "device": device,
                "run_name": run_name,
            },
        },
    )
    _write_asr_json(stage_dir / "asr_predictions.json", {"items": items})
    print(f"Wrote {stage_dir / 'asr_metrics.json'}")
    print(f"Wrote {stage_dir / 'asr_predictions.json'}")

    release_cuda_memory()
    return {
        "run_name": run_name,
        "metrics": asdict(metrics),
        "output_dir": str(stage_dir),
    }


def main() -> None:
    load_project_env(_pkg_root)
    # Prefer Whisper id from FrozenModelIdsConfig when env is unset via CLI default already.
    _ = FrozenModelIdsConfig.from_env()
    args = _build_parser().parse_args()
    _apply_device_override(args.device)
    run_whisper_asr(args)


if __name__ == "__main__":
    main()
