"""
Baseline audio-model ASR evaluation on LibriSpeech test-clean.

Runs one or more speech-to-text baselines on LibriSpeech ``test-clean`` and writes:

- ``asr_metrics.json`` with aggregate metrics and run metadata
- ``asr_predictions.json`` with per-sample predictions, WER, and BLEU-4

Example::

    uv run evaluation/audio_encoder_asr.py --num-samples 16
    uv run evaluation/audio_encoder_asr.py --model-id openai/whisper-large-v2 --num-samples all
    uv run evaluation/audio_encoder_asr.py --model-id Qwen/Qwen3-ASR-1.7B-hf --num-samples all
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_pkg_root, "src"))
sys.path.insert(0, _pkg_root)

from dataset import LibriSpeechConfig, LibriSpeechPairs, load_mono_waveform_16k
from encoder import WhisperConfig, load_whisper_asr_models
from training.utils.env import load_project_env
from training.utils.metrics import corpus_bleu4, normalize_asr_text, word_error_rate

DEFAULT_MODEL_IDS = (
    "openai/whisper-small",
    "openai/whisper-medium",
    "openai/whisper-large-v2",
    "Qwen/Qwen3-ASR-1.7B-hf",
)
DEFAULT_OUTPUT_DIR = os.path.join(_pkg_root, "outputs", "audio_encoder_asr")
DEFAULT_LOG_EVERY = 25


@dataclass(frozen=True)
class AsrMetrics:
    num_samples: int
    avg_wer: float
    bleu4: float


def _apply_device_override(device: str | None) -> None:
    """Force CPU eval when requested so model loading follows project conventions."""
    if device != "cpu":
        return
    os.environ["DEVICE"] = "cpu"
    os.environ["GPU_LOCK"] = "off"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _resolve_device_and_dtype(requested_device: str | None) -> tuple[str, torch.dtype]:
    if requested_device == "cpu":
        return "cpu", torch.float32
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def _resolve_num_samples(total_pairs: int, num_samples: int | str) -> int:
    if isinstance(num_samples, str) and num_samples.strip().lower() == "all":
        return total_pairs
    return max(1, int(num_samples))


def _select_asr_pairs(dataset_root: str, num_samples: int | str, seed: int) -> list[tuple[str, str]]:
    dataset = LibriSpeechPairs(dataset_root)
    pairs = list(dataset.pairs)
    resolved_samples = _resolve_num_samples(len(pairs), num_samples)
    if resolved_samples < len(pairs):
        rng = random.Random(seed)
        pairs = rng.sample(pairs, resolved_samples)
    return pairs


def _bleu4(reference: str, hypothesis: str) -> float:
    return corpus_bleu4([reference], [hypothesis])


def _utterance_id(audio_path: str) -> str:
    return Path(audio_path).stem


def _sanitize_model_name(model_id: str) -> str:
    return model_id.replace("/", "__").replace("-", "_")


def _default_run_name(model_id: str) -> str:
    return _sanitize_model_name(model_id)


def _model_family(model_id: str) -> str:
    if model_id.startswith("Qwen/Qwen3-ASR-"):
        return "qwen3_asr"
    return "whisper"


def _normalize_language_hint(model_id: str, language: str | None) -> str | None:
    if language is None:
        return None
    if _model_family(model_id) == "qwen3_asr" and len(language) > 2:
        return language.strip().title()
    return language


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _pipeline_kwargs_for_waveform(
    *,
    generate_kwargs: dict[str, Any],
    chunk_length_s: float | None,
    waveform_num_samples: int,
    sample_rate: int = 16000,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "batch_size": 1,
        "generate_kwargs": generate_kwargs,
        "return_timestamps": False,
    }
    if chunk_length_s is not None and (waveform_num_samples / float(sample_rate)) > chunk_length_s:
        kwargs["chunk_length_s"] = chunk_length_s
    return kwargs


@torch.no_grad()
def _run_whisper_asr_eval(
    *,
    model_id: str,
    pairs: list[tuple[str, str]],
    device: str,
    torch_dtype: torch.dtype,
    log_every: int,
    chunk_length_s: float | None,
    batch_size: int,
    language: str | None,
    task: str,
) -> tuple[AsrMetrics, list[dict[str, Any]]]:
    whisper = load_whisper_asr_models(
        WhisperConfig(
            model_id=model_id,
            device=device,
            torch_dtype=torch_dtype,
        )
    )
    refs: list[str] = []
    hyps: list[str] = []
    items: list[dict[str, Any]] = []

    generate_kwargs: dict[str, Any] = {"task": task}
    if language:
        generate_kwargs["language"] = language

    print(
        f"Generating transcripts for {len(pairs)} utterances "
        f"(model={model_id}, batch_size={batch_size}, chunk_length_s={chunk_length_s})...",
        flush=True,
    )
    for i, (audio_path, reference) in enumerate(pairs):
        if i == 0:
            print(f"  Starting utterance 1/{len(pairs)}...", flush=True)
        wave = load_mono_waveform_16k(audio_path)
        t0 = time.time()
        result = whisper.pipe(
            {"array": wave.numpy(), "sampling_rate": 16000},
            **_pipeline_kwargs_for_waveform(
                generate_kwargs=generate_kwargs,
                chunk_length_s=chunk_length_s,
                waveform_num_samples=int(wave.numel()),
            ),
        )
        elapsed = time.time() - t0

        ref = normalize_asr_text(reference)
        hyp = normalize_asr_text(str(result["text"]))
        wer = word_error_rate(ref, hyp)
        bleu4 = _bleu4(ref, hyp)

        items.append(
            {
                "utterance_id": _utterance_id(audio_path),
                "audio_path": audio_path,
                "reference": ref,
                "prediction": hyp,
                "wer": wer,
                "bleu4": bleu4,
                "latency_s": elapsed,
            }
        )
        refs.append(ref)
        hyps.append(hyp)

        if log_every > 0 and (i == 0 or (i + 1) % log_every == 0 or i + 1 == len(pairs)):
            print(
                f"  [{i + 1}/{len(pairs)}] WER={wer:.3f} BLEU-4={bleu4:.3f} ({elapsed:.1f}s)",
                flush=True,
            )

    avg_wer = sum(item["wer"] for item in items) / max(len(items), 1)
    bleu = corpus_bleu4(refs, hyps)
    return AsrMetrics(num_samples=len(items), avg_wer=float(avg_wer), bleu4=float(bleu)), items


@torch.no_grad()
def _run_qwen_asr_eval(
    *,
    model_id: str,
    pairs: list[tuple[str, str]],
    device: str,
    torch_dtype: torch.dtype,
    log_every: int,
    batch_size: int,
    language: str | None,
) -> tuple[AsrMetrics, list[dict[str, Any]]]:
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch_dtype,
    ).to(device)
    model.eval()

    refs: list[str] = []
    hyps: list[str] = []
    items: list[dict[str, Any]] = []

    batch_size = max(1, int(batch_size))
    print(
        f"Generating transcripts for {len(pairs)} utterances "
        f"(model={model_id}, batch_size={batch_size})...",
        flush=True,
    )

    for i0 in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i0 : i0 + batch_size]
        if i0 == 0:
            print(
                f"  Starting utterances {i0 + 1}-{i0 + len(batch_pairs)}/{len(pairs)}...",
                flush=True,
            )
        batch_audio = []
        batch_languages: list[str | None] | None = None
        if language is not None:
            batch_languages = []
        for audio_path, _ in batch_pairs:
            wave = load_mono_waveform_16k(audio_path)
            batch_audio.append(wave.numpy())
            if batch_languages is not None:
                batch_languages.append(language)

        t0 = time.time()
        inputs = processor.apply_transcription_request(
            audio=batch_audio,
            language=batch_languages if batch_languages is not None else None,
        ).to(device=model.device, dtype=model.dtype)
        output_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        predictions = processor.decode(
            generated_ids,
            return_format="transcription_only",
        )
        elapsed = time.time() - t0
        per_utt = elapsed / max(len(batch_pairs), 1)

        for j, ((audio_path, reference), prediction) in enumerate(
            zip(batch_pairs, predictions, strict=True)
        ):
            ref = normalize_asr_text(reference)
            hyp = normalize_asr_text(str(prediction))
            wer = word_error_rate(ref, hyp)
            bleu4 = _bleu4(ref, hyp)
            idx = i0 + j

            items.append(
                {
                    "utterance_id": _utterance_id(audio_path),
                    "audio_path": audio_path,
                    "reference": ref,
                    "prediction": hyp,
                    "wer": wer,
                    "bleu4": bleu4,
                    "latency_s": per_utt,
                }
            )
            refs.append(ref)
            hyps.append(hyp)

            if log_every > 0 and (
                idx == 0 or (idx + 1) % log_every == 0 or idx + 1 == len(pairs)
            ):
                print(
                    f"  [{idx + 1}/{len(pairs)}] WER={wer:.3f} BLEU-4={bleu4:.3f} "
                    f"(~{per_utt:.1f}s/utt, batch={len(batch_pairs)})",
                    flush=True,
                )

    avg_wer = sum(item["wer"] for item in items) / max(len(items), 1)
    bleu = corpus_bleu4(refs, hyps)
    return AsrMetrics(num_samples=len(items), avg_wer=float(avg_wer), bleu4=float(bleu)), items


def _run_asr_eval(
    *,
    model_id: str,
    pairs: list[tuple[str, str]],
    device: str,
    torch_dtype: torch.dtype,
    log_every: int,
    chunk_length_s: float | None,
    batch_size: int,
    language: str | None,
    task: str,
) -> tuple[AsrMetrics, list[dict[str, Any]]]:
    family = _model_family(model_id)
    if family == "qwen3_asr":
        return _run_qwen_asr_eval(
            model_id=model_id,
            pairs=pairs,
            device=device,
            torch_dtype=torch_dtype,
            log_every=log_every,
            batch_size=batch_size,
            language=language,
        )
    return _run_whisper_asr_eval(
        model_id=model_id,
        pairs=pairs,
        device=device,
        torch_dtype=torch_dtype,
        log_every=log_every,
        chunk_length_s=chunk_length_s,
        batch_size=batch_size,
        language=language,
        task=task,
    )


def _evaluate_one_model(
    *,
    model_id: str,
    dataset_root: str,
    output_dir: Path,
    num_samples: int | str,
    seed: int,
    device: str,
    torch_dtype: torch.dtype,
    log_every: int,
    chunk_length_s: float | None,
    batch_size: int,
    language: str | None,
    task: str,
    run_name_prefix: str | None,
) -> dict[str, Any]:
    run_name = _default_run_name(model_id)
    model_family = _model_family(model_id)
    if run_name_prefix:
        run_name = f"{run_name_prefix}_{run_name}"
    model_dir = output_dir / run_name

    print(f"\n{'=' * 60}\nASR eval - {run_name}\n  model: {model_id}\n{'=' * 60}")
    pairs = _select_asr_pairs(dataset_root, num_samples, seed)
    print(f"Evaluating {len(pairs)} utterances from {dataset_root}\n", flush=True)

    metrics, items = _run_asr_eval(
        model_id=model_id,
        pairs=pairs,
        device=device,
        torch_dtype=torch_dtype,
        log_every=log_every,
        chunk_length_s=chunk_length_s,
        batch_size=batch_size,
        language=language,
        task=task,
    )
    print(f"\n{run_name} results: avg_WER={metrics.avg_wer:.4f} BLEU-4={metrics.bleu4:.4f}")

    _write_json(
        model_dir / "asr_metrics.json",
        {
            "metrics": asdict(metrics),
            "meta": {
                "model_id": model_id,
                "model_family": model_family,
                "dataset_root": dataset_root,
                "num_samples": num_samples,
                "seed": seed,
                "device": device,
                "torch_dtype": str(torch_dtype),
                "chunk_length_s": chunk_length_s,
                "batch_size": batch_size,
                "language": language,
                "task": task,
            },
        },
    )
    _write_json(model_dir / "asr_predictions.json", {"items": items})
    return {
        "run_name": run_name,
        "model_id": model_id,
        "metrics": asdict(metrics),
        "output_dir": str(model_dir),
    }


def _build_parser() -> argparse.ArgumentParser:
    training_dir = os.path.join(_pkg_root, "training")
    default_test_root = LibriSpeechConfig.test_clean_root(training_dir)

    ap = argparse.ArgumentParser(description="Baseline ASR eval on LibriSpeech test-clean")
    ap.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=None,
        help="Force compute device; defaults to cuda if available, else cpu",
    )
    ap.add_argument("--dataset-root", type=str, default=default_test_root)
    ap.add_argument(
        "--model-id",
        dest="model_ids",
        action="append",
        default=None,
        help="ASR model to evaluate; pass multiple times to override the default model set",
    )
    ap.add_argument(
        "--num-samples",
        default="all",
        help="How many LibriSpeech test-clean utterances to score, or 'all'",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--run-name-prefix", type=str, default=None)
    ap.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    ap.add_argument(
        "--chunk-length-s",
        type=float,
        default=30.0,
        help="Chunk length passed to the HF ASR pipeline for long clips; set 0 to disable",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size for models that support batched decoding",
    )
    ap.add_argument(
        "--language",
        type=str,
        default="english",
        help="Language hint for supported models; set to empty string to disable",
    )
    ap.add_argument(
        "--task",
        choices=("transcribe", "translate"),
        default="transcribe",
        help="Generation task for Whisper-family models",
    )
    return ap


def main() -> None:
    load_project_env(_pkg_root)
    args = _build_parser().parse_args()
    _apply_device_override(args.device)

    device, torch_dtype = _resolve_device_and_dtype(args.device)
    model_ids = tuple(args.model_ids) if args.model_ids else DEFAULT_MODEL_IDS
    language = args.language.strip() if args.language is not None else None
    if language == "":
        language = None

    output_dir = Path(args.output_dir)
    print(f"Device: {device}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Models: {', '.join(model_ids)}")

    results = []
    for model_id in model_ids:
        results.append(
            _evaluate_one_model(
                model_id=model_id,
                dataset_root=args.dataset_root,
                output_dir=output_dir,
                num_samples=args.num_samples,
                seed=args.seed,
                device=device,
                torch_dtype=torch_dtype,
                log_every=max(1, int(args.log_every)),
                chunk_length_s=(None if float(args.chunk_length_s) <= 0 else float(args.chunk_length_s)),
                batch_size=max(1, int(args.batch_size)),
                language=_normalize_language_hint(model_id, language),
                task=args.task,
                run_name_prefix=args.run_name_prefix,
            )
        )

    print("\nCompleted baseline ASR evaluation.")
    for result in results:
        metrics = result["metrics"]
        print(
            f"  {result['model_id']}: avg_WER={metrics['avg_wer']:.4f} "
            f"BLEU-4={metrics['bleu4']:.4f} -> {result['output_dir']}"
        )


if __name__ == "__main__":
    main()
