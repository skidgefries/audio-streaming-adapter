"""
LibriSpeech test-clean ASR evaluation (WER + corpus BLEU-4).

Generates transcripts with the adapter + frozen Qwen pipeline, then compares
predictions to LibriSpeech references. Use this to compare Stage 1 (contrastive)
and Stage 2 (ASR distillation) checkpoints.

Stage 1: ``StreamingAdapter`` without rate controller, ``WhisperAdapterLLMPipeline``.
Stage 2: rate controller + ``TurnEndCommitGate``, ``WhisperAdapterLLMCommitGatePipeline``.

**Default (stages 1–2):** train-style ``audio tokens → generate`` (no im_end suffix at
inference; training CE still uses im_end + teacher forcing in ``adapter_asr_trainer``).
**Prompt path (Stage 3 goal):** pass ``--prompt-asr``
to use ``--asr-prompt`` + audio (chat template, ``enable_thinking=False``). Predictions
are raw model decode (whitespace-normalized only for WER).

Example::

    # Stage 1 only
    uv run evaluation/eval_librispeech_asr_metrics.py \\
        --checkpoint checkpoints/adapter_stage1.pt --stage 1

    # Stage 2 only
    uv run evaluation/eval_librispeech_asr_metrics.py \\
        --checkpoint checkpoints/adapter_stage2.pt --stage 2

    # Compare both (writes comparison JSON under --output-dir)
    uv run evaluation/eval_librispeech_asr_metrics.py --compare-stages
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_root = os.path.join(_pkg_root, "src")
sys.path.insert(0, _src_root)
sys.path.insert(0, _pkg_root)

from adapter.turn_end_commit_gate import TurnEndCommitGate
from adapter.streaming_adapter import StreamingAdapter
from adapter.windowing import AudioWaveformWindowizer
from adapter_llm_pipeline import (
    WhisperAdapterLLMCommitGatePipeline,
    WhisperAdapterLLMPipeline,
)
from training.utils.checkpointing import load_gate_state_dict_safe
from dataset import LibriSpeechPairs, load_mono_waveform_16k
from encoder import WhisperConfig, load_whisper_models
from llm.config import LlmGenerationParams
from training.utils.config import FrozenModelIdsConfig, Stage2Config
from training.utils.env import env_int, load_project_env
from training.utils.asr_prompt import (
    DEFAULT_ASR_PROMPT,
    PROMPT_CONDITIONING,
    TRAIN_STYLE_CONDITIONING,
)
from training.utils.loaders import default_device_and_dtype, load_frozen_qwen_causal_lm

_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class AsrMetrics:
    num_samples: int
    avg_wer: float
    bleu4: float


def _normalize_text(s: str) -> str:
    return _WS_RE.sub(" ", s.strip())


def _tokenize_words(s: str) -> list[str]:
    s = _normalize_text(s).lower()
    return s.split() if s else []


def _edit_distance(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, ta in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, tb in enumerate(b, start=1):
            cur = dp[j]
            cost = 0 if ta == tb else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref = _tokenize_words(reference)
    hyp = _tokenize_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / float(len(ref))


def corpus_bleu4(references: list[str], hypotheses: list[str]) -> float:
    """Corpus BLEU-4 with add-1 smoothing (same convention as SALMONN eval)."""
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


def _utterance_id(audio_path: str) -> str:
    return Path(audio_path).stem


def resolve_lm_conditioning(*, stage: int, prompt_asr: bool) -> tuple[bool, str]:
    """
    Stages 1–2 default to train-style ``[audio | BOS]`` (Stage 2 training layout).
    Prompt+audio is kept for Stage 3 / ``--prompt-asr``.
    """
    if prompt_asr or stage >= 3:
        return False, PROMPT_CONDITIONING
    return True, TRAIN_STYLE_CONDITIONING


def build_adapter(*, stage: int, stage2: Stage2Config) -> StreamingAdapter:
    use_rc = stage == 2 and stage2.use_rate_controller
    return StreamingAdapter(
        d_encoder=768,
        d_llm=4096,
        num_queries=4,
        num_layers=2,
        num_heads=4,
        d_ffn=2048,
        dropout=0.0,
        ema_alpha=0.8,
        learnable_ema=False,
        use_rate_controller=use_rc,
        rate_threshold=0.5,
        target_rate=stage2.rate_target,
    )


def load_checkpoint_into_models(
    checkpoint_path: str,
    *,
    stage: int,
    adapter: StreamingAdapter,
    gate: TurnEndCommitGate | None,
    device: str,
) -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    meta = {
        "checkpoint": checkpoint_path,
        "stage": stage,
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
    }
    if stage == 2 and gate is not None:
        if "gate_state_dict" not in ckpt:
            raise KeyError(
                f"Stage 2 checkpoint missing gate_state_dict: {checkpoint_path}"
            )
        load_gate_state_dict_safe(gate, ckpt)
    adapter.eval()
    if gate is not None:
        gate.eval()
    return meta


def build_pipeline(
    *,
    stage: int,
    checkpoint_path: str,
    device: str,
    torch_dtype: torch.dtype,
    model_ids: FrozenModelIdsConfig,
    stage2: Stage2Config,
    llm_device_map: str | None,
):
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
        device_map=llm_device_map or "auto",
    )

    windowizer = AudioWaveformWindowizer(
        sample_rate=16000,
        window_seconds=0.8,
        stride_seconds=0.4,
    )

    adapter = build_adapter(stage=stage, stage2=stage2).to(device, dtype=torch.bfloat16)
    gate: TurnEndCommitGate | None = None
    if stage == 2:
        gate = TurnEndCommitGate(
            d_llm=4096,
            hidden_dim=256,
            threshold=0.5,
            latency_weight=0.1,
            min_silence_ms=200.0,
            require_silence_for_commit=True,
            token_activity_threshold=8.0,
        ).to(device, dtype=torch.bfloat16)

    ckpt_meta = load_checkpoint_into_models(
        checkpoint_path,
        stage=stage,
        adapter=adapter,
        gate=gate,
        device=device,
    )

    base_kwargs = dict(
        whisper_processor=whisper.processor,
        whisper_model=whisper.model,
        windowizer=windowizer,
        streaming_adapter=adapter,
        llm_model=qwen.causal_lm,
        llm_tokenizer=qwen.tokenizer,
        device=device,
        torch_dtype=torch_dtype,
    )

    if stage == 2:
        assert gate is not None
        pipeline = WhisperAdapterLLMCommitGatePipeline(
            early_commit_gate=gate,
            **base_kwargs,
        )
        llm_pipe = pipeline._llm
    else:
        pipeline = WhisperAdapterLLMPipeline(**base_kwargs)
        llm_pipe = pipeline

    return pipeline, llm_pipe, ckpt_meta


@torch.no_grad()
def run_asr_eval(
    pipeline: WhisperAdapterLLMPipeline | WhisperAdapterLLMCommitGatePipeline,
    pairs: list[tuple[str, str]],
    *,
    asr_prompt: str,
    n_windows: int,
    generation: LlmGenerationParams,
    log_every: int,
    use_early_commit_truncation: bool = False,
    train_style_asr: bool = True,
    conditioning: str = TRAIN_STYLE_CONDITIONING,
) -> tuple[AsrMetrics, list[dict[str, Any]]]:
    refs: list[str] = []
    hyps: list[str] = []
    items: list[dict[str, Any]] = []

    n_win_msg = "all windows" if n_windows == -1 else f"{n_windows} window(s)"
    rep_pen = generation.repetition_penalty
    print(
        f"Generating transcripts for {len(pairs)} utterances "
        f"({n_win_msg}, beams={generation.num_beams}, "
        f"max_new_tokens={generation.max_new_tokens}, "
        f"repetition_penalty={rep_pen})...",
        flush=True,
    )
    if train_style_asr:
        print(f"  LM conditioning: {conditioning} (audio tokens → generate)", flush=True)
    else:
        print(
            f"  LM conditioning: {conditioning} "
            f"([prompt: {asr_prompt[:56]}{'...' if len(asr_prompt) > 56 else ''}] | audio tokens)",
            flush=True,
        )
    if use_early_commit_truncation:
        print("  Stage 2: early-commit truncation enabled (stop adding windows after gate commits).", flush=True)
    if n_windows == -1:
        print(
            "  Note: n_windows=-1 uses every adapter window per utterance; "
            "the first sample can take several minutes.",
            flush=True,
        )

    for i, (audio_path, reference) in enumerate(pairs):
        if i == 0:
            print(f"  Starting utterance 1/{len(pairs)}...", flush=True)
        wave = load_mono_waveform_16k(audio_path)
        t0 = time.time()
        if isinstance(pipeline, WhisperAdapterLLMCommitGatePipeline):
            result = pipeline.generate(
                wave,
                n_windows=n_windows,
                prompt=asr_prompt,
                generation=generation,
                use_early_commit_truncation=use_early_commit_truncation,
                train_style_asr=train_style_asr,
            )
        else:
            result = pipeline.generate(
                wave,
                n_windows=n_windows,
                prompt=asr_prompt,
                generation=generation,
                train_style_asr=train_style_asr,
            )
        elapsed = time.time() - t0

        ref = _normalize_text(reference)
        hyp = _normalize_text(result["text"])
        w = wer(ref, hyp)

        items.append(
            {
                "utterance_id": _utterance_id(audio_path),
                "audio_path": audio_path,
                "reference": ref,
                "prediction": hyp,
                "wer": w,
                "latency_s": elapsed,
                "num_windows_used": result.get("num_windows_used"),
            }
        )
        refs.append(ref)
        hyps.append(hyp)

        if log_every > 0 and (
            i == 0 or (i + 1) % log_every == 0 or i + 1 == len(pairs)
        ):
            print(
                f"  [{i + 1}/{len(pairs)}] WER={w:.3f} "
                f"windows={result.get('num_windows_used')} ({elapsed:.1f}s)",
                flush=True,
            )

    avg_wer = sum(p["wer"] for p in items) / max(len(items), 1)
    bleu = corpus_bleu4(refs, hyps)
    metrics = AsrMetrics(num_samples=len(items), avg_wer=float(avg_wer), bleu4=float(bleu))
    return metrics, items


def _select_pairs(
    dataset_root: str,
    num_samples: int,
    seed: int,
) -> list[tuple[str, str]]:
    dataset = LibriSpeechPairs(dataset_root)
    pairs = list(dataset.pairs)
    if num_samples < len(pairs):
        rng = random.Random(seed)
        pairs = rng.sample(pairs, num_samples)
    return pairs


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def evaluate_one_stage(
    *,
    stage: int,
    checkpoint_path: str,
    dataset_root: str,
    output_dir: Path,
    num_samples: int,
    seed: int,
    asr_prompt: str,
    n_windows: int,
    generation: LlmGenerationParams,
    device: str,
    torch_dtype: torch.dtype,
    model_ids: FrozenModelIdsConfig,
    stage2: Stage2Config,
    llm_device_map: str | None,
    log_every: int,
    use_early_commit_truncation: bool,
    prompt_asr: bool,
) -> dict[str, Any]:
    tag = f"stage{stage}"
    print(f"\n{'=' * 60}\nASR eval — {tag}\n  checkpoint: {checkpoint_path}\n{'=' * 60}")

    pairs = _select_pairs(dataset_root, num_samples, seed)
    print(f"Evaluating {len(pairs)} utterances from {dataset_root}\n")

    pipeline, _, ckpt_meta = build_pipeline(
        stage=stage,
        checkpoint_path=checkpoint_path,
        device=device,
        torch_dtype=torch_dtype,
        model_ids=model_ids,
        stage2=stage2,
        llm_device_map=llm_device_map,
    )
    print(
        f"Loaded {tag}: epoch={ckpt_meta.get('epoch')} "
        f"step={ckpt_meta.get('global_step')}\n"
    )

    early_trunc = stage == 2 and use_early_commit_truncation
    train_style_asr, conditioning = resolve_lm_conditioning(
        stage=stage, prompt_asr=prompt_asr
    )
    metrics, items = run_asr_eval(
        pipeline,
        pairs,
        asr_prompt=asr_prompt,
        n_windows=n_windows,
        generation=generation,
        log_every=log_every,
        use_early_commit_truncation=early_trunc,
        train_style_asr=train_style_asr,
        conditioning=conditioning,
    )

    print(f"\n{tag} results: avg_WER={metrics.avg_wer:.4f} BLEU-4={metrics.bleu4:.4f}")

    stage_dir = output_dir / tag
    _write_json(
        stage_dir / "asr_metrics.json",
        {
            "metrics": asdict(metrics),
            "meta": {
                **ckpt_meta,
                "dataset_root": dataset_root,
                "num_samples": num_samples,
                "seed": seed,
                "asr_prompt": asr_prompt,
                "n_windows": n_windows,
                "generation": asdict(generation),
                "use_early_commit_truncation": early_trunc,
                "conditioning": conditioning,
                "train_style_asr": train_style_asr,
                "device": device,
            },
        },
    )
    _write_json(stage_dir / "asr_predictions.json", {"items": items})

    return {
        "stage": stage,
        "checkpoint": checkpoint_path,
        "metrics": asdict(metrics),
        "output_dir": str(stage_dir),
    }


def main() -> None:
    load_project_env(_pkg_root)

    model_ids = FrozenModelIdsConfig.from_env()
    stage2 = Stage2Config.from_env()
    device, torch_dtype = default_device_and_dtype()

    default_test = os.path.join(
        _pkg_root, "datasets/librispeech_data/LibriSpeech/test-clean"
    )
    default_stage1 = os.path.join(_pkg_root, "checkpoints", "adapter_stage1.pt")
    default_stage2 = os.path.join(_pkg_root, "checkpoints", "adapter_stage2.pt")

    ap = argparse.ArgumentParser(description="LibriSpeech ASR metrics (WER, BLEU-4)")
    ap.add_argument("--dataset-root", type=str, default=default_test)
    ap.add_argument("--output-dir", type=str, default=os.path.join(_pkg_root, "outputs", "asr_eval"))
    ap.add_argument("--num-samples", type=int, default=env_int("ASR_EVAL_NUM_SAMPLES", 100))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint", type=str, default=None, help="Single checkpoint path")
    ap.add_argument("--stage", type=int, choices=[1, 2], default=None, help="1 or 2 (required with --checkpoint)")
    ap.add_argument(
        "--compare-stages",
        action="store_true",
        help=f"Run stage 1 ({default_stage1}) and stage 2 ({default_stage2})",
    )
    ap.add_argument("--stage1-checkpoint", type=str, default=default_stage1)
    ap.add_argument("--stage2-checkpoint", type=str, default=default_stage2)
    ap.add_argument("--asr-prompt", type=str, default=DEFAULT_ASR_PROMPT)
    ap.add_argument(
        "--n-windows",
        type=int,
        default=-1,
        help="Adapter windows per utterance (-1 = all windows, matches training stream)",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=496,
        help="Cap LM decode length; generation stops early on EOS when configured.",
    )
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--do-sample", action="store_true", default=False)
    ap.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.25,
        help="HF repetition_penalty (typical 1.1–1.3) to reduce decode loops.",
    )
    ap.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=4,
        help="Block repeated 4-grams during generation.",
    )
    ap.add_argument(
        "--prompt-asr",
        action="store_true",
        help=(
            "Prompt+audio conditioning (Stage 3 target). Default for stages 1–2 is "
            "train-style [audio | BOS] matching adapter_asr_trainer."
        ),
    )
    ap.add_argument(
        "--early-commit-truncation",
        action="store_true",
        default=False,
        help="Stage 2 only: stop adding windows after the early-commit gate fires.",
    )
    ap.add_argument("--llm-device-map", type=str, default="auto")
    ap.add_argument("--log-every", type=int, default=5)
    args = ap.parse_args()

    if args.compare_stages:
        runs = [
            (1, args.stage1_checkpoint),
            (2, args.stage2_checkpoint),
        ]
    elif args.checkpoint and args.stage:
        runs = [(args.stage, args.checkpoint)]
    else:
        ap.error("Provide --checkpoint and --stage, or use --compare-stages")

    out_dir = Path(args.output_dir)
    generation = LlmGenerationParams(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        num_beams=max(1, args.num_beams),
        temperature=0.7 if args.do_sample else None,
        repetition_penalty=float(args.repetition_penalty),
        no_repeat_ngram_size=int(args.no_repeat_ngram_size),
    )
    use_early_commit_truncation = args.early_commit_truncation
    llm_map = args.llm_device_map if args.llm_device_map != "none" else None

    summaries: list[dict[str, Any]] = []
    for stage, ckpt in runs:
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(ckpt)
        summaries.append(
            evaluate_one_stage(
                stage=stage,
                checkpoint_path=ckpt,
                dataset_root=args.dataset_root,
                output_dir=out_dir,
                num_samples=args.num_samples,
                seed=args.seed,
                asr_prompt=args.asr_prompt,
                n_windows=args.n_windows,
                generation=generation,
                device=device,
                torch_dtype=torch_dtype,
                model_ids=model_ids,
                stage2=stage2,
                llm_device_map=llm_map,
                log_every=args.log_every,
                use_early_commit_truncation=use_early_commit_truncation,
                prompt_asr=args.prompt_asr,
            )
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(summaries) == 2:
        m1 = summaries[0]["metrics"]
        m2 = summaries[1]["metrics"]
        comparison = {
            "stage1": summaries[0],
            "stage2": summaries[1],
            "delta_stage2_minus_stage1": {
                "avg_wer": m2["avg_wer"] - m1["avg_wer"],
                "bleu4": m2["bleu4"] - m1["bleu4"],
            },
            "interpretation": {
                "wer_lower_is_better": m2["avg_wer"] < m1["avg_wer"],
                "bleu_higher_is_better": m2["bleu4"] > m1["bleu4"],
            },
        }
        _write_json(out_dir / "asr_comparison.json", comparison)
        print("\n" + "=" * 60)
        print("STAGE 1 vs STAGE 2 (ASR)")
        print("=" * 60)
        print(f"  Stage 1  WER={m1['avg_wer']:.4f}  BLEU-4={m1['bleu4']:.4f}")
        print(f"  Stage 2  WER={m2['avg_wer']:.4f}  BLEU-4={m2['bleu4']:.4f}")
        print(f"  Δ WER    {comparison['delta_stage2_minus_stage1']['avg_wer']:+.4f}")
        print(f"  Δ BLEU-4 {comparison['delta_stage2_minus_stage1']['bleu4']:+.4f}")
        if comparison["interpretation"]["wer_lower_is_better"]:
            print("  Stage 2 improved WER.")
        else:
            print("  Stage 2 did not improve WER.")
        if comparison["interpretation"]["bleu_higher_is_better"]:
            print("  Stage 2 improved BLEU-4.")
        else:
            print("  Stage 2 did not improve BLEU-4.")
        print(f"\nWrote {out_dir / 'asr_comparison.json'}")


if __name__ == "__main__":
    main()
