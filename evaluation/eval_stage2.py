"""
Stage 2 evaluation on LibriSpeech test-clean.

Evaluates checkpoints from ``training/adapter_asr_trainer.py`` (StreamingAdapter +
TurnEndCommitGate, optional rate controller).

Choose what to run with ``--metric``:

- ``retrieval-cosine`` — R@1, R@5, R@10 via centered cosine similarity
- ``retrieval-nll`` — R@1, R@5, R@10 via frozen Qwen NLL (slow; supports resume)
- ``asr`` — avg WER + corpus BLEU-4 (per-utterance WER and BLEU-4 in ``asr_predictions.json``)

Training uses train-style LM conditioning (``[audio_tokens | BOS | teacher-forced text]``).
For ``--metric asr``, default inference is train-style ``[audio | im_end/BOS] → generate``
(``--append-im-end``). Use ``--prompt-asr`` to evaluate the Stage 3 prompt path.

Example::

    uv run evaluation/eval_stage2.py --metric retrieval-cosine --num-samples all
    uv run evaluation/eval_stage2.py --metric retrieval-nll --num-samples all
    uv run evaluation/eval_stage2.py --metric asr --compare-im-end
    uv run evaluation/eval_stage2.py --metric asr --prompt-asr
"""

from __future__ import annotations

import argparse
import os
import sys

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_pkg_root, "src"))
sys.path.insert(0, _pkg_root)

from evaluation.eval_stage1 import run_asr, run_retrieval_cosine, run_retrieval_nll
from src.dataset import LibriSpeechConfig
from training.utils.config import AsrExperimentConfig, CheckpointConfig, Stage2Config
from training.utils.env import env_int, load_project_env

STAGE = 2
EVAL_METRICS = ("retrieval-cosine", "retrieval-nll", "asr")


def _default_checkpoint(ckpt_cfg: CheckpointConfig) -> str:
    exp = AsrExperimentConfig.from_env(pkg_root=_pkg_root)
    return os.path.join(ckpt_cfg.dir, f"{exp.checkpoint_basename}.pt")


def _build_parser() -> argparse.ArgumentParser:
    stage2 = Stage2Config.from_env()
    ckpt_cfg = CheckpointConfig.from_env(pkg_root=_pkg_root)
    training_dir = os.path.join(_pkg_root, "training")
    default_checkpoint = _default_checkpoint(ckpt_cfg)
    default_test_root = LibriSpeechConfig.test_clean_root(training_dir)
    default_output_dir = os.path.join(_pkg_root, "outputs", "experiments")

    ap = argparse.ArgumentParser(description="Stage 2 eval (retrieval-cosine | retrieval-nll | asr)")
    ap.add_argument("--metric", required=True, choices=EVAL_METRICS)
    ap.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=None,
        help="Force compute device; cpu skips GPU lock (use while training holds GPUs)",
    )
    ap.add_argument("--checkpoint", type=str, default=default_checkpoint)
    ap.add_argument("--dataset-root", type=str, default=default_test_root)
    ap.add_argument("--num-samples", default=str(env_int("RETRIEVAL_NUM_UTTERANCES", 2620)))
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default=default_output_dir)
    ap.add_argument("--max-text-tokens", type=int, default=None)
    ap.add_argument("--candidate-batch-size", type=int, default=None)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--rank-only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--asr-prompt", type=str, default=None)
    ap.add_argument("--n-windows", type=int, default=-1)
    ap.add_argument("--max-new-tokens", type=int, default=stage2.val_max_new_tokens)
    ap.add_argument("--num-beams", type=int, default=stage2.val_num_beams)
    ap.add_argument("--do-sample", action="store_true", default=False)
    ap.add_argument("--repetition-penalty", type=float, default=stage2.val_repetition_penalty)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=4)
    ap.add_argument("--prompt-asr", action="store_true")
    ap.add_argument("--early-commit-truncation", action="store_true", default=False)
    ap.add_argument("--llm-device-map", type=str, default="auto")
    ap.add_argument("--log-every", type=int, default=stage2.val_log_every)
    im_end = ap.add_mutually_exclusive_group()
    im_end.add_argument("--append-im-end", dest="append_im_end", action="store_true", default=True)
    im_end.add_argument("--no-append-im-end", dest="append_im_end", action="store_false")
    ap.add_argument("--compare-im-end", action="store_true")
    return ap


def _apply_device_override(device: str | None) -> None:
    """Force CPU eval: skip GPU locks and hide CUDA devices from this process."""
    if device != "cpu":
        return
    os.environ["DEVICE"] = "cpu"
    os.environ["GPU_LOCK"] = "off"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.pop("LLM_DEVICE", None)
    os.environ.pop("LLM_MAX_MEMORY", None)


def _apply_stage2_defaults(args: argparse.Namespace) -> None:
    stage2 = Stage2Config.from_env()
    if args.metric == "retrieval-cosine" and args.max_text_tokens is None:
        args.max_text_tokens = stage2.max_text_tokens
    if args.metric == "retrieval-nll":
        if args.max_text_tokens is None:
            args.max_text_tokens = 512
        if args.num_samples == str(env_int("RETRIEVAL_NUM_UTTERANCES", 2620)):
            args.num_samples = str(env_int("RETRIEVAL_NLL_NUM_UTTERANCES", 2620))
    if args.metric == "asr" and args.num_samples == str(env_int("RETRIEVAL_NUM_UTTERANCES", 2620)):
        args.num_samples = str(env_int("ASR_EVAL_NUM_SAMPLES", 100))


def main() -> None:
    load_project_env(_pkg_root)
    args = _build_parser().parse_args()
    _apply_device_override(args.device)
    _apply_stage2_defaults(args)
    if args.metric == "retrieval-cosine":
        run_retrieval_cosine(stage=STAGE, args=args)
    elif args.metric == "retrieval-nll":
        run_retrieval_nll(stage=STAGE, args=args)
    elif args.metric == "asr":
        run_asr(stage=STAGE, args=args)
    else:
        raise SystemExit(f"Unknown metric: {args.metric}")


if __name__ == "__main__":
    main()
