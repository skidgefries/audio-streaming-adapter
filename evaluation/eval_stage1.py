"""Stage 1 evaluation — shared helpers and metric runners.

Stage 2 entry point (``eval_stage2.py``) imports runners from this module.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_root = os.path.join(_pkg_root, "src")
_training_dir = os.path.join(_pkg_root, "training")
sys.path.insert(0, _src_root)
sys.path.insert(0, _pkg_root)

from adapter.turn_end_commit_gate import TurnEndCommitGate
from adapter.streaming_adapter import StreamingAdapter
from adapter.windowing import AudioWaveformWindowizer
from adapter_llm_pipeline import (
    WhisperAdapterLLMCommitGatePipeline,
    WhisperAdapterLLMPipeline,
)
from dataset import LibriSpeechPairs, load_mono_waveform_16k
from encoder import WhisperConfig, load_whisper_models
from llm.config import LlmGenerationParams
from src.dataset import LibriSpeechConfig
from src.encoder.waveform_window_encoder import WhisperWindowFeatureExtractor
from training.utils.asr_prompt import (
    DEFAULT_ASR_PROMPT,
    PROMPT_CONDITIONING,
    TRAIN_STYLE_CONDITIONING,
    TRAIN_STYLE_NO_IM_END_CONDITIONING,
)
from training.utils.asr_only_validation import decode_asr_predictions_batch
from training.utils.checkpointing import (
    filter_adapter_state_dict,
    load_gate_state_dict_safe,
    resolve_gate_config_from_checkpoint,
)
from training.utils.config import (
    CheckpointConfig,
    DeviceConfig,
    FrozenModelIdsConfig,
    GateConfig,
    Stage2Config,
)
from training.utils.gate_training import build_turn_end_gate
from training.utils.devices import (
    apply_runtime_cuda_env,
    device_env_allows_qwen_spill,
    init_eval_device,
    llm_input_device,
    resolve_device,
    visible_gpu_count,
)
from training.utils.env import env_int, env_optional_int, env_str, load_project_env
from training.utils.loaders import (
    load_frozen_qwen_causal_lm,
    load_frozen_qwen_embeddings,
    load_frozen_vicuna_causal_lm,
    load_frozen_vicuna_embeddings,
)
from training.utils.stage1_validation import (
    compute_retrieval_metrics,
    compute_retrieval_metrics_from_cost_matrix,
)


def _ckpt_trainer_name(ckpt: dict) -> str | None:
    hp = ckpt.get("hyperparams")
    if isinstance(hp, dict):
        trainer = hp.get("trainer")
        return str(trainer) if trainer else None
    return None


def _ckpt_has_gate(ckpt: dict) -> bool:
    return ckpt.get("gate_state_dict") is not None


def _ckpt_has_rate_controller(ckpt: dict) -> bool:
    """True only when weights are present (align/asr-only trainers omit RC)."""
    sd = ckpt.get("adapter_state_dict") or {}
    return any(str(k).startswith("rate_controller.") for k in sd)


def _resolve_use_rate_controller(
    ckpt: dict,
    *,
    stage: int,
    stage2: Stage2Config | None,
) -> bool:
    """Match adapter architecture to the checkpoint (not just Stage2Config defaults)."""
    if stage != 2:
        return False
    trainer = _ckpt_trainer_name(ckpt)
    if trainer in (
        "adapter_asr_align_trainer",
        "adapter_asr_align_vicuna_trainer",
        "adapter_asr_only_trainer",
    ):
        return False
    if _ckpt_has_rate_controller(ckpt):
        return True
    if stage2 is not None:
        return bool(stage2.use_rate_controller) and _ckpt_has_gate(ckpt)
    return False


def _load_adapter_state_dict(adapter: StreamingAdapter, ckpt: dict) -> None:
    state = filter_adapter_state_dict(
        ckpt["adapter_state_dict"],
        num_queries=adapter.num_queries,
        use_rate_controller=bool(adapter.use_rate_controller),
    )
    # Align/asr-only checkpoints omit rate_controller; full stage-2 may include extras.
    adapter.load_state_dict(state, strict=False)


def resolve_run_name(checkpoint: str, explicit_run_name: str | None = None) -> str:
    if explicit_run_name:
        return explicit_run_name
    return Path(checkpoint).stem


def experiment_output_dir(base_output_dir: str | Path, run_name: str) -> Path:
    return Path(base_output_dir) / run_name


def write_eval_json(path: Path, metrics: dict[str, Any], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": metrics, "meta": meta}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def full_test_clean_utterance_count(dataset_root: str) -> int:
    return len(LibriSpeechPairs(dataset_root).pairs)


def resolve_num_utterances(dataset_root: str, num_utterances: int | str) -> int:
    if isinstance(num_utterances, str) and num_utterances.strip().lower() == "all":
        return full_test_clean_utterance_count(dataset_root)
    return int(num_utterances)


def resolve_num_samples(dataset_root: str, num_samples: int | str) -> int:
    if isinstance(num_samples, str) and num_samples.strip().lower() == "all":
        return full_test_clean_utterance_count(dataset_root)
    return int(num_samples)


# Default Qwen shard caps for eval (override via ``LLM_MAX_MEMORY`` in ``.env``).
# Whisper/adapter stay on ``cuda:0``; after release Qwen fills GPU 0, spills 2 GiB to GPU 1, then CPU.
EVAL_DEFAULT_LLM_MAX_MEMORY: dict[int | str, str] = {
    0: "14GiB",
    1: "2GiB",
    "cpu": "64GiB",
}


def _resolve_eval_llm_max_memory(num_cuda_devices: int) -> dict[int | str, str] | None:
    env_mem = DeviceConfig.from_env().llm_max_memory
    if env_mem:
        out: dict[int | str, str] = dict(env_mem)
        out.setdefault("cpu", "64GiB")
        return out
    if num_cuda_devices >= 2:
        return dict(EVAL_DEFAULT_LLM_MAX_MEMORY)
    if num_cuda_devices == 1:
        return {0: "14GiB", "cpu": "64GiB"}
    return None


def qwen_device_map_and_max_memory(
    *, num_cuda_devices: int | None = None
) -> tuple[str | None, dict[int | str, str] | None]:
    if resolve_device().type == "cpu":
        return None, None
    if not device_env_allows_qwen_spill():
        return None, None
    n = num_cuda_devices if num_cuda_devices is not None else visible_gpu_count()
    max_memory = _resolve_eval_llm_max_memory(n)
    if max_memory is None:
        return None, None
    return "sequential", max_memory


DEFAULT_STAGE1_LLM_ID = "lmsys/vicuna-7b-v1.5"


def _is_vicuna_model(model_id: str) -> bool:
    """True when the HF id is a Vicuna checkpoint (Stage 1 softmax trainer)."""
    return "vicuna" in str(model_id).lower()


def _resolve_eval_model_ids(stage: int) -> FrozenModelIdsConfig:
    """
    Stage 1 eval uses Vicuna-7B (same as ``adapter_contrastive_trainer_softmax``).

    ``LLM_MODEL_ID`` in ``.env`` stays Qwen for Stage 2; override Vicuna with
    ``VICUNA_MODEL_ID``.
    """
    ids = FrozenModelIdsConfig.from_env()
    if stage != 1:
        return ids
    vicuna_id = env_str("VICUNA_MODEL_ID", DEFAULT_STAGE1_LLM_ID) or DEFAULT_STAGE1_LLM_ID
    return FrozenModelIdsConfig(
        whisper_model_id=ids.whisper_model_id,
        llm_model_id=vicuna_id,
    )


def _load_eval_embeddings(
    *,
    model_id: str,
    device: str,
    torch_dtype: torch.dtype,
    device_map="auto",
    max_memory: dict[int, str] | None = None,
):
    """Load frozen text embeddings: Vicuna for Stage 1, Qwen otherwise."""
    if _is_vicuna_model(model_id):
        print(f"Loading Vicuna embedder ({model_id})...")
        return load_frozen_vicuna_embeddings(
            model_id=model_id,
            device=device,
            torch_dtype=torch_dtype,
            device_map=device_map,
            max_memory=max_memory,
        )
    print(f"Loading Qwen embedder ({model_id})...")
    return load_frozen_qwen_embeddings(
        model_id=model_id,
        device=device,
        torch_dtype=torch_dtype,
        device_map=device_map,
        max_memory=max_memory,
    )


def _load_eval_causal_lm(
    *,
    model_id: str,
    device: str,
    torch_dtype: torch.dtype,
    device_map="auto",
    max_memory: dict[int, str] | None = None,
):
    """Load frozen causal LM: Vicuna for Stage 1, Qwen otherwise."""
    if _is_vicuna_model(model_id):
        print(f"Loading Vicuna causal LM ({model_id})...")
        return load_frozen_vicuna_causal_lm(
            model_id=model_id,
            device=device,
            torch_dtype=torch_dtype,
            device_map=device_map,
            max_memory=max_memory,
        )
    print(f"Loading Qwen causal LM ({model_id})...")
    return load_frozen_qwen_causal_lm(
        model_id=model_id,
        device=device,
        torch_dtype=torch_dtype,
        device_map=device_map,
        max_memory=max_memory,
    )


def release_cuda_memory() -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def retrieval_nll_matrix_path(base_output_dir: str | Path, run_name: str) -> Path:
    return experiment_output_dir(base_output_dir, run_name) / "retrieval_nll_matrix.pt"


def retrieval_nll_audio_cache_path(base_output_dir: str | Path, run_name: str) -> Path:
    return experiment_output_dir(base_output_dir, run_name) / "retrieval_nll_audio_cache.pt"


def save_retrieval_nll_matrix_checkpoint(path: Path, matrix: torch.Tensor, rows_done: int, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": 1, "matrix": matrix.cpu(), "rows_done": rows_done, "meta": meta}, path)


def load_retrieval_nll_matrix_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def slice_rank_only_nll_matrix(
    saved: dict[str, Any],
    *,
    num_samples: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Return a square submatrix suitable for ranking from a (possibly partial) NLL cache.

    Each completed row ``i`` is scored against all candidates; for ranking we use
    ``matrix[:n, :n]`` where ``n = min(num_samples, rows_done, matrix_size)``.
    """
    matrix = saved["matrix"]
    rows_done = int(saved["rows_done"])
    matrix_n = int(matrix.shape[0])
    if rows_done <= 0:
        raise RuntimeError("No completed rows in saved NLL matrix")

    n_rank = min(int(num_samples), rows_done, matrix_n)
    if n_rank <= 0:
        raise RuntimeError(
            f"Cannot rank: num_samples={num_samples}, rows_done={rows_done}, matrix_n={matrix_n}"
        )

    info: dict[str, Any] = {
        "matrix_rows_total": matrix_n,
        "rows_done": rows_done,
        "num_ranked": n_rank,
        "matrix_complete": rows_done >= matrix_n,
        "partial_rank": n_rank < matrix_n or rows_done < matrix_n,
    }
    if rows_done < num_samples:
        print(
            f"  [WARN] Matrix incomplete ({rows_done}/{matrix_n} rows); "
            f"ranking on first {n_rank} completed queries only."
        )
    elif n_rank < matrix_n:
        print(f"  Ranking on first {n_rank}/{matrix_n} queries (--num-samples).")

    return matrix[:n_rank, :n_rank].clone(), info


def save_retrieval_nll_audio_cache(path: Path, audio_tokens_list: list[torch.Tensor], texts: list[str], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": 1, "audio_tokens_list": [t.cpu() for t in audio_tokens_list], "texts": texts, "meta": meta}, path)


def load_retrieval_nll_audio_cache(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def retrieval_nll_checkpoint_compatible(loaded_meta: dict[str, Any], expected_meta: dict[str, Any], *, extra_keys: tuple[str, ...] = ()) -> bool:
    keys = ("checkpoint", "dataset_root", "candidate_batch_size", "max_text_tokens", "stage", *extra_keys)
    return all(loaded_meta.get(key) == expected_meta.get(key) for key in keys)


WHISPER_DIM = 768
LLM_DIM = 4096

_WS_RE = re.compile(r"\s+")


def _eval_sample_count(args: argparse.Namespace) -> int:
    raw = getattr(args, "num_samples", None)
    if raw is None:
        raw = getattr(args, "num_utterances", "all")
    return resolve_num_utterances(args.dataset_root, raw)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _maybe_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _assert_module_device(module: torch.nn.Module, device: torch.device, name: str) -> None:
    param_device = next(module.parameters()).device
    if param_device != device:
        raise RuntimeError(f"{name} expected on {device}, found on {param_device}")


def _free_cuda(*objs: object, device: torch.device | None = None) -> None:
    for obj in objs:
        del obj
    release_cuda_memory()


def _build_retrieval_adapter(
    *,
    stage: int,
    stage2: Stage2Config | None,
    use_rate_controller: bool | None = None,
) -> StreamingAdapter:
    if use_rate_controller is None:
        use_rc = stage == 2 and stage2 is not None and stage2.use_rate_controller
    else:
        use_rc = use_rate_controller
    target_rate = stage2.rate_target if stage2 is not None else 0.5
    return StreamingAdapter(
        d_encoder=WHISPER_DIM,
        d_llm=LLM_DIM,
        num_queries=2,
        num_layers=2,
        num_heads=4,
        d_ffn=2048,
        dropout=0.0,
        ema_alpha=0.8,
        learnable_ema=False,
        use_rate_controller=use_rc,
        rate_threshold=0.5,
        target_rate=target_rate,
    )


def _interpret_cosine_r1(r1: float, *, stage: int) -> None:
    if stage == 1:
        if r1 >= 30:
            print("  R@1 ≥ 30% — solid alignment, ready for Stage 2")
        elif r1 >= 10:
            print("  R@1 ≥ 10% — marginal, Stage 2 may refine")
        else:
            print("  R@1 < 10% — alignment too weak, continue Stage 1")
        return
    if r1 >= 30:
        print("  R@1 ≥ 30% — strong audio–text alignment after Stage 2")
    elif r1 >= 10:
        print("  R@1 ≥ 10% — moderate alignment; compare with Stage 1 eval")
    else:
        print("  R@1 < 10% — weak alignment; check checkpoint and rate-controller settings")


def _print_retrieval_cosine_metrics(results: dict[str, float]) -> None:
    print("\n" + "=" * 50)
    print("RETRIEVAL METRICS — retrieval-cosine (Audio → Text)")
    print("=" * 50)
    print("  Retrieval:")
    for k in (1, 5, 10):
        print(f"    Recall@{k}: {results[f'R@{k}']:.2f}%")
    print(f"    MRR:          {results['MRR']:.2f}%")
    print(f"    Median Rank:  {results['median_rank']:.1f}")
    print(f"    Mean Rank:    {results['mean_rank']:.1f}")
    print("  Training objective:")
    print(f"    InfoNCE Loss: {results['infonce_loss']:.4f}")
    print("  Diagnostics:")
    print(f"    Alignment:    {results['alignment']:.4f}")
    print(f"    Uniformity:   {results['uniformity']:.4f}")
    print("=" * 50)


def _print_retrieval_nll_metrics(results: dict[str, float], *, stage: int) -> None:
    header = "RETRIEVAL METRICS — retrieval-nll (Audio → Text)"
    if stage == 2:
        header = "RETRIEVAL METRICS — retrieval-nll, Stage 2 (Audio → Text)"
    print("\n" + "=" * 50)
    print(header)
    print("=" * 50)
    print("  Retrieval:")
    for k in (1, 5, 10):
        print(f"    Recall@{k}: {results[f'R@{k}']:.2f}%")
    print(f"    MRR:          {results['MRR']:.2f}%")
    print(f"    Median Rank:  {results['median_rank']:.1f}")
    print(f"    Mean Rank:    {results['mean_rank']:.1f}")
    print("  Training objective:")
    print(f"    NLL (matched):     {results['nll']:.4f}")
    print(f"    NLL (neg mean):    {results['nll_neg_mean']:.4f}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Retrieval cosine
# ---------------------------------------------------------------------------


def _load_cosine_models(
    *,
    checkpoint_path: str,
    whisper_model_id: str,
    llm_model_id: str,
    device: torch.device,
    torch_dtype: torch.dtype,
    llm_map: str | None,
    stage: int,
    stage2: Stage2Config | None,
    max_memory: dict[int, str] | None = None,
):
    device_str = str(device)
    print("Loading Whisper...")
    audio = WhisperWindowFeatureExtractor(
        model_id=whisper_model_id,
        device=device_str,
        torch_dtype=torch_dtype,
    )

    llm_models = _load_eval_embeddings(
        model_id=llm_model_id,
        device=device_str,
        torch_dtype=torch_dtype,
        device_map=llm_map,
        max_memory=max_memory,
    )
    llm_device = llm_input_device(llm_models.embedder)

    label = "Stage 2 adapter" if stage == 2 else "adapter"
    print(f"Loading {label} from checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    use_rc = _resolve_use_rate_controller(ckpt, stage=stage, stage2=stage2)
    adapter = _build_retrieval_adapter(
        stage=stage, stage2=stage2, use_rate_controller=use_rc
    )
    adapter = adapter.to(device, dtype=torch_dtype)

    _load_adapter_state_dict(adapter, ckpt)
    adapter.eval()
    extra = ""
    if stage == 2 and stage2 is not None:
        trainer = _ckpt_trainer_name(ckpt)
        extra = (
            f" rate_controller={use_rc} "
            f"target_rate={stage2.rate_target}"
            + (f" trainer={trainer}" if trainer else "")
        )
    print(
        f"  Loaded {checkpoint_path}\n"
        f"  epoch={ckpt.get('epoch', '?')} step={ckpt.get('global_step', '?')}{extra}\n"
    )

    return audio, llm_models.tokenizer, llm_models.embedder, adapter, llm_device, ckpt

@torch.no_grad()
def _compute_cosine_embeddings(
    audio_extractor,
    tokenizer,
    text_embedder,
    adapter,
    pairs,
    *,
    device: torch.device,
    llm_device: torch.device,
    max_text_tokens: int,
):
    audio_vecs = []
    text_vecs = []


    for i, (audio_path, transcription) in enumerate(pairs):
        if i % 50 == 0:
            print(f"  Processing {i}/{len(pairs)}...")

        wave = load_mono_waveform_16k(audio_path)
        windows = audio_extractor.waveform_to_windows(wave)
        if len(windows) == 0:
            continue

        adapter.reset_streaming_state()
        chunks = []
        with _maybe_autocast(device):
            for w in windows:
                out = adapter.forward_window(w)
                chunks.append(out["tokens"])

        audio_tokens = torch.cat(chunks, dim=1)
        audio_pooled = audio_tokens.float().mean(dim=1)
        audio_vecs.append(audio_pooled.squeeze(0).cpu())

        text_tokens = tokenizer(
            transcription,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_text_tokens,
        )
        label_embeds = text_embedder(text_tokens.input_ids.to(llm_device)).float()
        text_pooled = label_embeds.mean(dim=1)
        text_vecs.append(text_pooled.squeeze(0).cpu())

    audio_bank = torch.stack(audio_vecs)
    text_bank = torch.stack(text_vecs)

    audio_bank = audio_bank - audio_bank.mean(dim=0, keepdim=True)
    text_bank = text_bank - text_bank.mean(dim=0, keepdim=True)
    audio_bank = F.normalize(audio_bank, dim=-1)
    text_bank = F.normalize(text_bank, dim=-1)

    return audio_bank, text_bank


def run_retrieval_cosine(*, stage: int, args: argparse.Namespace) -> None:
    """Run cosine-similarity retrieval eval for stage 1 or 2."""
    if stage not in (1, 2):
        raise ValueError(f"stage must be 1 or 2, got {stage}")

    model_ids = _resolve_eval_model_ids(stage)
    stage2 = Stage2Config.from_env() if stage == 2 else None

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    count = _eval_sample_count(args)
    run_name = resolve_run_name(args.checkpoint, args.run_name)

    device, torch_dtype, num_cuda = init_eval_device()
    llm_map, llm_max_memory = qwen_device_map_and_max_memory(num_cuda_devices=num_cuda)

    print(
        f"Device: {device} (visible CUDA devices: {num_cuda}; "
        f"Whisper/adapter on {device}, LLM={model_ids.llm_model_id} "
        f"device_map={llm_map!r})"
    )
    print(f"LLM device_map: {llm_map!r}")
    if llm_max_memory:
        print(f"LLM max_memory: {llm_max_memory!r}")

    print(f"Loading test-clean from {args.dataset_root}...")
    dataset = LibriSpeechPairs(args.dataset_root)
    pairs = dataset.pairs[:count]
    
    print(f"Evaluating on {len(pairs)} utterances (run={run_name})\n")
    

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    audio, tokenizer, text_embedder, adapter, llm_device, ckpt = _load_cosine_models(
        checkpoint_path=args.checkpoint,
        whisper_model_id=model_ids.whisper_model_id,
        llm_model_id=model_ids.llm_model_id,
        device=device,
        torch_dtype=torch_dtype,
        llm_map=llm_map,
        stage=stage,
        stage2=stage2,
        max_memory=llm_max_memory,
    )

    print("Computing embeddings...")
    audio_bank, text_bank = _compute_cosine_embeddings(
        audio,
        tokenizer,
        text_embedder,
        adapter,
        pairs,
        device=device,
        llm_device=llm_device,
        max_text_tokens=args.max_text_tokens,
    )
    print(f"  Audio bank: {audio_bank.shape}")
    print(f"  Text bank:  {text_bank.shape}\n")

    print("Computing retrieval metrics...")
    logit_scale = ckpt.get("contrastive_logit_scale")
    results = compute_retrieval_metrics(
        audio_bank,
        text_bank,
        ks=(1, 5, 10),
        logit_scale=logit_scale,
    )

    _print_retrieval_cosine_metrics(results)

    print("\nInterpretation:")
    _interpret_cosine_r1(results["R@1"], stage=stage)

    meta: dict[str, Any] = {
        "run_name": run_name,
        "checkpoint": args.checkpoint,
        "dataset_root": args.dataset_root,
        "num_utterances": len(pairs),
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
        "eval_type": "retrieval_cosine",
        "stage": stage,
    }
    if stage == 2 and stage2 is not None:
        meta["use_rate_controller"] = stage2.use_rate_controller

    out_path = experiment_output_dir(args.output_dir, run_name) / "retrieval_cosine.json"
    write_eval_json(out_path, results, meta)
    print(f"\nWrote {out_path}")

    release_cuda_memory()


# ---------------------------------------------------------------------------
# Retrieval NLL
# ---------------------------------------------------------------------------


def _load_nll_audio_encoder_and_adapter(
    *,
    device: torch.device,
    torch_dtype: torch.dtype,
    checkpoint_path: str,
    whisper_model_id: str,
    stage: int,
    stage2: Stage2Config | None,
):
    device_str = str(device)
    print("Loading Whisper...")
    audio = WhisperWindowFeatureExtractor(
        model_id=whisper_model_id,
        device=device_str,
        torch_dtype=torch_dtype,
    )

    label = "Stage 2 adapter" if stage == 2 else "adapter"
    print(f"Loading {label} from checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location=device)
    use_rc = _resolve_use_rate_controller(ckpt, stage=stage, stage2=stage2)
    adapter = _build_retrieval_adapter(
        stage=stage, stage2=stage2, use_rate_controller=use_rc
    )
    adapter = adapter.to(device, dtype=torch_dtype)

    _load_adapter_state_dict(adapter, ckpt)
    adapter.eval()
    _assert_module_device(audio.whisper, device, "Whisper")
    _assert_module_device(adapter, device, "Adapter")
    extra = ""
    if stage == 2 and stage2 is not None:
        trainer = _ckpt_trainer_name(ckpt)
        extra = (
            f" rate_controller={use_rc} "
            f"target_rate={stage2.rate_target}"
            + (f" trainer={trainer}" if trainer else "")
            + "\n"
        )
    print(
        f"  Loaded {checkpoint_path}\n"
        f"  epoch={ckpt.get('epoch', '?')} step={ckpt.get('global_step', '?')}\n"
        f"{extra}"
        f"  Whisper + adapter on {device}\n"
    )
    return audio, adapter, ckpt


def _load_nll_llm(*, device: torch.device, torch_dtype: torch.dtype, llm_model_id: str):
    device_str = str(device)
    device_map, max_memory = qwen_device_map_and_max_memory()
    if max_memory:
        print(f"  device_map={device_map!r} max_memory={max_memory!r}")
    llm_models = _load_eval_causal_lm(
        model_id=llm_model_id,
        device=device_str,
        torch_dtype=torch_dtype,
        device_map=device_map,
        max_memory=max_memory,
    )
    llm = llm_models.causal_lm
    llm_device = llm_input_device(llm)
    if device_map is None and llm_device != device:
        raise RuntimeError(f"LLM expected on {device}, found on {llm_device}")
    print(f"  Causal LM ({llm_model_id}) input device: {llm_device}\n")
    return llm_models.tokenizer, llm, llm_models.embedder, llm_device


@torch.no_grad()
def _compute_audio_prefix_tokens(audio_extractor, adapter, pairs, *, device: torch.device):
    audio_tokens_list = []
    kept_pairs = []

    for i, (audio_path, transcription) in enumerate(pairs):
        if i % 50 == 0:
            print(f"  Processing audio {i}/{len(pairs)}...")

        wave = load_mono_waveform_16k(audio_path)
        windows = audio_extractor.waveform_to_windows(wave)
        if len(windows) == 0:
            continue

        adapter.reset_streaming_state()
        chunks = []
        with _maybe_autocast(device):
            for w in windows:
                out = adapter.forward_window(w)
                chunks.append(out["tokens"])
        audio_tokens = torch.cat(chunks, dim=1)

        audio_tokens_list.append(audio_tokens.squeeze(0).cpu())
        kept_pairs.append((audio_path, transcription))

    return audio_tokens_list, kept_pairs


@torch.no_grad()
def _tokenize_candidates(tokenizer, texts, *, max_text_tokens: int):
    tok = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_text_tokens,
    )
    return tok.input_ids, tok.attention_mask


@torch.no_grad()
def _nll_matrix(
    audio_tokens_list,
    input_ids,
    attention_mask,
    llm,
    text_embedder,
    llm_device: torch.device,
    *,
    batch_size: int,
    autocast_device: torch.device,
    matrix_checkpoint_path: os.PathLike[str] | str | None = None,
    checkpoint_meta: dict | None = None,
    start_row: int = 0,
    out_scores: torch.Tensor | None = None,
):
    n = len(audio_tokens_list)
    assert input_ids.shape[0] == n
    assert attention_mask.shape[0] == n

    text_ids_shifted = input_ids[:, 1:].contiguous()
    text_mask_shifted = attention_mask[:, 1:].contiguous()

    pad_id = (
        llm.config.pad_token_id
        if getattr(llm.config, "pad_token_id", None) is not None
        else (text_embedder.weight.new_tensor([0], dtype=torch.long).item())
    )

    text_ids_shifted = text_ids_shifted.clone()
    text_ids_shifted[text_mask_shifted == 0] = int(pad_id)

    bos_token_id = (
        getattr(llm.config, "bos_token_id", None)
        if getattr(llm.config, "bos_token_id", None) is not None
        else getattr(llm.config, "eos_token_id", None)
    )
    if bos_token_id is None:
        bos_token_id = input_ids[0, 0].item()

    if out_scores is None:
        out_scores = torch.empty((n, n), dtype=torch.float32)
    else:
        if out_scores.shape != (n, n):
            raise ValueError(f"Expected checkpoint matrix {(n, n)}, got {tuple(out_scores.shape)}")

    for i in range(start_row, n):
        a = audio_tokens_list[i].to(llm_device)
        a_len = a.shape[0]
        a = a.unsqueeze(0)

        bos_ids = torch.tensor([[int(bos_token_id)]], device=llm_device, dtype=torch.long)
        bos_embed = text_embedder(bos_ids)

        for j0 in range(0, n, batch_size):
            j1 = min(n, j0 + batch_size)
            b = j1 - j0

            cand_ids = text_ids_shifted[j0:j1].to(llm_device)
            cand_mask = text_mask_shifted[j0:j1].to(llm_device)
            cand_embeds = text_embedder(cand_ids)

            prefix = torch.cat(
                [a.expand(b, -1, -1), bos_embed.expand(b, -1, -1)],
                dim=1,
            )
            inputs_embeds = torch.cat([prefix, cand_embeds], dim=1)

            labels_prefix = torch.full((b, a_len + 1), -100, device=llm_device, dtype=torch.long)
            labels_text = cand_ids.clone()
            labels_text[cand_mask == 0] = -100
            labels = torch.cat([labels_prefix, labels_text], dim=1)

            attn_prefix = torch.ones((b, a_len + 1), device=llm_device, dtype=torch.long)
            attn = torch.cat([attn_prefix, cand_mask], dim=1)

            with _maybe_autocast(autocast_device):
                out = llm(inputs_embeds=inputs_embeds, attention_mask=attn)
                logits = out.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_tok = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view(b, -1)
            valid = (shift_labels != -100).to(loss_tok.dtype)
            denom = valid.sum(dim=1).clamp_min(1.0)
            loss = (loss_tok * valid).sum(dim=1) / denom

            out_scores[i, j0:j1] = loss.detach().cpu()

            del inputs_embeds, logits, shift_logits, shift_labels, loss_tok, loss, out

        if matrix_checkpoint_path is not None and checkpoint_meta is not None:
            save_retrieval_nll_matrix_checkpoint(
                matrix_checkpoint_path,
                out_scores,
                i + 1,
                checkpoint_meta,
            )

        if (i + 1) % 25 == 0 or i == n - 1:
            print(f"  Scored {i + 1}/{n} audio queries...")

    return out_scores


def _nll_checkpoint_meta(
    *,
    run_name: str,
    checkpoint: str,
    dataset_root: str,
    num_utterances: int,
    candidate_batch_size: int,
    max_text_tokens: int,
    stage: int,
    use_rate_controller: bool | None = None,
) -> dict:
    meta: dict[str, Any] = {
        "run_name": run_name,
        "checkpoint": checkpoint,
        "dataset_root": dataset_root,
        "num_utterances": num_utterances,
        "candidate_batch_size": candidate_batch_size,
        "max_text_tokens": max_text_tokens,
        "stage": stage,
    }
    if use_rate_controller is not None:
        meta["use_rate_controller"] = use_rate_controller
    return meta


def _encode_and_cache_nll_audio(
    *,
    audio_cache_path,
    checkpoint_meta: dict,
    audio,
    adapter,
    pairs,
    device: torch.device,
):
    print("Computing audio prefix tokens...")
    audio_tokens_list, kept_pairs = _compute_audio_prefix_tokens(
        audio, adapter, pairs, device=device
    )
    if not audio_tokens_list:
        raise RuntimeError("No audio windows produced any prefix tokens; nothing to evaluate.")

    texts = [t for _, t in kept_pairs]
    print(f"  Kept {len(texts)} utterances after filtering\n")
    save_retrieval_nll_audio_cache(
        audio_cache_path,
        audio_tokens_list,
        texts,
        checkpoint_meta,
    )
    print(f"  Wrote audio cache {audio_cache_path}\n")
    return audio_tokens_list, texts


def run_retrieval_nll(*, stage: int, args: argparse.Namespace) -> None:
    """Run NLL-scored retrieval eval for stage 1 or 2."""
    if stage not in (1, 2):
        raise ValueError(f"stage must be 1 or 2, got {stage}")

    model_ids = _resolve_eval_model_ids(stage)
    stage2 = Stage2Config.from_env() if stage == 2 else None

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    count = _eval_sample_count(args)
    run_name = resolve_run_name(args.checkpoint, args.run_name)
    matrix_cache_path = retrieval_nll_matrix_path(args.output_dir, run_name)
    audio_cache_path = retrieval_nll_audio_cache_path(args.output_dir, run_name)
    use_rate_controller = stage2.use_rate_controller if stage2 is not None else None
    if args.candidate_batch_size is None:
        from training.utils.env import env_int

        args.candidate_batch_size = env_int("RETRIEVAL_NLL_BATCH_SIZE", 8)
    checkpoint_meta = _nll_checkpoint_meta(
        run_name=run_name,
        checkpoint=args.checkpoint,
        dataset_root=args.dataset_root,
        num_utterances=count,
        candidate_batch_size=args.candidate_batch_size,
        max_text_tokens=args.max_text_tokens,
        stage=stage,
        use_rate_controller=use_rate_controller,
    )
    resume_extra_keys = ("use_rate_controller",) if stage == 2 else ()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    texts: list[str] | None = None
    rank_info: dict[str, Any] | None = None

    if args.rank_only:
        saved = load_retrieval_nll_matrix_checkpoint(matrix_cache_path)
        if saved is None:
            raise FileNotFoundError(f"No saved matrix at {matrix_cache_path}")
        print(f"Ranking from saved matrix {matrix_cache_path}")
        nll_scores, rank_info = slice_rank_only_nll_matrix(saved, num_samples=count)
    else:
        device, torch_dtype, num_cuda = init_eval_device()
        llm_map, llm_max_memory = qwen_device_map_and_max_memory(num_cuda_devices=num_cuda)

        print(
            f"Device: {device} (visible CUDA devices: {num_cuda}; "
            f"Whisper/adapter on {device}, LLM={model_ids.llm_model_id} "
            f"device_map={llm_map!r})"
        )
        if llm_max_memory:
            print(f"LLM max_memory: {llm_max_memory!r}")

        print(f"Loading test-clean from {args.dataset_root}...")
        dataset = LibriSpeechPairs(args.dataset_root)
        pairs = dataset.pairs[:count]
        print(
            f"Preparing {len(pairs)} utterances (run={run_name}, "
            f"NLL batch={args.candidate_batch_size}, max_text_tokens={args.max_text_tokens})\n"
        )

        audio_tokens_list = None
        texts = None
        if args.resume:
            cached = load_retrieval_nll_audio_cache(audio_cache_path)
            if cached and retrieval_nll_checkpoint_compatible(
                cached["meta"],
                checkpoint_meta,
                extra_keys=resume_extra_keys,
            ):
                print(f"Resuming from audio cache {audio_cache_path}")
                audio_tokens_list = cached["audio_tokens_list"]
                texts = cached["texts"]

        if audio_tokens_list is None:
            audio, adapter, ckpt = _load_nll_audio_encoder_and_adapter(
                device=device,
                torch_dtype=torch_dtype,
                checkpoint_path=args.checkpoint,
                whisper_model_id=model_ids.whisper_model_id,
                stage=stage,
                stage2=stage2,
            )
            audio_tokens_list, texts = _encode_and_cache_nll_audio(
                audio_cache_path=audio_cache_path,
                checkpoint_meta=checkpoint_meta,
                audio=audio,
                adapter=adapter,
                pairs=pairs,
                device=device,
            )
            print("Releasing Whisper and adapter before loading causal LM...")
            _free_cuda(audio, adapter, device=device)

        checkpoint_meta["num_utterances"] = len(texts)

        tokenizer, llm, text_embedder, llm_device = _load_nll_llm(
            device=device,
            torch_dtype=torch_dtype,
            llm_model_id=model_ids.llm_model_id,
        )

        print("Tokenizing candidate transcripts...")
        input_ids, attention_mask = _tokenize_candidates(
            tokenizer, texts, max_text_tokens=args.max_text_tokens
        )

        start_row = 0
        partial_scores = None
        if args.resume:
            partial = load_retrieval_nll_matrix_checkpoint(matrix_cache_path)
            if partial and retrieval_nll_checkpoint_compatible(
                partial["meta"],
                checkpoint_meta,
                extra_keys=resume_extra_keys,
            ):
                rows_done = int(partial["rows_done"])
                n = len(audio_tokens_list)
                if partial["matrix"].shape[0] != n:
                    raise RuntimeError(
                        f"Checkpoint matrix size {partial['matrix'].shape[0]} "
                        f"does not match current audio cache size {n}"
                    )
                if 0 < rows_done < n:
                    start_row = rows_done
                    partial_scores = partial["matrix"]
                    print(
                        f"Resuming NLL matrix from row {start_row}/{n} "
                        f"({matrix_cache_path})"
                    )
                elif rows_done >= n:
                    print(f"NLL matrix already complete at {matrix_cache_path}")
                    partial_scores = partial["matrix"]

        if partial_scores is not None and start_row >= len(audio_tokens_list):
            nll_scores = partial_scores
        else:
            if start_row == 0:
                print("Computing NLL matrix (this can be slow)...")
            else:
                print(f"Computing remaining NLL rows {start_row + 1}-{len(audio_tokens_list)}...")
            nll_scores = _nll_matrix(
                audio_tokens_list,
                input_ids,
                attention_mask,
                llm,
                text_embedder,
                llm_device,
                batch_size=args.candidate_batch_size,
                autocast_device=device,
                matrix_checkpoint_path=matrix_cache_path,
                checkpoint_meta=checkpoint_meta,
                start_row=start_row,
                out_scores=partial_scores,
            )

    print("Computing retrieval metrics...")
    results = compute_retrieval_metrics_from_cost_matrix(nll_scores, ks=(1, 5, 10))

    _print_retrieval_nll_metrics(results, stage=stage)

    meta: dict[str, Any] = {
        "run_name": run_name,
        "checkpoint": args.checkpoint,
        "dataset_root": args.dataset_root,
        "num_utterances": nll_scores.shape[0],
        "candidate_batch_size": args.candidate_batch_size,
        "max_text_tokens": args.max_text_tokens,
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
        "eval_type": "retrieval_nll",
        "stage": stage,
        "rank_only": bool(args.rank_only),
    }
    if rank_info is not None:
        meta.update(rank_info)
    if stage == 2 and stage2 is not None:
        meta["use_rate_controller"] = stage2.use_rate_controller

    out_path = experiment_output_dir(args.output_dir, run_name) / "retrieval_nll.json"
    write_eval_json(out_path, results, meta)
    print(f"\nWrote {out_path}")
    release_cuda_memory()


# ---------------------------------------------------------------------------
# ASR (WER / BLEU-4)
# ---------------------------------------------------------------------------


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


def _wer(reference: str, hypothesis: str) -> float:
    ref = _tokenize_words(reference)
    hyp = _tokenize_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / float(len(ref))


def _bleu4(reference: str, hypothesis: str) -> float:
    """Sentence-level BLEU-4 (same smoothing as corpus BLEU)."""
    return _corpus_bleu4([reference], [hypothesis])


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


def _utterance_id(audio_path: str) -> str:
    return Path(audio_path).stem


def _resolve_lm_conditioning(
    *,
    stage: int,
    prompt_asr: bool,
    append_im_end: bool,
) -> tuple[bool, str]:
    if prompt_asr or stage >= 3:
        return False, PROMPT_CONDITIONING
    if append_im_end:
        return True, TRAIN_STYLE_CONDITIONING
    return True, TRAIN_STYLE_NO_IM_END_CONDITIONING


def _build_stage2_gate_from_checkpoint(
    checkpoint_path: str,
    *,
    device: str,
    torch_dtype: torch.dtype,
) -> tuple[TurnEndCommitGate, dict[str, Any]]:
    """Build turn-end gate from checkpoint hyperparams (aligned with adapter_asr_trainer)."""
    from dataclasses import replace

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    gate_defaults = GateConfig.from_env()
    gate_cfg = resolve_gate_config_from_checkpoint(ckpt, defaults=gate_defaults)
    if gate_cfg.silence_mode == "both":
        gate_cfg = replace(gate_cfg, active_silence_path=gate_defaults.active_silence_path)
    gate = build_turn_end_gate(
        d_llm=LLM_DIM,
        hidden_dim=gate_cfg.hidden_dim,
        threshold=gate_cfg.threshold,
        latency_weight=gate_cfg.latency_weight,
        min_silence_ms=gate_cfg.min_silence_ms,
        require_silence_for_commit=gate_cfg.require_silence_for_commit,
        token_activity_threshold=gate_cfg.token_activity_threshold,
        window_duration_sec=gate_cfg.window_seconds,
        silence_mode=gate_cfg.silence_mode,
        active_silence_path=gate_cfg.active_silence_path,
        learned_silence_hidden_dim=gate_cfg.learned_silence_hidden_dim,
        device=device,
        dtype=torch_dtype,
    )
    return gate, ckpt


def _build_asr_adapter(
    *,
    stage: int,
    stage2: Stage2Config,
    use_rate_controller: bool | None = None,
) -> StreamingAdapter:
    if use_rate_controller is None:
        use_rc = stage == 2 and stage2.use_rate_controller
    else:
        use_rc = use_rate_controller
    return StreamingAdapter(
        d_encoder=768,
        d_llm=4096,
        num_queries=2,
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


def _load_asr_checkpoint_into_models(
    checkpoint_path: str,
    *,
    stage: int,
    adapter: StreamingAdapter,
    gate: TurnEndCommitGate | None,
    device: str,
) -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    _load_adapter_state_dict(adapter, ckpt)
    meta = {
        "checkpoint": checkpoint_path,
        "stage": stage,
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
        "trainer": _ckpt_trainer_name(ckpt),
        "use_rate_controller": bool(adapter.use_rate_controller),
        "has_gate": gate is not None,
    }
    if stage == 2 and gate is not None:
        if "gate_state_dict" not in ckpt:
            raise KeyError(
                f"Stage 2 checkpoint missing gate_state_dict: {checkpoint_path}. "
                "Use --stage 1 for checkpoints without a gate."
            )
        load_gate_state_dict_safe(gate, ckpt)
    adapter.eval()
    if gate is not None:
        gate.eval()
    return meta


def _build_asr_pipeline(
    *,
    stage: int,
    checkpoint_path: str,
    device: str,
    torch_dtype: torch.dtype,
    model_ids: FrozenModelIdsConfig,
    stage2: Stage2Config,
    llm_device_map: str | None,
    llm_max_memory: dict[int | str, str] | None = None,
):
    whisper = load_whisper_models(
        cfg=WhisperConfig(
            model_id=model_ids.whisper_model_id,
            device=device,
            torch_dtype=torch_dtype,
        )
    )
    llm_models = _load_eval_causal_lm(
        model_id=model_ids.llm_model_id,
        device=device,
        torch_dtype=torch_dtype,
        device_map=llm_device_map,
        max_memory=llm_max_memory,
    )

    windowizer = AudioWaveformWindowizer(
        sample_rate=16000,
        window_seconds=0.8,
        stride_seconds=0.4,
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    use_rc = _resolve_use_rate_controller(ckpt, stage=stage, stage2=stage2)
    use_gate = stage == 2 and _ckpt_has_gate(ckpt)
    trainer = _ckpt_trainer_name(ckpt)

    adapter = _build_asr_adapter(
        stage=stage, stage2=stage2, use_rate_controller=use_rc
    ).to(device, dtype=torch_dtype)
    _load_adapter_state_dict(adapter, ckpt)
    adapter.eval()

    gate: TurnEndCommitGate | None = None
    if use_gate:
        gate, _ = _build_stage2_gate_from_checkpoint(
            checkpoint_path,
            device=device,
            torch_dtype=torch_dtype,
        )
        load_gate_state_dict_safe(gate, ckpt)
        gate.eval()
    elif stage == 2:
        print(
            "  Stage-2 checkpoint has no gate_state_dict "
            f"(trainer={trainer or 'unknown'}); "
            "evaluating with WhisperAdapterLLMPipeline (asr-align / asr-only style)."
        )

    ckpt_meta = {
        "checkpoint": checkpoint_path,
        "stage": stage,
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
        "trainer": trainer,
        "use_rate_controller": use_rc,
        "has_gate": use_gate,
    }

    base_kwargs = dict(
        whisper_processor=whisper.processor,
        whisper_model=whisper.model,
        windowizer=windowizer,
        streaming_adapter=adapter,
        llm_model=llm_models.causal_lm,
        llm_tokenizer=llm_models.tokenizer,
        device=device,
        torch_dtype=torch_dtype,
    )

    if use_gate:
        assert gate is not None
        pipeline = WhisperAdapterLLMCommitGatePipeline(
            early_commit_gate=gate,
            **base_kwargs,
        )
    else:
        pipeline = WhisperAdapterLLMPipeline(**base_kwargs)

    return pipeline, ckpt_meta


def _encode_asr_tokens_for_eval(
    pipeline: WhisperAdapterLLMPipeline | WhisperAdapterLLMCommitGatePipeline,
    wave: torch.Tensor,
    *,
    n_windows: int,
    use_early_commit_truncation: bool,
) -> tuple[torch.Tensor, int]:
    """Encode one waveform to compressed adapter tokens (no LLM decode)."""
    if isinstance(pipeline, WhisperAdapterLLMCommitGatePipeline):
        enc, windows, _ = pipeline.encode_waveform(wave)
        _ = enc
        window_list = WhisperAdapterLLMPipeline._select_windows(windows, n_windows)
        adapter = pipeline._llm.streaming_adapter
        gate = pipeline.early_commit_gate
        adapter.reset_streaming_state()
        silence_tracker = (
            gate.make_silence_tracker() if gate.silence_mode in ("rule", "both") else None
        )
        learned_silence_tracker = (
            gate.make_learned_silence_tracker()
            if gate.silence_mode in ("learned", "both")
            else None
        )
        chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for t, w in enumerate(window_list):
                wdev = w.to(device=pipeline._llm.device, dtype=pipeline._llm.torch_dtype)
                step = adapter.forward_window(wdev)
                chunks.append(step["tokens"])
                accumulated = torch.cat(chunks, dim=1)
                gr = gate(
                    accumulated,
                    t,
                    len(window_list),
                    endpoint_label=None,
                    silence_tracker=silence_tracker,
                    learned_silence_tracker=learned_silence_tracker,
                    window_tokens=step["tokens"],
                )
                if (
                    use_early_commit_truncation
                    and t < len(window_list) - 1
                    and gr["should_commit"].item() > 0.5
                ):
                    break
        if not chunks:
            raise RuntimeError("Commit-gate encode produced no audio tokens")
        tokens = torch.cat(chunks, dim=1)
        return tokens, len(chunks)

    _, windows, _ = pipeline.encode_waveform(wave)
    window_list = pipeline._select_windows(windows, n_windows)
    with torch.no_grad():
        adapter_out = pipeline.streaming_adapter(window_list)
    tokens = adapter_out["tokens"]
    return tokens, len(window_list)


@torch.no_grad()
def _run_asr_eval(
    pipeline: WhisperAdapterLLMPipeline | WhisperAdapterLLMCommitGatePipeline,
    pairs: list[tuple[str, str]],
    *,
    asr_prompt: str,
    n_windows: int,
    generation: LlmGenerationParams,
    log_every: int,
    batch_size: int = 16,
    use_early_commit_truncation: bool = False,
    train_style_asr: bool = True,
    append_im_end: bool = True,
    conditioning: str = TRAIN_STYLE_CONDITIONING,
) -> tuple[AsrMetrics, list[dict[str, Any]]]:
    refs: list[str] = []
    hyps: list[str] = []
    items: list[dict[str, Any]] = []

    batch_size = max(1, int(batch_size))
    # Prompt-ASR prefixes differ per length/template; keep sequential generate there.
    use_batched_decode = train_style_asr and batch_size > 1

    n_win_msg = "all windows" if n_windows == -1 else f"{n_windows} window(s)"
    rep_pen = generation.repetition_penalty
    print(
        f"Generating transcripts for {len(pairs)} utterances "
        f"(batch_size={batch_size if use_batched_decode else 1}, {n_win_msg}, "
        f"beams={generation.num_beams}, "
        f"max_new_tokens={generation.max_new_tokens}, "
        f"repetition_penalty={rep_pen})...",
        flush=True,
    )
    if train_style_asr:
        suffix = " | im_end/BOS" if append_im_end else ""
        print(f"  LM conditioning: {conditioning} ([audio{suffix}] → generate)", flush=True)
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
            "the first batch can take several minutes.",
            flush=True,
        )

    llm_pipeline = (
        pipeline._llm if isinstance(pipeline, WhisperAdapterLLMCommitGatePipeline) else pipeline
    )

    for i0 in range(0, len(pairs), batch_size if use_batched_decode else 1):
        batch_pairs = pairs[i0 : i0 + (batch_size if use_batched_decode else 1)]
        if i0 == 0:
            print(
                f"  Starting utterances {i0 + 1}-{i0 + len(batch_pairs)}/{len(pairs)}...",
                flush=True,
            )

        if not use_batched_decode:
            audio_path, reference = batch_pairs[0]
            wave = load_mono_waveform_16k(audio_path)
            t0 = time.time()
            gen_kwargs = dict(
                n_windows=n_windows,
                prompt=asr_prompt,
                generation=generation,
                train_style_asr=train_style_asr,
                append_im_end=append_im_end,
            )
            if isinstance(pipeline, WhisperAdapterLLMCommitGatePipeline):
                result = pipeline.generate(
                    wave,
                    use_early_commit_truncation=use_early_commit_truncation,
                    **gen_kwargs,
                )
            else:
                result = pipeline.generate(wave, **gen_kwargs)
            elapsed = time.time() - t0
            ref = _normalize_text(reference)
            hyp = _normalize_text(result["text"])
            w = _wer(ref, hyp)
            b = _bleu4(ref, hyp)
            items.append(
                {
                    "utterance_id": _utterance_id(audio_path),
                    "audio_path": audio_path,
                    "reference": ref,
                    "prediction": hyp,
                    "wer": w,
                    "bleu4": b,
                    "latency_s": elapsed,
                    "num_windows_used": result.get("num_windows_used"),
                }
            )
            refs.append(ref)
            hyps.append(hyp)
            i = i0
            if log_every > 0 and (
                i == 0 or (i + 1) % log_every == 0 or i + 1 == len(pairs)
            ):
                print(
                    f"  [{i + 1}/{len(pairs)}] WER={w:.3f} BLEU-4={b:.3f} "
                    f"windows={result.get('num_windows_used')} ({elapsed:.1f}s)",
                    flush=True,
                )
            continue

        t0 = time.time()
        tokens_list: list[torch.Tensor] = []
        windows_used: list[int] = []
        paths: list[str] = []
        references: list[str] = []
        for audio_path, reference in batch_pairs:
            wave = load_mono_waveform_16k(audio_path)
            tokens, n_used = _encode_asr_tokens_for_eval(
                pipeline,
                wave,
                n_windows=n_windows,
                use_early_commit_truncation=use_early_commit_truncation,
            )
            tokens_list.append(tokens)
            windows_used.append(n_used)
            paths.append(audio_path)
            references.append(reference)

        predictions = decode_asr_predictions_batch(
            llm_model=llm_pipeline.llm_model,
            llm_tokenizer=llm_pipeline.llm_tokenizer,
            audio_tokens_list=tokens_list,
            llm_device=llm_pipeline._llm_tensor_device(),
            torch_dtype=llm_pipeline.torch_dtype,
            generation=generation,
            append_im_end=append_im_end,
            trim_asr_tail=train_style_asr,
        )
        elapsed = time.time() - t0
        per_utt = elapsed / max(len(batch_pairs), 1)

        for j, (audio_path, reference, hyp_raw, n_used) in enumerate(
            zip(paths, references, predictions, windows_used, strict=True)
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
                    "num_windows_used": n_used,
                }
            )
            refs.append(ref)
            hyps.append(hyp)
            if log_every > 0 and (
                idx == 0 or (idx + 1) % log_every == 0 or idx + 1 == len(pairs)
            ):
                print(
                    f"  [{idx + 1}/{len(pairs)}] WER={w:.3f} BLEU-4={b:.3f} "
                    f"windows={n_used} (~{per_utt:.1f}s/utt, batch={len(batch_pairs)})",
                    flush=True,
                )

    avg_wer = sum(p["wer"] for p in items) / max(len(items), 1)
    bleu = _corpus_bleu4(refs, hyps)
    metrics = AsrMetrics(num_samples=len(items), avg_wer=float(avg_wer), bleu4=float(bleu))
    return metrics, items


def _select_asr_pairs(
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


def _write_asr_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _evaluate_asr_one_variant(
    *,
    stage: int,
    checkpoint_path: str,
    dataset_root: str,
    output_dir: Path,
    num_samples: int | str,
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
    batch_size: int,
    use_early_commit_truncation: bool,
    prompt_asr: bool,
    append_im_end: bool,
    run_name: str,
    llm_max_memory: dict[int | str, str] | None = None,
) -> dict[str, Any]:
    print(f"\n{'=' * 60}\nASR eval — {run_name}\n  checkpoint: {checkpoint_path}\n{'=' * 60}")

    resolved_samples = resolve_num_samples(dataset_root, num_samples)
    pairs = _select_asr_pairs(dataset_root, resolved_samples, seed)
    print(f"Evaluating {len(pairs)} utterances from {dataset_root}\n")

    pipeline, ckpt_meta = _build_asr_pipeline(
        stage=stage,
        checkpoint_path=checkpoint_path,
        device=device,
        torch_dtype=torch_dtype,
        model_ids=model_ids,
        stage2=stage2,
        llm_device_map=llm_device_map,
        llm_max_memory=llm_max_memory,
    )
    print(
        f"Loaded {run_name}: epoch={ckpt_meta.get('epoch')} "
        f"step={ckpt_meta.get('global_step')}\n"
    )

    early_trunc = (
        stage == 2
        and use_early_commit_truncation
        and bool(ckpt_meta.get("has_gate"))
    )
    train_style_asr, conditioning = _resolve_lm_conditioning(
        stage=stage, prompt_asr=prompt_asr, append_im_end=append_im_end
    )
    print(f"  append_im_end={append_im_end}")
    metrics, items = _run_asr_eval(
        pipeline,
        pairs,
        asr_prompt=asr_prompt,
        n_windows=n_windows,
        generation=generation,
        log_every=log_every,
        batch_size=batch_size,
        use_early_commit_truncation=early_trunc,
        train_style_asr=train_style_asr,
        append_im_end=append_im_end,
        conditioning=conditioning,
    )

    print(f"\n{run_name} results: avg_WER={metrics.avg_wer:.4f} BLEU-4={metrics.bleu4:.4f}")

    stage_dir = output_dir / run_name
    _write_asr_json(
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
                "append_im_end": append_im_end,
                "batch_size": batch_size,
                "device": device,
            },
        },
    )
    _write_asr_json(stage_dir / "asr_predictions.json", {"items": items})

    return {
        "stage": stage,
        "run_name": run_name,
        "checkpoint": checkpoint_path,
        "metrics": asdict(metrics),
        "output_dir": str(stage_dir),
    }


def run_asr(*, stage: int, args: argparse.Namespace) -> None:
    """Run LibriSpeech ASR eval (WER + BLEU-4) for a single stage 1 or 2 checkpoint."""
    if stage not in (1, 2):
        raise ValueError(f"stage must be 1 or 2, got {stage}")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    if args.asr_prompt is None:
        from training.utils.asr_prompt import DEFAULT_ASR_PROMPT

        args.asr_prompt = DEFAULT_ASR_PROMPT

    model_ids = _resolve_eval_model_ids(stage)
    stage2 = Stage2Config.from_env()
    device_obj, torch_dtype, _ = init_eval_device()
    device = str(device_obj)

    out_dir = Path(args.output_dir)
    generation = LlmGenerationParams(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        num_beams=max(1, args.num_beams),
        temperature=0.7 if args.do_sample else None,
        repetition_penalty=float(args.repetition_penalty),
        no_repeat_ngram_size=int(args.no_repeat_ngram_size),
    )
    llm_map, llm_max_memory = qwen_device_map_and_max_memory()
    if args.llm_device_map == "none":
        llm_map, llm_max_memory = None, None
    elif args.llm_device_map != "auto":
        llm_map = args.llm_device_map

    print(f"Device: {device}  LLM={model_ids.llm_model_id}  device_map: {llm_map!r}")
    if llm_max_memory:
        print(f"LLM max_memory: {llm_max_memory!r}")

    base_run_name = resolve_run_name(args.checkpoint, args.run_name)
    im_end_variants = [True, False] if args.compare_im_end else [args.append_im_end]

    for append_im_end in im_end_variants:
        if args.compare_im_end:
            suffix = "im_end" if append_im_end else "no_im_end"
            run_name = f"{base_run_name}_{suffix}"
        else:
            run_name = base_run_name

        _evaluate_asr_one_variant(
            stage=stage,
            checkpoint_path=args.checkpoint,
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
            llm_max_memory=llm_max_memory,
            log_every=args.log_every,
            batch_size=args.batch_size,
            use_early_commit_truncation=args.early_commit_truncation,
            prompt_asr=args.prompt_asr,
            append_im_end=append_im_end,
            run_name=run_name,
        )
        release_cuda_memory()



STAGE = 1
EVAL_METRICS = ("retrieval-cosine", "retrieval-nll", "asr")


def _build_parser() -> argparse.ArgumentParser:
    ckpt_cfg = CheckpointConfig.from_env(pkg_root=_pkg_root)
    default_checkpoint = os.path.join(ckpt_cfg.dir, "adapter_stage1.pt")
    default_test_root = LibriSpeechConfig.test_clean_root(_training_dir)
    default_output_dir = os.path.join(_pkg_root, "outputs", "experiments", "July-3-eval")

    ap = argparse.ArgumentParser(description="Stage 1 eval (retrieval-cosine | retrieval-nll | asr)")
    ap.add_argument("--metric", required=True, choices=EVAL_METRICS)
    ap.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=None,
        help="Force compute device; cpu skips GPU lock (use while training holds GPUs)",
    )
    ap.add_argument(
        "--num-threads",
        type=int,
        default=env_optional_int("EVAL_NUM_THREADS") or 16,
        help="Max CPU threads for PyTorch/BLAS (default 16; 0 = no cap)",
    )
    ap.add_argument("--checkpoint", type=str, default=default_checkpoint)
    ap.add_argument("--dataset-root", type=str, default=default_test_root)
    ap.add_argument("--num-samples", default=str(env_int("RETRIEVAL_NUM_UTTERANCES", 100)))
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default=default_output_dir)
    ap.add_argument("--max-text-tokens", type=int, default=None)
    ap.add_argument("--candidate-batch-size", type=int, default=None)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--rank-only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--asr-prompt", type=str, default=None)
    ap.add_argument("--n-windows", type=int, default=-1)
    ap.add_argument("--max-new-tokens", type=int, default=496)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--do-sample", action="store_true", default=False)
    ap.add_argument("--repetition-penalty", type=float, default=1.25)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=4)
    ap.add_argument("--prompt-asr", action="store_true")
    ap.add_argument("--early-commit-truncation", action="store_true", default=False)
    ap.add_argument("--llm-device-map", type=str, default="auto")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument(
        "--batch-size",
        type=int,
        default=env_int("VAL_BATCH_SIZE", 16),
        help="ASR decode batch size for train-style eval (default: VAL_BATCH_SIZE or 16)",
    )
    im_end = ap.add_mutually_exclusive_group()
    im_end.add_argument("--append-im-end", dest="append_im_end", action="store_true", default=True)
    im_end.add_argument("--no-append-im-end", dest="append_im_end", action="store_false")
    ap.add_argument("--compare-im-end", action="store_true")
    return ap


def _configure_cpu_threads(num_threads: int) -> None:
    """Cap PyTorch/BLAS thread pools (0 = leave defaults)."""
    if num_threads <= 0:
        return
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = str(num_threads)
    torch.set_num_threads(num_threads)
    interop = max(1, min(num_threads, 4))
    torch.set_num_interop_threads(interop)
    print(f"CPU threads capped at {num_threads} (interop={interop})")


def _apply_device_override(device: str | None) -> None:
    """Force CPU eval: skip GPU locks and hide CUDA devices from this process."""
    if device != "cpu":
        return
    os.environ["DEVICE"] = "cpu"
    os.environ["GPU_LOCK"] = "off"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.pop("LLM_DEVICE", None)
    os.environ.pop("LLM_MAX_MEMORY", None)


def _apply_stage1_defaults(args: argparse.Namespace) -> None:
    
    if args.metric == "retrieval-cosine" and args.max_text_tokens is None:
        args.max_text_tokens = 128

    
    if args.metric == "retrieval-nll":
        if args.max_text_tokens is None:
            args.max_text_tokens = 256
        if args.num_samples == str(env_int("RETRIEVAL_NUM_UTTERANCES", 100)):
            args.num_samples = str(env_int("RETRIEVAL_NLL_NUM_UTTERANCES", 2620))
    if args.metric == "asr" and args.num_samples == str(env_int("RETRIEVAL_NUM_UTTERANCES", 100)):
        args.num_samples = str(env_int("ASR_EVAL_NUM_SAMPLES", 100))


def main() -> None:
    load_project_env(_pkg_root)
    apply_runtime_cuda_env()
    args = _build_parser().parse_args()
    _apply_device_override(args.device)
    _configure_cpu_threads(args.num_threads)
    _apply_stage1_defaults(args)
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
