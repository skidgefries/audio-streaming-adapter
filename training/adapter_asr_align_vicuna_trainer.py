"""
Stage 2 (simplified, Vicuna): ASR + align + stability — train StreamingAdapter only.

Frozen Whisper encoder and frozen Vicuna-7B (``lmsys/vicuna-7b-v1.5``, HF cache if
present). No rate controller, no turn-end gate, no early-commit gate. Adapter uses
the EMA stability buffer (``ema_alpha``).

Loss (all backprop through adapter)::

    total = ASR + λ_align · InfoNCE(align) + λ_stability · stability_buffer_loss

InfoNCE uses ``TEMPERATURE`` from ``.env`` (default 0.07). Stage 1 warm-start path
is ``STAGE1_CHECKPOINT`` in ``.env``. Micro-batch ``BATCH_SIZE`` (default 16);
effective batch ``MACRO_BATCH_SIZE`` (default 128) via gradient accumulation.

**Run**::

    uv run training/adapter_asr_align_vicuna_trainer.py

GPU layout (overrides ``.env`` when unset): ``DEVICE=cpu`` (Whisper + adapter),
``LLM_DEVICE=cuda:0`` with ``LLM_MAX_MEMORY=0:14GiB,1:2GiB,cpu:64GiB`` (Vicuna on
free GPU 0; minimal 2 GiB spill to GPU 1 if needed). VRAM caps auto-shrink when a
GPU is busy. Override the LLM id with ``VICUNA_MODEL_ID`` (``.env``
``LLM_MODEL_ID`` is ignored so Qwen Stage 2 is not mixed in).
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from datetime import datetime, timezone

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_root = os.path.join(_pkg_root, "src")
sys.path.insert(0, _src_root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, _pkg_root)

from training.utils.env import apply_hf_hub_endpoint, env_bool, env_int, env_str, load_project_env

_env_path = load_project_env(_pkg_root)
if _env_path:
    print(f"Loaded environment from {_env_path}")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
# Whisper + adapter on CPU; frozen Vicuna on GPU (cuda:0 fill, cuda:1 2 GiB spill, then CPU).
os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("LLM_DEVICE", "cuda:0")
os.environ.setdefault("LLM_MAX_MEMORY", "0:14GiB,1:2GiB,cpu:64GiB")
os.environ.setdefault("BATCH_SIZE", "16")
os.environ.setdefault("MACRO_BATCH_SIZE", "128")
os.environ.setdefault("ASR_MICRO_BATCH_SIZE", "16")
os.environ.setdefault("MAX_WINDOWS_PER_UTT", "32")
os.environ.setdefault("MAX_TEXT_TOKENS", "128")
os.environ.setdefault("ENABLE_LLM_GRADIENT_CHECKPOINTING", "true")
os.environ.setdefault("GPU_LOCK", "true")
os.environ.setdefault("WANDB_ENABLED", "true")
os.environ.setdefault("WANDB_RUN_NAME", "adapter_asr_align_vicuna")
os.environ.setdefault("SAVE_EVERY_STEPS", "1000")
os.environ.setdefault("VAL_EVERY_STEPS", "1000")
os.environ.setdefault("VAL_MAX_UTTERANCES", "100")

_hf_endpoint = apply_hf_hub_endpoint(_pkg_root)
print(f"HF Hub endpoint: {_hf_endpoint}")

_hf_token = env_str("HF_TOKEN")
if _hf_token:
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", _hf_token)

from training.utils.devices import apply_runtime_cuda_env
from training.utils.gpu_reservation import reserve_gpus

apply_runtime_cuda_env()
reserve_gpus()

if env_str("CHECK_TORCH_COMPAT", "0") in ("1", "true", "yes", "on"):
    from training.utils.torch_compat import ensure_torch_compatible

    ensure_torch_compatible()

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from llm.config import LlmGenerationParams
from src.adapter.streaming_adapter import StreamingAdapter
from src.dataset import LibriSpeechConfig, LibriSpeechPairs, load_mono_waveform_16k
from src.encoder import WhisperWindowFeatureExtractor
from training.utils.asr_prompt import DEFAULT_ASR_PROMPT, PROMPT_CONDITIONING, TRAIN_STYLE_CONDITIONING
from training.utils.asr_only_validation import UtteranceEncodeResult, validate_asr_only
from training.utils.checkpointing import (
    TrainingCheckpoint,
    adapt_optimizer_state_dict_num_queries,
    filter_adapter_state_dict,
    maybe_upload_stage_epoch_checkpoint,
    resolve_resume_epoch_and_offset,
    save_checkpoint,
)
from training.utils.config import (
    CheckpointConfig,
    DataConfig,
    DeviceConfig,
    FrozenModelIdsConfig,
    HfCheckpointConfig,
    OptimConfig,
    Stage2Config,
    WandbConfig,
)
from training.utils.devices import (
    adjust_llm_max_memory_for_free_vram,
    cleanup_distributed,
    ensure_device_ready,
    init_training_context,
    llm_input_device,
    resolve_llm_load_plan,
)
from training.utils.loaders import load_frozen_vicuna_causal_lm
from training.utils.logging import WandbLogger
from training.utils.losses import (
    contrastive_infonce_loss_learnable_temperature,
    init_contrastive_logit_scale,
)
from training.utils.metrics import RunningMean
from training.utils.optimization import TrainingPipeline

_TRAINING_DIR = os.path.dirname(__file__)

_MODEL_IDS = FrozenModelIdsConfig.from_env()
WHISPER_DIM = 768
LLM_DIM = 4096  # Vicuna-7B hidden size
WHISPER_MODEL = _MODEL_IDS.whisper_model_id
# Dedicated env so .env LLM_MODEL_ID=Qwen/... does not override this trainer.
LLM_MODEL_ID = env_str("VICUNA_MODEL_ID", "lmsys/vicuna-7b-v1.5") or "lmsys/vicuna-7b-v1.5"

DATASET_ROOTS = LibriSpeechConfig.resolve_train_roots(
    _TRAINING_DIR,
    env_override=env_str("DATASET_ROOT"),
)
VAL_ROOT = LibriSpeechConfig.dev_clean_root(_TRAINING_DIR)

STAGE = Stage2Config.from_env()
DEVICE_CFG = DeviceConfig.from_env()
TRAIN_DTYPE = (
    torch.float32
    if DEVICE_CFG.device.strip().lower() == "cpu"
    else (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
)
LLM_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
TORCH_DTYPE = TRAIN_DTYPE
OPT = OptimConfig.from_env()
DATA = DataConfig.from_env(default_dataset_root=DATASET_ROOTS[0])
if DATA.macro_batch_size is None:
    DATA = replace(DATA, macro_batch_size=128)
CKPT = CheckpointConfig.from_env(pkg_root=_pkg_root)
HF_CKPT = HfCheckpointConfig.from_env()
WANDB = WandbConfig.from_env()

CHECKPOINT_BASENAME = (
    env_str("CHECKPOINT_BASENAME", "adapter_asr_align_vicuna") or "adapter_asr_align_vicuna"
)
SAVE_PATH = os.path.join(CKPT.dir, f"{CHECKPOINT_BASENAME}.pt")
SAVE_EVERY_STEPS = CKPT.save_every_steps if CKPT.save_every_steps is not None else 1000
LOG_EVERY_STEPS = max(1, DATA.batch_size)
VAL_HISTORY_PATH = os.path.join(CKPT.dir, f"{CHECKPOINT_BASENAME}_val_history.jsonl")
_stage1_rel = env_str("STAGE1_CHECKPOINT")
STAGE1_SAVE_PATH = (
    None
    if not _stage1_rel
    else (_stage1_rel if os.path.isabs(_stage1_rel) else os.path.join(_pkg_root, _stage1_rel))
)
LOAD_STAGE1_CHECKPOINT = env_bool("LOAD_STAGE1_CHECKPOINT", True)


def _load_adapter_from_checkpoint(
    adapter: StreamingAdapter,
    ckpt: dict,
    *,
    source: str,
    is_main: bool,
) -> None:
    adapter.load_state_dict(
        filter_adapter_state_dict(
            ckpt["adapter_state_dict"],
            num_queries=adapter.num_queries,
            use_rate_controller=False,
        ),
        strict=False,
    )
    if is_main:
        print(
            f"Loaded adapter weights from {source} "
            f"(stage {ckpt.get('stage', '?')}, epoch {ckpt.get('epoch', '?')})"
        )


def _maybe_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _asr_forward_loss(
    llm_model: torch.nn.Module,
    *,
    inputs_embeds: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    micro_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    n = inputs_embeds.shape[0]
    chunk = max(1, min(micro_batch_size, n))
    if chunk >= n:
        with _maybe_autocast(device):
            return llm_model(
                inputs_embeds=inputs_embeds,
                labels=labels,
                attention_mask=attention_mask,
            ).loss

    losses: list[torch.Tensor] = []
    for start in range(0, n, chunk):
        end = start + chunk
        with _maybe_autocast(device):
            out = llm_model(
                inputs_embeds=inputs_embeds[start:end],
                labels=labels[start:end],
                attention_mask=attention_mask[start:end],
            )
        losses.append(out.loss)
    return torch.stack(losses).mean()


def _forward_utterance_tokens(
    *,
    adapter: StreamingAdapter,
    audio_extractor: WhisperWindowFeatureExtractor,
    audio_path: str,
    train_device: torch.device,
    max_windows_per_utt: int | None,
) -> tuple[torch.Tensor | None, torch.Tensor, int]:
    """Encode one utterance; returns (tokens, stability_sum, num_windows)."""
    wave = load_mono_waveform_16k(audio_path)
    windows = audio_extractor.waveform_to_windows(wave)
    if max_windows_per_utt is not None:
        windows = windows[:max_windows_per_utt]
    if not windows:
        return None, torch.zeros((), device=train_device, dtype=torch.float32), 0

    adapter.reset_streaming_state()
    utterance_tokens: list[torch.Tensor] = []
    stability_sum = torch.zeros((), device=train_device, dtype=torch.float32)
    for window in windows:
        result = adapter.forward_window(window.to(device=train_device, dtype=TORCH_DTYPE))
        utterance_tokens.append(result["tokens"])
        stability_sum = stability_sum + result["stability_loss"].float()
    return torch.cat(utterance_tokens, dim=1), stability_sum, len(windows)


def _pipeline_begin_microbatch(pipeline: TrainingPipeline) -> None:
    if pipeline.accum_step == 0:
        pipeline.optimizer.zero_grad(set_to_none=True)


def _pipeline_accumulate_backward(
    pipeline: TrainingPipeline,
    loss: torch.Tensor,
    *,
    no_sync_modules: tuple[torch.nn.Module, ...] | None,
    sync_grads: bool,
    retain_graph: bool = False,
) -> None:
    scaled = loss / pipeline.gradient_accumulation_steps
    use_no_sync = no_sync_modules is not None and not sync_grads
    if use_no_sync:
        with ExitStack() as stack:
            for module in no_sync_modules:
                if isinstance(module, DDP):
                    stack.enter_context(module.no_sync())
            scaled.backward(retain_graph=retain_graph)
    else:
        scaled.backward(retain_graph=retain_graph)


def _pipeline_end_microbatch(
    pipeline: TrainingPipeline,
    params: list[torch.nn.Parameter],
) -> bool:
    """Apply optimizer/scheduler after manual backward(s). Returns True if stepped."""
    pipeline._accum_step += 1
    if pipeline._accum_step < pipeline.gradient_accumulation_steps:
        return False

    pipeline._accum_step = 0
    for p in params:
        if p.grad is not None:
            p.grad = torch.where(
                torch.isnan(p.grad) | torch.isinf(p.grad),
                torch.zeros_like(p.grad),
                p.grad,
            )

    has_nan = any(torch.isnan(p.grad).any() for p in params if p.grad is not None)
    if has_nan:
        print(f"[WARN] Step {pipeline.global_step}: NaN gradients detected, skipping update")
        pipeline.global_step += 1
        return True

    if pipeline.grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(params, pipeline.grad_clip_norm)
    pipeline.optimizer.step()
    if pipeline.scheduler is not None:
        pipeline.scheduler.step()
    pipeline.global_step += 1
    return True


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, DDP) else module


def _pad_audio_tokens(utterances: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(t.shape[1] for t in utterances)
    padded = []
    for tokens in utterances:
        pad_len = max_len - tokens.shape[1]
        if pad_len > 0:
            padding = torch.zeros(
                tokens.shape[0],
                pad_len,
                tokens.shape[2],
                device=tokens.device,
                dtype=tokens.dtype,
            )
            tokens = torch.cat([tokens, padding], dim=1)
        padded.append(tokens)
    return torch.cat(padded, dim=0)


def _build_inputs_for_asr(
    *,
    audio_tokens: torch.Tensor,
    gt_ids: torch.Tensor,
    gt_attention_mask: torch.Tensor,
    llm_tokenizer,
    text_embedder: torch.nn.Module,
    llm_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bos_token_id = (
        llm_tokenizer.bos_token_id
        if llm_tokenizer.bos_token_id is not None
        else llm_tokenizer.eos_token_id
    )
    bos_embed = text_embedder(
        torch.tensor([[bos_token_id]], device=llm_device).expand(audio_tokens.shape[0], -1)
    )
    inputs_embeds = torch.cat([audio_tokens.to(device=llm_device, dtype=LLM_DTYPE), bos_embed], dim=1)

    batch_size = audio_tokens.shape[0]
    audio_len = audio_tokens.shape[1]
    pre_text_labels = torch.full(
        (batch_size, audio_len + 1), -100, dtype=torch.long, device=llm_device
    )
    gt_shifted = gt_ids[:, 1:].to(llm_device)
    labels = torch.cat([pre_text_labels, gt_shifted], dim=1)
    inputs_embeds = torch.cat([inputs_embeds, text_embedder(gt_shifted)], dim=1)

    pre_text_mask = torch.ones((batch_size, audio_len + 1), device=llm_device)
    attention_mask = torch.cat([pre_text_mask, gt_attention_mask[:, 1:].to(llm_device)], dim=1)
    return inputs_embeds, labels, attention_mask


def _encode_utterance(
    *,
    adapter: StreamingAdapter,
    audio_extractor: WhisperWindowFeatureExtractor,
    audio_path: str,
    train_device: torch.device,
    max_windows_per_utt: int | None,
    collect_aux_metrics: bool = False,
) -> UtteranceEncodeResult | None:
    tokens, stability_sum, num_windows = _forward_utterance_tokens(
        adapter=adapter,
        audio_extractor=audio_extractor,
        audio_path=audio_path,
        train_device=train_device,
        max_windows_per_utt=max_windows_per_utt,
    )
    if tokens is None:
        return None
    stability_val = float(stability_sum.detach().item()) if collect_aux_metrics else 0.0
    return UtteranceEncodeResult(
        tokens=tokens,
        num_windows=num_windows,
        stability_loss=stability_val,
    )


def _append_val_history(
    *,
    history_path: str,
    global_step: int,
    epoch: int,
    metrics: dict[str, float],
    items: list[dict],
    predictions_path: str,
) -> None:
    """Append full validation payload to a running JSONL history file."""
    os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "global_step": global_step,
        "epoch": epoch + 1,
        "predictions_path": predictions_path,
        "metrics": metrics,
        "num_samples": len(items),
        "items": items,
    }
    with open(history_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  [val step {global_step}] history -> {history_path}", flush=True)


def _compute_val_aux_metrics(
    *,
    audio_tokens: torch.Tensor,
    gt_embeds: torch.Tensor,
    total_stability_loss: float,
    total_windows: int,
    logit_scale: torch.nn.Parameter,
) -> dict[str, float]:
    with torch.no_grad():
        align_loss = float(
            contrastive_infonce_loss_learnable_temperature(
                audio_tokens=audio_tokens.detach().float(),
                text_embeddings=gt_embeds.detach().float(),
                logit_scale=logit_scale,
            ).item()
        )
    stability_loss = total_stability_loss / float(total_windows) if total_windows > 0 else 0.0
    return {"align": align_loss, "stability": stability_loss}


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    warmup = max(0, OPT.warmup_steps)
    cosine_steps = max(1, total_steps - warmup)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup if warmup > 0 else 1,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=1e-6,
    )
    if warmup <= 0:
        return cosine_scheduler
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup],
    )


def _make_checkpoint(
    *,
    epoch: int,
    global_step: int,
    adapter: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    metrics: dict[str, float],
    logit_scale: torch.nn.Parameter,
) -> TrainingCheckpoint:
    return TrainingCheckpoint(
        stage=2,
        epoch=epoch,
        global_step=global_step,
        adapter_state_dict=_unwrap(adapter).state_dict(),
        gate_state_dict=None,
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),
        metrics=metrics,
        contrastive_logit_scale=float(logit_scale.item()),
        hyperparams={
            "trainer": "adapter_asr_align_vicuna_trainer",
            "llm_model_id": LLM_MODEL_ID,
            "stage_cfg": STAGE.__dict__,
            "grad_accum_steps": DATA.gradient_accumulation_steps(),
        },
    )


def _maybe_save_step_checkpoint(
    *,
    epoch: int,
    adapter: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    pipeline: TrainingPipeline,
    metrics: dict[str, float],
    logit_scale: torch.nn.Parameter,
) -> None:
    if SAVE_EVERY_STEPS <= 0:
        return
    step = pipeline.global_step
    if step <= 0 or step % SAVE_EVERY_STEPS != 0:
        return
    ckpt = _make_checkpoint(
        epoch=epoch + 1,
        global_step=step,
        adapter=adapter,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        logit_scale=logit_scale,
    )
    save_checkpoint(SAVE_PATH, ckpt)
    step_path = os.path.join(CKPT.dir, f"{CHECKPOINT_BASENAME}_step{step}.pt")
    save_checkpoint(step_path, ckpt)
    print(f"  Checkpoint (step {step}) -> {SAVE_PATH}\n              step file -> {step_path}")


def _resume_training_state(
    *,
    resume_ckpt: dict,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    pipeline: TrainingPipeline,
    micro_steps_per_epoch: int,
    grad_accum_steps: int,
    num_queries: int,
) -> tuple[int, int]:
    pipeline.global_step = int(resume_ckpt["global_step"])
    start_epoch, batch_offset = resolve_resume_epoch_and_offset(
        resume_ckpt=resume_ckpt,
        micro_steps_per_epoch=micro_steps_per_epoch,
        grad_accum_steps=grad_accum_steps,
        checkpoint_dir=CKPT.dir,
        checkpoint_basename=CHECKPOINT_BASENAME,
    )

    opt_state = resume_ckpt.get("optimizer_state_dict")
    if opt_state is not None:
        opt_state = adapt_optimizer_state_dict_num_queries(
            opt_state,
            adapter_state=resume_ckpt["adapter_state_dict"],
            num_queries=num_queries,
        )
        try:
            optimizer.load_state_dict(opt_state)
        except (ValueError, KeyError):
            print("[WARN] Optimizer state incompatible; restarting optimizer.")
    sched_state = resume_ckpt.get("scheduler_state_dict")
    if sched_state:
        try:
            scheduler.load_state_dict(sched_state)
        except (ValueError, KeyError):
            for _ in range(pipeline.global_step):
                scheduler.step()
    else:
        for _ in range(pipeline.global_step):
            scheduler.step()

    pipeline.reset_accumulation()
    return start_epoch, batch_offset


def train() -> None:
    ctx = init_training_context()
    train_device = ctx.device
    train_device_str = str(train_device)
    llm_device_str, vicuna_map, vicuna_max_memory = resolve_llm_load_plan(
        ctx=ctx,
        train_device=train_device,
        llm_device=DEVICE_CFG.llm_device,
        llm_max_memory=DEVICE_CFG.llm_max_memory,
    )
    llm_load_device = torch.device(llm_device_str)
    ensure_device_ready(llm_load_device)

    use_align = STAGE.lambda_align > 0
    use_stability = STAGE.lambda_stability > 0

    if ctx.is_main:
        print(f"Training device: {train_device_str} (Whisper + adapter)")
        print(f"LLM device: {llm_device_str}")
        print(f"Visible CUDA devices: {ctx.num_cuda_devices}")
        if train_device_str == "cpu" and llm_device_str.startswith("cuda"):
            print("Split layout: encode on CPU, ASR loss on GPU.")
        print(
            "Pipeline: window → encoder → adapter (stability buffer) → LLM | "
            "loss = ASR + λ_align·align + λ_stability·stability"
        )

    if llm_load_device.type == "cuda":
        with torch.cuda.device(llm_load_device):
            torch.cuda.empty_cache()

    vicuna_max_memory = adjust_llm_max_memory_for_free_vram(vicuna_max_memory)

    if ctx.is_main:
        print(f"Vicuna model: {LLM_MODEL_ID}")
        print(f"Vicuna device_map: {vicuna_map!r}")
        if vicuna_max_memory:
            print(f"Vicuna max_memory: {vicuna_max_memory!r}")

    vicuna_models = load_frozen_vicuna_causal_lm(
        model_id=LLM_MODEL_ID,
        device=llm_device_str,
        torch_dtype=LLM_DTYPE,
        device_map=vicuna_map,
        max_memory=vicuna_max_memory,
    )
    llm_tokenizer = vicuna_models.tokenizer
    llm_model = vicuna_models.causal_lm
    if llm_model is None:
        raise RuntimeError(f"Vicuna causal LM failed to load ({LLM_MODEL_ID})")
    text_embedder = vicuna_models.embedder
    llm_dim = int(getattr(vicuna_models, "hidden_size", LLM_DIM) or LLM_DIM)
    llm_device = llm_input_device(llm_model)

    if hasattr(llm_model, "enable_input_require_grads"):
        llm_model.enable_input_require_grads()

    if STAGE.enable_llm_gradient_checkpointing and hasattr(llm_model, "gradient_checkpointing_enable"):
        llm_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    llm_model.eval()
    for p in llm_model.parameters():
        p.requires_grad = False

    if ctx.is_main:
        print(f"Loading Whisper on {train_device_str}...")
    audio = WhisperWindowFeatureExtractor(
        model_id=WHISPER_MODEL, device=train_device_str, torch_dtype=TORCH_DTYPE
    )

    adapter = StreamingAdapter(
        d_encoder=WHISPER_DIM,
        d_llm=llm_dim,
        num_queries=2,
        num_layers=2,
        num_heads=4,
        d_ffn=2048,
        dropout=0.1,
        ema_alpha=0.8,
        learnable_ema=False,
        use_rate_controller=False,
    ).to(train_device, dtype=TORCH_DTYPE)
    adapter.train()

    resume_ckpt: dict | None = None
    resume_training_state = False
    adapter_init_path: str | None = None

    resume_path = CKPT.resume_checkpoint or SAVE_PATH
    if os.path.exists(resume_path):
        resume_meta = torch.load(resume_path, map_location="cpu")
        if int(resume_meta.get("stage", 2)) == 1:
            adapter_init_path = resume_path
            if ctx.is_main:
                print(f"RESUME_CHECKPOINT is Stage 1 ({resume_path}); adapter weights only.")
        else:
            resume_ckpt = torch.load(resume_path, map_location=train_device)
            resume_training_state = True
            if ctx.is_main:
                print(f"Will resume from checkpoint: {resume_path}")
    elif LOAD_STAGE1_CHECKPOINT:
        if not STAGE1_SAVE_PATH:
            raise ValueError(
                "LOAD_STAGE1_CHECKPOINT=true but STAGE1_CHECKPOINT is unset. "
                "Set STAGE1_CHECKPOINT in .env (or LOAD_STAGE1_CHECKPOINT=false)."
            )
        if not os.path.isfile(STAGE1_SAVE_PATH):
            raise FileNotFoundError(
                f"LOAD_STAGE1_CHECKPOINT=true but checkpoint missing: {STAGE1_SAVE_PATH}"
            )
        adapter_init_path = STAGE1_SAVE_PATH
        if ctx.is_main:
            print(f"Will init adapter from Stage 1: {STAGE1_SAVE_PATH}")
    elif ctx.is_main:
        print("Adapter: random init")

    if adapter_init_path is not None and not resume_training_state:
        init_ckpt = torch.load(adapter_init_path, map_location=train_device)
        _load_adapter_from_checkpoint(
            adapter,
            init_ckpt,
            source=adapter_init_path,
            is_main=ctx.is_main,
        )

    if ctx.world_size > 1:
        adapter = DDP(adapter, device_ids=[ctx.local_rank])

    logit_scale = init_contrastive_logit_scale(STAGE.temperature, device=train_device)

    if ctx.is_main:
        print(
            f"Models initialized:\n"
            f"  Encoder: frozen Whisper\n"
            f"  Adapter: trainable (stability buffer, no rate controller)\n"
            f"  LLM: frozen Vicuna ({LLM_MODEL_ID})\n"
            f"  Checkpoint: {CHECKPOINT_BASENAME}\n"
        )

    optimizer = torch.optim.AdamW(
        [*_unwrap(adapter).parameters(), logit_scale],
        lr=OPT.lr,
        weight_decay=OPT.weight_decay,
    )
    grad_accum_steps = DATA.gradient_accumulation_steps()
    effective_batch_size = DATA.batch_size * grad_accum_steps
    pipeline = TrainingPipeline(
        optimizer=optimizer,
        scheduler=None,
        grad_clip_norm=OPT.grad_clip_norm,
        gradient_accumulation_steps=grad_accum_steps,
    )

    dataset = LibriSpeechPairs(DATASET_ROOTS)
    sampler: DistributedSampler | None = None
    if ctx.world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=ctx.world_size, rank=ctx.rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=DATA.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=DATA.num_workers,
    )
    micro_steps_per_epoch = len(dataloader)
    optimizer_steps_per_epoch = max(1, (micro_steps_per_epoch + grad_accum_steps - 1) // grad_accum_steps)
    total_steps = max(1, STAGE.epochs * optimizer_steps_per_epoch)
    scheduler = _build_lr_scheduler(optimizer, total_steps=total_steps)
    pipeline.scheduler = scheduler

    if ctx.is_main:
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    logger = WandbLogger(
        enabled=WANDB.enabled and ctx.is_main,
        project=WANDB.project,
        entity=WANDB.entity,
        run_name=WANDB.run_name,
        config={
            "trainer": "adapter_asr_align_vicuna",
            "llm_model_id": LLM_MODEL_ID,
            "data": {**DATA.__dict__, "dataset_roots": DATASET_ROOTS, "val_root": VAL_ROOT},
            "optim": OPT.__dict__,
            "stage_cfg": STAGE.__dict__,
            "device": DEVICE_CFG.__dict__,
            "checkpoint_basename": CHECKPOINT_BASENAME,
            "stage1_checkpoint": STAGE1_SAVE_PATH,
            "world_size": ctx.world_size,
        },
    )

    start_epoch = 0
    resume_batch_offset = 0
    if resume_training_state and resume_ckpt is not None:
        _load_adapter_from_checkpoint(
            _unwrap(adapter),
            resume_ckpt,
            source=resume_path,
            is_main=ctx.is_main,
        )
        saved_logit_scale = resume_ckpt.get("contrastive_logit_scale")
        if saved_logit_scale is not None:
            logit_scale.data.fill_(float(saved_logit_scale))
        if ctx.is_main:
            start_epoch, resume_batch_offset = _resume_training_state(
                resume_ckpt=resume_ckpt,
                optimizer=optimizer,
                scheduler=scheduler,
                pipeline=pipeline,
                micro_steps_per_epoch=micro_steps_per_epoch,
                grad_accum_steps=grad_accum_steps,
                num_queries=_unwrap(adapter).num_queries,
            )
            print(
                f"  Resumed at epoch {start_epoch + 1}/{STAGE.epochs}, "
                f"global_step {pipeline.global_step}, "
                f"offset {resume_batch_offset}/{len(dataloader)}\n"
            )
    elif adapter_init_path is not None:
        # Warm-start logit scale from Stage 1 when present.
        init_meta = torch.load(adapter_init_path, map_location="cpu")
        saved_logit_scale = init_meta.get("contrastive_logit_scale")
        if saved_logit_scale is None:
            saved_logit_scale = init_meta.get("clap_logit_scale")
        if saved_logit_scale is not None:
            logit_scale.data.fill_(float(saved_logit_scale))
            if ctx.is_main:
                print(
                    f"  Loaded contrastive logit_scale from Stage 1 "
                    f"(temperature={logit_scale.exp().item():.4f})"
                )

    if ctx.is_main:
        print("\nStarting ASR + align + stability training (Vicuna)")
        print(f"  Dataset: {', '.join(os.path.basename(r) for r in DATASET_ROOTS)}")
        print(
            f"  Epochs: {STAGE.epochs} | micro-batch: {DATA.batch_size} | "
            f"effective: {effective_batch_size} | ASR CE chunk: {STAGE.asr_micro_batch_size}"
        )
        print(f"  λ_align: {STAGE.lambda_align} | λ_stability: {STAGE.lambda_stability}")
        print(f"  InfoNCE temperature (learnable, init): {logit_scale.exp().item():.4f}")
        print(f"  LLM: {LLM_MODEL_ID}")
        print(f"  Stage 1 init (STAGE1_CHECKPOINT): {STAGE1_SAVE_PATH}")
        print(f"  Logging: every {LOG_EVERY_STEPS} steps (batch size)")
        if SAVE_EVERY_STEPS > 0:
            print(f"  Checkpoints: every {SAVE_EVERY_STEPS} steps -> {SAVE_PATH}")
        if STAGE.val_enabled and os.path.isdir(VAL_ROOT):
            val_cap = STAGE.val_max_utterances if STAGE.val_max_utterances is not None else "all"
            print(f"  Validation: dev-clean every {STAGE.val_every_steps} steps (max {val_cap} utterances)")
            print(f"  Val history: {VAL_HISTORY_PATH}")
        print(f"  LM conditioning: {TRAIN_STYLE_CONDITIONING}")
        print(f"  Stage 3 target: {DEFAULT_ASR_PROMPT[:72]}...\n")

    adapter_module = _unwrap(adapter)

    for epoch in range(start_epoch, STAGE.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        m_total = RunningMean()
        m_asr = RunningMean()
        m_align = RunningMean()
        m_stab = RunningMean()

        if ctx.is_main:
            print(f"\n{'=' * 60}\nEpoch {epoch + 1}/{STAGE.epochs}\n{'=' * 60}\n")

        batch_start = resume_batch_offset if epoch == start_epoch else 0
        for step, batch in enumerate(dataloader):
            if step < batch_start:
                continue

            audio_paths, transcriptions = batch
            items = list(zip(audio_paths, transcriptions, strict=True))
            if not items:
                continue

            work: list[tuple[str, str, torch.Tensor, torch.Tensor, int]] = []
            for audio_path, text in items:
                tokens, stability_sum, num_windows = _forward_utterance_tokens(
                    adapter=adapter_module,
                    audio_extractor=audio,
                    audio_path=audio_path,
                    train_device=train_device,
                    max_windows_per_utt=DATA.max_windows_per_utt,
                )
                if tokens is None or num_windows == 0:
                    continue
                work.append((audio_path, text, tokens, stability_sum, num_windows))

            if not work:
                if ctx.is_main:
                    print(f"[WARN] Step {step}: empty batch; skipping.")
                continue

            n_valid = len(work)
            _pipeline_begin_microbatch(pipeline)
            no_sync_modules = (adapter,) if ctx.world_size > 1 else None
            at_accum_boundary = pipeline.is_accumulation_boundary

            # Align first on the same encode graph (retain_graph) so ASR can
            # reuse tokens without a second Whisper+adapter pass.
            align_loss = torch.tensor(0.0, device=train_device)
            if use_align and n_valid >= 2:
                align_tokens = [tokens for _path, _text, tokens, _stab, _nw in work]
                texts_for_align = [text for _path, text, _tokens, _stab, _nw in work]
                gt_align = llm_tokenizer(
                    texts_for_align,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=STAGE.max_text_tokens,
                ).to(train_device)
                with torch.no_grad():
                    gt_embeds = text_embedder(gt_align.input_ids.to(llm_device)).to(train_device)
                audio_tokens = _pad_audio_tokens(align_tokens)
                align_loss = contrastive_infonce_loss_learnable_temperature(
                    audio_tokens=audio_tokens.float(),
                    text_embeddings=gt_embeds.float(),
                    logit_scale=logit_scale,
                )
                _pipeline_accumulate_backward(
                    pipeline,
                    STAGE.lambda_align * align_loss,
                    no_sync_modules=no_sync_modules,
                    sync_grads=False,
                    retain_graph=True,
                )

            asr_vals: list[float] = []
            stab_vals: list[float] = []

            for utt_idx, (_path, text, tokens, stability_sum, num_windows) in enumerate(work):
                gt_single = llm_tokenizer(
                    [text],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=STAGE.max_text_tokens,
                ).to(train_device)

                inputs_embeds, labels, attention_mask = _build_inputs_for_asr(
                    audio_tokens=tokens,
                    gt_ids=gt_single.input_ids,
                    gt_attention_mask=gt_single.attention_mask,
                    llm_tokenizer=llm_tokenizer,
                    text_embedder=text_embedder,
                    llm_device=llm_device,
                )
                asr_loss = _asr_forward_loss(
                    llm_model,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    attention_mask=attention_mask,
                    micro_batch_size=STAGE.asr_micro_batch_size,
                    device=llm_device,
                )
                asr_vals.append(asr_loss.detach().item())

                utt_loss = asr_loss / float(n_valid)
                if use_stability:
                    stability_loss = stability_sum / float(num_windows)
                    stab_vals.append(float(stability_loss.detach().item()))
                    utt_loss = utt_loss + (STAGE.lambda_stability * stability_loss) / float(n_valid)

                is_last_utt = utt_idx == n_valid - 1
                _pipeline_accumulate_backward(
                    pipeline,
                    utt_loss,
                    no_sync_modules=no_sync_modules,
                    sync_grads=is_last_utt and at_accum_boundary,
                )

            asr_mean = sum(asr_vals) / len(asr_vals)
            stab_mean = sum(stab_vals) / len(stab_vals) if stab_vals else 0.0
            align_val = float(align_loss.detach().item())
            total_loss_val = (
                asr_mean
                + STAGE.lambda_align * align_val
                + (STAGE.lambda_stability * stab_mean if use_stability else 0.0)
            )

            optimizer_stepped = _pipeline_end_microbatch(
                pipeline,
                [*adapter_module.parameters(), logit_scale],
            )

            temperature_val = float(logit_scale.exp().item())
            m_total.update(total_loss_val)
            m_asr.update(asr_mean)
            m_align.update(align_val)
            m_stab.update(stab_mean)

            accum_suffix = ""
            if grad_accum_steps > 1:
                accum_done = grad_accum_steps if optimizer_stepped else pipeline.accum_step
                accum_suffix = f" | accum {accum_done}/{grad_accum_steps}"

            should_log = (
                (step + 1) % LOG_EVERY_STEPS == 0
                or (step + 1) == len(dataloader)
            )
            current_lr = scheduler.get_last_lr()[0]
            if ctx.is_main and should_log:
                print(
                    f"Micro {step:4d}/{len(dataloader)} | opt {pipeline.global_step:5d} | "
                    f"Loss: {total_loss_val:.4f} | ASR: {asr_mean:.4f} | "
                    f"Align: {align_val:.4f} | Stab: {stab_mean:.4f} | "
                    f"Temp: {temperature_val:.4f} | LR: {current_lr:.2e}{accum_suffix}"
                )

            if ctx.is_main and optimizer_stepped and should_log:
                logger.log(
                    {
                        "train/loss": total_loss_val,
                        "train/asr": asr_mean,
                        "train/align": align_val,
                        "train/stability": stab_mean,
                        "train/temperature": temperature_val,
                        "train/logit_scale": float(logit_scale.item()),
                        "train/lr": current_lr,
                    },
                    step=pipeline.global_step,
                )

            if (
                ctx.is_main
                and optimizer_stepped
                and STAGE.val_enabled
                and pipeline.global_step > 0
                and pipeline.global_step % STAGE.val_every_steps == 0
                and os.path.isdir(VAL_ROOT)
            ):
                val_predictions_path = os.path.join(
                    CKPT.dir,
                    f"{CHECKPOINT_BASENAME}_val_step{pipeline.global_step}_predictions.json",
                )
                val_generation = LlmGenerationParams(
                    max_new_tokens=STAGE.val_max_new_tokens,
                    do_sample=False,
                    num_beams=max(1, STAGE.val_num_beams),
                    repetition_penalty=STAGE.val_repetition_penalty,
                    no_repeat_ngram_size=4,
                )
                val_metrics, val_items = validate_asr_only(
                    adapter=adapter_module,
                    audio_extractor=audio,
                    llm_model=llm_model,
                    llm_tokenizer=llm_tokenizer,
                    text_embedder=text_embedder,
                    encode_utterance_fn=lambda **kwargs: _encode_utterance(
                        **kwargs,
                        collect_aux_metrics=True,
                    ),
                    build_inputs_for_asr_fn=_build_inputs_for_asr,
                    asr_forward_loss_fn=lambda **kwargs: _asr_forward_loss(llm_model, **kwargs),
                    compute_aux_loss_metrics_fn=lambda **kwargs: _compute_val_aux_metrics(
                        **kwargs,
                        logit_scale=logit_scale,
                    ),
                    val_root=VAL_ROOT,
                    train_device=train_device,
                    llm_device=llm_device,
                    max_utterances=STAGE.val_max_utterances,
                    max_windows_per_utt=DATA.max_windows_per_utt,
                    max_text_tokens=STAGE.max_text_tokens,
                    asr_micro_batch_size=STAGE.asr_micro_batch_size,
                    generation=val_generation,
                    global_step=pipeline.global_step,
                    batch_size=env_int("VAL_BATCH_SIZE", DATA.batch_size),
                    num_workers=DATA.num_workers,
                    predictions_path=val_predictions_path,
                    log_every=STAGE.val_log_every,
                    maybe_autocast_fn=_maybe_autocast,
                )
                temperature_val = float(logit_scale.exp().item())
                val_metrics["val/temperature"] = temperature_val
                val_metrics["val/logit_scale"] = float(logit_scale.item())
                print(
                    f"  [val step {pipeline.global_step}] "
                    f"temperature={temperature_val:.4f} "
                    f"logit_scale={float(logit_scale.item()):.4f}",
                    flush=True,
                )
                _append_val_history(
                    history_path=VAL_HISTORY_PATH,
                    global_step=pipeline.global_step,
                    epoch=epoch,
                    metrics=val_metrics,
                    items=val_items,
                    predictions_path=val_predictions_path,
                )
                logger.log(val_metrics, step=pipeline.global_step)

            if ctx.is_main and optimizer_stepped:
                _maybe_save_step_checkpoint(
                    epoch=epoch,
                    adapter=adapter,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    pipeline=pipeline,
                    metrics={
                        "loss": m_total.mean,
                        "asr": m_asr.mean,
                        "align": m_align.mean,
                        "stability": m_stab.mean,
                        "temperature": float(logit_scale.exp().item()),
                    },
                    logit_scale=logit_scale,
                )

            if llm_device.type == "cuda":
                with torch.cuda.device(llm_device):
                    torch.cuda.empty_cache()

        if ctx.is_main:
            epoch_metrics = {
                "loss": m_total.mean,
                "asr": m_asr.mean,
                "align": m_align.mean,
                "stability": m_stab.mean,
                "temperature": float(logit_scale.exp().item()),
            }
            ckpt = _make_checkpoint(
                epoch=epoch + 1,
                global_step=pipeline.global_step,
                adapter=adapter,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=epoch_metrics,
                logit_scale=logit_scale,
            )
            save_checkpoint(SAVE_PATH, ckpt)
            epoch_save_path = os.path.join(CKPT.dir, f"{CHECKPOINT_BASENAME}_epoch{epoch + 1}.pt")
            save_checkpoint(epoch_save_path, ckpt)
            maybe_upload_stage_epoch_checkpoint(
                epoch_save_path,
                stage=2,
                repo_id=HF_CKPT.repo_id,
                revision=HF_CKPT.revision,
                private=HF_CKPT.private,
                token=_hf_token,
                enabled=HF_CKPT.upload_enabled,
            )
            print(
                f"\nEpoch {epoch + 1} complete: loss={m_total.mean:.4f} "
                f"asr={m_asr.mean:.4f} align={m_align.mean:.4f} "
                f"stab={m_stab.mean:.4f}\nCheckpoint -> {SAVE_PATH}\n"
            )

    if ctx.is_main:
        print("ASR + align + stability training complete (Vicuna)!")
        logger.finish()
    cleanup_distributed()


if __name__ == "__main__":
    train()
