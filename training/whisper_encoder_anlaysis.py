"""
Stage 2 ASR distillation with selectable Whisper encoder layer embeddings.

Uses ``whisper_pipeline.LayerAwareWindowEncoder`` so the adapter trains on:
  - a single layer embedding
  - concatenated specific layers
  - mean of specific layers
  - learnable weighted sum of specific layers

Defaults (overridable via CLI / env):
  - LibriSpeech **train-clean-100** only
  - batch size **32**
  - **no** Stage 1 checkpoint warm-start
  - checkpoint basename ``adapter_stage2_layer_analysis``

Examples::

    uv run training/whisper_encoder_anlaysis.py --layers 11 --mode single
    uv run training/whisper_encoder_anlaysis.py --layers 6,8,11 --mode concat
    uv run training/whisper_encoder_anlaysis.py --layers 4,6,8,11 --mode mean
    uv run training/whisper_encoder_anlaysis.py --layers 4,6,8,11 --mode weighted_sum
    uv run training/whisper_encoder_anlaysis.py --layers 11 --mode single --load-stage1
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext
from dataclasses import replace

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_root = os.path.join(_pkg_root, "src")
sys.path.insert(0, _src_root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, _pkg_root)

from training.utils.env import apply_hf_hub_endpoint, env_bool, env_str, load_project_env

_env_path = load_project_env(_pkg_root)
if _env_path:
    print(f"Loaded environment from {_env_path}")

_hf_endpoint = apply_hf_hub_endpoint(_pkg_root)
print(f"HF Hub endpoint: {_hf_endpoint}")

_hf_token = env_str("HF_TOKEN")
if _hf_token:
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", _hf_token)


def _parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--layers",
        default=None,
        help="Comma-separated Whisper encoder layer indices (default: env "
        "WHISPER_LAYERS or 11). Example: 4,6,8,11",
    )
    ap.add_argument(
        "--mode",
        default=None,
        choices=["single", "concat", "mean", "weighted_sum"],
        help="How to combine selected layers (default: env WHISPER_LAYER_MODE or single)",
    )
    ap.add_argument(
        "--apply-final-norm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If the last encoder layer is selected, use post-final-LayerNorm "
        "hidden state (default: True)",
    )
    ap.add_argument(
        "--load-stage1",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Warm-start adapter from STAGE1_CHECKPOINT (default: False)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="DataLoader batch size (default: env BATCH_SIZE or 32)",
    )
    args, _unknown = ap.parse_known_args(argv)
    return args


_CLI = _parse_cli()

# Analysis-trainer defaults. Stage1 warm-start is off unless --load-stage1 or
# WHISPER_LOAD_STAGE1 is set (stock .env often has LOAD_STAGE1_CHECKPOINT=true).
os.environ.setdefault("CHECKPOINT_BASENAME", "adapter_stage2_layer_analysis")
if _CLI.batch_size is not None:
    os.environ["BATCH_SIZE"] = str(_CLI.batch_size)
else:
    os.environ.setdefault("BATCH_SIZE", "32")

if _CLI.load_stage1 is not None:
    os.environ["LOAD_STAGE1_CHECKPOINT"] = "true" if _CLI.load_stage1 else "false"
elif env_str("WHISPER_LOAD_STAGE1") is not None:
    os.environ["LOAD_STAGE1_CHECKPOINT"] = (
        "true" if env_bool("WHISPER_LOAD_STAGE1", False) else "false"
    )
else:
    os.environ["LOAD_STAGE1_CHECKPOINT"] = "false"

from training.utils.devices import apply_runtime_cuda_env

apply_runtime_cuda_env()

if env_str("CHECK_TORCH_COMPAT", "0") in ("1", "true", "yes", "on"):
    from training.utils.torch_compat import ensure_torch_compatible

    ensure_torch_compatible()

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.adapter.streaming_adapter import StreamingAdapter
from src.dataset import LibriSpeechConfig, LibriSpeechPairs, load_mono_waveform_16k
from training.utils.checkpointing import (
    TrainingCheckpoint,
    adapt_adapter_state_dict_num_queries,
    adapt_optimizer_state_dict_num_queries,
    checkpoint_grad_accum_steps,
    load_gate_state_dict_safe,
    maybe_upload_stage_epoch_checkpoint,
    resolve_gate_config_from_checkpoint,
    resolve_resume_epoch_and_offset,
    save_checkpoint,
)
from training.utils.config import (
    AsrExperimentConfig,
    CheckpointConfig,
    DataConfig,
    DeviceConfig,
    FrozenModelIdsConfig,
    GateConfig,
    HfCheckpointConfig,
    OptimConfig,
    Stage2Config,
    WandbConfig,
)
from training.utils.gate_training import (
    build_turn_end_gate,
    endpoint_label_for_timestep,
    make_silence_trackers,
)
from training.utils.devices import (
    cleanup_distributed,
    ensure_device_ready,
    init_training_context,
    llm_input_device,
    resolve_llm_load_plan,
)
from training.utils.logging import WandbLogger
from training.utils.losses import contrastive_infonce_loss
from training.utils.metrics import RunningMean
from training.utils.asr_prompt import DEFAULT_ASR_PROMPT, PROMPT_CONDITIONING, TRAIN_STYLE_CONDITIONING
from training.utils.loaders import default_device_and_dtype, load_frozen_qwen_causal_lm
from training.utils.optimization import TrainingPipeline
from training.utils.stage2_validation import validate_stage2_asr, wandb_val_log_dict
from whisper_pipeline import (
    LayerAwareWindowEncoder,
    LayerEmbedConfig,
    LearnableLayerAggregator,
    embedding_dim,
    parse_layers_arg,
)

_, TORCH_DTYPE = default_device_and_dtype()
_TRAINING_DIR = os.path.dirname(__file__)

# train-clean-100 only; DATASET_ROOT still overrides when set.
_dataset_override = env_str("DATASET_ROOT")
if _dataset_override:
    DATASET_ROOTS = LibriSpeechConfig.resolve_train_roots(
        _TRAINING_DIR, env_override=_dataset_override
    )
else:
    DATASET_ROOTS = LibriSpeechConfig.train_clean_100_roots(_TRAINING_DIR)
VAL_ROOT = LibriSpeechConfig.dev_clean_root(_TRAINING_DIR)

_MODEL_IDS = FrozenModelIdsConfig.from_env()
WHISPER_DIM = 768
LLM_DIM = 4096
WHISPER_MODEL = _MODEL_IDS.whisper_model_id
LLM_MODEL_ID = _MODEL_IDS.llm_model_id

STAGE = Stage2Config.from_env()
GATE = GateConfig.from_env()
DEVICE_CFG = DeviceConfig.from_env()
OPT = OptimConfig.from_env()
DATA = DataConfig.from_env(default_dataset_root=DATASET_ROOTS[0])
CKPT = CheckpointConfig.from_env(pkg_root=_pkg_root)
HF_CKPT = HfCheckpointConfig.from_env()
EXP = AsrExperimentConfig.from_env(pkg_root=_pkg_root)
WANDB = WandbConfig.from_env()

SAVE_PATH = os.path.join(CKPT.dir, f"{EXP.checkpoint_basename}.pt")
CHECKPOINT_BASENAME = EXP.checkpoint_basename
_stage1_rel = env_str("STAGE1_CHECKPOINT", "checkpoints/adapter_stage1.pt") or "checkpoints/adapter_stage1.pt"
STAGE1_SAVE_PATH = (
    _stage1_rel if os.path.isabs(_stage1_rel) else os.path.join(_pkg_root, _stage1_rel)
)

USE_RATE_CONTROLLER = STAGE.use_rate_controller
RATE_TARGET = STAGE.rate_target


def _build_layer_embed_config() -> LayerEmbedConfig:
    layers_raw = _CLI.layers if _CLI.layers is not None else env_str("WHISPER_LAYERS", "11")
    mode_raw = _CLI.mode if _CLI.mode is not None else (env_str("WHISPER_LAYER_MODE", "single") or "single")
    if _CLI.apply_final_norm is not None:
        apply_final_norm = _CLI.apply_final_norm
    else:
        apply_final_norm = env_bool("WHISPER_APPLY_FINAL_NORM", True)
    return LayerEmbedConfig(
        layers=parse_layers_arg(layers_raw or "11"),
        mode=mode_raw,
        apply_final_norm=apply_final_norm,
    )


LAYER_CFG = _build_layer_embed_config()
D_ENCODER = embedding_dim(WHISPER_DIM, LAYER_CFG)


def _maybe_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _stage2_epoch_metrics(
    m_total: RunningMean,
    m_asr: RunningMean,
    m_align: RunningMean,
    m_stab: RunningMean,
    m_sparse: RunningMean,
    m_rate: RunningMean,
    m_gate: RunningMean,
) -> dict[str, float]:
    return {
        "loss": m_total.mean,
        "asr": m_asr.mean,
        "align": m_align.mean,
        "stability": m_stab.mean,
        "sparse": m_sparse.mean,
        "rate": m_rate.mean,
        "gate": m_gate.mean,
    }


def _make_stage2_checkpoint(
    *,
    epoch: int,
    global_step: int,
    adapter: torch.nn.Module,
    gate: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    metrics: dict[str, float],
    gate_cfg: GateConfig = GATE,
    aggregator: LearnableLayerAggregator | None = None,
) -> TrainingCheckpoint:
    gate_state = _unwrap(gate).state_dict() if gate is not None else None
    return TrainingCheckpoint(
        stage=2,
        epoch=epoch,
        global_step=global_step,
        adapter_state_dict=_unwrap(adapter).state_dict(),
        gate_state_dict=gate_state,
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),
        metrics=metrics,
        hyperparams={
            "gate_cfg": gate_cfg.__dict__,
            "stage_cfg": STAGE.__dict__,
            "exp_cfg": EXP.__dict__,
            "grad_accum_steps": DATA.gradient_accumulation_steps(),
            "layer_cfg": LAYER_CFG.to_dict(),
            "d_encoder": D_ENCODER,
            "aggregator_state_dict": (
                aggregator.state_dict() if aggregator is not None else None
            ),
        },
    )


def _maybe_save_step_checkpoint(
    *,
    epoch: int,
    adapter: torch.nn.Module,
    gate: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    pipeline: TrainingPipeline,
    metrics: dict[str, float],
    gate_cfg: GateConfig = GATE,
    aggregator: LearnableLayerAggregator | None = None,
) -> None:
    interval = CKPT.save_every_steps
    if interval is None or interval <= 0:
        return
    step = pipeline.global_step
    if step <= 0 or step % interval != 0:
        return
    ckpt = _make_stage2_checkpoint(
        epoch=epoch + 1,
        global_step=step,
        adapter=adapter,
        gate=gate,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        gate_cfg=gate_cfg,
        aggregator=aggregator,
    )
    save_checkpoint(SAVE_PATH, ckpt)
    step_path = os.path.join(CKPT.dir, f"{CHECKPOINT_BASENAME}_step{step}.pt")
    save_checkpoint(step_path, ckpt)
    print(f"  Checkpoint (step {step}) -> {SAVE_PATH}\n              step file -> {step_path}")


def _enable_llm_gradient_checkpointing(llm_model: torch.nn.Module) -> None:
    if not STAGE.enable_llm_gradient_checkpointing:
        return
    if hasattr(llm_model, "gradient_checkpointing_enable"):
        llm_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    if hasattr(llm_model, "enable_input_require_grads"):
        llm_model.enable_input_require_grads()


def _asr_forward_loss(
    llm_model: torch.nn.Module,
    *,
    inputs_embeds: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    micro_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Run frozen LM loss; micro-batch to cap activation memory."""
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


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, DDP) else module


def _build_stage2_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup then cosine decay over the full training run."""
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


def _resume_stage2_training_state(
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
        optimizer.load_state_dict(opt_state)
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
    llm_device_str, qwen_map, qwen_max_memory = resolve_llm_load_plan(
        ctx=ctx,
        train_device=train_device,
        llm_device=DEVICE_CFG.llm_device,
        llm_max_memory=DEVICE_CFG.llm_max_memory,
    )
    llm_load_device = torch.device(llm_device_str)
    ensure_device_ready(llm_load_device)

    if ctx.is_main:
        print(f"Training device: {train_device_str} (configured: {DEVICE_CFG.device})")
        print(f"LLM device: {llm_device_str} (configured: {DEVICE_CFG.llm_device or train_device_str})")
        print(f"Visible CUDA devices: {ctx.num_cuda_devices}")
        print(f"Distributed: world_size={ctx.world_size} rank={ctx.rank}")
        print(f"Model parallel (LLM auto-shard): {ctx.model_parallel}")
        print(
            f"Layer embed: layers={LAYER_CFG.layers} mode={LAYER_CFG.mode} "
            f"apply_final_norm={LAYER_CFG.apply_final_norm} d_encoder={D_ENCODER}"
        )

    audio = LayerAwareWindowEncoder(
        model_id=WHISPER_MODEL,
        device=train_device_str,
        torch_dtype=TORCH_DTYPE,
        layer_cfg=LAYER_CFG,
    )
    aggregator = audio.aggregator
    if audio.d_encoder != D_ENCODER:
        raise RuntimeError(
            f"encoder d_encoder mismatch: config={D_ENCODER} audio={audio.d_encoder}"
        )

    if llm_load_device.type == "cuda":
        with torch.cuda.device(llm_load_device):
            torch.cuda.empty_cache()

    if ctx.is_main:
        print(f"Qwen device_map: {qwen_map!r}")
        if qwen_max_memory:
            print(f"Qwen max_memory: {qwen_max_memory!r}")

    qwen_models = load_frozen_qwen_causal_lm(
        model_id=LLM_MODEL_ID,
        device=llm_device_str,
        torch_dtype=TORCH_DTYPE,
        device_map=qwen_map,
        max_memory=qwen_max_memory,
    )
    llm_tokenizer = qwen_models.tokenizer
    llm_model = qwen_models.causal_lm
    text_embedder = qwen_models.embedder
    llm_device = llm_input_device(llm_model)
    _enable_llm_gradient_checkpointing(llm_model)
    llm_model.train()
    for p in llm_model.parameters():
        p.requires_grad = False
    if ctx.is_main:
        print(f"Qwen input embeddings device: {llm_device}")
        hf_map = getattr(llm_model, "hf_device_map", None)
        if hf_map:
            layer_devices = sorted(
                {f"cuda:{d}" if isinstance(d, int) else str(d) for d in hf_map.values()}
            )
            print(f"Qwen layer devices: {layer_devices}")
        if STAGE.enable_llm_gradient_checkpointing:
            print("Qwen gradient checkpointing: enabled")

    adapter = StreamingAdapter(
        d_encoder=D_ENCODER,
        d_llm=LLM_DIM,
        num_queries=2,
        num_layers=2,
        num_heads=4,
        d_ffn=2048,
        dropout=0.1,
        ema_alpha=0.8,
        learnable_ema=False,
        use_rate_controller=USE_RATE_CONTROLLER,
        rate_threshold=0.5,
        target_rate=RATE_TARGET,
    ).to(train_device, dtype=TORCH_DTYPE)
    adapter.train()

    if EXP.load_stage1_checkpoint:
        if D_ENCODER != WHISPER_DIM:
            raise ValueError(
                f"Cannot load Stage 1 checkpoint when d_encoder={D_ENCODER} "
                f"(Stage 1 expects {WHISPER_DIM}). Use mode=single|mean|weighted_sum "
                f"with a single-layer-width embedding, or omit --load-stage1."
            )
        if not os.path.isfile(STAGE1_SAVE_PATH):
            raise FileNotFoundError(
                f"LOAD_STAGE1_CHECKPOINT=true but checkpoint missing: {STAGE1_SAVE_PATH}"
            )
        stage1_ckpt = torch.load(STAGE1_SAVE_PATH, map_location=train_device)
        adapter.load_state_dict(
            adapt_adapter_state_dict_num_queries(
                stage1_ckpt["adapter_state_dict"],
                adapter.num_queries,
            ),
            strict=False,
        )
        if ctx.is_main:
            print(
                f"Loaded Stage 1 adapter weights from {STAGE1_SAVE_PATH} "
                f"(epoch {stage1_ckpt.get('epoch', '?')})"
            )
    elif ctx.is_main:
        print("Skipping Stage 1 checkpoint load (random adapter init)")

    gate_cfg = GATE
    gate: torch.nn.Module | None = None
    resume_ckpt = None
    resume_path = CKPT.resume_checkpoint or SAVE_PATH
    if os.path.exists(resume_path):
        resume_meta = torch.load(resume_path, map_location="cpu")
        gate_cfg = resolve_gate_config_from_checkpoint(resume_meta, defaults=GATE)
        if gate_cfg.silence_mode == "both":
            gate_cfg = replace(gate_cfg, active_silence_path=GATE.active_silence_path)
        if ctx.is_main:
            resume_ckpt = torch.load(resume_path, map_location=train_device)
            print(f"Will resume from checkpoint: {resume_path}")

    use_gate = EXP.train_gate or EXP.gate_checkpoint is not None
    if use_gate:
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
            device=train_device,
            dtype=TORCH_DTYPE,
        )
        if EXP.gate_checkpoint:
            if not os.path.isfile(EXP.gate_checkpoint):
                raise FileNotFoundError(f"GATE_CHECKPOINT not found: {EXP.gate_checkpoint}")
            gate_ckpt = torch.load(EXP.gate_checkpoint, map_location=train_device)
            load_gate_state_dict_safe(_unwrap(gate), gate_ckpt, warn=True)
            if ctx.is_main:
                print(f"Loaded gate weights from {EXP.gate_checkpoint}")
        if not EXP.train_gate:
            for p in _unwrap(gate).parameters():
                p.requires_grad = False
            _unwrap(gate).eval()

    if ctx.world_size > 1:
        adapter = DDP(adapter, device_ids=[ctx.local_rank])
        if gate is not None:
            gate = DDP(gate, device_ids=[ctx.local_rank])
        if aggregator is not None:
            aggregator = DDP(aggregator, device_ids=[ctx.local_rank])
            # Keep DDP module on the encoder so forward participates in grad sync.
            audio.aggregator = aggregator

    if ctx.is_main:
        gate_mode = "disabled"
        if gate is not None:
            gate_mode = "trainable" if EXP.train_gate else "frozen (pretrained)"
        print(
            f"Models initialized:\n  Adapter: trainable (d_encoder={D_ENCODER})\n"
            f"  Layer aggregator: {'trainable' if aggregator is not None else 'n/a'}\n"
            f"  Turn-end gate: {gate_mode}\n"
            f"  Gate labels: {gate_cfg.label_source}\n"
            f"  Gate silence mode: {gate_cfg.silence_mode}"
            f"{f' (active={gate_cfg.active_silence_path})' if gate_cfg.silence_mode == 'both' else ''}\n"
            f"  Rate controller: {USE_RATE_CONTROLLER}\n"
            f"  Checkpoint basename: {CHECKPOINT_BASENAME}\n"
        )

    trainable_params = list(_unwrap(adapter).parameters())
    if gate is not None and EXP.train_gate:
        trainable_params += list(_unwrap(gate).parameters())
    if aggregator is not None:
        trainable_params += list(_unwrap(aggregator).parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=OPT.lr, weight_decay=OPT.weight_decay)
    grad_accum_steps = DATA.gradient_accumulation_steps()
    effective_batch_size = DATA.batch_size * grad_accum_steps
    pipeline = TrainingPipeline(
        optimizer=optimizer,
        scheduler=None,
        grad_clip_norm=OPT.grad_clip_norm,
        gradient_accumulation_steps=grad_accum_steps,
    )

    if ctx.is_main:
        print(
            "Training LibriSpeech splits: "
            + ", ".join(os.path.basename(r) for r in DATASET_ROOTS)
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
    scheduler = _build_stage2_lr_scheduler(optimizer, total_steps=total_steps)
    pipeline.scheduler = scheduler

    if ctx.is_main:
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    logger = WandbLogger(
        enabled=WANDB.enabled and ctx.is_main,
        project=WANDB.project,
        entity=WANDB.entity,
        run_name=WANDB.run_name,
        config={
            "stage": 2,
            "layer_cfg": LAYER_CFG.to_dict(),
            "d_encoder": D_ENCODER,
            "data": {**DATA.__dict__, "dataset_roots": DATASET_ROOTS, "val_root": VAL_ROOT},
            "optim": OPT.__dict__,
            "stage_cfg": STAGE.__dict__,
            "exp_cfg": EXP.__dict__,
            "gate_cfg": gate_cfg.__dict__,
            "device": DEVICE_CFG.__dict__,
            "num_cuda_devices": ctx.num_cuda_devices,
            "model_parallel": ctx.model_parallel,
            "world_size": ctx.world_size,
        },
    )

    start_epoch = 0
    resume_batch_offset = 0
    if ctx.is_main and resume_ckpt is not None:
        _unwrap(adapter).load_state_dict(
            adapt_adapter_state_dict_num_queries(
                resume_ckpt["adapter_state_dict"],
                _unwrap(adapter).num_queries,
            )
        )
        if gate is not None:
            load_gate_state_dict_safe(_unwrap(gate), resume_ckpt, warn=False)
        hp = resume_ckpt.get("hyperparams") or {}
        agg_state = hp.get("aggregator_state_dict")
        if aggregator is not None and agg_state is not None:
            _unwrap(aggregator).load_state_dict(agg_state)
        start_epoch, resume_batch_offset = _resume_stage2_training_state(
            resume_ckpt=resume_ckpt,
            optimizer=optimizer,
            scheduler=scheduler,
            pipeline=pipeline,
            micro_steps_per_epoch=micro_steps_per_epoch,
            grad_accum_steps=grad_accum_steps,
            num_queries=_unwrap(adapter).num_queries,
        )
        resume_epoch_1idx = start_epoch + 1
        saved_accum = checkpoint_grad_accum_steps(resume_ckpt)
        accum_note = (
            f"legacy micro-step checkpoint"
            if saved_accum is None
            else f"grad_accum={saved_accum} at save time"
        )
        print(
            f"  Resumed at epoch {resume_epoch_1idx}/{STAGE.epochs}, "
            f"global_step {pipeline.global_step}, "
            f"dataloader offset {resume_batch_offset}/{len(dataloader)} ({accum_note})\n"
        )

    if ctx.is_main:
        print("\nStarting Stage 2 training: ASR Distillation (layer analysis)")
        print(f"  Dataset splits: {', '.join(os.path.basename(r) for r in DATASET_ROOTS)}")
        print(f"  Layers: {LAYER_CFG.layers} | mode: {LAYER_CFG.mode} | d_encoder: {D_ENCODER}")
        if STAGE.val_enabled:
            if os.path.isdir(VAL_ROOT):
                cap = STAGE.val_max_utterances
                cap_str = "all" if cap is None else str(cap)
                print(
                    f"  Validation: dev-clean ({VAL_ROOT}), every {STAGE.val_every_steps} steps, "
                    f"max {cap_str} utterances"
                )
            else:
                print(f"  Validation: dev-clean not found at {VAL_ROOT} (will skip until present)")
        if CKPT.save_every_steps:
            print(
                f"  Checkpoints: every {CKPT.save_every_steps} steps -> "
                f"{CKPT.dir}/{CHECKPOINT_BASENAME}_step<N>.pt (and {SAVE_PATH})"
            )
        print(f"  Epochs: {STAGE.epochs}")
        print(f"  Micro-batch size: {DATA.batch_size}")
        if grad_accum_steps > 1:
            print(
                f"  Effective batch size: {effective_batch_size} "
                f"(gradient accumulation: {grad_accum_steps} steps)"
            )
        else:
            print(f"  Effective batch size: {effective_batch_size}")
        print(f"  Load Stage 1 checkpoint: {EXP.load_stage1_checkpoint}")
        print(f"  Optimizer steps/epoch: {optimizer_steps_per_epoch}")
        print(f"  Learning rate: {OPT.lr}")
        print(f"  λ_align: {STAGE.lambda_align}")
        print(f"  λ_stability: {STAGE.lambda_stability}")
        print(f"  λ_rate: {STAGE.lambda_rate}")
        print(f"  λ_gate: {STAGE.lambda_gate}")
        print(f"  Gate label source: {gate_cfg.label_source}")
        print(f"  Rate target: {RATE_TARGET} tokens/window")
        max_windows_label = (
            "all" if DATA.max_windows_per_utt is None else str(DATA.max_windows_per_utt)
        )
        print(f"  Max windows/utt: {max_windows_label}")
        print(f"  Max text tokens: {STAGE.max_text_tokens}")
        print(f"  ASR micro-batch: {STAGE.asr_micro_batch_size}")
        print(f"  LM conditioning: {TRAIN_STYLE_CONDITIONING}")
        print(
            f"  Stage 3 target ({PROMPT_CONDITIONING}): {DEFAULT_ASR_PROMPT[:72]}...\n"
        )

    if GATE.label_source == "smart_turn":
        if ctx.is_main:
            print(
                "[ERROR] GATE_LABEL_SOURCE=smart_turn is not supported in "
                "whisper_encoder_anlaysis.py. Use: uv run training/gate_training.py"
            )
        cleanup_distributed()
        raise SystemExit(1)

    train_gate_in_loop = gate is not None and EXP.train_gate and STAGE.lambda_gate > 0
    use_aux_losses = (
        STAGE.lambda_align > 0
        or STAGE.lambda_stability > 0
        or STAGE.lambda_rate > 0
    )

    for epoch in range(start_epoch, STAGE.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        m_total = RunningMean()
        m_asr = RunningMean()
        m_align = RunningMean()
        m_stab = RunningMean()
        m_sparse = RunningMean()
        m_rate = RunningMean()
        m_gate = RunningMean()

        if ctx.is_main:
            print(f"\n{'=' * 60}\nEpoch {epoch + 1}/{STAGE.epochs}\n{'=' * 60}\n")

        batch_start = resume_batch_offset if epoch == start_epoch else 0
        for step, batch in enumerate(dataloader):
            if step < batch_start:
                continue
            audio_paths, transcriptions = batch
            batch_texts = list(transcriptions)

            gt_tokens = llm_tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=STAGE.max_text_tokens,
            ).to(train_device)
            gt_ids = gt_tokens.input_ids
            gt_attention_mask = gt_tokens.attention_mask

            with torch.no_grad():
                gt_embeds = text_embedder(gt_ids.to(llm_device)).to(train_device)

            audio_tokens_list: list[torch.Tensor] = []
            num_windows_list: list[int] = []
            gate_calls = 0
            total_stability_loss = torch.zeros((), device=train_device, dtype=torch.float32)
            total_sparse_loss = torch.zeros((), device=train_device, dtype=torch.float32)
            total_rate_loss = torch.zeros((), device=train_device, dtype=torch.float32)
            total_gate_loss = torch.zeros((), device=train_device, dtype=torch.float32)

            adapter_module = _unwrap(adapter)
            gate_module = _unwrap(gate) if gate is not None else None

            for p in audio_paths:
                wave = load_mono_waveform_16k(p)
                windows = audio.waveform_to_windows(wave)
                if DATA.max_windows_per_utt is not None:
                    windows = windows[: DATA.max_windows_per_utt]

                adapter_module.reset_streaming_state()
                silence_tracker, learned_silence_tracker = (
                    make_silence_trackers(gate_module) if gate_module is not None else (None, None)
                )
                utterance_tokens: list[torch.Tensor] = []
                for t, window in enumerate(windows):
                    result = adapter_module.forward_window(
                        window.to(device=train_device, dtype=TORCH_DTYPE)
                    )
                    utterance_tokens.append(result["tokens"])
                    if use_aux_losses:
                        total_stability_loss = total_stability_loss + result["stability_loss"].float()
                        if result["sparse_loss"] is not None:
                            total_sparse_loss = total_sparse_loss + result["sparse_loss"].float()
                        if result["rate_loss"] is not None:
                            total_rate_loss = total_rate_loss + result["rate_loss"].float()

                    if train_gate_in_loop and gate_module is not None:
                        accumulated = torch.cat(utterance_tokens, dim=1)
                        endpoint = endpoint_label_for_timestep(
                            t,
                            len(windows),
                            batch_size=accumulated.shape[0],
                            device=train_device,
                        )
                        win_tokens = utterance_tokens[t]
                        gate_result = gate_module(
                            accumulated,
                            t,
                            len(windows),
                            endpoint_label=endpoint,
                            silence_tracker=silence_tracker,
                            learned_silence_tracker=learned_silence_tracker,
                            window_tokens=win_tokens,
                        )
                        total_gate_loss = total_gate_loss + gate_result["gate_loss"].float()
                        gate_calls += 1

                if utterance_tokens:
                    tokens = torch.cat(utterance_tokens, dim=1)
                    audio_tokens_list.append(tokens)
                    num_windows_list.append(len(windows))

            if not audio_tokens_list:
                if ctx.is_main:
                    print(f"[WARN] Step {step}: empty batch (no audio windows); skipping.")
                continue

            max_tokens = max(t.shape[1] for t in audio_tokens_list)
            padded_tokens = []
            for tokens in audio_tokens_list:
                pad_len = max_tokens - tokens.shape[1]
                padding = torch.zeros(
                    tokens.shape[0],
                    pad_len,
                    tokens.shape[2],
                    device=tokens.device,
                    dtype=tokens.dtype,
                )
                padded = torch.cat([tokens, padding], dim=1)
                padded_tokens.append(padded)

            audio_tokens = torch.cat(padded_tokens, dim=0)

            bos_token_id = (
                llm_tokenizer.bos_token_id if llm_tokenizer.bos_token_id is not None else llm_tokenizer.eos_token_id
            )

            bos_embed = text_embedder(
                torch.tensor([[bos_token_id]], device=llm_device).expand(audio_tokens.shape[0], -1)
            )
            inputs_embeds = torch.cat([audio_tokens.to(llm_device), bos_embed], dim=1)

            batch_size = audio_tokens.shape[0]
            audio_len = audio_tokens.shape[1]
            pre_text_labels = torch.full((batch_size, audio_len + 1), -100, dtype=torch.long, device=llm_device)
            gt_shifted = gt_ids[:, 1:].to(llm_device)
            labels = torch.cat([pre_text_labels, gt_shifted], dim=1)
            inputs_embeds = torch.cat([inputs_embeds, text_embedder(gt_shifted)], dim=1)

            pre_text_mask = torch.ones((batch_size, audio_len + 1), device=llm_device)
            llm_attention_mask = torch.cat([pre_text_mask, gt_attention_mask[:, 1:].to(llm_device)], dim=1)

            asr_loss = _asr_forward_loss(
                llm_model,
                inputs_embeds=inputs_embeds,
                labels=labels,
                attention_mask=llm_attention_mask,
                micro_batch_size=STAGE.asr_micro_batch_size,
                device=train_device,
            )

            align_loss = contrastive_infonce_loss(
                audio_tokens=audio_tokens.float(),
                text_embeddings=gt_embeds.float(),
                temperature=STAGE.temperature,
            ) if STAGE.lambda_align > 0 else torch.tensor(0.0, device=train_device)

            total_windows = sum(num_windows_list)
            stability_loss = (
                total_stability_loss / float(total_windows)
                if total_windows > 0 and STAGE.lambda_stability > 0
                else torch.tensor(0.0, device=train_device)
            )
            sparse_loss = (
                total_sparse_loss / float(total_windows)
                if total_windows > 0
                else torch.tensor(0.0, device=train_device)
            )
            rate_loss = (
                total_rate_loss / float(total_windows)
                if total_windows > 0 and STAGE.lambda_rate > 0
                else torch.tensor(0.0, device=train_device)
            )
            gate_loss_mean = (
                total_gate_loss / float(gate_calls)
                if gate_calls > 0 and train_gate_in_loop
                else torch.tensor(0.0, device=train_device)
            )

            total_loss = asr_loss
            if STAGE.lambda_align > 0:
                total_loss = total_loss + STAGE.lambda_align * align_loss
            if STAGE.lambda_stability > 0:
                total_loss = total_loss + STAGE.lambda_stability * stability_loss
            if STAGE.lambda_rate > 0:
                total_loss = total_loss + STAGE.lambda_rate * rate_loss
            if train_gate_in_loop:
                total_loss = total_loss + STAGE.lambda_gate * gate_loss_mean

            no_sync_modules: tuple[torch.nn.Module, ...] | None = None
            if ctx.world_size > 1:
                modules = [adapter]
                if gate is not None and EXP.train_gate:
                    modules.append(gate)
                if aggregator is not None:
                    modules.append(aggregator)
                no_sync_modules = tuple(modules)
            optimizer_stepped = pipeline.step(
                total_loss,
                trainable_params,
                no_sync_modules=no_sync_modules,
            )

            m_total.update(total_loss.item())
            m_asr.update(asr_loss.item())
            m_align.update(align_loss.item())
            m_stab.update(float(stability_loss.item()))
            m_sparse.update(sparse_loss.item())
            m_rate.update(rate_loss.item())
            m_gate.update(float(gate_loss_mean.item()))

            accum_suffix = ""
            if grad_accum_steps > 1:
                accum_done = grad_accum_steps if optimizer_stepped else pipeline.accum_step
                accum_suffix = f" | accum {accum_done}/{grad_accum_steps}"

            if ctx.is_main and step % 1 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"Micro {step:4d}/{len(dataloader)} | opt {pipeline.global_step:5d} | "
                    f"Loss: {total_loss.item():.4f} | "
                    f"ASR: {asr_loss.item():.4f} | Align: {align_loss.item():.4f} | "
                    f"Stab: {stability_loss.float().item():.4f} | "
                    f"Rate: {rate_loss.item():.4f} | Gate: {gate_loss_mean.float().item():.4f} | "
                    f"Sparse(metric): {sparse_loss.item():.4f} | LR: {current_lr:.2e}"
                    f"{accum_suffix}"
                )

            if ctx.is_main and optimizer_stepped:
                log_payload = {
                    "train/loss": total_loss.item(),
                    "train/asr": asr_loss.item(),
                    "train/align": align_loss.item(),
                    "train/stability": stability_loss.float().item(),
                    "train/rate": rate_loss.item(),
                    "train/gate": gate_loss_mean.float().item(),
                    "train/sparse_metric": sparse_loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                }
                if aggregator is not None:
                    log_payload.update(
                        {f"train/{k}": v for k, v in _unwrap(aggregator).weight_dict().items()}
                    )
                logger.log(log_payload, step=pipeline.global_step)

            if (
                ctx.is_main
                and optimizer_stepped
                and STAGE.val_enabled
                and pipeline.global_step > 0
                and pipeline.global_step % STAGE.val_every_steps == 0
                and gate is not None
            ):
                if os.path.isdir(VAL_ROOT):
                    val_metrics = validate_stage2_asr(
                        adapter=_unwrap(adapter),
                        gate=_unwrap(gate),
                        audio_extractor=audio,
                        llm_model=llm_model,
                        llm_tokenizer=llm_tokenizer,
                        text_embedder=text_embedder,
                        asr_forward_loss_fn=lambda **kwargs: _asr_forward_loss(llm_model, **kwargs),
                        val_root=VAL_ROOT,
                        train_device=train_device,
                        llm_device=llm_device,
                        batch_size=DATA.batch_size,
                        num_workers=DATA.num_workers,
                        max_utterances=STAGE.val_max_utterances,
                        max_windows_per_utt=DATA.max_windows_per_utt,
                        max_text_tokens=STAGE.max_text_tokens,
                        asr_micro_batch_size=STAGE.asr_micro_batch_size,
                        lambda_align=STAGE.lambda_align,
                        lambda_stability=STAGE.lambda_stability,
                        lambda_rate=STAGE.lambda_rate,
                        lambda_gate=STAGE.lambda_gate,
                        temperature=STAGE.temperature,
                        maybe_autocast_fn=_maybe_autocast,
                        torch_dtype=TORCH_DTYPE,
                        global_step=pipeline.global_step,
                    )
                    logger.log(wandb_val_log_dict(val_metrics), step=pipeline.global_step)
                else:
                    print(f"  [WARN] Skipping validation — dev-clean not found at {VAL_ROOT}")

            if ctx.is_main and optimizer_stepped:
                _maybe_save_step_checkpoint(
                    epoch=epoch,
                    adapter=adapter,
                    gate=gate,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    pipeline=pipeline,
                    metrics=_stage2_epoch_metrics(
                        m_total, m_asr, m_align, m_stab, m_sparse, m_rate, m_gate
                    ),
                    gate_cfg=gate_cfg,
                    aggregator=_unwrap(aggregator) if aggregator is not None else None,
                )

            if train_device.type == "cuda":
                torch.cuda.empty_cache()

        if ctx.is_main:
            epoch_metrics = _stage2_epoch_metrics(
                m_total, m_asr, m_align, m_stab, m_sparse, m_rate, m_gate
            )
            agg_mod = _unwrap(aggregator) if aggregator is not None else None
            save_checkpoint(
                SAVE_PATH,
                _make_stage2_checkpoint(
                    epoch=epoch + 1,
                    global_step=pipeline.global_step,
                    adapter=adapter,
                    gate=gate,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metrics=epoch_metrics,
                    gate_cfg=gate_cfg,
                    aggregator=agg_mod,
                ),
            )

            epoch_save_path = os.path.join(CKPT.dir, f"{CHECKPOINT_BASENAME}_epoch{epoch + 1}.pt")
            save_checkpoint(
                epoch_save_path,
                _make_stage2_checkpoint(
                    epoch=epoch + 1,
                    global_step=pipeline.global_step,
                    adapter=adapter,
                    gate=gate,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metrics=epoch_metrics,
                    gate_cfg=gate_cfg,
                    aggregator=agg_mod,
                ),
            )
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
                f"\nEpoch {epoch + 1} complete: loss={m_total.mean:.4f} asr={m_asr.mean:.4f} "
                f"align={m_align.mean:.4f} stab={m_stab.mean:.4f}\nCheckpoint -> {SAVE_PATH}\n"
            )

    if ctx.is_main:
        print("Stage 2 layer-analysis training complete!")
        logger.finish()
    cleanup_distributed()


if __name__ == "__main__":
    train()
