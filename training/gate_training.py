"""
Standalone turn-end gate training on Smart Turn endpoint labels.

Frozen Whisper + frozen adapter provide audio tokens; only :class:`TurnEndCommitGate`
is trained. Saves ``checkpoints/{CHECKPOINT_BASENAME}.pt`` and per-epoch copies.

Example::

    GATE_LABEL_SOURCE=smart_turn EPOCHS=5 CHECKPOINT_BASENAME=gate_smart_turn \\
        ADAPTER_CHECKPOINT=checkpoints/adapter_stage1.pt \\
        uv run training/gate_training.py
"""

from __future__ import annotations

import os
import sys

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_root = os.path.join(_pkg_root, "src")
sys.path.insert(0, _src_root)
sys.path.insert(0, _pkg_root)

from training.utils.env import apply_hf_hub_endpoint, env_str, load_project_env

_env_path = load_project_env(_pkg_root)
if _env_path:
    print(f"Loaded environment from {_env_path}")

apply_hf_hub_endpoint(_pkg_root)

_hf_token = env_str("HF_TOKEN")
if _hf_token:
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", _hf_token)

from training.utils.devices import apply_runtime_cuda_env

apply_runtime_cuda_env()

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.adapter.streaming_adapter import StreamingAdapter
from src.dataset import SmartTurnGateDataset, smart_turn_collate
from src.encoder import WhisperWindowFeatureExtractor
from training.utils.checkpointing import (
    TrainingCheckpoint,
    load_gate_state_dict_safe,
    resolve_gate_config_from_checkpoint,
    save_checkpoint,
)
from training.utils.config import (
    AsrExperimentConfig,
    CheckpointConfig,
    DataConfig,
    DeviceConfig,
    FrozenModelIdsConfig,
    GateConfig,
    OptimConfig,
    Stage2Config,
    WandbConfig,
)
from training.utils.devices import cleanup_distributed, init_training_context
from training.utils.gate_training import build_turn_end_gate, make_silence_trackers
from training.utils.loaders import default_device_and_dtype
from training.utils.logging import WandbLogger
from training.utils.metrics import RunningMean
from training.utils.optimization import TrainingPipeline

_, TORCH_DTYPE = default_device_and_dtype()
_TRAINING_DIR = os.path.dirname(__file__)

_MODEL_IDS = FrozenModelIdsConfig.from_env()
WHISPER_DIM = 768
LLM_DIM = 4096
WHISPER_MODEL = _MODEL_IDS.whisper_model_id

STAGE = Stage2Config.from_env()
GATE = GateConfig.from_env()
DEVICE_CFG = DeviceConfig.from_env()
OPT = OptimConfig.from_env()
DATA = DataConfig.from_env(default_dataset_root=_TRAINING_DIR)
CKPT = CheckpointConfig.from_env(pkg_root=_pkg_root)
EXP = AsrExperimentConfig.from_env(pkg_root=_pkg_root)
WANDB = WandbConfig.from_env()

SAVE_PATH = os.path.join(CKPT.dir, f"{EXP.checkpoint_basename}.pt")

_default_adapter = env_str("ADAPTER_CHECKPOINT", "checkpoints/adapter_stage1.pt") or (
    "checkpoints/adapter_stage1.pt"
)
ADAPTER_CHECKPOINT = EXP.adapter_checkpoint or (
    _default_adapter
    if os.path.isabs(_default_adapter)
    else os.path.join(_pkg_root, _default_adapter)
)


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, DDP) else module


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


def train() -> None:
    if GATE.label_source != "smart_turn":
        print(
            "[WARN] GATE_LABEL_SOURCE is not smart_turn; "
            "gate_training.py expects Smart Turn labels. Continuing anyway."
        )

    ctx = init_training_context()
    train_device = ctx.device
    train_device_str = str(train_device)

    if ctx.is_main:
        print(f"Gate training device: {train_device_str}")
        print(f"  Adapter checkpoint (frozen): {ADAPTER_CHECKPOINT}")
        print(f"  Save path: {SAVE_PATH}")

    audio = WhisperWindowFeatureExtractor(
        model_id=WHISPER_MODEL,
        device=train_device_str,
        torch_dtype=TORCH_DTYPE,
    )

    adapter = StreamingAdapter(
        d_encoder=WHISPER_DIM,
        d_llm=LLM_DIM,
        num_queries=2,
        num_layers=2,
        num_heads=4,
        d_ffn=2048,
        dropout=0.0,
        ema_alpha=0.8,
        learnable_ema=False,
        use_rate_controller=False,
    ).to(train_device, dtype=TORCH_DTYPE)

    if not os.path.isfile(ADAPTER_CHECKPOINT):
        raise FileNotFoundError(f"Adapter checkpoint not found: {ADAPTER_CHECKPOINT}")
    adapter_ckpt = torch.load(ADAPTER_CHECKPOINT, map_location=train_device)
    adapter.load_state_dict(adapter_ckpt["adapter_state_dict"], strict=False)
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad = False

    if ctx.is_main:
        print(
            f"Loaded frozen adapter from {ADAPTER_CHECKPOINT} "
            f"(epoch {adapter_ckpt.get('epoch', '?')})"
        )

    gate_cfg = GATE
    resume_ckpt = None
    resume_path = CKPT.resume_checkpoint or SAVE_PATH
    if os.path.exists(resume_path):
        resume_meta = torch.load(resume_path, map_location="cpu")
        gate_cfg = resolve_gate_config_from_checkpoint(resume_meta, defaults=GATE)
        if ctx.is_main:
            resume_ckpt = torch.load(resume_path, map_location=train_device)
            print(f"Will resume gate training from: {resume_path}")

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

    if ctx.world_size > 1:
        adapter = DDP(adapter, device_ids=[ctx.local_rank])
        gate = DDP(gate, device_ids=[ctx.local_rank])

    gate_module = _unwrap(gate)
    adapter_module = _unwrap(adapter)

    dataset = SmartTurnGateDataset(
        GATE.smart_turn_dataset,
        split=GATE.smart_turn_split,
        max_samples=GATE.smart_turn_max_samples,
    )
    sampler: DistributedSampler | None = None
    if ctx.world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=ctx.world_size, rank=ctx.rank, shuffle=True
        )
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=DATA.num_workers,
        collate_fn=smart_turn_collate,
    )

    trainable_params = list(gate_module.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=OPT.lr, weight_decay=OPT.weight_decay)
    grad_accum_steps = DATA.gradient_accumulation_steps()
    pipeline = TrainingPipeline(
        optimizer=optimizer,
        scheduler=None,
        grad_clip_norm=OPT.grad_clip_norm,
        gradient_accumulation_steps=grad_accum_steps,
    )

    total_steps = max(1, STAGE.epochs * len(dataloader))
    scheduler = _build_lr_scheduler(optimizer, total_steps=total_steps)
    pipeline.scheduler = scheduler

    if ctx.is_main:
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    logger = WandbLogger(
        enabled=WANDB.enabled and ctx.is_main,
        project=WANDB.project,
        entity=WANDB.entity,
        run_name=WANDB.run_name or EXP.checkpoint_basename,
        config={
            "task": "gate_training",
            "gate_cfg": gate_cfg.__dict__,
            "stage_cfg": STAGE.__dict__,
            "exp_cfg": EXP.__dict__,
            "adapter_checkpoint": ADAPTER_CHECKPOINT,
        },
    )

    start_epoch = 0
    if ctx.is_main and resume_ckpt is not None:
        load_gate_state_dict_safe(gate_module, resume_ckpt, warn=False)
        if resume_ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        pipeline.global_step = int(resume_ckpt.get("global_step", 0))
        start_epoch = max(0, int(resume_ckpt.get("epoch", 1)) - 1)

    if ctx.is_main:
        print("\nStarting Smart Turn gate training (adapter frozen)")
        print(f"  Dataset: {GATE.smart_turn_dataset} [{GATE.smart_turn_split}]")
        print(f"  Clips: {len(dataset)}")
        print(f"  Epochs: {STAGE.epochs}\n")

    for epoch in range(start_epoch, STAGE.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        m_gate = RunningMean()

        for step, batch in enumerate(dataloader):
            waveforms, endpoint_labels = batch
            wave = waveforms[0].to(train_device)
            endpoint = endpoint_labels[0, 0].item()

            windows = audio.waveform_to_windows(wave)
            if DATA.max_windows_per_utt is not None:
                windows = windows[: DATA.max_windows_per_utt]
            if not windows:
                continue

            adapter_module.reset_streaming_state()
            silence_tracker, learned_silence_tracker = make_silence_trackers(gate_module)
            utterance_tokens: list[torch.Tensor] = []
            total_gate_loss = torch.zeros((), device=train_device, dtype=torch.float32)
            gate_calls = 0

            with torch.no_grad():
                for window in windows:
                    result = adapter_module.forward_window(
                        window.to(device=train_device, dtype=TORCH_DTYPE)
                    )
                    utterance_tokens.append(result["tokens"])

            for t in range(len(utterance_tokens)):
                accumulated = torch.cat(utterance_tokens[: t + 1], dim=1)
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

            gate_loss_mean = total_gate_loss / float(max(gate_calls, 1))
            total_loss = STAGE.lambda_gate * gate_loss_mean
            pipeline.step(total_loss, trainable_params)
            m_gate.update(float(gate_loss_mean.item()))

            if ctx.is_main and step % 10 == 0:
                print(
                    f"Step {step:4d}/{len(dataloader)} | Gate: {gate_loss_mean.item():.4f} | "
                    f"endpoint={endpoint:.0f}"
                )
                logger.log({"train/gate": gate_loss_mean.item()}, step=pipeline.global_step)

        if ctx.is_main:
            ckpt = TrainingCheckpoint(
                stage=2,
                epoch=epoch + 1,
                global_step=pipeline.global_step,
                adapter_state_dict=adapter_module.state_dict(),
                gate_state_dict=gate_module.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                metrics={"gate": m_gate.mean},
                hyperparams={
                    "gate_cfg": gate_cfg.__dict__,
                    "stage_cfg": STAGE.__dict__,
                    "exp_cfg": EXP.__dict__,
                    "adapter_checkpoint": ADAPTER_CHECKPOINT,
                },
            )
            save_checkpoint(SAVE_PATH, ckpt)
            epoch_path = os.path.join(
                CKPT.dir, f"{EXP.checkpoint_basename}_epoch{epoch + 1}.pt"
            )
            save_checkpoint(epoch_path, ckpt)
            print(
                f"\nEpoch {epoch + 1} gate training complete: gate={m_gate.mean:.4f}\n"
                f"  Checkpoint -> {SAVE_PATH}\n"
                f"  Epoch file -> {epoch_path}\n"
            )

    if ctx.is_main:
        print("Gate training complete!")
        logger.finish()
    cleanup_distributed()


if __name__ == "__main__":
    train()
