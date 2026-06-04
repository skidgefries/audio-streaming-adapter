"""
Stage 2: ASR distillation — frozen Whisper + frozen Qwen, train StreamingAdapter + TurnEndCommitGate.

The turn-end gate (Component 3) replaces separate VAD/turn detection: it classifies accumulated
adapter tokens and triggers LLM generation when ``should_commit`` fires. See ``docs/EARLY_COMMIT.md``.

LM conditioning during training is **train-style**: ``[audio_tokens | BOS | teacher-forced
transcript embeddings]`` with CE loss on the reference text (no ASR instruction prompt).
Prompt-based inference (``--asr-prompt`` + audio) is the target for Stage 3; use
``evaluation/eval_librispeech_asr_metrics.py --prompt-asr`` to evaluate that path.

Gate label sources (``GATE_LABEL_SOURCE``):
  - ``synthetic`` (default): final window label=1 on LibriSpeech full utterances
  - ``smart_turn``: gate-only fine-tune on pipecat Smart Turn ``endpoint_bool`` labels

# Previous gate import (reference):
# from src.adapter.early_commit_gate import EarlyCommitGate

Configuration is loaded from ``.env`` in the package root (see ``.env.example``).

**Single GPU**::

    uv run training/adapter_asr_trainer.py

**Multi-GPU** (frozen Qwen sharded automatically via ``device_map="auto"``)::

    uv run torchrun --standalone --nnodes=1 --nproc_per_node=1 training/adapter_asr_trainer.py

Or use ``bash scripts/setup_remote_training.sh``, which sources ``.env``, checks PyTorch
CUDA compatibility, downloads data/checkpoints, and picks ``uv run`` vs ``torchrun``.
"""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_root = os.path.join(_pkg_root, "src")
sys.path.insert(0, _src_root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, _pkg_root)

from training.utils.env import apply_hf_hub_endpoint, env_str, load_project_env

_env_path = load_project_env(_pkg_root)
if _env_path:
    print(f"Loaded environment from {_env_path}")

_hf_endpoint = apply_hf_hub_endpoint(_pkg_root)
print(f"HF Hub endpoint: {_hf_endpoint}")

_hf_token = env_str("HF_TOKEN")
if _hf_token:
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", _hf_token)

from training.utils.devices import apply_runtime_cuda_env

apply_runtime_cuda_env()

if env_str("CHECK_TORCH_COMPAT", "0") in ("1", "true", "yes", "on"):
    from training.utils.torch_compat import ensure_torch_compatible

    ensure_torch_compatible()

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from src.adapter.streaming_adapter import StreamingAdapter
from src.adapter.turn_end_commit_gate import TurnEndCommitGate
from src.dataset import (
    LibriSpeechConfig,
    LibriSpeechPairs,
    SmartTurnGateDataset,
    load_mono_waveform_16k,
    smart_turn_collate,
)
from src.encoder import WhisperWindowFeatureExtractor
from training.utils.checkpointing import (
    TrainingCheckpoint,
    load_gate_state_dict_safe,
    maybe_upload_stage_epoch_checkpoint,
    save_checkpoint,
)
from training.utils.config import (
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
from training.utils.gate_training import endpoint_label_for_timestep, make_silence_trackers
from training.utils.devices import (
    cleanup_distributed,
    init_training_context,
    llm_device_map,
    llm_input_device,
)
from training.utils.logging import WandbLogger
from training.utils.losses import contrastive_infonce_loss
from training.utils.metrics import RunningMean
from training.utils.asr_prompt import DEFAULT_ASR_PROMPT, PROMPT_CONDITIONING, TRAIN_STYLE_CONDITIONING
from training.utils.loaders import default_device_and_dtype, load_frozen_qwen_causal_lm
from training.utils.optimization import TrainingPipeline
from training.utils.stage2_validation import validate_stage2_asr, wandb_val_log_dict

_, TORCH_DTYPE = default_device_and_dtype()
_TRAINING_DIR = os.path.dirname(__file__)
DATASET_ROOTS = LibriSpeechConfig.train_clean_100_and_360_roots(_TRAINING_DIR)
VAL_ROOT = LibriSpeechConfig.dev_clean_root(_TRAINING_DIR)

_MODEL_IDS = FrozenModelIdsConfig.from_env()
WHISPER_DIM = 768
LLM_DIM = 4096
WHISPER_MODEL = _MODEL_IDS.whisper_model_id
LLM_MODEL_ID = _MODEL_IDS.llm_model_id

_TRAINING_DIR = os.path.dirname(__file__)
DATASET_ROOTS = LibriSpeechConfig.resolve_train_roots(
    _TRAINING_DIR,
    env_override=env_str("DATASET_ROOT"),
)
VAL_ROOT = LibriSpeechConfig.dev_clean_root(_TRAINING_DIR)

STAGE = Stage2Config.from_env()
GATE = GateConfig.from_env()
DEVICE_CFG = DeviceConfig.from_env()
OPT = OptimConfig.from_env()
DATA = DataConfig.from_env(default_dataset_root=DATASET_ROOTS[0])
CKPT = CheckpointConfig.from_env(pkg_root=_pkg_root)
HF_CKPT = HfCheckpointConfig.from_env()
WANDB = WandbConfig(enabled=True, project="audio-streaming-adapter", run_name="stage2-bigger-dataset")

SAVE_PATH = os.path.join(CKPT.dir, "adapter_stage2.pt")
_stage1_rel = env_str("STAGE1_CHECKPOINT", "checkpoints/adapter_stage1.pt") or "checkpoints/adapter_stage1.pt"
STAGE1_SAVE_PATH = (
    _stage1_rel if os.path.isabs(_stage1_rel) else os.path.join(_pkg_root, _stage1_rel)
)

USE_RATE_CONTROLLER = STAGE.use_rate_controller
RATE_TARGET = STAGE.rate_target


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
    samples_seen: int,
    batch_size: int,
    adapter: torch.nn.Module,
    gate: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    metrics: dict[str, float],
) -> TrainingCheckpoint:
    return TrainingCheckpoint(
        stage=2,
        epoch=epoch,
        global_step=global_step,
        adapter_state_dict=_unwrap(adapter).state_dict(),
        gate_state_dict=_unwrap(gate).state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),
        metrics=metrics,
        hyperparams={"batch_size": batch_size, "samples_seen": samples_seen},
    )


def _maybe_save_step_checkpoint(
    *,
    epoch: int,
    samples_seen: int,
    adapter: torch.nn.Module,
    gate: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    pipeline: TrainingPipeline,
    metrics: dict[str, float],
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
        samples_seen=samples_seen,
        batch_size=DATA.batch_size,
        adapter=adapter,
        gate=gate,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
    )
    save_checkpoint(SAVE_PATH, ckpt)
    step_path = os.path.join(CKPT.dir, f"adapter_stage2_step{step}.pt")
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


def _make_stage2_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    eta_min: float = 1e-5,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup then linear decay to eta_min."""
    warmup_steps = min(max(warmup_steps, 1), max(total_steps - 1, 1))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    decay_steps = max(total_steps - warmup_steps, 1)
    base_lr = optimizer.param_groups[0]["lr"]
    end_factor = eta_min / base_lr if base_lr > 0 else 1.0
    decay = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=end_factor,
        total_iters=decay_steps,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, decay],
        milestones=[warmup_steps],
    )


@dataclass(frozen=True)
class Stage2ResumeState:
    """Where to continue in the dataloader (derived from utterances seen)."""

    start_epoch: int = 0
    skip_steps_in_epoch: int = 0
    samples_seen: int = 0


def _resume_stage2_if_present(
    *,
    save_path: str,
    train_device: torch.device,
    adapter: torch.nn.Module,
    gate: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    pipeline: TrainingPipeline,
    steps_per_epoch: int,
    batch_size: int,
) -> Stage2ResumeState:
    """Restore weights; resume dataloader position from utterances seen (batch-size aware)."""
    if not os.path.exists(save_path):
        return Stage2ResumeState()

    print(f"Resuming from checkpoint: {save_path}")
    ckpt = torch.load(save_path, map_location=train_device)
    _unwrap(adapter).load_state_dict(ckpt["adapter_state_dict"])
    load_gate_state_dict_safe(_unwrap(gate), ckpt)
    if ckpt.get("optimizer_state_dict") is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except ValueError as exc:
            print(
                f"  [WARN] Could not load optimizer state ({exc}); "
                "using fresh AdamW moments (adapter/gate weights still restored)."
            )

    pipeline.global_step = int(ckpt.get("global_step", 0))
    hp = ckpt.get("hyperparams") or {}
    ckpt_batch = int(hp.get("batch_size", 1)) or 1
    samples_seen = int(hp.get("samples_seen", pipeline.global_step * ckpt_batch))
    steps_per_epoch = max(steps_per_epoch, 1)
    batches_done = samples_seen // max(batch_size, 1)
    start_epoch = batches_done // steps_per_epoch
    skip_steps = batches_done % steps_per_epoch
    if ckpt_batch != batch_size:
        print(
            f"  [INFO] Checkpoint batch_size={ckpt_batch} -> current={batch_size}; "
            f"resume at utterance {samples_seen} (batch {skip_steps + 1}/{steps_per_epoch} "
            f"in epoch {start_epoch + 1})."
        )

    sched_sd = ckpt.get("scheduler_state_dict")
    if sched_sd is not None:
        try:
            scheduler.load_state_dict(sched_sd)
        except (ValueError, KeyError) as exc:
            print(
                f"  [WARN] Could not load scheduler state ({exc}); "
                f"fast-forwarding {pipeline.global_step} scheduler steps instead."
            )
            for _ in range(pipeline.global_step):
                scheduler.step()
    elif pipeline.global_step > 0:
        for _ in range(pipeline.global_step):
            scheduler.step()

    print(
        f"  Resumed at epoch {start_epoch + 1}/{STAGE.epochs}, "
        f"batch {skip_steps + 1}/{steps_per_epoch}, utterances {samples_seen}, "
        f"global_step {pipeline.global_step}, lr={scheduler.get_last_lr()[0]:.2e}\n"
    )
    return Stage2ResumeState(
        start_epoch=start_epoch,
        skip_steps_in_epoch=skip_steps,
        samples_seen=samples_seen,
    )


def train_smart_turn_gate(ctx, *, train_device, audio, adapter, gate, pipeline, optimizer, scheduler, logger) -> None:
    """Gate-only fine-tune on Smart Turn endpoint labels; adapter frozen."""
    adapter_module = _unwrap(adapter)
    gate_module = _unwrap(gate)
    for p in adapter_module.parameters():
        p.requires_grad = False
    adapter_module.eval()

    dataset = SmartTurnGateDataset(
        GATE.smart_turn_dataset,
        split=GATE.smart_turn_split,
        max_samples=GATE.smart_turn_max_samples,
    )
    sampler: DistributedSampler | None = None
    if ctx.world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=ctx.world_size, rank=ctx.rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=DATA.num_workers,
        collate_fn=smart_turn_collate,
    )

    trainable_params = list(gate_module.parameters())
    if ctx.is_main:
        print("\nStarting Stage 2 gate fine-tune: Smart Turn labels (adapter frozen)")
        print(f"  Dataset: {GATE.smart_turn_dataset} [{GATE.smart_turn_split}]")
        print(f"  Clips: {len(dataset)}\n")

    for epoch in range(STAGE.epochs):
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
                for t, window in enumerate(windows):
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
            save_checkpoint(
                SAVE_PATH,
                TrainingCheckpoint(
                    stage=2,
                    epoch=epoch + 1,
                    global_step=pipeline.global_step,
                    adapter_state_dict=adapter_module.state_dict(),
                    gate_state_dict=gate_module.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(),
                    metrics={"gate": m_gate.mean},
                ),
            )
            print(f"\nEpoch {epoch + 1} gate fine-tune complete: gate={m_gate.mean:.4f}\n")


def train() -> None:
    ctx = init_training_context()
    train_device = ctx.device
    train_device_str = str(train_device)

    if ctx.is_main:
        print(f"Training device: {train_device_str} (configured: {DEVICE_CFG.device})")
        print(f"Visible CUDA devices: {ctx.num_cuda_devices}")
        print(f"Distributed: world_size={ctx.world_size} rank={ctx.rank}")
        print(f"Model parallel (LLM auto-shard): {ctx.model_parallel}")

    audio = WhisperWindowFeatureExtractor(
        model_id=WHISPER_MODEL, device=train_device_str, torch_dtype=TORCH_DTYPE
    )

    qwen_map = llm_device_map(ctx)
    if ctx.is_main:
        print(f"Qwen device_map: {qwen_map!r}")

    qwen_models = load_frozen_qwen_causal_lm(
        model_id=LLM_MODEL_ID,
        device=train_device_str,
        torch_dtype=TORCH_DTYPE,
        device_map=qwen_map,
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
        if STAGE.enable_llm_gradient_checkpointing:
            print("Qwen gradient checkpointing: enabled")

    adapter = StreamingAdapter(
        d_encoder=WHISPER_DIM,
        d_llm=LLM_DIM,
        num_queries=4,
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

    stage1_ckpt = torch.load(STAGE1_SAVE_PATH, map_location=train_device)
    adapter.load_state_dict(stage1_ckpt["adapter_state_dict"], strict=False)
    if ctx.is_main:
        print(
            f"Loaded Stage 1 adapter weights from {STAGE1_SAVE_PATH} "
            f"(epoch {stage1_ckpt.get('epoch', '?')})"
        )

    # gate = EarlyCommitGate(d_llm=LLM_DIM, hidden_dim=256, threshold=0.5, latency_weight=0.1)
    gate = TurnEndCommitGate(
        d_llm=LLM_DIM,
        hidden_dim=GATE.hidden_dim,
        threshold=GATE.threshold,
        latency_weight=GATE.latency_weight,
        min_silence_ms=GATE.min_silence_ms,
        require_silence_for_commit=GATE.require_silence_for_commit,
        token_activity_threshold=GATE.token_activity_threshold,
        window_duration_sec=GATE.window_seconds,
        silence_mode=GATE.silence_mode,
        active_silence_path=GATE.active_silence_path,
        learned_silence_hidden_dim=GATE.learned_silence_hidden_dim,
    ).to(train_device, dtype=TORCH_DTYPE)
    gate.train()

    if ctx.world_size > 1:
        adapter = DDP(adapter, device_ids=[ctx.local_rank])
        gate = DDP(gate, device_ids=[ctx.local_rank])

    if ctx.is_main:
        print(
            f"Models initialized:\n  Adapter: trainable\n  Turn-end gate: trainable\n"
            f"  Gate labels: {GATE.label_source}\n"
            f"  Gate silence mode: {GATE.silence_mode}"
            f"{f' (active={GATE.active_silence_path})' if GATE.silence_mode == 'both' else ''}\n"
            f"  Rate controller: {USE_RATE_CONTROLLER}\n"
        )

    trainable_params = list(_unwrap(adapter).parameters()) + list(_unwrap(gate).parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=OPT.lr, weight_decay=OPT.weight_decay)

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

    total_steps = max(STAGE.epochs * len(dataloader), 1)
    scheduler = _make_stage2_lr_scheduler(
        optimizer, total_steps=total_steps, warmup_steps=OPT.warmup_steps
    )
    pipeline = TrainingPipeline(optimizer=optimizer, scheduler=scheduler, grad_clip_norm=OPT.grad_clip_norm)

    if ctx.is_main:
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    logger = WandbLogger(
        enabled=WANDB.enabled and ctx.is_main,
        project=WANDB.project,
        entity=WANDB.entity,
        run_name=WANDB.run_name,
        config={
            "stage": 2,
            "data": {**DATA.__dict__, "dataset_roots": DATASET_ROOTS, "val_root": VAL_ROOT},
            "optim": OPT.__dict__,
            "stage_cfg": STAGE.__dict__,
            "gate_cfg": GATE.__dict__,
            "device": DEVICE_CFG.__dict__,
            "num_cuda_devices": ctx.num_cuda_devices,
            "model_parallel": ctx.model_parallel,
            "world_size": ctx.world_size,
        },
    )

    if ctx.is_main:
        resume = _resume_stage2_if_present(
            save_path=SAVE_PATH,
            train_device=train_device,
            adapter=adapter,
            gate=gate,
            optimizer=optimizer,
            scheduler=scheduler,
            pipeline=pipeline,
            steps_per_epoch=len(dataloader),
            batch_size=DATA.batch_size,
        )
    else:
        resume = Stage2ResumeState()

    samples_seen = resume.samples_seen

    if ctx.is_main:
        print("\nStarting Stage 2 training: ASR Distillation")
        print(f"  Dataset splits: {', '.join(os.path.basename(r) for r in DATASET_ROOTS)}")
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
                f"{CKPT.dir}/adapter_stage2_step<N>.pt (and {SAVE_PATH})"
            )
        print(f"  Epochs: {STAGE.epochs}")
        print(f"  Batch size: {DATA.batch_size}")
        print(f"  Learning rate: {OPT.lr} (warmup {OPT.warmup_steps} steps, then linear to 1e-6)")
        print(f"  LR schedule total steps: {total_steps}")
        print(f"  λ_align: {STAGE.lambda_align}")
        print(f"  λ_stability: {STAGE.lambda_stability}")
        print(f"  λ_rate: {STAGE.lambda_rate}")
        print(f"  λ_gate: {STAGE.lambda_gate}")
        print(f"  Gate label source: {GATE.label_source}")
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
        train_smart_turn_gate(
            ctx,
            train_device=train_device,
            audio=audio,
            adapter=adapter,
            gate=gate,
            pipeline=pipeline,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
        )
        if ctx.is_main:
            print("Stage 2 Smart Turn gate fine-tune complete!")
            logger.finish()
        cleanup_distributed()
        return

    for epoch in range(resume.start_epoch, STAGE.epochs):
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

        steps_per_epoch = max((len(dataset) + DATA.batch_size - 1) // DATA.batch_size, 1)
        skip_batches = resume.skip_steps_in_epoch if epoch == resume.start_epoch else 0
        sample_offset = resume.samples_seen if epoch == resume.start_epoch else 0
        epoch_dataset: Dataset = dataset
        if sample_offset > 0:
            epoch_dataset = Subset(dataset, range(sample_offset, len(dataset)))
            if ctx.is_main:
                print(
                    f"Resuming epoch {epoch + 1} at batch {skip_batches + 1}/{steps_per_epoch} "
                    f"(utterance {sample_offset + 1}/{len(dataset)}, global_step {pipeline.global_step})\n"
                )
        epoch_loader = DataLoader(
            epoch_dataset,
            batch_size=DATA.batch_size,
            shuffle=sampler is None and skip_batches == 0,
            sampler=None if skip_batches > 0 else sampler,
            num_workers=DATA.num_workers,
        )

        for step, batch in enumerate(epoch_loader, start=skip_batches):
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
            gate_module = _unwrap(gate)

            for p in audio_paths:
                wave = load_mono_waveform_16k(p)
                windows = audio.waveform_to_windows(wave)
                if DATA.max_windows_per_utt is not None:
                    windows = windows[: DATA.max_windows_per_utt]

                adapter_module.reset_streaming_state()
                silence_tracker, learned_silence_tracker = make_silence_trackers(gate_module)
                utterance_tokens: list[torch.Tensor] = []
                for t, window in enumerate(windows):
                    result = adapter_module.forward_window(
                        window.to(device=train_device, dtype=TORCH_DTYPE)
                    )
                    utterance_tokens.append(result["tokens"])
                    total_stability_loss = total_stability_loss + result["stability_loss"].float()
                    if result["sparse_loss"] is not None:
                        total_sparse_loss = total_sparse_loss + result["sparse_loss"].float()
                    if result["rate_loss"] is not None:
                        total_rate_loss = total_rate_loss + result["rate_loss"].float()

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
            )

            total_windows = sum(num_windows_list)
            stability_loss = (
                total_stability_loss / float(total_windows)
                if total_windows > 0
                else torch.tensor(0.0, device=train_device)
            )
            sparse_loss = (
                total_sparse_loss / float(total_windows)
                if total_windows > 0
                else torch.tensor(0.0, device=train_device)
            )
            rate_loss = (
                total_rate_loss / float(total_windows)
                if total_windows > 0
                else torch.tensor(0.0, device=train_device)
            )
            gate_loss_mean = (
                total_gate_loss / float(gate_calls) if gate_calls > 0 else torch.tensor(0.0, device=train_device)
            )

            total_loss = (
                asr_loss
                + STAGE.lambda_align * align_loss
                + STAGE.lambda_stability * stability_loss
                + STAGE.lambda_rate * rate_loss
                + STAGE.lambda_gate * gate_loss_mean
            )

            pipeline.step(total_loss, trainable_params)
            samples_seen += len(audio_paths)

            m_total.update(total_loss.item())
            m_asr.update(asr_loss.item())
            m_align.update(align_loss.item())
            m_stab.update(float(stability_loss.item()))
            m_sparse.update(sparse_loss.item())
            m_rate.update(rate_loss.item())
            m_gate.update(float(gate_loss_mean.item()))

            if ctx.is_main and step % 1 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"Step {pipeline.global_step:5d}/{total_steps} "
                    f"(epoch {epoch + 1} batch {step + 1}/{len(dataloader)}) | Loss: {total_loss.item():.4f} | "
                    f"ASR: {asr_loss.item():.4f} | Align: {align_loss.item():.4f} | "
                    f"Stab: {stability_loss.float().item():.4f} | "
                    f"Rate: {rate_loss.item():.4f} | Gate: {gate_loss_mean.float().item():.4f} | "
                    f"Sparse(metric): {sparse_loss.item():.4f} | LR: {current_lr:.2e}"
                )
                logger.log(
                    {
                        "train/loss": total_loss.item(),
                        "train/asr": asr_loss.item(),
                        "train/align": align_loss.item(),
                        "train/stability": stability_loss.float().item(),
                        "train/rate": rate_loss.item(),
                        "train/gate": gate_loss_mean.float().item(),
                        "train/sparse_metric": sparse_loss.item(),
                        "train/lr": current_lr,
                    },
                    step=pipeline.global_step,
                )

            if (
                ctx.is_main
                and STAGE.val_enabled
                and pipeline.global_step > 0
                and pipeline.global_step % STAGE.val_every_steps == 0
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

            if ctx.is_main:
                _maybe_save_step_checkpoint(
                    epoch=epoch,
                    samples_seen=samples_seen,
                    adapter=adapter,
                    gate=gate,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    pipeline=pipeline,
                    metrics=_stage2_epoch_metrics(
                        m_total, m_asr, m_align, m_stab, m_sparse, m_rate, m_gate
                    ),
                )

            if train_device.type == "cuda":
                torch.cuda.empty_cache()

        if ctx.is_main:
            epoch_metrics = _stage2_epoch_metrics(
                m_total, m_asr, m_align, m_stab, m_sparse, m_rate, m_gate
            )
            save_checkpoint(
                SAVE_PATH,
                _make_stage2_checkpoint(
                    epoch=epoch + 1,
                    global_step=pipeline.global_step,
                    samples_seen=samples_seen,
                    batch_size=DATA.batch_size,
                    adapter=adapter,
                    gate=gate,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metrics=epoch_metrics,
                ),
            )

            epoch_save_path = os.path.join(CKPT.dir, f"adapter_stage2_epoch{epoch + 1}.pt")
            save_checkpoint(
                epoch_save_path,
                _make_stage2_checkpoint(
                    epoch=epoch + 1,
                    global_step=pipeline.global_step,
                    samples_seen=samples_seen,
                    batch_size=DATA.batch_size,
                    adapter=adapter,
                    gate=gate,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metrics=epoch_metrics,
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
        print("Stage 2 training complete!")
        logger.finish()
    cleanup_distributed()


if __name__ == "__main__":
    train()
