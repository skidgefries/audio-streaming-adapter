"""
ASR-only training: window → encoder → adapter → frozen LLM.

Trains the StreamingAdapter with causal LM loss on teacher-forced transcripts.
By default initializes adapter weights from a Stage 1 checkpoint
(``STAGE1_CHECKPOINT``, default ``checkpoints/stage1_final_checkpoint.pt``).
No gate, no rate controller. Align/stability/rate/sparse/gate losses are computed
and logged for monitoring but only ASR loss drives backpropagation. Validation on
dev-clean reports the same auxiliary losses (batched), plus WER and BLEU-4.

Pipeline per utterance:
  1. AudioWaveformWindowizer (0.8s / 0.4s stride)
  2. Whisper encode per window (WhisperWindowFeatureExtractor)
  3. StreamingAdapter.forward_window(encoder_features)
  4. Frozen Qwen CE loss on [audio_tokens | BOS | transcript]

**Single process (2 GPUs recommended)**::

    uv run training/adapter_asr_only_trainer.py

Whisper + adapter run on ``cuda:0``. Frozen Qwen is loaded with
``device_map='sequential'`` and ``LLM_MAX_MEMORY=0:10GiB,1:2GiB,cpu:64GiB`` (10 GiB Qwen on GPU 0,
2 GiB spill on GPU 1, remainder on CPU; leaves ~5 GiB on GPU 0 for Whisper + adapter). Visible GPUs are reserved exclusively for the run
(``GPU_LOCK=true``); other training/eval scripts block until this process exits.
Weights & Biases logging is enabled by default
(``WANDB_ENABLED=true``); set ``WANDB_API_KEY`` in ``.env``.

**Multi-GPU data parallel**::

    uv run torchrun --standalone --nnodes=1 --nproc_per_node=1 training/adapter_asr_only_trainer.py
"""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_root = os.path.join(_pkg_root, "src")
sys.path.insert(0, _src_root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, _pkg_root)

from training.utils.env import apply_hf_hub_endpoint, env_bool, env_int, env_str, load_project_env

_env_path = load_project_env(_pkg_root)
if _env_path:
    print(f"Loaded environment from {_env_path}")

# ASR-only defaults: Whisper/adapter on cuda:0; Qwen sequential cuda:0 → cuda:1 → cpu.
# GPU 0 is ~16 GiB — capping Qwen at 14+ GiB leaves no room for Whisper/adapter activations.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ["DEVICE"] = "cuda:0"
os.environ["LLM_DEVICE"] = "cuda:0"
os.environ["LLM_MAX_MEMORY"] = "0:12GiB,1:2GiB,cpu:64GiB"
os.environ.setdefault("GPU_LOCK", "true")
os.environ.setdefault("VAL_MAX_UTTERANCES", "all")
os.environ.setdefault("WANDB_ENABLED", "true")
os.environ.setdefault("WANDB_RUN_NAME", "adapter_asr_only")

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
    checkpoint_grad_accum_steps,
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
    cleanup_distributed,
    ensure_device_ready,
    init_training_context,
    llm_input_device,
    resolve_llm_load_plan,
)
from training.utils.loaders import default_device_and_dtype, load_frozen_qwen_causal_lm
from training.utils.logging import WandbLogger
from training.utils.losses import contrastive_infonce_loss
from training.utils.metrics import RunningMean
from training.utils.optimization import TrainingPipeline

_, TORCH_DTYPE = default_device_and_dtype()
_TRAINING_DIR = os.path.dirname(__file__)

_MODEL_IDS = FrozenModelIdsConfig.from_env()
WHISPER_DIM = 768
LLM_DIM = 4096
WHISPER_MODEL = _MODEL_IDS.whisper_model_id
LLM_MODEL_ID = _MODEL_IDS.llm_model_id

DATASET_ROOTS = LibriSpeechConfig.resolve_train_roots(
    _TRAINING_DIR,
    env_override=env_str("DATASET_ROOT"),
)
VAL_ROOT = LibriSpeechConfig.dev_clean_root(_TRAINING_DIR)

STAGE = Stage2Config.from_env()
DEVICE_CFG = DeviceConfig.from_env()
OPT = OptimConfig.from_env()
DATA = DataConfig.from_env(default_dataset_root=DATASET_ROOTS[0])
CKPT = CheckpointConfig.from_env(pkg_root=_pkg_root)
HF_CKPT = HfCheckpointConfig.from_env()
WANDB = WandbConfig.from_env()

# CHECKPOINT_BASENAME = env_str("CHECKPOINT_BASENAME", "adapter_asr_only") or "adapter_asr_only"
CHECKPOINT_BASENAME = "adapter_stage2_infoNCE"
SAVE_PATH = os.path.join(CKPT.dir, f"{CHECKPOINT_BASENAME}.pt")
SAVE_EVERY_STEPS = CKPT.save_every_steps if CKPT.save_every_steps is not None else 500
_stage1_rel = (
    # env_str("STAGE1_CHECKPOINT", "checkpoints/stage1_final_checkpoint.pt")
    # or "checkpoints/stage1_final_checkpoint.pt"
    "checkpoints/adapter_infoNCE_stage1_with_centering.pt"
)

STAGE1_SAVE_PATH = (
    _stage1_rel if os.path.isabs(_stage1_rel) else os.path.join(_pkg_root, _stage1_rel)
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
    """Build [audio | BOS | teacher-forced text] embeddings and labels."""
    bos_token_id = (
        llm_tokenizer.bos_token_id
        if llm_tokenizer.bos_token_id is not None
        else llm_tokenizer.eos_token_id
    )
    bos_embed = text_embedder(
        torch.tensor([[bos_token_id]], device=llm_device).expand(audio_tokens.shape[0], -1)
    )
    inputs_embeds = torch.cat([audio_tokens.to(llm_device), bos_embed], dim=1)

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
    """window → encoder → adapter for one utterance."""
    wave = load_mono_waveform_16k(audio_path)
    windows = audio_extractor.waveform_to_windows(wave)
    if max_windows_per_utt is not None:
        windows = windows[:max_windows_per_utt]
    if not windows:
        return None

    adapter.reset_streaming_state()
    utterance_tokens: list[torch.Tensor] = []
    stability_sum = 0.0
    sparse_sum = 0.0
    rate_sum = 0.0
    for window in windows:
        result = adapter.forward_window(window.to(device=train_device, dtype=TORCH_DTYPE))
        utterance_tokens.append(result["tokens"])
        if collect_aux_metrics:
            stability_sum += float(result["stability_loss"].detach().float().item())
            if result["sparse_loss"] is not None:
                sparse_sum += float(result["sparse_loss"].detach().float().item())
            if result["rate_loss"] is not None:
                rate_sum += float(result["rate_loss"].detach().float().item())

    return UtteranceEncodeResult(
        tokens=torch.cat(utterance_tokens, dim=1),
        num_windows=len(windows),
        stability_loss=stability_sum,
        sparse_loss=sparse_sum,
        rate_loss=rate_sum,
    )


def _encode_utterance_tokens(
    *,
    adapter: StreamingAdapter,
    audio_extractor: WhisperWindowFeatureExtractor,
    audio_path: str,
    train_device: torch.device,
    max_windows_per_utt: int | None,
) -> torch.Tensor | None:
    encoded = _encode_utterance(
        adapter=adapter,
        audio_extractor=audio_extractor,
        audio_path=audio_path,
        train_device=train_device,
        max_windows_per_utt=max_windows_per_utt,
    )
    return encoded.tokens if encoded is not None else None


def _compute_aux_loss_metrics(
    *,
    audio_tokens: torch.Tensor,
    gt_embeds: torch.Tensor,
    total_stability_loss: float,
    total_windows: int,
    total_sparse_loss: float = 0.0,
    total_rate_loss: float = 0.0,
) -> dict[str, float]:
    """Auxiliary losses. Training may pass sparse/rate for monitoring; val uses align+stab only."""
    with torch.no_grad():
        align_loss = float(
            contrastive_infonce_loss(
                audio_tokens=audio_tokens.detach().float(),
                text_embeddings=gt_embeds.detach().float(),
                temperature=STAGE.temperature,
            ).item()
        )

    if total_windows > 0:
        stability_loss = total_stability_loss / float(total_windows)
        sparse_loss = total_sparse_loss / float(total_windows)
        rate_loss = total_rate_loss / float(total_windows)
    else:
        stability_loss = 0.0
        sparse_loss = 0.0
        rate_loss = 0.0

    return {
        "align": align_loss,
        "stability": stability_loss,
        "sparse": sparse_loss,
        "rate": rate_loss,
        "gate": 0.0,
    }


def _build_stage2_lr_scheduler(
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
        hyperparams={
            "trainer": "adapter_asr_only_trainer",
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
            print(
                "[WARN] Optimizer state incompatible with current adapter "
                "(e.g. rate_controller removed); restarting optimizer."
            )
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
        print("Pipeline: window → encoder → adapter → LLM (ASR backprop only; aux losses logged)")

    audio = WhisperWindowFeatureExtractor(
        model_id=WHISPER_MODEL, device=train_device_str, torch_dtype=TORCH_DTYPE
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

    if STAGE.enable_llm_gradient_checkpointing and hasattr(llm_model, "gradient_checkpointing_enable"):
        llm_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    if hasattr(llm_model, "enable_input_require_grads"):
        llm_model.enable_input_require_grads()

    llm_model.train()
    for p in llm_model.parameters():
        p.requires_grad = False

    adapter = StreamingAdapter(
        d_encoder=WHISPER_DIM,
        d_llm=LLM_DIM,
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
                print(
                    f"RESUME_CHECKPOINT is Stage 1 ({resume_path}); "
                    "loading adapter weights only (fresh ASR optimizer/scheduler)."
                )
        else:
            resume_ckpt = torch.load(resume_path, map_location=train_device)
            resume_training_state = True
            if ctx.is_main:
                print(f"Will resume ASR-only training from checkpoint: {resume_path}")
    elif LOAD_STAGE1_CHECKPOINT:
        if not os.path.isfile(STAGE1_SAVE_PATH):
            raise FileNotFoundError(
                f"LOAD_STAGE1_CHECKPOINT=true but checkpoint missing: {STAGE1_SAVE_PATH}"
            )
        adapter_init_path = STAGE1_SAVE_PATH
        if ctx.is_main:
            print(f"Will init adapter from Stage 1 checkpoint: {STAGE1_SAVE_PATH}")
    elif ctx.is_main:
        print("Adapter: random init (no Stage 1 or ASR-only checkpoint found)")

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

    if ctx.is_main:
        print(
            f"Models initialized:\n"
            f"  Encoder: frozen Whisper ({WHISPER_MODEL})\n"
            f"  Adapter: trainable"
            f"{' (Stage 1 init)' if adapter_init_path and not resume_training_state else ''}"
            f"{' (resuming ASR-only)' if resume_training_state else ''}"
            f" (fixed 2 tokens/window, no rate controller)\n"
            f"  LLM: frozen ({LLM_MODEL_ID})\n"
            f"  Checkpoint basename: {CHECKPOINT_BASENAME}\n"
        )

    optimizer = torch.optim.AdamW(_unwrap(adapter).parameters(), lr=OPT.lr, weight_decay=OPT.weight_decay)
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
            "trainer": "adapter_asr_only",
            "data": {**DATA.__dict__, "dataset_roots": DATASET_ROOTS, "val_root": VAL_ROOT},
            "optim": OPT.__dict__,
            "stage_cfg": STAGE.__dict__,
            "device": DEVICE_CFG.__dict__,
            "checkpoint_basename": CHECKPOINT_BASENAME,
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
                f"dataloader offset {resume_batch_offset}/{len(dataloader)}\n"
            )

    if ctx.is_main:
        print("\nStarting ASR-only training")
        print(f"  Dataset splits: {', '.join(os.path.basename(r) for r in DATASET_ROOTS)}")
        print(f"  Epochs: {STAGE.epochs}")
        print(f"  Micro-batch size: {DATA.batch_size}")
        print(f"  Effective batch size: {effective_batch_size}")
        print(f"  Learning rate: {OPT.lr}")
        print(f"  Max text tokens: {STAGE.max_text_tokens}")
        print(f"  ASR micro-batch: {STAGE.asr_micro_batch_size}")
        print(
            "  Aux losses: align/stability/rate/sparse/gate logged (monitoring only; "
            "rate/sparse/gate are 0 without rate controller / early-commit gate)"
        )
        if STAGE.val_enabled and os.path.isdir(VAL_ROOT):
            cap = STAGE.val_max_utterances
            cap_str = "all" if cap is None else str(cap)
            print(
                f"  Validation: dev-clean ({VAL_ROOT}), every {STAGE.val_every_steps} steps, "
                f"{cap_str} utterances, losses (ASR+aux) + decode WER/BLEU-4"
            )
        if SAVE_EVERY_STEPS > 0:
            print(
                f"  Checkpoints: every {SAVE_EVERY_STEPS} steps -> "
                f"{CKPT.dir}/{CHECKPOINT_BASENAME}_step<N>.pt (and {SAVE_PATH})"
            )
        print(f"  LM conditioning: {TRAIN_STYLE_CONDITIONING}")
        print(f"  Stage 3 target ({PROMPT_CONDITIONING}): {DEFAULT_ASR_PROMPT[:72]}...\n")

    adapter_module = _unwrap(adapter)

    for epoch in range(start_epoch, STAGE.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

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
            gt_tokens = llm_tokenizer(
                list(transcriptions),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=STAGE.max_text_tokens,
            ).to(train_device)

            with torch.no_grad():
                gt_embeds = text_embedder(gt_tokens.input_ids.to(llm_device)).to(train_device)

            audio_tokens_list: list[torch.Tensor] = []
            total_windows = 0
            total_stability_loss = 0.0
            total_sparse_loss = 0.0
            total_rate_loss = 0.0
            for p in audio_paths:
                encoded = _encode_utterance(
                    adapter=adapter_module,
                    audio_extractor=audio,
                    audio_path=p,
                    train_device=train_device,
                    max_windows_per_utt=DATA.max_windows_per_utt,
                    collect_aux_metrics=True,
                )
                if encoded is not None:
                    audio_tokens_list.append(encoded.tokens)
                    total_windows += encoded.num_windows
                    total_stability_loss += encoded.stability_loss
                    total_sparse_loss += encoded.sparse_loss
                    total_rate_loss += encoded.rate_loss

            if not audio_tokens_list:
                if ctx.is_main:
                    print(f"[WARN] Step {step}: empty batch (no audio windows); skipping.")
                continue

            audio_tokens = _pad_audio_tokens(audio_tokens_list)
            aux_metrics = _compute_aux_loss_metrics(
                audio_tokens=audio_tokens,
                gt_embeds=gt_embeds,
                total_stability_loss=total_stability_loss,
                total_sparse_loss=total_sparse_loss,
                total_rate_loss=total_rate_loss,
                total_windows=total_windows,
            )
            inputs_embeds, labels, attention_mask = _build_inputs_for_asr(
                audio_tokens=audio_tokens,
                gt_ids=gt_tokens.input_ids,
                gt_attention_mask=gt_tokens.attention_mask,
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

            no_sync_modules = (adapter,) if ctx.world_size > 1 else None
            optimizer_stepped = pipeline.step(
                asr_loss,
                list(adapter_module.parameters()),
                no_sync_modules=no_sync_modules,
            )

            m_asr.update(asr_loss.item())
            m_align.update(aux_metrics["align"])
            m_stab.update(aux_metrics["stability"])
            m_sparse.update(aux_metrics["sparse"])
            m_rate.update(aux_metrics["rate"])
            m_gate.update(aux_metrics["gate"])

            accum_suffix = ""
            if grad_accum_steps > 1:
                accum_done = grad_accum_steps if optimizer_stepped else pipeline.accum_step
                accum_suffix = f" | accum {accum_done}/{grad_accum_steps}"

            if ctx.is_main:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"Micro {step:4d}/{len(dataloader)} | opt {pipeline.global_step:5d} | "
                    f"ASR: {asr_loss.item():.4f} | Align: {aux_metrics['align']:.4f} | "
                    f"Stab: {aux_metrics['stability']:.4f} | "
                    f"Rate: {aux_metrics['rate']:.4f} | Gate: {aux_metrics['gate']:.4f} | "
                    f"Sparse: {aux_metrics['sparse']:.4f} | LR: {current_lr:.2e}{accum_suffix}"
                )

            if ctx.is_main and optimizer_stepped:
                logger.log(
                    {
                        "train/asr": asr_loss.item(),
                        "train/align": aux_metrics["align"],
                        "train/stability": aux_metrics["stability"],
                        "train/rate": aux_metrics["rate"],
                        "train/gate": aux_metrics["gate"],
                        "train/sparse_metric": aux_metrics["sparse"],
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
                val_metrics, _ = validate_asr_only(
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
                    compute_aux_loss_metrics_fn=_compute_aux_loss_metrics,
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
                logger.log(val_metrics, step=pipeline.global_step)

            if ctx.is_main and optimizer_stepped:
                _maybe_save_step_checkpoint(
                    epoch=epoch,
                    adapter=adapter,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    pipeline=pipeline,
                    metrics={
                        "loss": m_asr.mean,
                        "asr": m_asr.mean,
                        "align": m_align.mean,
                        "stability": m_stab.mean,
                        "rate": m_rate.mean,
                        "gate": m_gate.mean,
                        "sparse": m_sparse.mean,
                    },
                )

            if llm_device.type == "cuda":
                with torch.cuda.device(llm_device):
                    torch.cuda.empty_cache()

        if ctx.is_main:
            epoch_metrics = {
                "loss": m_asr.mean,
                "asr": m_asr.mean,
                "align": m_align.mean,
                "stability": m_stab.mean,
                "rate": m_rate.mean,
                "gate": m_gate.mean,
                "sparse": m_sparse.mean,
            }
            ckpt = _make_checkpoint(
                epoch=epoch + 1,
                global_step=pipeline.global_step,
                adapter=adapter,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=epoch_metrics,
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
                f"\nEpoch {epoch + 1} complete: "
                f"ASR={m_asr.mean:.4f} align={m_align.mean:.4f} "
                f"stab={m_stab.mean:.4f} rate={m_rate.mean:.4f} "
                f"gate={m_gate.mean:.4f} sparse={m_sparse.mean:.4f}\n"
                f"Checkpoint -> {SAVE_PATH}\n"
            )

    if ctx.is_main:
        print("ASR-only training complete!")
        logger.finish()
    cleanup_distributed()


if __name__ == "__main__":
    train()
