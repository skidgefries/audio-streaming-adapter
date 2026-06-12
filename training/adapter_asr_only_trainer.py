"""
ASR-only training: window → encoder → adapter → frozen LLM.

Trains the StreamingAdapter from scratch (random init) with causal LM loss on
teacher-forced transcripts. No Stage 1 alignment checkpoint, no gate, no
contrastive/align/stability/rate auxiliary losses, no rate controller.

Pipeline per utterance:
  1. AudioWaveformWindowizer (0.8s / 0.4s stride)
  2. Whisper encode per window (WhisperWindowFeatureExtractor)
  3. StreamingAdapter.forward_window(encoder_features)
  4. Frozen Qwen CE loss on [audio_tokens | BOS | transcript]

**Single process (2 GPUs recommended)**::

    uv run training/adapter_asr_only_trainer.py

Whisper + adapter run on ``cuda:0``. Frozen Qwen is loaded with
``device_map='sequential'`` and ``LLM_MAX_MEMORY=0:14GiB,1:5GiB`` (14 GiB on GPU 0,
5 GiB spill on GPU 1). Visible GPUs are reserved exclusively for the run
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

from training.utils.env import apply_hf_hub_endpoint, env_str, load_project_env

_env_path = load_project_env(_pkg_root)
if _env_path:
    print(f"Loaded environment from {_env_path}")

# ASR-only defaults: train Whisper/adapter on cuda:0; shard Qwen across cuda:0 (14GiB) + cuda:1 (5GiB).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ["DEVICE"] = "cuda:0"
os.environ["LLM_DEVICE"] = "cuda:0"
os.environ["LLM_MAX_MEMORY"] = "0:14GiB,1:5GiB"
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
from training.utils.asr_validation import validate_asr_only
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

CHECKPOINT_BASENAME = env_str("CHECKPOINT_BASENAME", "adapter_asr_only") or "adapter_asr_only"
SAVE_PATH = os.path.join(CKPT.dir, f"{CHECKPOINT_BASENAME}.pt")
SAVE_EVERY_STEPS = CKPT.save_every_steps if CKPT.save_every_steps is not None else 500


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


def _encode_utterance_tokens(
    *,
    adapter: StreamingAdapter,
    audio_extractor: WhisperWindowFeatureExtractor,
    audio_path: str,
    train_device: torch.device,
    max_windows_per_utt: int | None,
) -> torch.Tensor | None:
    """window → encoder → adapter for one utterance."""
    wave = load_mono_waveform_16k(audio_path)
    windows = audio_extractor.waveform_to_windows(wave)
    if max_windows_per_utt is not None:
        windows = windows[:max_windows_per_utt]
    if not windows:
        return None

    adapter.reset_streaming_state()
    utterance_tokens: list[torch.Tensor] = []
    for window in windows:
        result = adapter.forward_window(window.to(device=train_device, dtype=TORCH_DTYPE))
        utterance_tokens.append(result["tokens"])
    return torch.cat(utterance_tokens, dim=1)


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
        print("Pipeline: window → encoder → adapter → LLM (ASR loss only)")

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

    if ctx.is_main:
        print("Adapter: random init (from scratch, no Stage 1 alignment checkpoint)")

    resume_ckpt = None
    resume_path = CKPT.resume_checkpoint or SAVE_PATH
    if os.path.exists(resume_path) and ctx.is_main:
        resume_ckpt = torch.load(resume_path, map_location=train_device)
        print(f"Will resume from checkpoint: {resume_path}")

    if ctx.world_size > 1:
        adapter = DDP(adapter, device_ids=[ctx.local_rank])

    if ctx.is_main:
        print(
            f"Models initialized:\n"
            f"  Encoder: frozen Whisper ({WHISPER_MODEL})\n"
            f"  Adapter: trainable from scratch (fixed 2 tokens/window, no rate controller)\n"
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
    if ctx.is_main and resume_ckpt is not None:
        _unwrap(adapter).load_state_dict(
            filter_adapter_state_dict(
                resume_ckpt["adapter_state_dict"],
                num_queries=_unwrap(adapter).num_queries,
                use_rate_controller=False,
            )
        )
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
        if STAGE.val_enabled and os.path.isdir(VAL_ROOT):
            cap = STAGE.val_max_utterances
            cap_str = "all" if cap is None else str(cap)
            print(
                f"  Validation: dev-clean ({VAL_ROOT}), every {STAGE.val_every_steps} steps, "
                f"{cap_str} utterances, decode + WER/BLEU-4"
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

        m_loss = RunningMean()

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

            audio_tokens_list: list[torch.Tensor] = []
            for p in audio_paths:
                tokens = _encode_utterance_tokens(
                    adapter=adapter_module,
                    audio_extractor=audio,
                    audio_path=p,
                    train_device=train_device,
                    max_windows_per_utt=DATA.max_windows_per_utt,
                )
                if tokens is not None:
                    audio_tokens_list.append(tokens)

            if not audio_tokens_list:
                if ctx.is_main:
                    print(f"[WARN] Step {step}: empty batch (no audio windows); skipping.")
                continue

            audio_tokens = _pad_audio_tokens(audio_tokens_list)
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

            m_loss.update(asr_loss.item())

            accum_suffix = ""
            if grad_accum_steps > 1:
                accum_done = grad_accum_steps if optimizer_stepped else pipeline.accum_step
                accum_suffix = f" | accum {accum_done}/{grad_accum_steps}"

            if ctx.is_main:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"Micro {step:4d}/{len(dataloader)} | opt {pipeline.global_step:5d} | "
                    f"ASR: {asr_loss.item():.4f} | LR: {current_lr:.2e}{accum_suffix}"
                )

            if ctx.is_main and optimizer_stepped:
                logger.log(
                    {"train/asr": asr_loss.item(), "train/lr": current_lr},
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
                    encode_utterance_tokens_fn=_encode_utterance_tokens,
                    build_inputs_for_asr_fn=_build_inputs_for_asr,
                    asr_forward_loss_fn=lambda **kwargs: _asr_forward_loss(llm_model, **kwargs),
                    val_root=VAL_ROOT,
                    train_device=train_device,
                    llm_device=llm_device,
                    max_utterances=STAGE.val_max_utterances,
                    max_windows_per_utt=DATA.max_windows_per_utt,
                    max_text_tokens=STAGE.max_text_tokens,
                    asr_micro_batch_size=STAGE.asr_micro_batch_size,
                    generation=val_generation,
                    global_step=pipeline.global_step,
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
                    metrics={"loss": m_loss.mean, "asr": m_loss.mean},
                )

            if llm_device.type == "cuda":
                with torch.cuda.device(llm_device):
                    torch.cuda.empty_cache()

        if ctx.is_main:
            epoch_metrics = {"loss": m_loss.mean, "asr": m_loss.mean}
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
            print(f"\nEpoch {epoch + 1} complete: ASR={m_loss.mean:.4f}\nCheckpoint -> {SAVE_PATH}\n")

    if ctx.is_main:
        print("ASR-only training complete!")
        logger.finish()
    cleanup_distributed()


if __name__ == "__main__":
    train()
