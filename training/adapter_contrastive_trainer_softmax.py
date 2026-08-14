"""
Stage 1 Training: Audio-Text Alignment (Softmax InfoNCE)

Trains the streaming adapter using softmax InfoNCE contrastive learning to align
audio tokens with text embeddings from frozen Vicuna-7B (``lmsys/vicuna-7b-v1.5``).

Loss: L = L_align (InfoNCE) + λ_stability · L_stability

Large-batch InfoNCE uses GradCache: micro-batches fit in VRAM (``BATCH_SIZE``,
typically 8) while InfoNCE runs on the full macro-batch (``MACRO_BATCH_SIZE``,
default 128) so in-batch negatives match the effective contrastive batch.

Multi-GPU (recommended)::

    uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \\
        training/adapter_contrastive_trainer_softmax.py

Each rank encodes ``MACRO_BATCH_SIZE / world_size`` utterances; pools are
all-gathered so InfoNCE still sees the full macro-batch. Single-process with two
visible GPUs places Whisper/adapter on ``cuda:0`` and text embeddings on ``cuda:1``.

Reproducibility: ``RANDOM_SEED`` (default 42).
Override Vicuna id with ``VICUNA_MODEL_ID`` (default ``lmsys/vicuna-7b-v1.5``).
"""

import os
import sys
from contextlib import nullcontext
from dataclasses import replace

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_root = os.path.join(_pkg_root, "src")
sys.path.insert(0, _src_root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, _pkg_root)

from training.utils.cuda_memory_reserve import CudaVramFence
from training.utils.env import apply_hf_hub_endpoint, env_int, env_str, load_project_env

_env_path = load_project_env(_pkg_root)
if _env_path:
    print(f"Loaded environment from {_env_path}")

# Prefer both free GPUs. Under torchrun, each rank binds to LOCAL_RANK.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
os.environ.pop("LLM_MAX_MEMORY", None)
if "LOCAL_RANK" not in os.environ:
    # Single-process: adapter/Whisper on GPU 0; text embedder on GPU 1 when visible.
    os.environ["DEVICE"] = "cuda:0"
    _visible = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    os.environ.setdefault("LLM_DEVICE", "cuda:1" if len(_visible) >= 2 else "cuda:0")

_hf_endpoint = apply_hf_hub_endpoint(_pkg_root)
print(f"HF Hub endpoint: {_hf_endpoint}")

_hf_token = env_str("HF_TOKEN")
if _hf_token:
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", _hf_token)

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.adapter.streaming_adapter import StreamingAdapter
from src.dataset import LibriSpeechConfig, load_mono_waveform_16k, LibriSpeechPairs
from src.encoder.waveform_window_encoder import WhisperWindowFeatureExtractor
from training.utils.checkpointing import TrainingCheckpoint, save_checkpoint
from training.utils.config import (
    CheckpointConfig,
    DataConfig,
    OptimConfig,
    Stage1Config,
    DeviceConfig,
    WandbConfig,
)
from training.utils.grad_cache import (
    all_reduce_mean_grads,
    grad_cache_infonce_backward,
    pool_audio_tokens_for_infonce,
)
from training.utils.logging import WandbLogger
from training.utils.losses import init_contrastive_logit_scale
from training.utils.metrics import RunningMean
from training.utils.loaders import load_frozen_vicuna_embeddings
from training.utils.reproducibility import make_torch_generator, seed_worker, set_seed
from training.utils.stage1_validation import validate_stage1_contrastive, wandb_val_log_dict
from training.utils.devices import (
    cleanup_distributed,
    ensure_device_ready,
    init_training_context,
    llm_input_device,
    resolve_llm_load_plan,
)

DEVICE_CFG = DeviceConfig.from_env()
TORCH_DTYPE = torch.float32 if DEVICE_CFG.device == "cpu" or not torch.cuda.is_available() else torch.bfloat16
RANDOM_SEED = env_int("RANDOM_SEED", 42)


def _maybe_autocast(device: str):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


WHISPER_DIM = 768
LLM_DIM = 4096  # Vicuna-7B hidden size
WHISPER_MODEL = "openai/whisper-small"
# Dedicated env so .env LLM_MODEL_ID=Qwen/... does not override this trainer.
LLM_MODEL_ID = env_str("VICUNA_MODEL_ID", "lmsys/vicuna-7b-v1.5") or "lmsys/vicuna-7b-v1.5"

STAGE = Stage1Config.from_env()
# Grad clip off for now so Align can move (was 2.0; grad_norm was pinned at 0.5 earlier).
OPT = OptimConfig(lr=1e-4, weight_decay=0.01, grad_clip_norm=0.0, warmup_steps=1000)
_TRAINING_DIR = os.path.dirname(__file__)
DATASET_ROOTS = LibriSpeechConfig.resolve_train_roots(
    _TRAINING_DIR,
    env_override=env_str("DATASET_ROOT"),
)
VAL_ROOT = LibriSpeechConfig.dev_clean_root(_TRAINING_DIR)
DATA = DataConfig.from_env(default_dataset_root=DATASET_ROOTS[0])
# GradCache defaults for this trainer: micro-batch 8 (VRAM), InfoNCE macro-batch 128.
_stage1_overrides: dict = {}
if not env_str("BATCH_SIZE"):
    _stage1_overrides["batch_size"] = 8
if DATA.macro_batch_size is None:
    _stage1_overrides["macro_batch_size"] = 128
if _stage1_overrides:
    DATA = replace(DATA, **_stage1_overrides)
# Prefer Align until contrastive signal moves; override only when env unset.
if not env_str("LAMBDA_STABILITY"):
    STAGE = replace(STAGE, lambda_stability=0.01)
MICRO_BATCH_SIZE = DATA.batch_size
MACRO_BATCH_SIZE = int(DATA.macro_batch_size)
if MACRO_BATCH_SIZE % MICRO_BATCH_SIZE != 0:
    raise ValueError(
        f"MACRO_BATCH_SIZE ({MACRO_BATCH_SIZE}) must be a multiple of "
        f"BATCH_SIZE ({MICRO_BATCH_SIZE})"
    )
CKPT = CheckpointConfig.from_env(pkg_root=_pkg_root)
WANDB = WandbConfig(
    enabled=True,
    project="audio-streaming-adapter",
    run_name="stage1-softmax-infonce-vicuna7b",
)

CHECKPOINT_BASENAME = env_str("CHECKPOINT_BASENAME", "adapter_softmax_infonce_vicuna_stage1") or (
    "adapter_softmax_infonce_vicuna_stage1"
)
SAVE_PATH = os.path.join(CKPT.dir, f"{CHECKPOINT_BASENAME}.pt")


def _pad_tokens(utterances: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(t.shape[1] for t in utterances)
    padded = []
    for t in utterances:
        if t.shape[1] < max_len:
            pad = torch.zeros(1, max_len - t.shape[1], t.shape[2], device=t.device, dtype=t.dtype)
            t = torch.cat([t, pad], dim=1)
        padded.append(t)
    return torch.cat(padded, dim=0)


def _chunk_paths(paths: list[str], micro_batch_size: int) -> list[list[str]]:
    return [paths[i : i + micro_batch_size] for i in range(0, len(paths), micro_batch_size)]


def _trainable_params(
    adapter: torch.nn.Module,
    logit_scale: torch.nn.Parameter,
) -> list[torch.nn.Parameter]:
    return [*adapter.parameters(), logit_scale]


def _stage1_hyperparams(*, world_size: int, effective_seed: int) -> dict:
    return {
        "random_seed": RANDOM_SEED,
        "effective_seed": effective_seed,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "macro_batch_size": MACRO_BATCH_SIZE,
        "world_size": world_size,
        "grad_cache": True,
        "loss": "softmax_infonce",
    }


def _make_stage1_checkpoint(
    *,
    epoch: int,
    global_step: int,
    adapter: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    metrics: dict[str, float],
    logit_scale: torch.nn.Parameter,
    hyperparams: dict,
) -> TrainingCheckpoint:
    return TrainingCheckpoint(
        stage=1,
        epoch=epoch,
        global_step=global_step,
        adapter_state_dict=adapter.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),
        metrics=metrics,
        hyperparams=hyperparams,
        contrastive_logit_scale=float(logit_scale.item()),
    )


def _maybe_save_step_checkpoint(
    *,
    epoch: int,
    global_step: int,
    adapter: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    metrics: dict[str, float],
    logit_scale: torch.nn.Parameter,
    hyperparams: dict,
    is_main: bool,
) -> None:
    if not is_main:
        return
    interval = CKPT.save_every_steps
    if interval is None or interval <= 0:
        return
    if global_step <= 0 or global_step % interval != 0:
        return
    ckpt = _make_stage1_checkpoint(
        epoch=epoch + 1,
        global_step=global_step,
        adapter=adapter,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        logit_scale=logit_scale,
        hyperparams=hyperparams,
    )
    save_checkpoint(SAVE_PATH, ckpt)
    step_path = os.path.join(CKPT.dir, f"{CHECKPOINT_BASENAME}_step{global_step}.pt")
    save_checkpoint(step_path, ckpt)
    print(f"  Checkpoint (step {global_step}) -> {SAVE_PATH}\n              step file -> {step_path}")


def _save_epoch_checkpoint(
    *,
    epoch: int,
    global_step: int,
    adapter: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    metrics: dict[str, float],
    logit_scale: torch.nn.Parameter,
    hyperparams: dict,
    is_main: bool,
) -> None:
    if not is_main:
        return
    interval = CKPT.save_every_epochs
    if interval is None or interval <= 0:
        return
    epoch_num = epoch + 1
    if epoch_num % interval != 0:
        return
    ckpt = _make_stage1_checkpoint(
        epoch=epoch_num,
        global_step=global_step,
        adapter=adapter,
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=metrics,
        logit_scale=logit_scale,
        hyperparams=hyperparams,
    )
    save_checkpoint(SAVE_PATH, ckpt)
    epoch_path = os.path.join(CKPT.dir, f"{CHECKPOINT_BASENAME}_epoch{epoch_num}.pt")
    save_checkpoint(epoch_path, ckpt)
    print(
        f"  Checkpoint (epoch {epoch_num}) -> {SAVE_PATH}\n"
        f"              epoch file -> {epoch_path}"
    )


def _run_stage1_validation(
    *,
    label: str,
    epoch: int,
    global_step: int,
    adapter: torch.nn.Module,
    audio: WhisperWindowFeatureExtractor,
    llm_tokenizer: object,
    text_embedder: torch.nn.Module,
    logit_scale: torch.nn.Parameter,
    train_device_str: str,
    logger: WandbLogger,
) -> None:
    if not os.path.isdir(VAL_ROOT):
        print(f"  [WARN] Skipping validation ({label}) — dev-clean not found at {VAL_ROOT}")
        return
    print(f"\n  [val {label}]")
    val_metrics = validate_stage1_contrastive(
        adapter=adapter,
        audio_extractor=audio,
        llm_tokenizer=llm_tokenizer,
        text_embedder=text_embedder,
        val_root=VAL_ROOT,
        device=train_device_str,
        batch_size=MICRO_BATCH_SIZE,
        num_workers=DATA.num_workers,
        logit_scale=logit_scale,
        lambda_stability=STAGE.lambda_stability,
        max_utterances=STAGE.val_max_utterances,
        pad_tokens_fn=_pad_tokens,
        maybe_autocast_fn=_maybe_autocast,
        epoch=epoch,
    )
    logger.log(wandb_val_log_dict(val_metrics), step=global_step)


def train():
    ctx = init_training_context()
    train_device = ctx.device
    train_device_str = str(train_device)
    effective_seed = set_seed(RANDOM_SEED, rank=ctx.rank)
    hyperparams = _stage1_hyperparams(world_size=ctx.world_size, effective_seed=effective_seed)

    if MACRO_BATCH_SIZE % ctx.world_size != 0:
        raise ValueError(
            f"MACRO_BATCH_SIZE ({MACRO_BATCH_SIZE}) must be divisible by "
            f"world_size ({ctx.world_size})"
        )
    local_macro_batch = MACRO_BATCH_SIZE // ctx.world_size
    if local_macro_batch % MICRO_BATCH_SIZE != 0:
        raise ValueError(
            f"Per-rank macro batch ({local_macro_batch}) must be a multiple of "
            f"BATCH_SIZE ({MICRO_BATCH_SIZE})"
        )
    micro_steps_per_local_macro = local_macro_batch // MICRO_BATCH_SIZE

    llm_device_str, qwen_map, qwen_max_memory = resolve_llm_load_plan(
        ctx=ctx,
        train_device=train_device,
        llm_device=DEVICE_CFG.llm_device,
        llm_max_memory=DEVICE_CFG.llm_max_memory,
    )
    # DDP: keep text embedder on the same GPU as the adapter (no cross-rank HF map).
    if ctx.world_size > 1:
        llm_device_str = train_device_str
        qwen_map = None
        qwen_max_memory = None

    ensure_device_ready(train_device)
    llm_load_device = torch.device(llm_device_str)
    if llm_load_device.type == "cuda":
        ensure_device_ready(llm_load_device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if ctx.is_main:
        print(f"Using device: {train_device_str} ({TORCH_DTYPE})")
        print(f"LLM device: {llm_device_str}")
        print(f"Visible CUDA devices: {ctx.num_cuda_devices}")
        print(f"World size: {ctx.world_size} | random_seed={RANDOM_SEED} (effective={effective_seed})")
        if ctx.world_size > 1:
            print(
                f"DDP GradCache: local macro {local_macro_batch}/rank, "
                f"global InfoNCE batch {MACRO_BATCH_SIZE}"
            )
        elif ctx.num_cuda_devices >= 2:
            print(
                "Single-process 2-GPU: Whisper/adapter on "
                f"{train_device_str}, text embeddings on {llm_device_str}"
            )
        if qwen_map:
            print(f"LLM device_map: {qwen_map!r}")
        if qwen_max_memory:
            print(f"LLM max_memory: {qwen_max_memory!r}")
        print(f"LLM model: {LLM_MODEL_ID}")

    audio = WhisperWindowFeatureExtractor(model_id=WHISPER_MODEL, device=train_device_str, torch_dtype=TORCH_DTYPE)

    llm_models = load_frozen_vicuna_embeddings(
        model_id=LLM_MODEL_ID,
        device=llm_device_str,
        torch_dtype=TORCH_DTYPE,
        device_map=qwen_map,
        max_memory=qwen_max_memory,
    )
    llm_tokenizer = llm_models.tokenizer
    text_embedder = llm_models.embedder
    llm_dim = int(getattr(llm_models, "hidden_size", LLM_DIM) or LLM_DIM)
    llm_embed_device = llm_input_device(text_embedder)
    if ctx.is_main:
        print(f"Text embedder device: {llm_embed_device}")
        print(f"LLM hidden size: {llm_dim}")

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
    logit_scale = init_contrastive_logit_scale(STAGE.temperature, device=train_device)

    if ctx.is_main:
        print("StreamingAdapter initialized:")
        print(f"  Encoder dim: {WHISPER_DIM}")
        print(f"  LLM dim: {llm_dim}")
        print(f"  Max tokens/window: {adapter.num_queries}\n")

    # Soft VRAM fence: hold free memory between steps so neighbors cannot grab dips.
    # Enable with CUDA_MEM_FENCE=true (optional CUDA_MEM_LEAVE_FREE_GB, default 1.5).
    vram_fence = CudaVramFence.from_env(train_device)
    vram_fence.acquire()

    dataset = LibriSpeechPairs(DATASET_ROOTS)
    sampler: DistributedSampler | None = None
    if ctx.world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=ctx.world_size,
            rank=ctx.rank,
            shuffle=True,
            seed=RANDOM_SEED,
            drop_last=True,
        )
    # Per-rank batch is the local share of the global InfoNCE macro-batch.
    dataloader = DataLoader(
        dataset,
        batch_size=local_macro_batch,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=DATA.num_workers,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(effective_seed),
    )

    optimizer = torch.optim.AdamW(
        _trainable_params(adapter, logit_scale),
        lr=OPT.lr,
        weight_decay=OPT.weight_decay,
    )

    total_steps = STAGE.epochs * len(dataloader)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=OPT.warmup_steps,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - OPT.warmup_steps),
        eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[OPT.warmup_steps],
    )

    if ctx.is_main:
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    logger = WandbLogger(
        enabled=WANDB.enabled and ctx.is_main,
        project=WANDB.project,
        entity=WANDB.entity,
        run_name=WANDB.run_name,
        config={
            "stage": 1,
            "loss": "softmax_infonce",
            "grad_cache": True,
            "random_seed": RANDOM_SEED,
            "effective_seed": effective_seed,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "macro_batch_size": MACRO_BATCH_SIZE,
            "local_macro_batch_size": local_macro_batch,
            "world_size": ctx.world_size,
            "data": {**DATA.__dict__, "dataset_roots": DATASET_ROOTS, "val_root": VAL_ROOT},
            "optim": OPT.__dict__,
            "stage_cfg": STAGE.__dict__,
        },
    )

    start_epoch = 0
    global_step = 0
    if os.path.exists(SAVE_PATH):
        if ctx.is_main:
            print(f"Resuming from checkpoint: {SAVE_PATH}")
        ckpt = torch.load(SAVE_PATH, map_location=train_device)
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        saved_logit_scale = ckpt.get("contrastive_logit_scale")
        if saved_logit_scale is not None:
            logit_scale.data.fill_(float(saved_logit_scale))
        opt_state = ckpt.get("optimizer_state_dict")
        if opt_state is not None:
            try:
                optimizer.load_state_dict(opt_state)
            except ValueError:
                if ctx.is_main:
                    print(
                        "  [WARN] Optimizer state mismatch (e.g. new logit_scale param); "
                        "starting optimizer fresh."
                    )
        start_epoch = max(0, int(ckpt["epoch"]) - 1)
        global_step = ckpt["global_step"]
        for _ in range(global_step):
            scheduler.step()
        for param_group in optimizer.param_groups:
            param_group["lr"] = OPT.lr
        if ctx.is_main:
            print(
                f"  Resumed at epoch {start_epoch + 1}/{STAGE.epochs} "
                f"(checkpoint epoch={ckpt['epoch']}), global_step {global_step}\n"
            )

    if ctx.is_main:
        print("Starting Stage 1 training: Audio-Text Alignment (softmax InfoNCE + GradCache)")
        print(f"  Dataset splits: {', '.join(os.path.basename(r) for r in DATASET_ROOTS)}")
        if STAGE.val_enabled:
            if os.path.isdir(VAL_ROOT):
                cap = STAGE.val_max_utterances
                cap_str = "all" if cap is None else str(cap)
                print(
                    f"  Validation: dev-clean ({VAL_ROOT}), every {STAGE.val_every_steps} steps "
                    f"and every {STAGE.val_every_epochs} epoch(s), "
                    f"max {cap_str} utterances/run"
                )
            else:
                print(f"  Validation: dev-clean not found at {VAL_ROOT} (will skip until present)")
        if CKPT.save_every_steps:
            print(
                f"  Checkpoints: every {CKPT.save_every_steps} steps -> "
                f"{CHECKPOINT_BASENAME}_step<N>.pt (+ latest {SAVE_PATH})"
            )
        if CKPT.save_every_epochs:
            print(
                f"  Checkpoints: every {CKPT.save_every_epochs} epoch(s) -> "
                f"{CHECKPOINT_BASENAME}_epoch<N>.pt (+ latest {SAVE_PATH})"
            )
        print(f"  Checkpoint basename: {CHECKPOINT_BASENAME}")
        print(f"  Epochs: {STAGE.epochs}")
        print(f"  Micro-batch size (VRAM): {MICRO_BATCH_SIZE}")
        print(
            f"  InfoNCE macro-batch: {MACRO_BATCH_SIZE} global "
            f"({local_macro_batch}/rank × {ctx.world_size}, "
            f"{micro_steps_per_local_macro} encode chunks/rank)"
        )
        print(f"  Learning rate: {OPT.lr}")
        print(f"  grad_clip_norm: {OPT.grad_clip_norm}")
        print(f"  λ_stability: {STAGE.lambda_stability}")
        print(f"  Temperature (initial): {logit_scale.exp().item():.4f}")
        print(f"  random_seed: {RANDOM_SEED}\n")

    if start_epoch >= STAGE.epochs:
        if ctx.is_main:
            print(
                f"Checkpoint is at epoch {start_epoch + 1}, but only {STAGE.epochs} "
                f"epoch(s) configured. Increase STAGE.epochs to continue training."
            )
        vram_fence.release(silent=False)
        logger.finish()
        cleanup_distributed()
        return

    def _encode_micro(paths: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        utterances: list[torch.Tensor] = []
        stab_sum = torch.zeros((), device=train_device, dtype=torch.float32)
        with _maybe_autocast(train_device_str):
            for p in paths:
                wave = load_mono_waveform_16k(p)
                windows = audio.waveform_to_windows(wave)
                adapter.reset_streaming_state()
                chunks = []
                for w in windows:
                    out = adapter.forward_window(w)
                    chunks.append(out["tokens"])
                    stab_sum = stab_sum + out["stability_loss"].float()
                utterances.append(torch.cat(chunks, dim=1))
            audio_tokens = _pad_tokens(utterances)
        if torch.isnan(audio_tokens).any():
            raise ValueError("audio_tokens contains NaN during encode_micro")
        return pool_audio_tokens_for_infonce(audio_tokens), stab_sum

    try:
        for epoch in range(start_epoch, STAGE.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            m_total = RunningMean()
            m_align = RunningMean()
            m_stab = RunningMean()

            if ctx.is_main:
                print(f"\n{'=' * 60}")
                print(f"Epoch {epoch + 1}/{STAGE.epochs}")
                print(f"{'=' * 60}\n")

            for step, batch in enumerate(dataloader):
                vram_fence.release()
                audio_paths, transcriptions = batch
                paths = list(audio_paths)
                batch_texts = list(transcriptions)
                micro_chunks = _chunk_paths(paths, MICRO_BATCH_SIZE)

                text_tokens = llm_tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                ).to(llm_embed_device)

                with torch.no_grad():
                    label_embeds = text_embedder(text_tokens.input_ids).float().to(train_device)

                if torch.isnan(label_embeds).any():
                    raise ValueError(f"label_embeds contains NaN at step {step}")

                total_loss, align_loss, stability_loss, diag = grad_cache_infonce_backward(
                    adapter=adapter,
                    encode_micro=_encode_micro,
                    micro_path_chunks=micro_chunks,
                    text_embeddings=label_embeds,
                    logit_scale=logit_scale,
                    lambda_stability=STAGE.lambda_stability,
                    optimizer=optimizer,
                )

                if torch.isnan(stability_loss):
                    raise ValueError(f"stability_loss contains NaN at step {step}")
                if torch.isnan(total_loss):
                    raise ValueError(f"total_loss contains NaN at step {step}")

                params = _trainable_params(adapter, logit_scale)
                all_reduce_mean_grads(params)

                if any(torch.isnan(p.grad).any() for p in params if p.grad is not None):
                    raise ValueError(f"Gradients contain NaN at step {step}")

                for param in params:
                    if param.grad is not None:
                        param.grad = torch.where(
                            torch.isnan(param.grad) | torch.isinf(param.grad),
                            torch.zeros_like(param.grad),
                            param.grad,
                        )

                if OPT.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, OPT.grad_clip_norm)
                optimizer.step()
                scheduler.step()

                total_norm = sum(
                    param.grad.data.norm(2).item() ** 2 for param in params if param.grad is not None
                ) ** 0.5
                if total_norm > 5.0 and ctx.is_main:
                    print(f"[WARN] Step {step}: Large gradient norm {total_norm:.2f}")

                m_total.update(float(total_loss.item()))
                m_align.update(float(align_loss.item()))
                m_stab.update(float(stability_loss.item()))
                global_step += 1

                current_lr = scheduler.get_last_lr()[0]
                if ctx.is_main:
                    # Log every optimizer step to W&B; print every 10 steps to the console.
                    logger.log(
                        {
                            "train/loss": float(total_loss.item()),
                            "train/align": float(align_loss.item()),
                            "train/stability": float(stability_loss.item()),
                            "train/lr": current_lr,
                            "train/grad_norm": total_norm,
                            "train/grad_norm_align": diag["align_grad_norm"],
                            "train/grad_norm_stab": diag["stab_grad_norm"],
                            "train/temperature": diag["logit_scale"],
                            "train/macro_batch_size": MACRO_BATCH_SIZE,
                            "train/micro_batch_size": MICRO_BATCH_SIZE,
                            "train/world_size": ctx.world_size,
                            "train/epoch": epoch + 1,
                            "train/step_in_epoch": step,
                            "diag/pos_sim": diag["pos_sim"],
                            "diag/neg_sim": diag["neg_sim"],
                            "diag/pos_minus_neg": diag["pos_minus_neg"],
                            "diag/audio_std": diag["audio_std"],
                            "diag/text_std": diag["text_std"],
                        },
                        step=global_step,
                    )
                    if step % 10 == 0:
                        print(
                            f"Step {step:4d}/{len(dataloader)} | "
                            f"Loss: {float(total_loss.item()):.4f} | "
                            f"Align: {float(align_loss.item()):.4f} | "
                            f"Stab: {float(stability_loss.item()):.4f} | "
                            f"Δpos-neg: {diag['pos_minus_neg']:+.4f} | "
                            f"gA: {diag['align_grad_norm']:.3f} | "
                            f"gS: {diag['stab_grad_norm']:.3f} | "
                            f"gTot: {total_norm:.3f} | "
                            f"LR: {current_lr:.2e} | "
                            f"T: {diag['logit_scale']:.4f} | "
                            f"InfoNCE-B: {MACRO_BATCH_SIZE}"
                        )

                step_metrics: dict[str, float] = {
                    "loss": m_total.mean,
                    "align": m_align.mean,
                    "stability": m_stab.mean,
                    "temperature": logit_scale.exp().item(),
                }

                if (
                    STAGE.val_enabled
                    and ctx.is_main
                    and global_step > 0
                    and global_step % STAGE.val_every_steps == 0
                ):
                    _run_stage1_validation(
                        label=f"step {global_step}",
                        epoch=epoch,
                        global_step=global_step,
                        adapter=adapter,
                        audio=audio,
                        llm_tokenizer=llm_tokenizer,
                        text_embedder=text_embedder,
                        logit_scale=logit_scale,
                        train_device_str=train_device_str,
                        logger=logger,
                    )

                _maybe_save_step_checkpoint(
                    epoch=epoch,
                    global_step=global_step,
                    adapter=adapter,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metrics=step_metrics,
                    logit_scale=logit_scale,
                    hyperparams=hyperparams,
                    is_main=ctx.is_main,
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Re-hold leftover VRAM so other processes cannot occupy the post-step dip.
                vram_fence.acquire()

            epoch_metrics: dict[str, float] = {
                "loss": m_total.mean,
                "align": m_align.mean,
                "stability": m_stab.mean,
                "temperature": logit_scale.exp().item(),
            }

            if (
                STAGE.val_enabled
                and ctx.is_main
                and STAGE.val_every_epochs > 0
                and (epoch + 1) % STAGE.val_every_epochs == 0
            ):
                _run_stage1_validation(
                    label=f"epoch {epoch + 1}",
                    epoch=epoch,
                    global_step=global_step,
                    adapter=adapter,
                    audio=audio,
                    llm_tokenizer=llm_tokenizer,
                    text_embedder=text_embedder,
                    logit_scale=logit_scale,
                    train_device_str=train_device_str,
                    logger=logger,
                )

            _save_epoch_checkpoint(
                epoch=epoch,
                global_step=global_step,
                adapter=adapter,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=epoch_metrics,
                logit_scale=logit_scale,
                hyperparams=hyperparams,
                is_main=ctx.is_main,
            )

            if ctx.is_main:
                print(f"\nEpoch {epoch + 1} complete:")
                print(f"  Train Loss: {m_total.mean:.4f}")
                print(f"  Train Align: {m_align.mean:.4f}")
                print(f"  Train Stability: {m_stab.mean:.4f}\n")

        vram_fence.release(silent=False)
        if ctx.is_main:
            final_metrics: dict[str, float] = {
                "loss": m_total.mean,
                "align": m_align.mean,
                "stability": m_stab.mean,
                "temperature": logit_scale.exp().item(),
            }
            final_ckpt = _make_stage1_checkpoint(
                epoch=STAGE.epochs,
                global_step=global_step,
                adapter=adapter,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=final_metrics,
                logit_scale=logit_scale,
                hyperparams=hyperparams,
            )
            save_checkpoint(SAVE_PATH, final_ckpt)
            final_path = os.path.join(CKPT.dir, f"{CHECKPOINT_BASENAME}_final.pt")
            save_checkpoint(final_path, final_ckpt)
            print(f"Final checkpoint saved to {SAVE_PATH}")
            print(f"              final file -> {final_path}")
            print("Stage 1 training complete!")
        logger.finish()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    train()
