"""
Stage 1 Training: Audio-Text Alignment (Softmax InfoNCE)

Trains the streaming adapter using softmax InfoNCE contrastive learning to align
audio tokens with text embeddings from the frozen LLM.

Loss: L = L_align (InfoNCE) + λ_stability · L_stability
"""

import os
import sys
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

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
from training.utils.logging import WandbLogger
from training.utils.losses import (
    contrastive_infonce_loss_learnable_temperature,
    init_contrastive_logit_scale,
)
from training.utils.metrics import RunningMean
from training.utils.loaders import load_frozen_qwen_embeddings
from training.utils.stage1_validation import validate_stage1_contrastive, wandb_val_log_dict
from training.utils.devices import (
    ensure_device_ready,
    init_training_context,
    resolve_llm_load_plan,
)

DEVICE_CFG = DeviceConfig.from_env()
TORCH_DTYPE = torch.float32 if DEVICE_CFG.device == "cpu" or not torch.cuda.is_available() else torch.bfloat16


def _maybe_autocast(device: str):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


WHISPER_DIM = 768
LLM_DIM = 4096
WHISPER_MODEL = "openai/whisper-small"
LLM_MODEL_ID = "Qwen/Qwen3-8B"

STAGE = Stage1Config()
OPT = OptimConfig(lr=1e-4, weight_decay=0.01, grad_clip_norm=0.5, warmup_steps=1000)
_TRAINING_DIR = os.path.dirname(__file__)
DATASET_ROOTS = LibriSpeechConfig.resolve_train_roots(
    _TRAINING_DIR,
    env_override=env_str("DATASET_ROOT"),
)
VAL_ROOT = LibriSpeechConfig.dev_clean_root(_TRAINING_DIR)
DATA = DataConfig.from_env(default_dataset_root=DATASET_ROOTS[0])
CKPT = CheckpointConfig(dir="checkpoints", save_every_steps=500)
WANDB = WandbConfig(
    enabled=True,
    project="audio-streaming-adapter",
    run_name="stage1-softmax-infonce",
)

CHECKPOINT_BASENAME = "adapter_softmax_infonce_stage1"
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


def _trainable_params(
    adapter: torch.nn.Module,
    logit_scale: torch.nn.Parameter,
) -> list[torch.nn.Parameter]:
    return [*adapter.parameters(), logit_scale]


def _make_stage1_checkpoint(
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
        stage=1,
        epoch=epoch,
        global_step=global_step,
        adapter_state_dict=adapter.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),
        metrics=metrics,
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
) -> None:
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
    )
    save_checkpoint(SAVE_PATH, ckpt)
    step_path = os.path.join(CKPT.dir, f"{CHECKPOINT_BASENAME}_step{global_step}.pt")
    save_checkpoint(step_path, ckpt)
    print(f"  Checkpoint (step {global_step}) -> {SAVE_PATH}\n              step file -> {step_path}")


def train():
    ctx = init_training_context()
    train_device = ctx.device
    train_device_str = str(train_device)
    llm_device_str, qwen_map, qwen_max_memory = resolve_llm_load_plan(
        ctx=ctx,
        train_device=train_device,
        llm_device=DEVICE_CFG.llm_device,
        llm_max_memory=DEVICE_CFG.llm_max_memory,
    )
    ensure_device_ready(train_device)
    llm_load_device = torch.device(llm_device_str)
    if llm_load_device.type == "cuda":
        ensure_device_ready(llm_load_device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Using device: {train_device_str} ({TORCH_DTYPE})")
    print(f"LLM device: {llm_device_str}")
    print(f"Visible CUDA devices: {ctx.num_cuda_devices}")
    if ctx.num_cuda_devices >= 2:
        print("Stage 1 single-GPU mode: Whisper/adapter/Qwen pinned to primary GPU; sibling GPU left free.")
    if qwen_map:
        print(f"Qwen device_map: {qwen_map!r}")
    if qwen_max_memory:
        print(f"Qwen max_memory: {qwen_max_memory!r}")

    audio = WhisperWindowFeatureExtractor(model_id=WHISPER_MODEL, device=train_device_str, torch_dtype=TORCH_DTYPE)

    qwen_models = load_frozen_qwen_embeddings(
        model_id=LLM_MODEL_ID,
        device=llm_device_str,
        torch_dtype=TORCH_DTYPE,
        device_map=qwen_map,
        max_memory=qwen_max_memory,
    )
    llm_tokenizer = qwen_models.tokenizer
    text_embedder = qwen_models.embedder

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
    logit_scale = init_contrastive_logit_scale(STAGE.temperature, device=train_device)

    print("StreamingAdapter initialized:")
    print(f"  Encoder dim: {WHISPER_DIM}")
    print(f"  LLM dim: {LLM_DIM}")
    print(f"  Max tokens/window: {adapter.num_queries}\n")

    dataset = LibriSpeechPairs(DATASET_ROOTS)
    dataloader = DataLoader(
        dataset,
        batch_size=DATA.batch_size,
        shuffle=True,
        num_workers=DATA.num_workers,
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
        T_max=total_steps - OPT.warmup_steps,
        eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[OPT.warmup_steps],
    )

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    logger = WandbLogger(
        enabled=WANDB.enabled,
        project=WANDB.project,
        entity=WANDB.entity,
        run_name=WANDB.run_name,
        config={
            "stage": 1,
            "loss": "softmax_infonce",
            "data": {**DATA.__dict__, "dataset_roots": DATASET_ROOTS, "val_root": VAL_ROOT},
            "optim": OPT.__dict__,
            "stage_cfg": STAGE.__dict__,
        },
    )

    start_epoch = 0
    global_step = 0
    if os.path.exists(SAVE_PATH):
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
        print(
            f"  Resumed at epoch {start_epoch + 1}/{STAGE.epochs} "
            f"(checkpoint epoch={ckpt['epoch']}), global_step {global_step}\n"
        )

    print("Starting Stage 1 training: Audio-Text Alignment (softmax InfoNCE)")
    print(f"  Dataset splits: {', '.join(os.path.basename(r) for r in DATASET_ROOTS)}")
    if STAGE.val_enabled:
        if os.path.isdir(VAL_ROOT):
            cap = STAGE.val_max_utterances
            cap_str = "all" if cap is None else str(cap)
            print(
                f"  Validation: dev-clean ({VAL_ROOT}), every {STAGE.val_every_steps} steps, "
                f"max {cap_str} utterances/run"
            )
        else:
            print(f"  Validation: dev-clean not found at {VAL_ROOT} (will skip until present)")
    if CKPT.save_every_steps:
        print(
            f"  Checkpoints: every {CKPT.save_every_steps} steps -> "
            f"{SAVE_PATH} + {CHECKPOINT_BASENAME}_step<N>.pt"
        )
    print(f"  Epochs: {STAGE.epochs}")
    print(f"  Batch size: {DATA.batch_size}")
    print(f"  Learning rate: {OPT.lr}")
    print(f"  λ_stability: {STAGE.lambda_stability}")
    print(f"  Temperature (initial): {logit_scale.exp().item():.4f}\n")

    if start_epoch >= STAGE.epochs:
        print(
            f"Checkpoint is at epoch {start_epoch + 1}, but only {STAGE.epochs} "
            f"epoch(s) configured. Increase STAGE.epochs to continue training."
        )
        logger.finish()
        return

    for epoch in range(start_epoch, STAGE.epochs):
        m_total = RunningMean()
        m_align = RunningMean()
        m_stab = RunningMean()

        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{STAGE.epochs}")
        print(f"{'=' * 60}\n")

        for step, batch in enumerate(dataloader):
            audio_paths, transcriptions = batch
            batch_texts = list(transcriptions)

            text_tokens = llm_tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(train_device)

            with torch.no_grad():
                label_embeds = text_embedder(text_tokens.input_ids).float()

            if torch.isnan(label_embeds).any():
                raise ValueError(f"label_embeds contains NaN at step {step}")

            with _maybe_autocast(train_device_str):
                utterances = []
                stab_sum = torch.zeros((), device=train_device, dtype=torch.float32)
                for p in audio_paths:
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
                    raise ValueError(f"audio_tokens contains NaN at step {step}")

                align_loss, diag = contrastive_infonce_loss_learnable_temperature(
                    audio_tokens=audio_tokens,
                    text_embeddings=label_embeds,
                    logit_scale=logit_scale,
                    return_diagnostics=True,
                )
                stability_loss = stab_sum / float(len(audio_paths))

            if torch.isnan(stability_loss):
                raise ValueError(f"stability_loss contains NaN at step {step}")

            total_loss = align_loss + STAGE.lambda_stability * stability_loss
            if torch.isnan(total_loss):
                raise ValueError(f"total_loss contains NaN at step {step}")

            optimizer.zero_grad()
            total_loss.backward()

            params = _trainable_params(adapter, logit_scale)
            if any(torch.isnan(p.grad).any() for p in params if p.grad is not None):
                raise ValueError(f"Gradients contain NaN at step {step}")

            for param in params:
                if param.grad is not None:
                    param.grad = torch.where(
                        torch.isnan(param.grad) | torch.isinf(param.grad),
                        torch.zeros_like(param.grad),
                        param.grad,
                    )

            torch.nn.utils.clip_grad_norm_(params, OPT.grad_clip_norm)
            optimizer.step()
            scheduler.step()

            total_norm = sum(
                param.grad.data.norm(2).item() ** 2 for param in params if param.grad is not None
            ) ** 0.5
            if total_norm > 5.0:
                print(f"[WARN] Step {step}: Large gradient norm {total_norm:.2f}")

            m_total.update(total_loss.item())
            m_align.update(align_loss.item())
            m_stab.update(stability_loss.item())
            global_step += 1

            if step % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"Step {step:4d}/{len(dataloader)} | "
                    f"Loss: {total_loss.item():.4f} | "
                    f"Align: {align_loss.item():.4f} | "
                    f"Stab: {stability_loss.item():.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Temperature: {diag['logit_scale']:.4f}"
                )
                logger.log(
                    {
                        "train/loss": total_loss.item(),
                        "train/align": align_loss.item(),
                        "train/stability": stability_loss.item(),
                        "train/lr": current_lr,
                        "train/grad_norm": total_norm,
                        "train/temperature": diag["logit_scale"],
                        "diag/pos_sim": diag["pos_sim"],
                        "diag/neg_sim": diag["neg_sim"],
                        "diag/pos_minus_neg": diag["pos_minus_neg"],
                        "diag/audio_std": diag["audio_std"],
                        "diag/text_std": diag["text_std"],
                    },
                    step=global_step,
                )

            step_metrics: dict[str, float] = {
                "loss": m_total.mean,
                "align": m_align.mean,
                "stability": m_stab.mean,
                "temperature": logit_scale.exp().item(),
            }

            if (
                STAGE.val_enabled
                and global_step > 0
                and global_step % STAGE.val_every_steps == 0
            ):
                if os.path.isdir(VAL_ROOT):
                    print(f"\n  [val step {global_step}]")
                    val_metrics = validate_stage1_contrastive(
                        adapter=adapter,
                        audio_extractor=audio,
                        llm_tokenizer=llm_tokenizer,
                        text_embedder=text_embedder,
                        val_root=VAL_ROOT,
                        device=train_device_str,
                        batch_size=DATA.batch_size,
                        num_workers=DATA.num_workers,
                        logit_scale=logit_scale,
                        lambda_stability=STAGE.lambda_stability,
                        max_utterances=STAGE.val_max_utterances,
                        pad_tokens_fn=_pad_tokens,
                        maybe_autocast_fn=_maybe_autocast,
                        epoch=epoch,
                    )
                    logger.log(wandb_val_log_dict(val_metrics), step=global_step)
                else:
                    print(f"  [WARN] Skipping validation — dev-clean not found at {VAL_ROOT}")

            _maybe_save_step_checkpoint(
                epoch=epoch,
                global_step=global_step,
                adapter=adapter,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=step_metrics,
                logit_scale=logit_scale,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"\nEpoch {epoch + 1} complete:")
        print(f"  Train Loss: {m_total.mean:.4f}")
        print(f"  Train Align: {m_align.mean:.4f}")
        print(f"  Train Stability: {m_stab.mean:.4f}\n")

    final_metrics: dict[str, float] = {
        "loss": m_total.mean,
        "align": m_align.mean,
        "stability": m_stab.mean,
        "temperature": logit_scale.exp().item(),
    }
    save_checkpoint(
        SAVE_PATH,
        _make_stage1_checkpoint(
            epoch=STAGE.epochs,
            global_step=global_step,
            adapter=adapter,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=final_metrics,
            logit_scale=logit_scale,
        ),
    )
    print(f"Final checkpoint saved to {SAVE_PATH}")
    print("Stage 1 training complete!")
    logger.finish()


if __name__ == "__main__":
    train()
