"""
Stage 2: ASR distillation — frozen Whisper + frozen Qwen, train StreamingAdapter + EarlyCommitGate.

Shared building blocks live in `training.*` and `dataset.*`.

Inference that mirrors this loop (per-window adapter + ``L_gate`` / EarlyCommitGate) should use
:class:`adapter_llm_pipeline.WhisperAdapterLLMCommitGatePipeline`, not
:class:`adapter_llm_pipeline.WhisperAdapterLLMPipeline`.

Two-GPU layout (recommended for Qwen3-8B)::

    CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        uv run training/adapter_asr_trainer.py

Whisper, adapter, and gate on GPU 0; frozen Qwen sharded ~50/50 on both GPUs.
"""

from __future__ import annotations

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

from src.adapter.streaming_adapter import StreamingAdapter
from src.adapter.early_commit_gate import EarlyCommitGate
from src.dataset import LibriSpeechConfig, LibriSpeechPairs, load_mono_waveform_16k
from src.encoder import WhisperWindowFeatureExtractor
from training.utils.checkpointing import TrainingCheckpoint, save_checkpoint
from training.utils.config import (
    CheckpointConfig,
    DataConfig,
    OptimConfig,
    Stage2Config,
    Stage2DeviceConfig,
    WandbConfig,
)
from training.utils.devices import (
    default_train_device,
    init_cuda_for_device_map,
    llm_input_device,
    qwen_balanced_max_memory,
)
from training.utils.logging import WandbLogger
from training.utils.losses import contrastive_infonce_loss
from training.utils.metrics import RunningMean
from training.utils.loaders import default_device_and_dtype, load_frozen_qwen_causal_lm
from training.utils.optimization import TrainingPipeline

_, TORCH_DTYPE = default_device_and_dtype()

WHISPER_DIM = 768
LLM_DIM = 4096
WHISPER_MODEL = "openai/whisper-small"
LLM_MODEL_ID = "Qwen/Qwen3-8B"

STAGE = Stage2Config()
DEV = Stage2DeviceConfig()
OPT = OptimConfig(lr=5e-5, weight_decay=0.01, grad_clip_norm=1.0, warmup_steps=1000)
DATA = DataConfig(
    dataset_root=LibriSpeechConfig.default_train_clean_100_from_training_dir(os.path.dirname(__file__)).root,
    batch_size=2,
    num_workers=2,
    max_windows_per_utt=None,
)
CKPT = CheckpointConfig(dir="checkpoints", save_every_epochs=1)
WANDB = WandbConfig(enabled=True, project="audio-streaming-adapter", run_name="stage2-asr")

SAVE_PATH = os.path.join(CKPT.dir, "adapter_stage2.pt")
STAGE1_SAVE_PATH = os.path.join(CKPT.dir, "adapter_stage1.pt")

USE_RATE_CONTROLLER = STAGE.use_rate_controller
RATE_TARGET = STAGE.rate_target


def _maybe_autocast():
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _stage2_devices(num_gpus: int) -> tuple[str, str, str | dict | None, dict[int, str] | None]:
    """Whisper/adapter device and Qwen ``device_map`` / ``max_memory`` for the GPU count."""
    if not torch.cuda.is_available():
        cpu = default_train_device()
        return cpu, cpu, None, None
    train_device = DEV.train_device
    whisper_device = DEV.whisper_device
    if num_gpus >= 2:
        llm_device_map: str | dict | None = DEV.llm_device_map or "auto"
        llm_max_memory = qwen_balanced_max_memory(reserve_on_gpu0_gib=DEV.reserve_train_gpu_gib)
    else:
        whisper_device = train_device
        llm_device_map = None
        llm_max_memory = None
    return whisper_device, train_device, llm_device_map, llm_max_memory


def train() -> None:
    num_gpus = init_cuda_for_device_map()
    whisper_device, train_device, llm_device_map, llm_max_memory = _stage2_devices(num_gpus)

    torch.cuda.empty_cache()
    print(f"Visible CUDA devices: {num_gpus}")
    print(f"Whisper device: {whisper_device}")
    print(f"Train device (adapter / gate): {train_device}")
    print(f"Qwen device_map: {llm_device_map!r}")
    if llm_max_memory is not None:
        print(f"Qwen max_memory (balanced ~50/50): {llm_max_memory}")

    audio = WhisperWindowFeatureExtractor(
        model_id=WHISPER_MODEL, device=whisper_device, torch_dtype=TORCH_DTYPE
    )

    qwen_models = load_frozen_qwen_causal_lm(
        model_id=LLM_MODEL_ID,
        device=train_device,
        torch_dtype=TORCH_DTYPE,
        device_map=llm_device_map,
        max_memory=llm_max_memory,
    )
    llm_tokenizer = qwen_models.tokenizer
    llm_model = qwen_models.causal_lm
    text_embedder = qwen_models.embedder
    llm_device = llm_input_device(llm_model)
    print(f"Qwen input embeddings device: {llm_device}")

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
    print(f"Loaded Stage 1 adapter weights from {STAGE1_SAVE_PATH} (epoch {stage1_ckpt.get('epoch', '?')})")

    gate = EarlyCommitGate(
        d_llm=LLM_DIM,
        hidden_dim=256,
        threshold=0.5,
        latency_weight=0.1,
    ).to(train_device, dtype=TORCH_DTYPE)
    gate.train()

    print(
        f"Models initialized:\n  Adapter: trainable\n  Early-commit gate: trainable\n  Rate controller: {USE_RATE_CONTROLLER}\n"
    )

    trainable_params = list(adapter.parameters()) + list(gate.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=OPT.lr, weight_decay=OPT.weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=OPT.warmup_steps,
    )
    pipeline = TrainingPipeline(optimizer=optimizer, scheduler=scheduler, grad_clip_norm=OPT.grad_clip_norm)

    dataset = LibriSpeechPairs(DATA.dataset_root)
    dataloader = DataLoader(dataset, batch_size=DATA.batch_size, shuffle=True, num_workers=DATA.num_workers)

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    logger = WandbLogger(
        enabled=WANDB.enabled,
        project=WANDB.project,
        entity=WANDB.entity,
        run_name=WANDB.run_name,
        config={
            "stage": 2,
            "data": DATA.__dict__,
            "optim": OPT.__dict__,
            "stage_cfg": STAGE.__dict__,
            "device_cfg": DEV.__dict__,
            "num_gpus": num_gpus,
        },
    )

    start_epoch = 0
    if os.path.exists(SAVE_PATH):
        print(f"Resuming from checkpoint: {SAVE_PATH}")
        ckpt = torch.load(SAVE_PATH, map_location=train_device)
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        gate.load_state_dict(ckpt["gate_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        pipeline.global_step = ckpt["global_step"]
        for _ in range(pipeline.global_step):
            scheduler.step()
        print(f"  Resumed at epoch {start_epoch}, global_step {pipeline.global_step}\n")
    else:
        start_epoch = 0

    print("\nStarting Stage 2 training: ASR Distillation")
    print(f"  Epochs: {STAGE.epochs}")
    print(f"  Batch size: {DATA.batch_size}")
    print(f"  Learning rate: {OPT.lr}")
    print(f"  λ_align: {STAGE.lambda_align}")
    print(f"  λ_stability: {STAGE.lambda_stability}")
    print(f"  λ_rate: {STAGE.lambda_rate}")
    print(f"  λ_gate: {STAGE.lambda_gate}")
    print(f"  Rate target: {RATE_TARGET} tokens/window\n")

    for epoch in range(start_epoch, STAGE.epochs):
        m_total = RunningMean()
        m_asr = RunningMean()
        m_align = RunningMean()
        m_stab = RunningMean()
        m_sparse = RunningMean()
        m_rate = RunningMean()
        m_gate = RunningMean()

        print(f"\n{'=' * 60}\nEpoch {epoch + 1}/{STAGE.epochs}\n{'=' * 60}\n")

        for step, batch in enumerate(dataloader):
            audio_paths, transcriptions = batch
            batch_texts = list(transcriptions)

            gt_tokens = llm_tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
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

            for p in audio_paths:
                wave = load_mono_waveform_16k(p)
                windows = audio.waveform_to_windows(wave)
                if DATA.max_windows_per_utt is not None:
                    windows = windows[: DATA.max_windows_per_utt]

                adapter.reset_streaming_state()
                utterance_tokens: list[torch.Tensor] = []
                for t, window in enumerate(windows):
                    result = adapter.forward_window(window)
                    utterance_tokens.append(result["tokens"])
                    total_stability_loss = total_stability_loss + result["stability_loss"].float()
                    if result["sparse_loss"] is not None:
                        total_sparse_loss = total_sparse_loss + result["sparse_loss"].float()
                    if result["rate_loss"] is not None:
                        total_rate_loss = total_rate_loss + result["rate_loss"].float()
                    if t > 0:
                        accumulated = torch.cat(utterance_tokens[:t], dim=1)
                        gate_result = gate(accumulated, t, len(windows))
                        total_gate_loss = total_gate_loss + gate_result["gate_loss"].float()
                        gate_calls += 1

                if utterance_tokens:
                    tokens = torch.cat(utterance_tokens, dim=1)
                    audio_tokens_list.append(tokens)
                    num_windows_list.append(len(windows))

            if not audio_tokens_list:
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

            with _maybe_autocast():
                asr_output = llm_model(
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    attention_mask=llm_attention_mask,
                )
                asr_loss = asr_output.loss

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

            m_total.update(total_loss.item())
            m_asr.update(asr_loss.item())
            m_align.update(align_loss.item())
            m_stab.update(float(stability_loss.item()))
            m_sparse.update(sparse_loss.item())
            m_rate.update(rate_loss.item())
            m_gate.update(float(gate_loss_mean.item()))

            if step % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"Step {step:4d}/{len(dataloader)} | Loss: {total_loss.item():.4f} | "
                    f"ASR: {asr_loss.item():.4f} | Align: {align_loss.item():.4f} | "
                    f"Stab: {float(stability_loss):.4f} | "
                    f"Rate: {rate_loss.item():.4f} | Gate: {float(gate_loss_mean):.4f} | "
                    f"Sparse(metric): {sparse_loss.item():.4f} | LR: {current_lr:.2e}"
                )
                logger.log(
                    {
                        "train/loss": total_loss.item(),
                        "train/asr": asr_loss.item(),
                        "train/align": align_loss.item(),
                        "train/stability": float(stability_loss),
                        "train/rate": rate_loss.item(),
                        "train/gate": float(gate_loss_mean),
                        "train/sparse_metric": sparse_loss.item(),
                        "train/lr": current_lr,
                    },
                    step=pipeline.global_step,
                )

            if train_device.startswith("cuda"):
                torch.cuda.empty_cache()

        save_checkpoint(
            SAVE_PATH,
            TrainingCheckpoint(
                stage=2,
                epoch=epoch + 1,
                global_step=pipeline.global_step,
                adapter_state_dict=adapter.state_dict(),
                gate_state_dict=gate.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                metrics={
                    "loss": m_total.mean,
                    "asr": m_asr.mean,
                    "align": m_align.mean,
                    "stability": m_stab.mean,
                    "sparse": m_sparse.mean,
                    "rate": m_rate.mean,
                    "gate": m_gate.mean,
                },
            ),
        )

        epoch_save_path = os.path.join(CKPT.dir, f"adapter_stage2_epoch{epoch + 1}.pt")

        save_checkpoint(
            epoch_save_path,
            TrainingCheckpoint(
                stage=2,
                epoch=epoch + 1,
                global_step=pipeline.global_step,
                adapter_state_dict=adapter.state_dict(),
                gate_state_dict=gate.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                metrics={
                    "loss": m_total.mean,
                    "asr": m_asr.mean,
                    "align": m_align.mean,
                    "stability": m_stab.mean,
                    "sparse": m_sparse.mean,
                    "rate": m_rate.mean,
                    "gate": m_gate.mean,
                },
            ),
        )

        print(
            f"\nEpoch {epoch + 1} complete: loss={m_total.mean:.4f} asr={m_asr.mean:.4f} "
            f"align={m_align.mean:.4f} stab={m_stab.mean:.4f}\nCheckpoint -> {SAVE_PATH}\n"
        )

    print("Stage 2 training complete!")
    logger.finish()


if __name__ == "__main__":
    train()
