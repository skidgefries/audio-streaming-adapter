"""
Stage 3: task distillation (teacher vs student frozen LLMs), train adapter + gate.

Reuses `training.utils.losses`, `training.utils.optimization`, `dataset`, and Whisper window features.

``L_gate`` is the early-commit gate loss (``EarlyCommitGate`` in ``adapter/early_commit_gate.py``),
not the adapter rate-controller gates. For inference aligned with this trainer, use
``WhisperAdapterLLMCommitGatePipeline`` in ``adapter_llm_pipeline.py``.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_root = os.path.join(_pkg_root, "src")
sys.path.insert(0, _src_root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, _pkg_root)

from training.utils.env import env_str, load_project_env

_env_path = load_project_env(_pkg_root)
if _env_path:
    print(f"Loaded environment from {_env_path}")

_hf_token = env_str("HF_TOKEN")
if _hf_token:
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", _hf_token)

from src.encoder import WhisperWindowFeatureExtractor
from src.adapter.streaming_adapter import StreamingAdapter
from src.adapter.early_commit_gate import EarlyCommitGate
from src.dataset import LibriSpeechConfig, LibriSpeechPairs, load_mono_waveform_16k
from src.llm import QwenConfig, load_qwen_models
from training.utils.checkpointing import (
    TrainingCheckpoint,
    load_adapter_state_dict,
    maybe_upload_stage_epoch_checkpoint,
    save_checkpoint,
)
from training.utils.config import (
    CheckpointConfig,
    DataConfig,
    HfCheckpointConfig,
    OptimConfig,
    Stage3Config,
    Stage3DeviceConfig,
    WandbConfig,
)
from training.utils.logging import WandbLogger
from training.utils.losses import kl_distill_loss, prefix_consistency_loss, revision_penalty_loss
from training.utils.metrics import RunningMean
from training.utils.optimization import TrainingPipeline

torch.cuda.empty_cache()

WHISPER_DIM = 768
WHISPER_MODEL = "openai/whisper-small"
STUDENT_LLM_ID = "Qwen/Qwen3-8B"
TEACHER_LLM_ID = "Qwen/Qwen3-8B"

STAGE = Stage3Config()
DEV = Stage3DeviceConfig()
OPT = OptimConfig(lr=3e-5, weight_decay=0.01, grad_clip_norm=1.0, warmup_steps=0)
DATA = DataConfig(
    dataset_root=LibriSpeechConfig.default_train_clean_100_from_training_dir(os.path.dirname(__file__)).root,
    batch_size=1,
    num_workers=0,
    max_windows_per_utt=None,
)
CKPT = CheckpointConfig(dir="checkpoints")
HF_CKPT = HfCheckpointConfig.from_env()
WANDB = WandbConfig(enabled=False, project="audio-streaming-adapter", run_name="stage3-task")

SAVE_PATH = os.path.join(CKPT.dir, "adapter_stage3.pt")
adapter_PATH = os.path.join(CKPT.dir, "adapter_stage3.pt")

TASK_TYPE = "asr"


def _pick_student_device() -> str:
    if DEV.student is not None:
        return DEV.student
    return "cuda" if torch.cuda.is_available() else "cpu"


def train() -> None:
    student_device = _pick_student_device()
    teacher_device = DEV.teacher
    student_dtype = torch.float16 if student_device.startswith("cuda") else torch.float32

    print(f"Student device: {student_device}, teacher device: {teacher_device}")

    audio = WhisperWindowFeatureExtractor(
        model_id=WHISPER_MODEL,
        device=student_device,
        torch_dtype=student_dtype,
    )

    student_qwen = load_qwen_models(
        cfg=QwenConfig(
            model_id=STUDENT_LLM_ID,
            device=student_device,
            torch_dtype=student_dtype,
            device_map=None,
            embeddings_only=False,
        )
    )
    student_tokenizer = student_qwen.tokenizer
    student_model = student_qwen.causal_lm
    if student_model is None:
        raise RuntimeError("Expected causal LM for stage 3 student")
    student_model.to(student_device)
    student_embedder = student_model.get_input_embeddings()

    teacher_qwen = load_qwen_models(
        cfg=QwenConfig(
            model_id=TEACHER_LLM_ID,
            device=teacher_device,
            torch_dtype=torch.float32,
            device_map=None,
            embeddings_only=False,
        )
    )
    teacher_tokenizer = teacher_qwen.tokenizer
    teacher_model = teacher_qwen.causal_lm
    if teacher_model is None:
        raise RuntimeError("Expected causal LM for stage 3 teacher")
    teacher_model.to(teacher_device)
    teacher_embedder = teacher_model.get_input_embeddings()

    adapter = StreamingAdapter(
        d_encoder=WHISPER_DIM,
        d_llm=4096,
        num_queries=4,
        num_layers=2,
        num_heads=4,
        d_ffn=2048,
        dropout=0.1,
        ema_alpha=0.8,
        learnable_ema=False,
        use_rate_controller=STAGE.use_rate_controller,
        rate_threshold=0.5,
        target_rate=STAGE.rate_target,
    ).to(student_device, dtype=student_dtype)
    adapter.train()

    if os.path.isfile(adapter_PATH):
        print(f"Loading adapter weights from {adapter_PATH}")
        adapter.load_state_dict(load_adapter_state_dict(adapter_PATH))

    gate = EarlyCommitGate(
        d_llm=4096,
        hidden_dim=256,
        threshold=0.5,
        latency_weight=0.1,
    ).to(student_device, dtype=student_dtype)
    gate.train()

    trainable_params = list(adapter.parameters()) + list(gate.parameters())
    optimizer = torch.optim.SGD(trainable_params, lr=OPT.lr, weight_decay=OPT.weight_decay)

    dataset = LibriSpeechPairs(DATA.dataset_root)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, STAGE.epochs * max(1, len(dataset))),
        eta_min=1e-6,
    )
    pipeline = TrainingPipeline(optimizer=optimizer, scheduler=scheduler, grad_clip_norm=OPT.grad_clip_norm)

    dataloader = DataLoader(dataset, batch_size=DATA.batch_size, shuffle=True, num_workers=DATA.num_workers)

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    logger = WandbLogger(
        enabled=WANDB.enabled,
        project=WANDB.project,
        entity=WANDB.entity,
        run_name=WANDB.run_name,
        config={"stage": 3, "data": DATA.__dict__, "optim": OPT.__dict__, "stage_cfg": STAGE.__dict__},
    )

    print("\nStarting Stage 3 training: Task Distillation")
    print(f"  Task: {TASK_TYPE} | Epochs: {STAGE.epochs} | batch: {DATA.batch_size} | lr: {OPT.lr}\n")

    for epoch in range(STAGE.epochs):
        m_total = RunningMean()
        m_task = RunningMean()
        m_asr = RunningMean()
        m_stab = RunningMean()
        m_rate = RunningMean()
        m_gate = RunningMean()
        m_prefix = RunningMean()
        m_rev = RunningMean()

        print(f"\n{'=' * 60}\nEpoch {epoch + 1}/{STAGE.epochs}\n{'=' * 60}\n")

        for step, batch in enumerate(dataloader):
            audio_paths, transcriptions = batch
            audio_path = audio_paths[0]
            transcription = transcriptions[0]

            wave = load_mono_waveform_16k(audio_path)
            windows = audio.waveform_to_windows(wave)
            if DATA.max_windows_per_utt is not None:
                windows = windows[: DATA.max_windows_per_utt]
            if not windows:
                continue

            teacher_input = teacher_tokenizer(
                f"Transcribe the following audio: {transcription}",
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(teacher_device)

            with torch.inference_mode():
                teacher_output = teacher_model.generate(
                    **teacher_input,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=teacher_tokenizer.pad_token_id,
                )
                teacher_response_tokens = teacher_output[:, teacher_input.input_ids.shape[1] :]
                teacher_ids = teacher_response_tokens
                teacher_inputs_embeds = teacher_embedder(torch.cat([teacher_input.input_ids, teacher_ids], dim=1))
                teacher_full_output = teacher_model(inputs_embeds=teacher_inputs_embeds[:, :-1, :])
                teacher_logits = teacher_full_output.logits.to(student_device)

            student_gt_tokens = student_tokenizer(
                transcription,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(student_device)
            student_gt_ids = student_gt_tokens.input_ids

            adapter.reset_streaming_state()
            all_audio_tokens: list[torch.Tensor] = []
            generation_history: list[torch.Tensor] = []
            total_stability_loss = torch.zeros((), device=student_device, dtype=torch.float32)
            total_rate_loss = torch.zeros((), device=student_device, dtype=torch.float32)
            total_gate_loss = torch.zeros((), device=student_device, dtype=torch.float32)

            for t, window in enumerate(windows):
                result = adapter.forward_window(window)
                all_audio_tokens.append(result["tokens"])
                total_stability_loss = total_stability_loss + result["stability_loss"].float()
                if result["rate_loss"] is not None:
                    total_rate_loss = total_rate_loss + result["rate_loss"].float()

                accumulated = torch.cat(all_audio_tokens, dim=1)
                gate_result = gate(accumulated, t, len(windows))
                total_gate_loss = total_gate_loss + gate_result["gate_loss"].float()

                if gate_result["should_commit"].item() > 0.5 or t == len(windows) - 1:
                    bos_id = (
                        student_tokenizer.bos_token_id
                        if student_tokenizer.bos_token_id is not None
                        else student_tokenizer.eos_token_id
                    )
                    bos_embed = student_embedder(torch.tensor([[bos_id]], device=student_device))
                    inputs_embeds = torch.cat([accumulated, bos_embed], dim=1)
                    with torch.no_grad():
                        output = student_model(inputs_embeds=inputs_embeds)
                        student_logits = output.logits[:, -1:, :]
                    generation_history.append(student_logits)

            final_tokens = torch.cat(all_audio_tokens, dim=1)
            bos_id = (
                student_tokenizer.bos_token_id
                if student_tokenizer.bos_token_id is not None
                else student_tokenizer.eos_token_id
            )
            bos_embed = student_embedder(torch.tensor([[bos_id]], device=student_device))
            inputs_embeds_full = torch.cat([final_tokens, bos_embed], dim=1)

            with torch.no_grad():
                student_full_output = student_model(inputs_embeds=inputs_embeds_full)
                student_full_logits = student_full_output.logits

            max_len = max(student_full_logits.shape[1], teacher_logits.shape[1])
            student_logits_padded = F.pad(student_full_logits, (0, 0, 0, max_len - student_full_logits.shape[1]))
            teacher_logits_padded = F.pad(teacher_logits, (0, 0, 0, max_len - teacher_logits.shape[1]))

            mask = torch.ones(student_full_logits.shape[:2], device=student_device)
            mask = F.pad(mask, (0, max_len - mask.shape[1]))

            task_loss = kl_distill_loss(
                student_logits=student_logits_padded,
                teacher_logits=teacher_logits_padded,
                temperature=STAGE.kl_temperature,
                mask=mask,
            )

            gt_embeds = student_embedder(student_gt_ids)
            asr_loss = F.mse_loss(final_tokens.mean(dim=1), gt_embeds.mean(dim=1))

            stability_loss = total_stability_loss / float(len(windows))
            rate_loss = total_rate_loss / float(len(windows))
            gate_loss = total_gate_loss / float(len(windows))

            prefix_loss = torch.tensor(0.0, device=student_device)
            if len(generation_history) > 1:
                prefix_loss = prefix_consistency_loss(
                    partial_logits=generation_history[0],
                    full_logits=generation_history[-1],
                    prefix_length=1,
                )

            revision_loss = revision_penalty_loss(
                generation_history,
                mask=None,
                device=student_device,
            )

            total_loss = (
                task_loss
                + STAGE.lambda_asr * asr_loss
                + STAGE.lambda_stability * stability_loss
                + STAGE.lambda_rate * rate_loss
                + STAGE.lambda_gate * gate_loss
            )

            pipeline.step(total_loss, trainable_params)

            m_total.update(total_loss.item())
            m_task.update(task_loss.item())
            m_asr.update(asr_loss.item())
            m_stab.update(float(stability_loss.item()))
            m_rate.update(float(rate_loss.item()))
            m_gate.update(float(gate_loss.item()))
            m_prefix.update(float(prefix_loss.item()))
            m_rev.update(float(revision_loss.item()))

            if step % 10 == 0:
                lr = scheduler.get_last_lr()[0]
                tok_per_win = final_tokens.shape[1] / len(windows)
                print(
                    f"Step {step:4d}/{len(dataloader)} | Loss: {total_loss.item():.4f} | Task: {task_loss.item():.4f} | "
                    f"ASR: {asr_loss.item():.4f} | Stab: {float(stability_loss):.4f} | Rate: {float(rate_loss):.4f} | "
                    f"Gate: {float(gate_loss):.4f} | Prefix: {float(prefix_loss):.4f} | Toks/win: {tok_per_win:.2f} | LR: {lr:.2e}"
                )
                logger.log(
                    {
                        "train/loss": total_loss.item(),
                        "train/task": task_loss.item(),
                        "train/asr": asr_loss.item(),
                        "train/stability": float(stability_loss),
                        "train/rate": float(rate_loss),
                        "train/gate": float(gate_loss),
                        "train/prefix": float(prefix_loss),
                        "train/revision": float(revision_loss),
                        "train/lr": lr,
                    },
                    step=pipeline.global_step,
                )

            if student_device.startswith("cuda"):
                torch.cuda.empty_cache()

        save_checkpoint(
            SAVE_PATH,
            TrainingCheckpoint(
                stage=3,
                epoch=epoch + 1,
                global_step=pipeline.global_step,
                adapter_state_dict=adapter.state_dict(),
                gate_state_dict=gate.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                metrics={
                    "loss": m_total.mean,
                    "task": m_task.mean,
                    "asr": m_asr.mean,
                    "stability": m_stab.mean,
                    "rate": m_rate.mean,
                    "gate": m_gate.mean,
                    "prefix": m_prefix.mean,
                    "revision": m_rev.mean,
                },
            ),
        )

        epoch_save_path = os.path.join(CKPT.dir, f"adapter_stage3_epoch{epoch + 1}.pt")
        save_checkpoint(
            epoch_save_path,
            TrainingCheckpoint(
                stage=3,
                epoch=epoch + 1,
                global_step=pipeline.global_step,
                adapter_state_dict=adapter.state_dict(),
                gate_state_dict=gate.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                metrics={
                    "loss": m_total.mean,
                    "task": m_task.mean,
                    "asr": m_asr.mean,
                    "stability": m_stab.mean,
                    "rate": m_rate.mean,
                    "gate": m_gate.mean,
                    "prefix": m_prefix.mean,
                    "revision": m_rev.mean,
                },
            ),
        )
        maybe_upload_stage_epoch_checkpoint(
            epoch_save_path,
            stage=3,
            repo_id=HF_CKPT.repo_id,
            revision=HF_CKPT.revision,
            private=HF_CKPT.private,
            token=_hf_token,
            enabled=HF_CKPT.upload_enabled,
        )
        print(f"\nEpoch {epoch + 1} done. Checkpoint -> {SAVE_PATH}\n")

    print("Stage 3 training complete!")
    logger.finish()


if __name__ == "__main__":
    train()
