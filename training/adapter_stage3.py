"""
Stage 3 (simplified): ASR answer distillation — train StreamingAdapter only.

Follows Stage-2 layout (``adapter_asr_align_trainer.py``) but drops align / stability /
rate / gate. Frozen Whisper encoder and frozen Qwen LLM.

Student prefix: ``[DEFAULT_ASR_PROMPT | audio_tokens | answer]`` (``PROMPT_CONDITIONING``).
Teacher: frozen LLM on GT transcript tokens only.
Loss is on answer token positions only::

    L = α · CE + (1 − α) · KL(teacher ‖ student)

Warm-start defaults to Stage-2 ``checkpoints/adapter_asr_align.pt``.

**Run**::

    uv run training/adapter_stage3.py

External packages (already in project): ``torch``, ``transformers``.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import ExitStack, nullcontext
from datetime import datetime, timezone

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_root = os.path.join(_pkg_root, "src")
sys.path.insert(0, _src_root)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, _pkg_root)

from training.utils.env import apply_hf_hub_endpoint, env_bool, env_int, env_str, load_project_env

_env_path = load_project_env(_pkg_root)
if _env_path:
    print(f"Loaded environment from {_env_path}")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("DEVICE", "cuda")
os.environ.setdefault("BATCH_SIZE", "16")
os.environ.setdefault("ASR_MICRO_BATCH_SIZE", "1")
os.environ.setdefault("MAX_WINDOWS_PER_UTT", "32")
os.environ.setdefault("MAX_TEXT_TOKENS", "128")
os.environ.setdefault("ENABLE_LLM_GRADIENT_CHECKPOINTING", "true")
os.environ.setdefault("GPU_LOCK", "true")
os.environ.setdefault("WANDB_ENABLED", "true")
os.environ.setdefault("WANDB_RUN_NAME", "adapter_stage3")
os.environ.setdefault("SAVE_EVERY_STEPS", "1000")
os.environ.setdefault("VAL_EVERY_STEPS", "1000")
os.environ.setdefault("VAL_MAX_UTTERANCES", "100")
# CE_ALPHA / KL_TEMPERATURE defaults live only in Stage3Config (config.py).

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

from llm.config import LlmGenerationParams
from src.adapter.streaming_adapter import StreamingAdapter
from src.dataset import LibriSpeechConfig, LibriSpeechPairs, load_mono_waveform_16k
from src.encoder import WhisperWindowFeatureExtractor
from training.utils.asr_prompt import DEFAULT_ASR_PROMPT, PROMPT_CONDITIONING
from training.utils.asr_only_validation import UtteranceEncodeResult
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
    Stage3Config,
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
from training.utils.loaders import load_frozen_qwen_causal_lm
from training.utils.logging import WandbLogger
from training.utils.losses import kl_distill_loss
from training.utils.metrics import RunningMean, corpus_bleu4, normalize_asr_text, word_error_rate
from training.utils.optimization import TrainingPipeline

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

STAGE = Stage3Config.from_env()
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
CKPT = CheckpointConfig.from_env(pkg_root=_pkg_root)
HF_CKPT = HfCheckpointConfig.from_env()
WANDB = WandbConfig.from_env()

CHECKPOINT_BASENAME = env_str("CHECKPOINT_BASENAME", "adapter_stage3") or "adapter_stage3"
SAVE_PATH = os.path.join(CKPT.dir, f"{CHECKPOINT_BASENAME}.pt")
SAVE_EVERY_STEPS = CKPT.save_every_steps if CKPT.save_every_steps is not None else 1000
LOG_EVERY_STEPS = max(1, DATA.batch_size)
VAL_HISTORY_PATH = os.path.join(CKPT.dir, f"{CHECKPOINT_BASENAME}_val_history.jsonl")

_stage2_rel = (
    env_str("STAGE2_CHECKPOINT", "checkpoints/adapter_asr_align.pt")
    or "checkpoints/adapter_asr_align.pt"
)
STAGE2_SAVE_PATH = (
    _stage2_rel if os.path.isabs(_stage2_rel) else os.path.join(_pkg_root, _stage2_rel)
)
LOAD_STAGE2_CHECKPOINT = env_bool("LOAD_STAGE2_CHECKPOINT", True)


def _load_adapter_from_checkpoint(
    adapter: StreamingAdapter,
    ckpt: dict,
    *,
    source: str,
    is_main: bool,
) -> None:
    """Load adapter weights from a Stage-2/3 checkpoint (no rate controller)."""
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
    """Enable CUDA autocast in bf16 when running on GPU."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _tokenize_asr_prompt(llm_tokenizer, *, device: torch.device) -> torch.Tensor:
    """
    Tokenize ``DEFAULT_ASR_PROMPT`` as a Qwen chat user turn.

    Returns:
        ``input_ids`` of shape ``(1, P)``.
    """
    messages = [{"role": "user", "content": DEFAULT_ASR_PROMPT}]
    if hasattr(llm_tokenizer, "apply_chat_template"):
        batch = llm_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if hasattr(batch, "input_ids"):
            return batch.input_ids.to(device)
        return batch.to(device)
    return llm_tokenizer(DEFAULT_ASR_PROMPT, return_tensors="pt").input_ids.to(device)


def _forward_utterance_tokens(
    *,
    adapter: StreamingAdapter,
    audio_extractor: WhisperWindowFeatureExtractor,
    audio_path: str,
    train_device: torch.device,
    max_windows_per_utt: int | None,
) -> tuple[torch.Tensor | None, int]:
    """
    Encode one utterance through Whisper + adapter.

    Returns:
        ``(audio_tokens, num_windows)`` or ``(None, 0)`` if empty.
    """
    wave = load_mono_waveform_16k(audio_path)
    windows = audio_extractor.waveform_to_windows(wave)
    if max_windows_per_utt is not None:
        windows = windows[:max_windows_per_utt]
    if not windows:
        return None, 0

    adapter.reset_streaming_state()
    utterance_tokens: list[torch.Tensor] = []
    for window in windows:
        result = adapter.forward_window(window.to(device=train_device, dtype=TORCH_DTYPE))
        utterance_tokens.append(result["tokens"])
    return torch.cat(utterance_tokens, dim=1), len(windows)


def _build_student_inputs(
    *,
    prompt_ids: torch.Tensor,
    audio_tokens: torch.Tensor,
    gt_ids: torch.Tensor,
    gt_attention_mask: torch.Tensor,
    text_embedder: torch.nn.Module,
    llm_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Build student ``[prompt | audio | answer]`` embeds with answer-only labels.

    Args:
        prompt_ids: Shared ASR prompt ids ``(1, P)`` (broadcast to batch).
        audio_tokens: Adapter tokens ``(B, A, D)``.
        gt_ids: Tokenized GT transcript ``(B, L)``.
        gt_attention_mask: Attention mask for ``gt_ids``.

    Returns:
        ``(inputs_embeds, labels, attention_mask, cond_len)`` where ``cond_len`` is
        ``P + A`` (positions before answer embeds).
    """
    batch_size = audio_tokens.shape[0]
    audio_len = audio_tokens.shape[1]
    if prompt_ids.shape[0] == 1 and batch_size > 1:
        prompt_ids = prompt_ids.expand(batch_size, -1)
    prompt_ids = prompt_ids.to(llm_device)
    prompt_len = prompt_ids.shape[1]
    prompt_embeds = text_embedder(prompt_ids)

    audio = audio_tokens.to(device=llm_device, dtype=LLM_DTYPE)
    gt_shifted = gt_ids[:, 1:].to(llm_device)
    gt_mask_shifted = gt_attention_mask[:, 1:].to(llm_device)
    answer_embeds = text_embedder(gt_shifted)

    inputs_embeds = torch.cat([prompt_embeds, audio, answer_embeds], dim=1)
    cond_len = prompt_len + audio_len
    pre_text_labels = torch.full(
        (batch_size, cond_len), -100, dtype=torch.long, device=llm_device
    )
    labels = torch.cat([pre_text_labels, gt_shifted], dim=1)

    prompt_mask = torch.ones((batch_size, prompt_len), device=llm_device)
    audio_mask = torch.ones((batch_size, audio_len), device=llm_device)
    attention_mask = torch.cat([prompt_mask, audio_mask, gt_mask_shifted], dim=1)
    return inputs_embeds, labels, attention_mask, cond_len


def _answer_ce_and_kl(
    llm_model: torch.nn.Module,
    *,
    student_inputs_embeds: torch.Tensor,
    student_labels: torch.Tensor,
    student_attention_mask: torch.Tensor,
    teacher_input_ids: torch.Tensor,
    teacher_attention_mask: torch.Tensor,
    kl_temperature: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute answer-only CE and KL against a text-only teacher.

    Teacher runs on GT transcript ids only (no audio / prompt). Student and teacher
    answer logits are aligned on the shifted transcript targets.
    """
    with _maybe_autocast(device):
        student_out = llm_model(
            inputs_embeds=student_inputs_embeds,
            attention_mask=student_attention_mask,
        )
        student_shift_logits = student_out.logits[:, :-1, :].float()
        student_shift_labels = student_labels[:, 1:]
        answer_mask = (student_shift_labels != -100).to(dtype=student_shift_logits.dtype)

        if answer_mask.sum() <= 0:
            zero = student_shift_logits.sum() * 0.0
            return zero, zero

        flat_logits = student_shift_logits.reshape(-1, student_shift_logits.size(-1))
        flat_labels = student_shift_labels.reshape(-1)
        ce_loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)

        with torch.no_grad():
            teacher_out = llm_model(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
            )
            teacher_shift_logits = teacher_out.logits[:, :-1, :].float()
            teacher_mask = teacher_attention_mask[:, 1:] > 0

        # Answer positions: student (prompt|audio|answer) vs teacher (GT transcript only).
        kl_terms: list[torch.Tensor] = []
        for b in range(student_shift_logits.shape[0]):
            s_logits = student_shift_logits[b, answer_mask[b].bool()]
            t_logits = teacher_shift_logits[b, teacher_mask[b]]
            n = min(int(s_logits.shape[0]), int(t_logits.shape[0]))
            if n == 0:
                continue
            kl_terms.append(
                kl_distill_loss(
                    student_logits=s_logits[:n].unsqueeze(0),
                    teacher_logits=t_logits[:n].detach().unsqueeze(0),
                    temperature=kl_temperature,
                    mask=torch.ones((1, n), device=device, dtype=s_logits.dtype),
                )
            )
        kl_loss = torch.stack(kl_terms).mean() if kl_terms else ce_loss * 0.0

    return ce_loss, kl_loss


def _pipeline_begin_microbatch(pipeline: TrainingPipeline) -> None:
    """Zero optimizer grads at the start of an accumulation window."""
    if pipeline.accum_step == 0:
        pipeline.optimizer.zero_grad(set_to_none=True)


def _pipeline_accumulate_backward(
    pipeline: TrainingPipeline,
    loss: torch.Tensor,
    *,
    no_sync_modules: tuple[torch.nn.Module, ...] | None,
    sync_grads: bool,
) -> None:
    """Scale loss by accumulation steps and backward (optional DDP no_sync)."""
    scaled = loss / pipeline.gradient_accumulation_steps
    use_no_sync = no_sync_modules is not None and not sync_grads
    if use_no_sync:
        with ExitStack() as stack:
            for module in no_sync_modules:
                if isinstance(module, DDP):
                    stack.enter_context(module.no_sync())
            scaled.backward()
    else:
        scaled.backward()


def _pipeline_end_microbatch(
    pipeline: TrainingPipeline,
    params: list[torch.nn.Parameter],
) -> bool:
    """
    Apply optimizer/scheduler after manual backward(s).

    Returns:
        True if an optimizer step was taken.
    """
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
    """Return the underlying module when wrapped in DDP."""
    return module.module if isinstance(module, DDP) else module


def _encode_utterance(
    *,
    adapter: StreamingAdapter,
    audio_extractor: WhisperWindowFeatureExtractor,
    audio_path: str,
    train_device: torch.device,
    max_windows_per_utt: int | None,
    collect_aux_metrics: bool = False,
) -> UtteranceEncodeResult | None:
    """Encode utterance for validation; stability unused in Stage 3."""
    del collect_aux_metrics  # Stage 3 does not track stability.
    tokens, num_windows = _forward_utterance_tokens(
        adapter=adapter,
        audio_extractor=audio_extractor,
        audio_path=audio_path,
        train_device=train_device,
        max_windows_per_utt=max_windows_per_utt,
    )
    if tokens is None:
        return None
    return UtteranceEncodeResult(tokens=tokens, num_windows=num_windows, stability_loss=0.0)


def _decode_prompt_audio(
    *,
    llm_model: torch.nn.Module,
    llm_tokenizer,
    text_embedder: torch.nn.Module,
    prompt_ids: torch.Tensor,
    audio_tokens: torch.Tensor,
    llm_device: torch.device,
    generation: LlmGenerationParams,
) -> str:
    """Greedy/beam decode from ``[prompt | audio]`` prefix (Stage-3 eval layout)."""
    from llm.config import build_hf_generation_config
    from adapter_llm_pipeline import _truncate_asr_chat_tail

    prompt_embeds = text_embedder(prompt_ids.to(llm_device))
    audio = audio_tokens.to(device=llm_device, dtype=LLM_DTYPE)
    inputs_embeds = torch.cat([prompt_embeds, audio], dim=1)
    attention_mask = torch.ones(
        inputs_embeds.shape[0],
        inputs_embeds.shape[1],
        dtype=torch.long,
        device=llm_device,
    )
    gen_cfg = build_hf_generation_config(
        model=llm_model,
        tokenizer=llm_tokenizer,
        params=generation,
    )
    with _maybe_autocast(llm_device):
        out_ids = llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=gen_cfg,
        )
    text = llm_tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
    return _truncate_asr_chat_tail(text)


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


@torch.no_grad()
def _validate_stage3(
    *,
    adapter: StreamingAdapter,
    audio_extractor: WhisperWindowFeatureExtractor,
    llm_model: torch.nn.Module,
    llm_tokenizer,
    text_embedder: torch.nn.Module,
    prompt_ids: torch.Tensor,
    val_root: str,
    train_device: torch.device,
    llm_device: torch.device,
    max_utterances: int | None,
    max_windows_per_utt: int | None,
    max_text_tokens: int,
    kl_temperature: float,
    alpha: float,
    generation: LlmGenerationParams,
    global_step: int,
    log_every: int,
    predictions_path: str | None,
) -> tuple[dict[str, float], list[dict]]:
    """
    Dev-clean validation: answer CE/KL plus prompt|audio decode WER/BLEU.
    """
    from torch.utils.data import Subset

    adapter_was_training = adapter.training
    adapter.eval()

    val_full = LibriSpeechPairs(val_root)
    n_val = len(val_full) if max_utterances is None else min(max_utterances, len(val_full))
    val_dataset = val_full if n_val == len(val_full) else Subset(val_full, range(n_val))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    m_ce = RunningMean()
    m_kl = RunningMean()
    m_total = RunningMean()
    refs: list[str] = []
    hyps: list[str] = []
    items: list[dict] = []

    print(
        f"\n  [val step {global_step}] dev-clean ({n_val} utts) "
        f"CE+KL + {PROMPT_CONDITIONING} decode...",
        flush=True,
    )

    for step, batch in enumerate(val_loader):
        audio_paths, transcriptions = batch
        audio_path, text = audio_paths[0], transcriptions[0]
        encoded = _encode_utterance(
            adapter=adapter,
            audio_extractor=audio_extractor,
            audio_path=audio_path,
            train_device=train_device,
            max_windows_per_utt=max_windows_per_utt,
        )
        if encoded is None:
            continue

        gt = llm_tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_text_tokens,
        )
        student_embeds, labels, attn, _cond = _build_student_inputs(
            prompt_ids=prompt_ids,
            audio_tokens=encoded.tokens,
            gt_ids=gt.input_ids,
            gt_attention_mask=gt.attention_mask,
            text_embedder=text_embedder,
            llm_device=llm_device,
        )
        ce_loss, kl_loss = _answer_ce_and_kl(
            llm_model,
            student_inputs_embeds=student_embeds,
            student_labels=labels,
            student_attention_mask=attn,
            teacher_input_ids=gt.input_ids.to(llm_device),
            teacher_attention_mask=gt.attention_mask.to(llm_device),
            kl_temperature=kl_temperature,
            device=llm_device,
        )
        total = alpha * ce_loss + (1.0 - alpha) * kl_loss
        m_ce.update(float(ce_loss.item()))
        m_kl.update(float(kl_loss.item()))
        m_total.update(float(total.item()))

        prediction = _decode_prompt_audio(
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer,
            text_embedder=text_embedder,
            prompt_ids=prompt_ids,
            audio_tokens=encoded.tokens,
            llm_device=llm_device,
            generation=generation,
        )
        reference = normalize_asr_text(text)
        prediction = normalize_asr_text(prediction)
        wer = word_error_rate(reference, prediction)
        items.append(
            {
                "utterance_id": os.path.splitext(os.path.basename(audio_path))[0],
                "audio_path": audio_path,
                "reference": reference,
                "prediction": prediction,
                "wer": wer,
                "ce": float(ce_loss.item()),
                "kl": float(kl_loss.item()),
            }
        )
        refs.append(reference)
        hyps.append(prediction)
        samples_done = len(items)
        if log_every > 0 and (
            samples_done == 1 or samples_done % log_every == 0 or samples_done == n_val
        ):
            print(
                f"  [{samples_done}/{n_val}] WER={wer:.3f} CE={ce_loss.item():.4f} "
                f"KL={kl_loss.item():.4f} ref={reference[:48]}{'...' if len(reference) > 48 else ''}",
                flush=True,
            )

    if adapter_was_training:
        adapter.train()
    if not items:
        raise RuntimeError(f"No validation samples found under {val_root}")

    avg_wer = sum(item["wer"] for item in items) / len(items)
    bleu4 = corpus_bleu4(refs, hyps)
    metrics = {
        "val/loss": m_total.mean,
        "val/ce": m_ce.mean,
        "val/kl": m_kl.mean,
        "val/wer": avg_wer,
        "val/bleu4": bleu4,
        "val/num_samples": float(len(items)),
    }
    print(
        f"  [val step {global_step}] "
        f"loss={m_total.mean:.4f} CE={m_ce.mean:.4f} KL={m_kl.mean:.4f} | "
        f"WER={avg_wer:.4f} BLEU-4={bleu4:.4f} ({len(items)} samples)",
        flush=True,
    )
    if predictions_path and items:
        out_dir = os.path.dirname(predictions_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        with open(predictions_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "global_step": global_step,
                    "num_samples": len(items),
                    "metrics": metrics,
                    "items": items,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        print(f"  [val step {global_step}] predictions -> {predictions_path}", flush=True)
    return metrics, items


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Warmup + cosine LR schedule matching Stage 2."""
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
    """Build a Stage-3 training checkpoint (adapter only; no gate)."""
    return TrainingCheckpoint(
        stage=3,
        epoch=epoch,
        global_step=global_step,
        adapter_state_dict=_unwrap(adapter).state_dict(),
        gate_state_dict=None,
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(),
        metrics=metrics,
        hyperparams={
            "trainer": "adapter_stage3",
            "stage_cfg": STAGE.__dict__,
            "grad_accum_steps": DATA.gradient_accumulation_steps(),
            "ce_alpha": STAGE.alpha,
            "kl_temperature": STAGE.kl_temperature,
            "conditioning": PROMPT_CONDITIONING,
            "asr_prompt": DEFAULT_ASR_PROMPT,
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
    """Save rolling + step checkpoint every ``SAVE_EVERY_STEPS`` optimizer steps."""
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
    """Restore optimizer/scheduler/step and return ``(start_epoch, batch_offset)``."""
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
    """Run Stage-3 ASR answer distillation training."""
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

    alpha = float(STAGE.alpha)
    alpha = min(1.0, max(0.0, alpha))
    kl_temperature = float(STAGE.kl_temperature)

    if ctx.is_main:
        print(f"Training device: {train_device_str} (Whisper + adapter)")
        print(f"LLM device: {llm_device_str}")
        print(f"Visible CUDA devices: {ctx.num_cuda_devices}")
        print(
            f"Pipeline: window → encoder → adapter → LLM | "
            f"L = {alpha:.2f}·CE + {1.0 - alpha:.2f}·KL  "
            f"(answer tokens only, {PROMPT_CONDITIONING})"
        )

    if llm_load_device.type == "cuda":
        with torch.cuda.device(llm_load_device):
            torch.cuda.empty_cache()

    qwen_max_memory = adjust_llm_max_memory_for_free_vram(qwen_max_memory)
    if ctx.is_main:
        print(f"Qwen device_map: {qwen_map!r}")
        if qwen_max_memory:
            print(f"Qwen max_memory: {qwen_max_memory!r}")

    qwen_models = load_frozen_qwen_causal_lm(
        model_id=LLM_MODEL_ID,
        device=llm_device_str,
        torch_dtype=LLM_DTYPE,
        device_map=qwen_map,
        max_memory=qwen_max_memory,
    )
    llm_tokenizer = qwen_models.tokenizer
    llm_model = qwen_models.causal_lm
    text_embedder = qwen_models.embedder
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
        resume_stage = int(resume_meta.get("stage", 3))
        if resume_stage < 3:
            adapter_init_path = resume_path
            if ctx.is_main:
                print(f"RESUME_CHECKPOINT is Stage {resume_stage} ({resume_path}); adapter weights only.")
        else:
            resume_ckpt = torch.load(resume_path, map_location=train_device)
            resume_training_state = True
            if ctx.is_main:
                print(f"Will resume from checkpoint: {resume_path}")
    elif LOAD_STAGE2_CHECKPOINT:
        if not os.path.isfile(STAGE2_SAVE_PATH):
            raise FileNotFoundError(
                f"LOAD_STAGE2_CHECKPOINT=true but checkpoint missing: {STAGE2_SAVE_PATH}"
            )
        adapter_init_path = STAGE2_SAVE_PATH
        if ctx.is_main:
            print(f"Will init adapter from Stage 2: {STAGE2_SAVE_PATH}")
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

    if ctx.is_main:
        print(
            "Models initialized:\n"
            "  Encoder: frozen Whisper\n"
            "  Adapter: trainable (no rate controller / gate)\n"
            "  LLM: frozen (student + text-only teacher)\n"
            f"  Checkpoint: {CHECKPOINT_BASENAME}\n"
        )

    optimizer = torch.optim.AdamW(
        list(_unwrap(adapter).parameters()),
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
        os.makedirs(os.path.dirname(SAVE_PATH) or ".", exist_ok=True)

    logger = WandbLogger(
        enabled=WANDB.enabled and ctx.is_main,
        project=WANDB.project,
        entity=WANDB.entity,
        run_name=WANDB.run_name,
        config={
            "trainer": "adapter_stage3",
            "data": {**DATA.__dict__, "dataset_roots": DATASET_ROOTS, "val_root": VAL_ROOT},
            "optim": OPT.__dict__,
            "stage_cfg": STAGE.__dict__,
            "device": DEVICE_CFG.__dict__,
            "checkpoint_basename": CHECKPOINT_BASENAME,
            "stage2_checkpoint": STAGE2_SAVE_PATH,
            "conditioning": PROMPT_CONDITIONING,
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
                f"offset {resume_batch_offset}/{len(dataloader)}\n"
            )

    prompt_ids = _tokenize_asr_prompt(llm_tokenizer, device=llm_device)

    if ctx.is_main:
        print("\nStarting Stage 3 ASR answer distillation")
        print(f"  Dataset: {', '.join(os.path.basename(r) for r in DATASET_ROOTS)}")
        print(f"  Epochs: {STAGE.epochs} | batch: {DATA.batch_size} | effective: {effective_batch_size}")
        print(f"  α (CE): {alpha} | (1-α) KL: {1.0 - alpha} | T_KL: {kl_temperature}")
        print(f"  Stage 2 init: {STAGE2_SAVE_PATH}")
        print(f"  Conditioning: {PROMPT_CONDITIONING}")
        print(f"  Prompt: {DEFAULT_ASR_PROMPT[:72]}...")
        print(f"  Logging: every {LOG_EVERY_STEPS} steps (batch size)")
        if SAVE_EVERY_STEPS > 0:
            print(f"  Checkpoints: every {SAVE_EVERY_STEPS} steps -> {SAVE_PATH}")
        if STAGE.val_enabled and os.path.isdir(VAL_ROOT):
            val_cap = STAGE.val_max_utterances if STAGE.val_max_utterances is not None else "all"
            print(f"  Validation: dev-clean every {STAGE.val_every_steps} steps (max {val_cap})")
            print(f"  Val history: {VAL_HISTORY_PATH}")
        print()

    adapter_module = _unwrap(adapter)

    for epoch in range(start_epoch, STAGE.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        m_total = RunningMean()
        m_ce = RunningMean()
        m_kl = RunningMean()

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

            work: list[tuple[str, str, torch.Tensor, int]] = []
            for audio_path, text in items:
                tokens, num_windows = _forward_utterance_tokens(
                    adapter=adapter_module,
                    audio_extractor=audio,
                    audio_path=audio_path,
                    train_device=train_device,
                    max_windows_per_utt=DATA.max_windows_per_utt,
                )
                if tokens is None or num_windows == 0:
                    continue
                work.append((audio_path, text, tokens, num_windows))

            if not work:
                if ctx.is_main:
                    print(f"[WARN] Step {step}: empty batch; skipping.")
                continue

            n_valid = len(work)
            _pipeline_begin_microbatch(pipeline)
            no_sync_modules = (adapter,) if ctx.world_size > 1 else None
            at_accum_boundary = pipeline.is_accumulation_boundary

            ce_vals: list[float] = []
            kl_vals: list[float] = []

            for utt_idx, (_path, text, tokens, _num_windows) in enumerate(work):
                gt_single = llm_tokenizer(
                    [text],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=STAGE.max_text_tokens,
                )
                student_embeds, labels, attn, _cond = _build_student_inputs(
                    prompt_ids=prompt_ids,
                    audio_tokens=tokens,
                    gt_ids=gt_single.input_ids,
                    gt_attention_mask=gt_single.attention_mask,
                    text_embedder=text_embedder,
                    llm_device=llm_device,
                )
                ce_loss, kl_loss = _answer_ce_and_kl(
                    llm_model,
                    student_inputs_embeds=student_embeds,
                    student_labels=labels,
                    student_attention_mask=attn,
                    teacher_input_ids=gt_single.input_ids.to(llm_device),
                    teacher_attention_mask=gt_single.attention_mask.to(llm_device),
                    kl_temperature=kl_temperature,
                    device=llm_device,
                )
                utt_loss = (alpha * ce_loss + (1.0 - alpha) * kl_loss) / float(n_valid)
                ce_vals.append(float(ce_loss.detach().item()))
                kl_vals.append(float(kl_loss.detach().item()))

                is_last_utt = utt_idx == n_valid - 1
                _pipeline_accumulate_backward(
                    pipeline,
                    utt_loss,
                    no_sync_modules=no_sync_modules,
                    sync_grads=is_last_utt and at_accum_boundary,
                )

            ce_mean = sum(ce_vals) / len(ce_vals)
            kl_mean = sum(kl_vals) / len(kl_vals)
            total_loss_val = alpha * ce_mean + (1.0 - alpha) * kl_mean

            optimizer_stepped = _pipeline_end_microbatch(
                pipeline,
                list(adapter_module.parameters()),
            )

            m_total.update(total_loss_val)
            m_ce.update(ce_mean)
            m_kl.update(kl_mean)

            accum_suffix = ""
            if grad_accum_steps > 1:
                accum_done = grad_accum_steps if optimizer_stepped else pipeline.accum_step
                accum_suffix = f" | accum {accum_done}/{grad_accum_steps}"

            should_log = (step + 1) % LOG_EVERY_STEPS == 0 or (step + 1) == len(dataloader)
            current_lr = scheduler.get_last_lr()[0]
            if ctx.is_main and should_log:
                print(
                    f"Micro {step:4d}/{len(dataloader)} | opt {pipeline.global_step:5d} | "
                    f"Loss: {total_loss_val:.4f} | CE: {ce_mean:.4f} | "
                    f"KL: {kl_mean:.4f} | LR: {current_lr:.2e}{accum_suffix}"
                )

            if ctx.is_main and optimizer_stepped and should_log:
                logger.log(
                    {
                        "train/loss": total_loss_val,
                        "train/ce": ce_mean,
                        "train/kl": kl_mean,
                        "train/alpha": alpha,
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
                val_metrics, val_items = _validate_stage3(
                    adapter=adapter_module,
                    audio_extractor=audio,
                    llm_model=llm_model,
                    llm_tokenizer=llm_tokenizer,
                    text_embedder=text_embedder,
                    prompt_ids=prompt_ids,
                    val_root=VAL_ROOT,
                    train_device=train_device,
                    llm_device=llm_device,
                    max_utterances=STAGE.val_max_utterances,
                    max_windows_per_utt=DATA.max_windows_per_utt,
                    max_text_tokens=STAGE.max_text_tokens,
                    kl_temperature=kl_temperature,
                    alpha=alpha,
                    generation=val_generation,
                    global_step=pipeline.global_step,
                    log_every=STAGE.val_log_every,
                    predictions_path=val_predictions_path,
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
                        "ce": m_ce.mean,
                        "kl": m_kl.mean,
                    },
                )

            if llm_device.type == "cuda":
                with torch.cuda.device(llm_device):
                    torch.cuda.empty_cache()

        if ctx.is_main:
            epoch_metrics = {
                "loss": m_total.mean,
                "ce": m_ce.mean,
                "kl": m_kl.mean,
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
                stage=3,
                repo_id=HF_CKPT.repo_id,
                revision=HF_CKPT.revision,
                private=HF_CKPT.private,
                token=_hf_token,
                enabled=HF_CKPT.upload_enabled,
            )
            print(
                f"\nEpoch {epoch + 1} complete: loss={m_total.mean:.4f} "
                f"ce={m_ce.mean:.4f} kl={m_kl.mean:.4f}\nCheckpoint -> {SAVE_PATH}\n"
            )

    if ctx.is_main:
        print("Stage 3 ASR answer distillation complete!")
        logger.finish()
    cleanup_distributed()


if __name__ == "__main__":
    train()
