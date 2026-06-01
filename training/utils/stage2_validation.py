"""Stage 2 ASR distillation validation on LibriSpeech dev-clean."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.dataset import LibriSpeechPairs, load_mono_waveform_16k
from training.utils.gate_training import endpoint_label_for_timestep, make_silence_trackers
from training.utils.losses import contrastive_infonce_loss
from training.utils.metrics import RunningMean


def _pad_audio_tokens(utterances: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(t.shape[1] for t in utterances)
    padded = []
    for t in utterances:
        if t.shape[1] < max_len:
            pad = torch.zeros(1, max_len - t.shape[1], t.shape[2], device=t.device, dtype=t.dtype)
            t = torch.cat([t, pad], dim=1)
        padded.append(t)
    return torch.cat(padded, dim=0)


@torch.no_grad()
def validate_stage2_asr(
    *,
    adapter: torch.nn.Module,
    gate: torch.nn.Module,
    audio_extractor: Any,
    llm_model: torch.nn.Module,
    llm_tokenizer: Any,
    text_embedder: torch.nn.Module,
    asr_forward_loss_fn: Callable[..., torch.Tensor],
    val_root: str,
    train_device: torch.device,
    llm_device: torch.device,
    batch_size: int,
    num_workers: int,
    max_utterances: int | None,
    max_windows_per_utt: int | None,
    max_text_tokens: int,
    asr_micro_batch_size: int,
    lambda_align: float,
    lambda_stability: float,
    lambda_rate: float,
    lambda_gate: float,
    temperature: float,
    maybe_autocast_fn: Callable[[torch.device], AbstractContextManager],
    torch_dtype: torch.dtype,
    global_step: int,
) -> dict[str, float]:
    """
    Run dev-clean validation with the same losses as ``adapter_asr_trainer`` (no backward).
    """
    adapter_was_training = adapter.training
    gate_was_training = gate.training
    adapter.eval()
    gate.eval()

    val_full = LibriSpeechPairs(val_root)
    n_val = len(val_full) if max_utterances is None else min(max_utterances, len(val_full))
    val_dataset: Dataset = val_full if n_val == len(val_full) else Subset(val_full, range(n_val))
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    m_total = RunningMean()
    m_asr = RunningMean()
    m_align = RunningMean()
    m_stab = RunningMean()
    m_sparse = RunningMean()
    m_rate = RunningMean()
    m_gate = RunningMean()
    m_pos_sim = RunningMean()
    m_neg_sim = RunningMean()
    m_pos_minus_neg = RunningMean()

    print(
        f"\n  [val step {global_step}] dev-clean ({n_val} utterances, "
        f"{len(val_loader)} batches)..."
    )

    for batch_idx, batch in enumerate(val_loader):
        audio_paths, transcriptions = batch
        batch_texts = list(transcriptions)

        gt_tokens = llm_tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_text_tokens,
        ).to(train_device)
        gt_ids = gt_tokens.input_ids
        gt_attention_mask = gt_tokens.attention_mask
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
            windows = audio_extractor.waveform_to_windows(wave)
            if max_windows_per_utt is not None:
                windows = windows[:max_windows_per_utt]

            adapter.reset_streaming_state()
            silence_tracker, learned_silence_tracker = make_silence_trackers(gate)
            utterance_tokens: list[torch.Tensor] = []
            for t, window in enumerate(windows):
                result = adapter.forward_window(window.to(device=train_device, dtype=torch_dtype))
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
                gate_result = gate(
                    accumulated,
                    t,
                    len(windows),
                    endpoint_label=endpoint,
                    silence_tracker=silence_tracker,
                    learned_silence_tracker=learned_silence_tracker,
                    window_tokens=utterance_tokens[t],
                )
                total_gate_loss = total_gate_loss + gate_result["gate_loss"].float()
                gate_calls += 1

            if utterance_tokens:
                audio_tokens_list.append(torch.cat(utterance_tokens, dim=1))
                num_windows_list.append(len(windows))

        if not audio_tokens_list:
            continue

        audio_tokens = _pad_audio_tokens(audio_tokens_list)

        bos_token_id = (
            llm_tokenizer.bos_token_id
            if llm_tokenizer.bos_token_id is not None
            else llm_tokenizer.eos_token_id
        )
        bos_embed = text_embedder(
            torch.tensor([[bos_token_id]], device=llm_device).expand(audio_tokens.shape[0], -1)
        )
        inputs_embeds = torch.cat([audio_tokens.to(llm_device), bos_embed], dim=1)

        batch_n = audio_tokens.shape[0]
        audio_len = audio_tokens.shape[1]
        pre_text_labels = torch.full((batch_n, audio_len + 1), -100, dtype=torch.long, device=llm_device)
        gt_shifted = gt_ids[:, 1:].to(llm_device)
        labels = torch.cat([pre_text_labels, gt_shifted], dim=1)
        inputs_embeds = torch.cat([inputs_embeds, text_embedder(gt_shifted)], dim=1)

        pre_text_mask = torch.ones((batch_n, audio_len + 1), device=llm_device)
        llm_attention_mask = torch.cat([pre_text_mask, gt_attention_mask[:, 1:].to(llm_device)], dim=1)

        with maybe_autocast_fn(train_device):
            asr_loss = asr_forward_loss_fn(
                inputs_embeds=inputs_embeds,
                labels=labels,
                attention_mask=llm_attention_mask,
                micro_batch_size=asr_micro_batch_size,
                device=train_device,
            )
            align_loss, diag = contrastive_infonce_loss(
                audio_tokens=audio_tokens.float(),
                text_embeddings=gt_embeds.float(),
                temperature=temperature,
                return_diagnostics=True,
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
            + lambda_align * align_loss
            + lambda_stability * stability_loss
            + lambda_rate * rate_loss
            + lambda_gate * gate_loss_mean
        )

        m_total.update(total_loss.item())
        m_asr.update(asr_loss.item())
        m_align.update(align_loss.item())
        m_stab.update(float(stability_loss.item()))
        m_sparse.update(sparse_loss.item())
        m_rate.update(rate_loss.item())
        m_gate.update(float(gate_loss_mean.item()))
        m_pos_sim.update(diag["pos_sim"])
        m_neg_sim.update(diag["neg_sim"])
        m_pos_minus_neg.update(diag["pos_minus_neg"])

        if batch_idx % 20 == 0 and batch_idx > 0:
            print(f"    val batch {batch_idx}/{len(val_loader)}")

    if adapter_was_training:
        adapter.train()
    if gate_was_training:
        gate.train()

    if m_total.count == 0:
        raise RuntimeError(f"No validation samples found under {val_root}")

    metrics: dict[str, float] = {
        "loss": m_total.mean,
        "asr": m_asr.mean,
        "align": m_align.mean,
        "stability": m_stab.mean,
        "sparse_metric": m_sparse.mean,
        "rate": m_rate.mean,
        "gate": m_gate.mean,
        "pos_sim": m_pos_sim.mean,
        "neg_sim": m_neg_sim.mean,
        "pos_minus_neg": m_pos_minus_neg.mean,
        "num_utterances": float(n_val),
    }

    print(
        f"  Val loss={metrics['loss']:.4f} asr={metrics['asr']:.4f} align={metrics['align']:.4f} "
        f"stab={metrics['stability']:.4f} gate={metrics['gate']:.4f}"
    )
    return metrics


def wandb_val_log_dict(metrics: dict[str, float]) -> dict[str, float]:
    """Map validation metrics to Wandb keys (``val/...``), aligned with Stage 1 style."""
    return {
        "val/loss": metrics["loss"],
        "val/asr": metrics["asr"],
        "val/align": metrics["align"],
        "val/stability": metrics["stability"],
        "val/rate": metrics["rate"],
        "val/gate": metrics["gate"],
        "val/sparse_metric": metrics["sparse_metric"],
        "val/pos_sim": metrics["pos_sim"],
        "val/neg_sim": metrics["neg_sim"],
        "val/pos_minus_neg": metrics["pos_minus_neg"],
        "val/num_utterances": metrics["num_utterances"],
    }
