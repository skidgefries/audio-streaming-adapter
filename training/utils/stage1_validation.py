"""Stage 1 contrastive validation on LibriSpeech dev-clean."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from src.dataset import LibriSpeechPairs, load_mono_waveform_16k
from training.utils.losses import contrastive_infonce_loss
from training.utils.metrics import RunningMean


def compute_retrieval_recall(
    audio_bank: torch.Tensor,
    text_bank: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Audio→text recall@k (percent) from centered, L2-normalized banks."""
    n = audio_bank.shape[0]
    sim_matrix = audio_bank @ text_bank.T
    ranked = sim_matrix.argsort(dim=1, descending=True)
    results: dict[str, float] = {}
    for k in ks:
        top_k = ranked[:, :k]
        correct = torch.arange(n).unsqueeze(1)
        hits = (top_k == correct).any(dim=1).float()
        results[f"recall_at_{k}"] = hits.mean().item() * 100.0
    return results


@torch.no_grad()
def validate_stage1_contrastive(
    *,
    adapter: torch.nn.Module,
    audio_extractor: Any,
    llm_tokenizer: Any,
    text_embedder: torch.nn.Module,
    val_root: str,
    device: str,
    batch_size: int,
    num_workers: int,
    temperature: float,
    lambda_stability: float,
    max_utterances: int | None,
    pad_tokens_fn: Callable[[list[torch.Tensor]], torch.Tensor],
    maybe_autocast_fn: Callable[[str], AbstractContextManager],
    epoch: int,
) -> dict[str, float]:
    """
    Run dev-clean validation: contrastive loss + stability + retrieval@k.

    Uses the same forward path and loss as ``adapter_contrastive_trainer``.
    """
    was_training = adapter.training
    adapter.eval()

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
    m_align = RunningMean()
    m_stab = RunningMean()
    m_pos_sim = RunningMean()
    m_neg_sim = RunningMean()
    m_pos_minus_neg = RunningMean()
    m_audio_std = RunningMean()
    m_text_std = RunningMean()

    audio_vecs: list[torch.Tensor] = []
    text_vecs: list[torch.Tensor] = []

    print(f"\n  Validating on dev-clean ({n_val} utterances, {len(val_loader)} batches)...")

    for batch_idx, batch in enumerate(val_loader):
        audio_paths, transcriptions = batch
        batch_texts = list(transcriptions)

        text_tokens = llm_tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        label_embeds = text_embedder(text_tokens.input_ids).float()

        utterances: list[torch.Tensor] = []
        stab_sum = torch.zeros((), device=device, dtype=torch.float32)

        with maybe_autocast_fn(device):
            for p in audio_paths:
                wave = load_mono_waveform_16k(p)
                windows = audio_extractor.waveform_to_windows(wave)
                adapter.reset_streaming_state()
                chunks = []
                for w in windows:
                    out = adapter.forward_window(w)
                    chunks.append(out["tokens"])
                    stab_sum = stab_sum + out["stability_loss"].float()
                if chunks:
                    utterances.append(torch.cat(chunks, dim=1))

            if not utterances:
                continue

            audio_tokens = pad_tokens_fn(utterances)

            align_loss, diag = contrastive_infonce_loss(
                audio_tokens=audio_tokens,
                text_embeddings=label_embeds,
                temperature=temperature,
                return_diagnostics=True,
            )
            stability_loss = stab_sum / float(len(audio_paths))
            total_loss = align_loss + lambda_stability * stability_loss

        m_total.update(total_loss.item())
        m_align.update(align_loss.item())
        m_stab.update(stability_loss.item())
        m_pos_sim.update(diag["pos_sim"])
        m_neg_sim.update(diag["neg_sim"])
        m_pos_minus_neg.update(diag["pos_minus_neg"])
        m_audio_std.update(diag["audio_std"])
        m_text_std.update(diag["text_std"])

        # Pooled embeddings for retrieval (same pooling as contrastive loss)
        a_pooled = audio_tokens.float().mean(dim=1).cpu()
        t_pooled = label_embeds.float().mean(dim=1).cpu()
        audio_vecs.append(a_pooled)
        text_vecs.append(t_pooled)

        if batch_idx % 20 == 0 and batch_idx > 0:
            print(f"    val batch {batch_idx}/{len(val_loader)}")

    if not audio_vecs:
        if was_training:
            adapter.train()
        raise RuntimeError(f"No validation samples found under {val_root}")

    audio_bank = torch.cat(audio_vecs, dim=0)
    text_bank = torch.cat(text_vecs, dim=0)
    audio_bank = audio_bank - audio_bank.mean(dim=0, keepdim=True)
    text_bank = text_bank - text_bank.mean(dim=0, keepdim=True)
    audio_bank = F.normalize(audio_bank, dim=-1)
    text_bank = F.normalize(text_bank, dim=-1)
    recall = compute_retrieval_recall(audio_bank, text_bank)

    if was_training:
        adapter.train()

    metrics: dict[str, float] = {
        "loss": m_total.mean,
        "align": m_align.mean,
        "stability": m_stab.mean,
        "pos_sim": m_pos_sim.mean,
        "neg_sim": m_neg_sim.mean,
        "pos_minus_neg": m_pos_minus_neg.mean,
        "audio_std": m_audio_std.mean,
        "text_std": m_text_std.mean,
        **recall,
        "epoch": float(epoch + 1),
        "num_utterances": float(n_val),
    }

    print(
        f"  Val loss={metrics['loss']:.4f} align={metrics['align']:.4f} "
        f"stab={metrics['stability']:.4f} | "
        f"R@1={metrics['recall_at_1']:.2f}% R@5={metrics['recall_at_5']:.2f}% "
        f"R@10={metrics['recall_at_10']:.2f}%"
    )
    return metrics


def wandb_val_log_dict(metrics: dict[str, float]) -> dict[str, float]:
    """Map validation metrics to Wandb keys (``val/...``)."""
    return {
        "val/loss": metrics["loss"],
        "val/align": metrics["align"],
        "val/stability": metrics["stability"],
        "val/pos_sim": metrics["pos_sim"],
        "val/neg_sim": metrics["neg_sim"],
        "val/pos_minus_neg": metrics["pos_minus_neg"],
        "val/audio_std": metrics["audio_std"],
        "val/text_std": metrics["text_std"],
        "val/recall_at_1": metrics["recall_at_1"],
        "val/recall_at_5": metrics["recall_at_5"],
        "val/recall_at_10": metrics["recall_at_10"],
        "val/num_utterances": metrics["num_utterances"],
    }
