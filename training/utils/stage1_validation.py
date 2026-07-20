"""Stage 1 contrastive validation on LibriSpeech dev-clean."""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from src.dataset import LibriSpeechPairs, load_mono_waveform_16k
from training.utils.losses import (
    clap_loss_learnable_temperature,
    contrastive_infonce_loss,
    contrastive_infonce_loss_learnable_temperature,
    sigmoid_loss_learnable_temperature,
)
from training.utils.metrics import RunningMean
from training.utils.devices import llm_input_device


def compute_retrieval_recall(
    audio_bank: torch.Tensor,
    text_bank: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Audio→text recall@k (percent) from centered, L2-normalized banks."""
    metrics = compute_retrieval_metrics(
        audio_bank,
        text_bank,
        ks=ks,
        higher_is_better=True,
    )
    return {f"recall_at_{k}": metrics[f"R@{k}"] for k in ks}


def _correct_match_ranks(scores: torch.Tensor, *, higher_is_better: bool) -> torch.Tensor:
    """1-indexed rank of the diagonal match in each row."""
    n = scores.shape[0]
    sorted_idx = scores.argsort(dim=1, descending=higher_is_better)
    targets = torch.arange(n, device=scores.device).unsqueeze(1)
    rank0 = (sorted_idx == targets).float().argmax(dim=1)
    return rank0.to(torch.float32) + 1.0


def compute_alignment(audio_bank: torch.Tensor, text_bank: torch.Tensor) -> float:
    """
    Positive-pair alignment (Wang & Isola): mean squared L2 distance on the hypersphere.

    For unit-norm vectors, ``||a - t||^2 = 2 - 2 cos(a, t)``. Lower is better.
    """
    pos_sim = (audio_bank * text_bank).sum(dim=1)
    return (2.0 - 2.0 * pos_sim).mean().item()


def compute_uniformity(vectors: torch.Tensor, *, t: float = 2.0) -> float:
    """
    Uniformity on the hypersphere (Wang & Isola): ``log mean exp(-t ||z_i - z_j||^2)``.

    Lower is better (more uniform spread). ``vectors`` should already be L2-normalized.
    """
    n = vectors.shape[0]
    if n < 2:
        return 0.0
    sim = vectors @ vectors.T
    sq_dist = (2.0 - 2.0 * sim).clamp_min(0.0)
    mask = ~torch.eye(n, dtype=torch.bool, device=vectors.device)
    return torch.log(torch.exp(-t * sq_dist[mask]).mean()).item()


def compute_infonce_loss(
    audio_bank: torch.Tensor,
    text_bank: torch.Tensor,
    *,
    logit_scale: float | None = None,
    temperature: float = 0.07,
) -> float:
    """Batch InfoNCE (softmax cross-entropy) on cosine similarity logits."""
    n = audio_bank.shape[0]
    scale = math.exp(logit_scale) if logit_scale is not None else (1.0 / temperature)
    logits = scale * (audio_bank @ text_bank.T)
    labels = torch.arange(n, device=logits.device)
    return F.cross_entropy(logits, labels).item()


def compute_retrieval_metrics(
    audio_bank: torch.Tensor,
    text_bank: torch.Tensor,
    *,
    ks: tuple[int, ...] = (1, 5, 10),
    higher_is_better: bool = True,
    logit_scale: float | None = None,
    temperature: float = 0.07,
    include_alignment: bool = True,
    include_uniformity: bool = True,
) -> dict[str, float]:
    """
    Full audio→text retrieval metrics from centered, L2-normalized embedding banks.

    Returns recall@k (%), MRR (%), median/mean rank (1-indexed), InfoNCE loss,
    and optional alignment / uniformity diagnostics.
    """
    n = audio_bank.shape[0]
    scores = audio_bank @ text_bank.T
    ranks = _correct_match_ranks(scores, higher_is_better=higher_is_better)

    results: dict[str, float] = {}
    sorted_idx = scores.argsort(dim=1, descending=higher_is_better)
    for k in ks:
        top_k = sorted_idx[:, :k]
        correct = torch.arange(n, device=scores.device).unsqueeze(1)
        hits = (top_k == correct).any(dim=1).float()
        results[f"R@{k}"] = hits.mean().item() * 100.0

    results["MRR"] = (1.0 / ranks).mean().item() * 100.0
    results["median_rank"] = ranks.median().item()
    results["mean_rank"] = ranks.mean().item()
    results["infonce_loss"] = compute_infonce_loss(
        audio_bank,
        text_bank,
        logit_scale=logit_scale,
        temperature=temperature,
    )

    if include_alignment:
        results["alignment"] = compute_alignment(audio_bank, text_bank)
    if include_uniformity:
        combined = torch.cat([audio_bank, text_bank], dim=0)
        results["uniformity"] = compute_uniformity(combined)

    return results


def compute_retrieval_metrics_from_cost_matrix(
    cost_matrix: torch.Tensor,
    *,
    ks: tuple[int, ...] = (1, 5, 10),
    include_alignment: bool = False,
    include_uniformity: bool = False,
) -> dict[str, float]:
    """
    Retrieval metrics when scores are costs (lower is better), e.g. LM NLL matrix.

    ``nll`` is the mean matched-pair (diagonal) cost; ``nll_neg_mean`` is the mean
    off-diagonal cost.
    """
    n = cost_matrix.shape[0]
    ranks = _correct_match_ranks(cost_matrix, higher_is_better=False)
    sorted_idx = cost_matrix.argsort(dim=1, descending=False)

    results: dict[str, float] = {}
    for k in ks:
        top_k = sorted_idx[:, :k]
        correct = torch.arange(n, device=cost_matrix.device).unsqueeze(1)
        hits = (top_k == correct).any(dim=1).float()
        results[f"R@{k}"] = hits.mean().item() * 100.0

    results["MRR"] = (1.0 / ranks).mean().item() * 100.0
    results["median_rank"] = ranks.median().item()
    results["mean_rank"] = ranks.mean().item()

    diag = cost_matrix.diagonal()
    results["nll"] = diag.mean().item()
    if n > 1:
        off_diag = cost_matrix[~torch.eye(n, dtype=torch.bool, device=cost_matrix.device)]
        results["nll_neg_mean"] = off_diag.mean().item()
    else:
        results["nll_neg_mean"] = float("nan")

    if include_alignment:
        results["alignment"] = float("nan")
    if include_uniformity:
        results["uniformity"] = float("nan")

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
    logit_scale: torch.nn.Parameter | torch.Tensor,
    lambda_stability: float,
    max_utterances: int | None,
    pad_tokens_fn: Callable[[list[torch.Tensor]], torch.Tensor],
    maybe_autocast_fn: Callable[[str], AbstractContextManager],
    epoch: int,
    logit_bias: torch.nn.Parameter | torch.Tensor | None = None,
) -> dict[str, float]:
    """
    Run dev-clean validation: contrastive loss + stability + retrieval@k.

    Uses the same forward path and loss as ``adapter_contrastive_trainer``.
    When ``logit_bias`` is provided, uses SigLIP-style sigmoid loss instead of InfoNCE.
    """
    was_training = adapter.training
    adapter.eval()
    llm_embed_device = llm_input_device(text_embedder)
    train_device = torch.device(device)

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
    m_temperature = RunningMean()
    m_logit_bias = RunningMean()
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
        ).to(llm_embed_device)

        label_embeds = text_embedder(text_tokens.input_ids).float().to(train_device)

        utterances: list[torch.Tensor] = []
        stab_sum = torch.zeros((), device=train_device, dtype=torch.float32)

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

            if logit_bias is None:
                align_loss, diag = contrastive_infonce_loss_learnable_temperature(
                    audio_tokens=audio_tokens,
                    text_embeddings=label_embeds,
                    logit_scale=logit_scale,
                    return_diagnostics=True,
                )
            else:
                align_loss, diag = sigmoid_loss_learnable_temperature(
                    audio_tokens=audio_tokens,
                    text_embeddings=label_embeds,
                    logit_scale=logit_scale,
                    logit_bias=logit_bias,
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
        m_temperature.update(diag["logit_scale"])
        if "logit_bias" in diag:
            m_logit_bias.update(diag["logit_bias"])

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
        "temperature": m_temperature.mean,
    }
    if m_logit_bias.count > 0:
        metrics["logit_bias"] = m_logit_bias.mean

    print(
        f"  Val loss={metrics['loss']:.4f} align={metrics['align']:.4f} "
        f"stab={metrics['stability']:.4f} | "
        f"R@1={metrics['recall_at_1']:.2f}% R@5={metrics['recall_at_5']:.2f}% "
        f"R@10={metrics['recall_at_10']:.2f}%"
    )
    return metrics

@torch.no_grad()
def validate_stage1_clap(
    *,
    adapter: torch.nn.Module,
    audio_extractor: Any,
    llm_tokenizer: Any,
    text_embedder: torch.nn.Module,
    val_root: str,
    device: str,
    batch_size: int,
    num_workers: int,
    logit_scale: torch.nn.Parameter | torch.Tensor,
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
    llm_embed_device = llm_input_device(text_embedder)
    train_device = torch.device(device)

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
    m_temperature = RunningMean()
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
        ).to(llm_embed_device)

        label_embeds = text_embedder(text_tokens.input_ids).float().to(train_device)

        utterances: list[torch.Tensor] = []
        stab_sum = torch.zeros((), device=train_device, dtype=torch.float32)

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

            align_loss, diag = clap_loss_learnable_temperature(
                audio_tokens=audio_tokens,
                text_embeddings=label_embeds,
                logit_scale=logit_scale,
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
        m_temperature.update(diag["logit_scale"])

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
        "temperature": m_temperature.mean,
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
        "val/temperature": metrics["temperature"],
        **({"val/logit_bias": metrics["logit_bias"]} if "logit_bias" in metrics else {}),
    }
