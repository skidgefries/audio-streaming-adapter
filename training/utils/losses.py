from __future__ import annotations

import torch
import torch.nn.functional as F


def contrastive_infonce_loss(
    *,
    audio_tokens: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    InfoNCE loss on pooled representations.

    audio_tokens: (B, T_a, D)
    text_embeddings: (B, T_t, D)
    """
    b = audio_tokens.shape[0]
    a = F.normalize(audio_tokens.float().mean(dim=1), dim=-1)
    t = F.normalize(text_embeddings.float().mean(dim=1), dim=-1)
    logits = (a @ t.T) / float(temperature)
    return F.cross_entropy(logits, torch.arange(b, device=logits.device))


def kl_distill_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    KL(teacher || student) distillation, scaled by T^2.

    Logits may be (B, V) or (B, L, V). For sequence logits, pass mask (B, L) with 1
    for valid positions; when mask is None, all positions are averaged equally.
    """
    student_log_prob = F.log_softmax(student_logits / float(temperature), dim=-1)
    teacher_prob = F.softmax(teacher_logits / float(temperature), dim=-1)
    kl = F.kl_div(student_log_prob, teacher_prob, reduction="none", log_target=False)
    kl = kl.sum(dim=-1)
    if student_logits.ndim == 2:
        kl_mean = kl.mean()
    elif mask is not None:
        kl = kl * mask
        denom = mask.sum().clamp_min(1.0)
        kl_mean = kl.sum() / denom
    else:
        kl_mean = kl.mean()
    return (float(temperature) ** 2) * kl_mean

def prefix_consistency_loss(
    *,
    partial_logits: torch.Tensor,
    full_logits: torch.Tensor,
    prefix_length: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    partial_prefix = partial_logits[:, :prefix_length, :]
    full_prefix = full_logits[:, :prefix_length, :]
    partial_log_prob = F.log_softmax(partial_prefix, dim=-1)
    full_prob = F.softmax(full_prefix, dim=-1)
    loss = F.kl_div(partial_log_prob, full_prob, reduction="none", log_target=False)
    loss = loss.sum(dim=-1)
    if mask is not None:
        prefix_mask = mask[:, :prefix_length]
        loss = loss * prefix_mask
        loss = loss.sum() / prefix_mask.sum().clamp_min(1.0)
    else:
        loss = loss.mean()
    return loss


def revision_penalty_loss(
    generation_history: list[torch.Tensor],
    *,
    mask: torch.Tensor | None = None,
    device: torch.device | str,
) -> torch.Tensor:
    if len(generation_history) < 2:
        return torch.tensor(0.0, device=device)
    total = torch.tensor(0.0, device=device)
    steps = len(generation_history) - 1
    for t in range(steps):
        prob_t = F.softmax(generation_history[t], dim=-1)
        prob_t1 = F.softmax(generation_history[t + 1], dim=-1)
        diff = (prob_t - prob_t1) ** 2
        loss = diff.sum(dim=-1)
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / mask.sum().clamp_min(1.0)
        else:
            loss = loss.mean()
        total = total + loss
    return total / float(steps)

