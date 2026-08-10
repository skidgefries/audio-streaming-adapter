from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F



def contrastive_infonce_loss(
    *,
    audio_tokens: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float = 0.07,
    return_diagnostics: bool = False
) -> torch.Tensor:
    """
    InfoNCE loss on pooled representations.

    audio_tokens: (B, T_a, D)
    text_embeddings: (B, T_t, D)
    """
    b = audio_tokens.shape[0]
    a_pooled = audio_tokens.float().mean(dim=1)
    t_pooled = text_embeddings.float().mean(dim=1)
    a_pooled = a_pooled - a_pooled.mean(dim=0, keepdim=True)
    t_pooled = t_pooled - t_pooled.mean(dim=0, keepdim=True)
    a = F.normalize(a_pooled, dim=-1)
    t = F.normalize(t_pooled, dim=-1)
    logits = float(temperature) * (a @ t.T)
    loss = F.cross_entropy(logits, torch.arange(b, device=logits.device))
    if return_diagnostics:
        with torch.no_grad():
            sim_matrix = a @ t.T  # (B, B) — cosine similarities
            pos_sim = sim_matrix.diagonal().mean().item()         # mean of diagonal
            neg_sim = (sim_matrix.sum() - sim_matrix.diagonal().sum()) / (b * b - b)  # mean off-diagonal
            neg_sim = neg_sim.item()
            pos_minus_neg = pos_sim - neg_sim
            audio_std = a.std(dim=0).mean().item()   
            text_std = t.std(dim=0).mean().item() 
        return loss, {"pos_sim": pos_sim, "neg_sim": neg_sim, "pos_minus_neg": pos_minus_neg, "audio_std": audio_std, "text_std": text_std}

    return loss


def init_contrastive_logit_scale(
    initial_temperature: float,
    *,
    device: torch.device | str,
) -> nn.Parameter:
    """
    Learnable log-scale for InfoNCE logits.

  Effective temperature applied to cosine similarities is ``exp(logit_scale)``,
    initialized to match ``contrastive_infonce_loss(..., temperature=initial_temperature)``.
    """
    return nn.Parameter(
        torch.tensor(math.log(float(1.0/initial_temperature)), device=device, dtype=torch.float32)
    )


def contrastive_infonce_loss_learnable_temperature(
    *,
    audio_tokens: torch.Tensor,
    text_embeddings: torch.Tensor,
    logit_scale: nn.Parameter | torch.Tensor,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """
    InfoNCE loss on pooled representations with learnable logit scale.

    audio_tokens: (B, T_a, D)
    text_embeddings: (B, T_t, D)
    logit_scale: learnable scalar; logits use ``logit_scale.exp() * (a @ t.T)``
    """
    b = audio_tokens.shape[0]
    a_pooled_mean = audio_tokens.float().mean(dim=1)
    t_pooled_mean = text_embeddings.float().mean(dim=1)
    # a_pooled_std = a_pooled_mean - a_pooled_mean.mean(dim=0, keepdim=True)
    # t_pooled_std = t_pooled_mean - t_pooled_mean.mean(dim=0, keepdim=True)
    a_pooled = a_pooled_mean - a_pooled_mean.mean(dim=0, keepdim=True)
    t_pooled = t_pooled_mean - t_pooled_mean.mean(dim=0, keepdim=True)
    # a_pooled = torch.cat([a_pooled_mean, a_pooled_std], dim=-1)
    # t_pooled = torch.cat([t_pooled_mean, t_pooled_std], dim=-1)
    a = F.normalize(a_pooled, dim=-1)
    t = F.normalize(t_pooled, dim=-1)
    scale = logit_scale.exp()
    logits = scale * (a @ t.T)
    # loss = F.cross_entropy(logits, torch.arange(b, device=logits.device))
    loss = F.cross_entropy(logits, torch.arange(b, device=logits.device))

    if return_diagnostics:
        with torch.no_grad():
            sim_matrix = a @ t.T  # (B, B) — cosine similarities
            pos_sim = sim_matrix.diagonal().mean().item()
            neg_sim = (sim_matrix.sum() - sim_matrix.diagonal().sum()) / (b * b - b)
            neg_sim = neg_sim.item()
            pos_minus_neg = pos_sim - neg_sim
            audio_std = a.std(dim=0).mean().item()
            text_std = t.std(dim=0).mean().item()
        return loss, {
            "pos_sim": pos_sim,
            "neg_sim": neg_sim,
            "pos_minus_neg": pos_minus_neg,
            "audio_std": audio_std,
            "text_std": text_std,
            "logit_scale": logit_scale.exp().item(),
        }

    return loss


def clap_loss_learnable_temperature(
    *,
    audio_tokens: torch.Tensor,
    text_embeddings: torch.Tensor,
    logit_scale: nn.Parameter | torch.Tensor,
    return_diagnostics: bool = False,
) -> torch.Tensor:
    """
    Symmetric CLAP contrastive loss over an audio-text similarity matrix.

    For batch size N with joint embeddings Ea, Et in R^{N x d}:
      C = tau * (Et @ Ea^T)
      L = 0.5 * (l_text(C) + l_audio(C))
    where l_k averages log diag(softmax(C)) along the text and audio axes.

    audio_tokens: (B, T_a, D) or (B, D)
    text_embeddings: (B, T_t, D) or (B, D)
    initial_temperature: tau scaling factor for logits
    """
    if audio_tokens.ndim == 3:
        a_pooled = audio_tokens.float().mean(dim=1)
    else:
        a_pooled = audio_tokens.float()
    if text_embeddings.ndim == 3:
        t_pooled = text_embeddings.float().mean(dim=1)
    else:
        t_pooled = text_embeddings.float()

    a_pooled = a_pooled - a_pooled.mean(dim=0, keepdim=True)
    t_pooled = t_pooled - t_pooled.mean(dim=0, keepdim=True)

    a_pooled = F.normalize(a_pooled, dim=-1)
    t_pooled = F.normalize(t_pooled, dim=-1)

    loss_device = logit_scale.device
    a_pooled = a_pooled.to(loss_device)
    t_pooled = t_pooled.to(loss_device)

    n = a_pooled.shape[0]
    scale = logit_scale.exp()
    c = scale * (t_pooled @ a_pooled.T)
    labels = torch.arange(n, device=c.device)
    loss = 0.5 * (F.cross_entropy(c, labels) + F.cross_entropy(c.T, labels))
    # print(f"CE loss: {loss.item()}")

    if return_diagnostics:
        with torch.no_grad():
            sim_matrix = t_pooled @ a_pooled.T
            pos_sim = sim_matrix.diagonal().mean().item()
            neg_sim = (sim_matrix.sum() - sim_matrix.diagonal().sum()) / (n * n - n)
            neg_sim = neg_sim.item()
            pos_minus_neg = pos_sim - neg_sim
            audio_std = a_pooled.std(dim=0).mean().item()
            text_std = t_pooled.std(dim=0).mean().item()
        return loss, {
            "pos_sim": pos_sim,
            "neg_sim": neg_sim,
            "pos_minus_neg": pos_minus_neg,
            "audio_std": audio_std,
            "text_std": text_std,
            "logit_scale": logit_scale.exp().item(),
        }

    return loss



def clap_loss(
    *,
    audio_tokens: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float,
    return_diagnostics: bool = False,
) -> torch.Tensor:
    """
    Symmetric CLAP contrastive loss over an audio-text similarity matrix.

    For batch size N with joint embeddings Ea, Et in R^{N x d}:
      C = tau * (Et @ Ea^T)
      L = 0.5 * (l_text(C) + l_audio(C))
    where l_k averages log diag(softmax(C)) along the text and audio axes.

    audio_tokens: (B, T_a, D) or (B, D)
    text_embeddings: (B, T_t, D) or (B, D)
    temperature: tau scaling factor for logits
    """
    if audio_tokens.ndim == 3:
        ea = audio_tokens.float().mean(dim=1)
    else:
        ea = audio_tokens.float()
    if text_embeddings.ndim == 3:
        et = text_embeddings.float().mean(dim=1)
    else:
        et = text_embeddings.float()

    ea = ea - ea.mean(dim=0, keepdim=True)
    et = et - et.mean(dim=0, keepdim=True)

    ea = F.normalize(ea, dim=-1)
    et = F.normalize(et, dim=-1)

    n = ea.shape[0]
    c = float(temperature) * (et @ ea.T)
    labels = torch.arange(n, device=c.device)
    loss = 0.5 * (F.cross_entropy(c, labels) + F.cross_entropy(c.T, labels))
    # print(f"CE loss: {loss.item()}")

    if return_diagnostics:
        with torch.no_grad():
            sim_matrix = ea @ et.T
            pos_sim = sim_matrix.diagonal().mean().item()
            neg_sim = (sim_matrix.sum() - sim_matrix.diagonal().sum()) / (n * n - n)
            neg_sim = neg_sim.item()
            pos_minus_neg = pos_sim - neg_sim
            audio_std = ea.std(dim=0).mean().item()
            text_std = et.std(dim=0).mean().item()
        return loss, {
            "pos_sim": pos_sim,
            "neg_sim": neg_sim,
            "pos_minus_neg": pos_minus_neg,
            "audio_std": audio_std,
            "text_std": text_std,
        }

    return loss


def init_sigmoid_logit_bias(
    *,
    device: torch.device | str,
    initial_bias: float = -10.0,
) -> nn.Parameter:
    """
    Learnable bias for SigLIP-style sigmoid contrastive loss.

    Initialized to ``initial_bias`` (default -10) so early training is not
    dominated by the many negative pairs in each batch.
    """
    return nn.Parameter(
        torch.tensor(float(initial_bias), device=device, dtype=torch.float32)
    )


def sigmoid_loss_learnable_temperature(
    *,
    audio_tokens: torch.Tensor,
    text_embeddings: torch.Tensor,
    logit_scale: nn.Parameter | torch.Tensor,
    logit_bias: nn.Parameter | torch.Tensor | None = None,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """
    SigLIP-style sigmoid loss over pooled audio-text similarities.

    Each pair (i, j) is a binary match/non-match: +1 on the diagonal,
    -1 off-diagonal. For batch size B and cosine similarity matrix S:

      logits = exp(logit_scale) * S (+ logit_bias if provided)
      loss = mean_i sum_j -log sigmoid(z_ij * logits_ij)

    audio_tokens: (B, T_a, D)
    text_embeddings: (B, T_t, D)
    logit_scale: learnable scalar; logits use ``logit_scale.exp() * (a @ t.T)``
    logit_bias: optional learnable scalar bias added to all logits
    """
    b = audio_tokens.shape[0]
    a_pooled_mean = audio_tokens.float().mean(dim=1)
    t_pooled_mean = text_embeddings.float().mean(dim=1)
    a_pooled = a_pooled_mean - a_pooled_mean.mean(dim=0, keepdim=True)
    t_pooled = t_pooled_mean - t_pooled_mean.mean(dim=0, keepdim=True)
    a = F.normalize(a_pooled, dim=-1)
    t = F.normalize(t_pooled, dim=-1)

    scale = logit_scale.exp()
    logits = scale * (a @ t.T)
    if logit_bias is not None:
        logits = logits + logit_bias

    eye = torch.eye(b, device=logits.device, dtype=logits.dtype)
    labels = -torch.ones_like(logits) + 2 * eye
    nll = -F.logsigmoid(labels * logits).sum(dim=-1)
    loss = nll.mean()

    if return_diagnostics:
        with torch.no_grad():
            sim_matrix = a @ t.T
            pos_sim = sim_matrix.diagonal().mean().item()
            neg_sim = (sim_matrix.sum() - sim_matrix.diagonal().sum()) / (b * b - b)
            neg_sim = neg_sim.item()
            pos_minus_neg = pos_sim - neg_sim
            audio_std = a.std(dim=0).mean().item()
            text_std = t.std(dim=0).mean().item()
            diag: dict[str, float] = {
                "pos_sim": pos_sim,
                "neg_sim": neg_sim,
                "pos_minus_neg": pos_minus_neg,
                "audio_std": audio_std,
                "text_std": text_std,
                "logit_scale": logit_scale.exp().item(),
            }
            if logit_bias is not None:
                diag["logit_bias"] = float(logit_bias.item())
            return loss, diag

    return loss


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

