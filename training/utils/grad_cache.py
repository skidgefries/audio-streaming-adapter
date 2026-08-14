"""
Gradient caching for large-batch contrastive InfoNCE under micro-batch VRAM limits.

Encodes representations in small chunks without storing the full graph, runs InfoNCE
on the full macro-batch (true in-batch negatives), then re-encodes each micro-batch
with a representation-gradient surrogate so adapter grads match large-batch InfoNCE.

With ``torch.distributed`` (``WORLD_SIZE > 1``), local pools are all-gathered so InfoNCE
sees the global macro-batch across ranks; only local representation grads are applied.

Cache and recompute run under ``adapter.eval()`` so dropout masks match across passes
(the adapter has no BatchNorm; eval only disables dropout). Prior ``training`` mode is
restored before returning.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn

from training.utils.losses import contrastive_infonce_loss_learnable_temperature


def pool_audio_tokens_for_infonce(audio_tokens: torch.Tensor) -> torch.Tensor:
    """Match ``contrastive_infonce_loss*`` temporal pooling: ``(B, T, D) -> (B, D)``."""
    return audio_tokens.float().mean(dim=1)


def _dist_world() -> tuple[bool, int, int]:
    if not dist.is_available() or not dist.is_initialized():
        return False, 1, 0
    return True, dist.get_world_size(), dist.get_rank()


def all_gather_cat(local: torch.Tensor) -> torch.Tensor:
    """Concatenate ``local`` tensors from all ranks along dim 0 (no-op if single process)."""
    enabled, world_size, _rank = _dist_world()
    if not enabled or world_size == 1:
        return local
    parts = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(parts, local.contiguous())
    return torch.cat(parts, dim=0)


def all_reduce_mean_grads(params: Sequence[nn.Parameter]) -> None:
    """Average gradients across ranks (GradCache bypasses DDP autograd hooks)."""
    enabled, world_size, _rank = _dist_world()
    if not enabled or world_size == 1:
        return
    for p in params:
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)


def _param_grad_l2_norm(params: Sequence[nn.Parameter]) -> float:
    total_sq = 0.0
    for p in params:
        if p.grad is None:
            continue
        total_sq += float(p.grad.data.norm(2).item() ** 2)
    return total_sq**0.5


def _clone_param_grads(params: Sequence[nn.Parameter]) -> list[torch.Tensor | None]:
    return [None if p.grad is None else p.grad.detach().clone() for p in params]


def _add_param_grads(
    params: Sequence[nn.Parameter],
    saved: Sequence[torch.Tensor | None],
) -> None:
    for p, g in zip(params, saved):
        if g is None:
            continue
        if p.grad is None:
            p.grad = g.clone()
        else:
            p.grad.add_(g)


def infonce_audio_representation_grads(
    *,
    audio_pooled: torch.Tensor,
    text_embeddings: torch.Tensor,
    logit_scale: nn.Parameter,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Run full-macro InfoNCE on cached audio pools; return per-sample audio grads.

    ``audio_pooled`` should be detached ``(B, D)``. ``text_embeddings`` may be
    ``(B, T, D)`` or already pooled ``(B, D)``. Populates ``logit_scale.grad``.
    Returns ``(audio_grads, align_loss, diagnostics)``.
    """
    if audio_pooled.ndim != 2:
        raise ValueError(f"Expected audio_pooled (B, D), got {tuple(audio_pooled.shape)}")
    if text_embeddings.ndim == 2:
        text_tokens = text_embeddings.unsqueeze(1)
    elif text_embeddings.ndim == 3:
        text_tokens = text_embeddings
    else:
        raise ValueError(
            f"Expected text_embeddings (B, D) or (B, T, D), got {tuple(text_embeddings.shape)}"
        )
    cached = audio_pooled.detach().requires_grad_(True)
    align_loss, diag = contrastive_infonce_loss_learnable_temperature(
        audio_tokens=cached.unsqueeze(1),
        text_embeddings=text_tokens,
        logit_scale=logit_scale,
        return_diagnostics=True,
    )
    align_loss.backward()
    if cached.grad is None:
        raise RuntimeError("InfoNCE GradCache: missing audio representation gradients")
    return cached.grad.detach(), align_loss.detach(), diag


def grad_cache_infonce_backward(
    *,
    adapter: nn.Module,
    encode_micro: Callable[[Sequence[str]], tuple[torch.Tensor, torch.Tensor]],
    micro_path_chunks: Sequence[Sequence[str]],
    text_embeddings: torch.Tensor,
    logit_scale: nn.Parameter,
    lambda_stability: float,
    optimizer: torch.optim.Optimizer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Two-pass GradCache update for InfoNCE + mean stability.

    ``encode_micro(paths) -> (pooled (bs, D), stab_sum)`` must match training encode
    (same pooling as InfoNCE).

    Runs cache + recompute under ``adapter.eval()`` so dropout is disabled and the
    representation used for ``∂L/∂z`` matches the surrogate recompute (no BatchNorm
    in the adapter). Restores the prior ``adapter.training`` mode before returning.

    Align and stability are backpropped in separate recompute passes so diagnostics
    include per-term adapter grad norms (``align_grad_norm``, ``stab_grad_norm``).

    Under distributed training, pools and text are all-gathered so InfoNCE uses the
    global batch; surrogate backward uses only this rank's representation grads.
    Stability is normalized by the global batch size.

    Returns ``(total_loss, align_loss, stability_loss, diagnostics)`` as detached scalars
    for logging; gradients are accumulated on adapter + ``logit_scale``.
    """
    if not micro_path_chunks:
        raise ValueError("micro_path_chunks must be non-empty")
    local_n = sum(len(chunk) for chunk in micro_path_chunks)
    if local_n == 0:
        raise ValueError("macro batch has no utterances")
    if text_embeddings.shape[0] != local_n:
        raise ValueError(
            f"text_embeddings batch {text_embeddings.shape[0]} != local size {local_n}"
        )

    enabled, world_size, rank = _dist_world()
    adapter_params = [p for p in adapter.parameters() if p.requires_grad]
    was_training = adapter.training
    adapter.eval()
    try:
        optimizer.zero_grad(set_to_none=True)

        pooled_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for paths in micro_path_chunks:
                pooled, _stab = encode_micro(paths)
                if pooled.shape[0] != len(paths):
                    raise ValueError(
                        f"encode_micro returned {pooled.shape[0]} pools for {len(paths)} paths"
                    )
                pooled_chunks.append(pooled.detach())

        audio_pooled_local = torch.cat(pooled_chunks, dim=0)
        if torch.isnan(audio_pooled_local).any():
            raise ValueError("GradCache cache pass: audio_pooled contains NaN")

        if text_embeddings.ndim == 3:
            text_pooled_local = text_embeddings.float().mean(dim=1)
        elif text_embeddings.ndim == 2:
            text_pooled_local = text_embeddings.float()
        else:
            raise ValueError(
                f"Expected text_embeddings (B, D) or (B, T, D), got {tuple(text_embeddings.shape)}"
            )

        audio_pooled = all_gather_cat(audio_pooled_local)
        text_pooled = all_gather_cat(text_pooled_local)
        global_n = int(audio_pooled.shape[0])
        if global_n != local_n * world_size:
            raise RuntimeError(
                f"GradCache gather size {global_n} != local_n ({local_n}) * world_size ({world_size})"
            )

        audio_grads_global, align_loss, diag = infonce_audio_representation_grads(
            audio_pooled=audio_pooled,
            text_embeddings=text_pooled,
            logit_scale=logit_scale,
        )
        local_start = rank * local_n
        audio_grads = audio_grads_global[local_start : local_start + local_n]
        logit_scale_grad = (
            None if logit_scale.grad is None else logit_scale.grad.detach().clone()
        )

        # Align-only recompute (surrogate); measure adapter grad norm, then save grads.
        optimizer.zero_grad(set_to_none=True)
        offset = 0
        for paths in micro_path_chunks:
            n = len(paths)
            pooled, _stab = encode_micro(paths)
            rep_grad = audio_grads[offset : offset + n]
            offset += n
            surrogate = (pooled.float() * rep_grad).sum()
            surrogate.backward()
        if offset != local_n:
            raise RuntimeError(f"GradCache align offset {offset} != local_n {local_n}")

        align_grad_norm = _param_grad_l2_norm(adapter_params)
        saved_align_grads = _clone_param_grads(adapter_params)

        # Stability-only recompute; measure stab grad norm, then add align grads back.
        for p in adapter_params:
            p.grad = None
        stab_total = torch.zeros((), device=audio_pooled_local.device, dtype=torch.float32)
        for paths in micro_path_chunks:
            _pooled, stab_sum = encode_micro(paths)
            stab_total = stab_total + stab_sum.float()
            stability_term = float(lambda_stability) * (stab_sum.float() / float(global_n))
            stability_term.backward()

        stab_grad_norm = _param_grad_l2_norm(adapter_params)
        _add_param_grads(adapter_params, saved_align_grads)

        if logit_scale_grad is not None:
            logit_scale.grad = logit_scale_grad

        if enabled and world_size > 1:
            dist.all_reduce(stab_total, op=dist.ReduceOp.SUM)

        stability_loss = (stab_total / float(global_n)).detach()
        total_loss = (align_loss + float(lambda_stability) * stability_loss).detach()
        diag = {
            **diag,
            "align_grad_norm": float(align_grad_norm),
            "stab_grad_norm": float(stab_grad_norm),
        }
        return total_loss, align_loss, stability_loss, diag
    finally:
        adapter.train(was_training)
