from __future__ import annotations

from collections.abc import Iterable

import torch


class TrainingPipeline:
    """
    Shared optimizer step: backward, grad clip, optimizer, scheduler, step counter.

    Stage scripts keep their own forward/loss logic; this keeps the training *flow* identical.
    """

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        grad_clip_norm: float,
    ) -> None:
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip_norm = float(grad_clip_norm)
        self.global_step = 0

    def step(self, loss: torch.Tensor, params: Iterable[torch.nn.Parameter]) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, self.grad_clip_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.global_step += 1
