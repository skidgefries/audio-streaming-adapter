# from __future__ import annotations

# from collections.abc import Iterable

# import torch


# class TrainingPipeline:
#     """
#     Shared optimizer step: backward, grad clip, optimizer, scheduler, step counter.

#     Stage scripts keep their own forward/loss logic; this keeps the training *flow* identical.
#     """

#     def __init__(
#         self,
#         *,
#         optimizer: torch.optim.Optimizer,
#         scheduler: torch.optim.lr_scheduler.LRScheduler | None,
#         grad_clip_norm: float,
#     ) -> None:
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.grad_clip_norm = float(grad_clip_norm)
#         self.global_step = 0

#     def step(self, loss: torch.Tensor, params: Iterable[torch.nn.Parameter]) -> None:
#         self.optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         if self.grad_clip_norm > 0:
#             torch.nn.utils.clip_grad_norm_(params, self.grad_clip_norm)
#         self.optimizer.step()
#         if self.scheduler is not None:
#             self.scheduler.step()
#         self.global_step += 1



from __future__ import annotations

from collections.abc import Iterable
from contextlib import ExitStack

import torch
from torch.nn.parallel import DistributedDataParallel as DDP


class TrainingPipeline:
    """
    Shared optimizer step: backward, grad clip, optimizer, scheduler, step counter.

    Stage scripts keep their own forward/loss logic; this keeps the training *flow* identical.
    Supports gradient accumulation when ``gradient_accumulation_steps > 1``.
    """

    def __init__(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        grad_clip_norm: float,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip_norm = float(grad_clip_norm)
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        self.global_step = 0
        self._accum_step = 0

    def reset_accumulation(self) -> None:
        """Clear partial accumulation (e.g. after resume on an optimizer boundary)."""
        self._accum_step = 0
        self.optimizer.zero_grad(set_to_none=True)

    @property
    def accum_step(self) -> int:
        """Micro-batches accumulated toward the next optimizer update."""
        return self._accum_step

    @property
    def is_accumulation_boundary(self) -> bool:
        """True when the next backward completes an accumulation cycle."""
        return self._accum_step + 1 >= self.gradient_accumulation_steps

    def step(
        self,
        loss: torch.Tensor,
        params: Iterable[torch.nn.Parameter],
        *,
        no_sync_modules: Iterable[torch.nn.Module] | None = None,
    ) -> bool:
        """
        Backward on ``loss``; optimizer/scheduler step only every N micro-batches.

        Returns True when an optimizer update was applied.
        """
        if self._accum_step == 0:
            self.optimizer.zero_grad(set_to_none=True)

        scaled_loss = loss / self.gradient_accumulation_steps
        use_no_sync = no_sync_modules is not None and not self.is_accumulation_boundary
        if use_no_sync:
            with ExitStack() as stack:
                for module in no_sync_modules:
                    if isinstance(module, DDP):
                        stack.enter_context(module.no_sync())
                scaled_loss.backward()
        else:
            scaled_loss.backward()

        self._accum_step += 1
        if self._accum_step < self.gradient_accumulation_steps:
            return False

        self._accum_step = 0

        # Zero out NaN/inf gradients before clipping
        for p in params:
            if p.grad is not None:
                p.grad = torch.where(
                    torch.isnan(p.grad) | torch.isinf(p.grad),
                    torch.zeros_like(p.grad),
                    p.grad,
                )

        # Skip update if gradients are all NaN
        has_nan = any(
            torch.isnan(p.grad).any()
            for p in params
            if p.grad is not None
        )
        if has_nan:
            print(f"[WARN] Step {self.global_step}: NaN gradients detected, skipping update")
            self.global_step += 1
            return True

        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, self.grad_clip_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.global_step += 1
        return True