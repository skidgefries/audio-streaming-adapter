"""Hold free CUDA VRAM so other processes cannot steal it between training steps.

This is not a true NVIDIA exclusive lock. It allocates a float32 tensor that fills
most currently-free memory (minus a headroom), then frees it before the next step
that needs activations.

Enable with ``CUDA_MEM_FENCE=true`` (optional ``CUDA_MEM_LEAVE_FREE_GB``, default 1.5).
"""

from __future__ import annotations

import torch

from training.utils.env import env_bool, env_float


class CudaVramFence:
    """
    Soft VRAM reservation via a large empty tensor.

    Typical use::

        fence = CudaVramFence.from_env(device)
        # after models are loaded
        fence.acquire()
        for batch in loader:
            fence.release()   # free reserved bytes for this step
            train_step(batch)
            fence.acquire()   # reclaim free VRAM so neighbors cannot grab dips
        fence.release()
    """

    def __init__(
        self,
        device: torch.device | str,
        *,
        leave_free_gb: float = 1.5,
        enabled: bool = True,
        verbose: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.leave_free_bytes = max(0, int(leave_free_gb * (1024**3)))
        self.enabled = bool(enabled) and self.device.type == "cuda" and torch.cuda.is_available()
        self.verbose = verbose
        self._reserve: torch.Tensor | None = None
        self._announced = False

    @classmethod
    def from_env(cls, device: torch.device | str) -> CudaVramFence:
        return cls(
            device,
            leave_free_gb=env_float("CUDA_MEM_LEAVE_FREE_GB", 1.5),
            enabled=env_bool("CUDA_MEM_FENCE", False),
        )

    @property
    def held_bytes(self) -> int:
        if self._reserve is None:
            return 0
        return int(self._reserve.numel() * self._reserve.element_size())

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def acquire(self) -> int:
        """Fill free VRAM down to ``leave_free_gb``. Returns bytes reserved."""
        if not self.enabled:
            return 0
        self.release(silent=True)
        free_b, total_b = torch.cuda.mem_get_info(self.device)
        target = max(0, int(free_b) - self.leave_free_bytes)
        # float32 elements; keep a small alignment margin for allocator fragmentation
        n = (target // 4) - (1 << 16)
        if n <= 0:
            return 0
        try:
            self._reserve = torch.empty(n, dtype=torch.float32, device=self.device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            free_b, _ = torch.cuda.mem_get_info(self.device)
            n = max(0, (int(free_b) - self.leave_free_bytes) // 4 - (1 << 16))
            if n <= 0:
                return 0
            try:
                self._reserve = torch.empty(n, dtype=torch.float32, device=self.device)
            except torch.cuda.OutOfMemoryError:
                return 0
        held = self.held_bytes
        if not self._announced:
            free_after, _ = torch.cuda.mem_get_info(self.device)
            self._log(
                f"  [vram-fence] enabled on {self.device}: hold free VRAM between steps "
                f"(leave_free={self.leave_free_bytes / (1024**3):.2f} GiB). "
                f"First reserve={held / (1024**3):.2f} GiB, "
                f"free_now={free_after / (1024**3):.2f}/{total_b / (1024**3):.2f} GiB"
            )
            self._announced = True
        return held

    def release(self, *, silent: bool = True) -> None:
        if self._reserve is None:
            return
        self._reserve = None
        torch.cuda.empty_cache()
        if not silent:
            free_b, total_b = torch.cuda.mem_get_info(self.device)
            self._log(
                f"  [vram-fence] released on {self.device} "
                f"(free_now={free_b / (1024**3):.2f}/{total_b / (1024**3):.2f} GiB)"
            )
