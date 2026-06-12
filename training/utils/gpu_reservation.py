"""Exclusive GPU file locks — hold devices until the owning process exits."""

from __future__ import annotations

import atexit
import fcntl
import os
import signal
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from training.utils.env import env_bool, env_str, package_root


def gpu_lock_enabled() -> bool:
    """True unless ``GPU_LOCK=off|false|0|no``."""
    return env_bool("GPU_LOCK", True)


def _gpu_count_without_torch() -> int:
    visible = env_str("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        ids = [x.strip() for x in visible.split(",") if x.strip()]
        if ids:
            return len(ids)
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return len([ln for ln in out.stdout.splitlines() if ln.strip()])
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return 0


def visible_physical_gpu_indices() -> list[int]:
    """Physical GPU indices selected by ``CUDA_VISIBLE_DEVICES`` (or all GPUs)."""
    visible = env_str("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        ids = [int(x.strip()) for x in visible.split(",") if x.strip()]
        if ids:
            return ids
    count = _gpu_count_without_torch()
    return list(range(count)) if count > 0 else []


def gpu_indices_for_process(*, local_rank: int, world_size: int) -> list[int]:
    """GPUs this process should lock (all visible in MP; one per rank in DDP)."""
    visible = visible_physical_gpu_indices()
    if not visible:
        return []
    if world_size > 1:
        if local_rank >= len(visible):
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {len(visible)} GPU(s) visible "
                f"({visible}); increase CUDA_VISIBLE_DEVICES or lower nproc_per_node."
            )
        return [visible[local_rank]]
    return visible


def _lock_dir() -> Path:
    raw = env_str("GPU_LOCK_DIR")
    if raw:
        return Path(raw)
    return Path(package_root()) / ".gpu_locks"


@dataclass
class _HeldLock:
    gpu_index: int
    path: Path
    handle: object


class GpuReservation:
    """Blocking exclusive ``flock`` locks on one or more GPU indices."""

    def __init__(
        self,
        gpu_indices: list[int],
        *,
        lock_dir: Path | None = None,
        label: str | None = None,
    ) -> None:
        self.gpu_indices = list(gpu_indices)
        self.lock_dir = lock_dir or _lock_dir()
        self.label = label or _default_label()
        self._held: list[_HeldLock] = []

    def acquire(self, *, blocking: bool = True) -> None:
        if not self.gpu_indices:
            return
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        acquired: list[_HeldLock] = []
        try:
            for idx in sorted(self.gpu_indices):
                path = self.lock_dir / f"gpu{idx}.lock"
                handle = path.open("a+", encoding="utf-8")
                flags = fcntl.LOCK_EX
                if not blocking:
                    flags |= fcntl.LOCK_NB
                if blocking:
                    holder = _read_lock_holder(path)
                    if holder and holder != f"pid={os.getpid()}":
                        print(
                            f"Waiting for GPU {idx} ({holder})...",
                            flush=True,
                        )
                try:
                    fcntl.flock(handle.fileno(), flags)
                except BlockingIOError as exc:
                    handle.close()
                    raise RuntimeError(
                        f"GPU {idx} is reserved by another process ({path})"
                    ) from exc
                _write_lock_metadata(handle, gpu_index=idx, label=self.label)
                acquired.append(_HeldLock(gpu_index=idx, path=path, handle=handle))
                print(f"Reserved GPU {idx} ({self.label})", flush=True)
        except Exception:
            for held in acquired:
                _release_one(held)
            raise
        self._held = acquired

    def release(self) -> None:
        for held in reversed(self._held):
            _release_one(held)
        self._held.clear()

    def __enter__(self) -> GpuReservation:
        self.acquire(blocking=True)
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


_ACTIVE: GpuReservation | None = None
_CLEANUP_REGISTERED = False


def _default_label() -> str:
    argv = Path(sys.argv[0]).name if sys.argv else "python"
    return f"pid={os.getpid()} cmd={argv}"


def _write_lock_metadata(handle: object, *, gpu_index: int, label: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    handle.seek(0)
    handle.truncate()
    handle.write(f"gpu={gpu_index}\nlabel={label}\nacquired_at={ts}\n")
    handle.flush()


def _read_lock_holder(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith("label="):
            return line.removeprefix("label=").strip()
    return path.name


def _release_one(held: _HeldLock) -> None:
    try:
        fcntl.flock(held.handle.fileno(), fcntl.LOCK_UN)
    finally:
        held.handle.close()


def _register_cleanup() -> None:
    global _CLEANUP_REGISTERED
    if _CLEANUP_REGISTERED:
        return
    atexit.register(release_gpu_reservation)

    def _signal_handler(signum: int, _frame: object) -> None:
        release_gpu_reservation()
        raise SystemExit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _signal_handler)
        except (ValueError, OSError):
            pass
    _CLEANUP_REGISTERED = True


def reserve_gpus(
    *,
    local_rank: int = 0,
    world_size: int = 1,
    blocking: bool = True,
    label: str | None = None,
) -> GpuReservation | None:
    """
    Acquire exclusive locks on the GPUs used by this process.

    Returns ``None`` when ``GPU_LOCK=off`` or no GPUs are visible.
    Other processes calling this on the same indices block until release.
    """
    global _ACTIVE
    if not gpu_lock_enabled():
        return None
    if _ACTIVE is not None:
        return _ACTIVE

    indices = gpu_indices_for_process(local_rank=local_rank, world_size=world_size)
    if not indices:
        return None

    reservation = GpuReservation(indices, label=label)
    reservation.acquire(blocking=blocking)
    _ACTIVE = reservation
    _register_cleanup()
    return reservation


def release_gpu_reservation() -> None:
    global _ACTIVE
    if _ACTIVE is not None:
        _ACTIVE.release()
        _ACTIVE = None


def wait_for_gpu_reservation(
    *,
    local_rank: int = 0,
    world_size: int = 1,
    label: str | None = None,
) -> GpuReservation | None:
    """Blocking alias used before CUDA init in training/eval entry points."""
    return reserve_gpus(
        local_rank=local_rank,
        world_size=world_size,
        blocking=True,
        label=label,
    )
