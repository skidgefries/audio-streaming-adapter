"""CUDA / distributed helpers for training scripts."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from training.utils.env import env_str


@dataclass(frozen=True)
class TrainingContext:
    """Resolved device layout for a training run."""

    device: torch.device
    rank: int
    world_size: int
    local_rank: int
    is_main: bool
    num_cuda_devices: int
    model_parallel: bool


def _gpu_count_without_torch() -> int:
    """GPU count from env or ``nvidia-smi`` only (safe before ``import torch``)."""
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
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0


def apply_runtime_cuda_env() -> None:
    """Apply ``PYTORCH_CUDA_ALLOC_CONF`` default when unset."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def resolve_device() -> torch.device:
    """
    Primary compute device.

    Uses ``DEVICE`` from the environment: ``cuda`` (default when available),
    ``cpu``, or an explicit index such as ``cuda:1``.
    """
    raw = (env_str("DEVICE", "cuda") or "cuda").strip().lower()
    if raw == "cpu":
        return torch.device("cpu")
    if raw.startswith("cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        if raw == "cuda":
            return torch.device("cuda:0")
        return torch.device(raw)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def visible_gpu_count() -> int:
    """Number of CUDA devices visible to the current process."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return _gpu_count_without_torch()


def init_cuda_devices() -> int:
    """Initialize CUDA and touch each visible device. Returns visible GPU count."""
    if not torch.cuda.is_available():
        return 0
    torch.cuda.init()
    n = torch.cuda.device_count()
    for i in range(n):
        torch.zeros((), device=f"cuda:{i}")
    return n


def _init_process_group() -> None:
    if dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def init_training_context() -> TrainingContext:
    """
    Initialize distributed (when launched with ``torchrun``) and resolve devices.

    Multi-GPU **model parallel** (single process, ``device_map="auto"`` for the LLM)
    is used when two or more GPUs are visible and ``WORLD_SIZE == 1``.

    When ``WORLD_SIZE > 1``, each rank uses ``cuda:{local_rank}`` for data-parallel
    training of the trainable modules.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1:
        _init_process_group()
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        num_cuda = visible_gpu_count()
        model_parallel = False
    else:
        num_cuda = init_cuda_devices()
        device = resolve_device()
        model_parallel = num_cuda >= 2 and device.type == "cuda"

    return TrainingContext(
        device=device,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        is_main=rank == 0,
        num_cuda_devices=num_cuda,
        model_parallel=model_parallel,
    )


def llm_device_map(ctx: TrainingContext) -> str | None:
    """HuggingFace ``device_map`` for the frozen LLM, or ``None`` for a single device."""
    if ctx.device.type != "cuda":
        return None
    if ctx.model_parallel:
        return "auto"
    return None


def llm_input_device(model: torch.nn.Module) -> torch.device:
    """Device for ``inputs_embeds`` when the causal LM uses ``hf_device_map``."""
    hf_map = getattr(model, "hf_device_map", None)
    if hf_map:
        for key in ("model.embed_tokens", "embed_tokens", "transformer.wte"):
            if key in hf_map:
                dev = hf_map[key]
                if isinstance(dev, int):
                    return torch.device(f"cuda:{dev}")
                return torch.device(dev)
    return next(model.parameters()).device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
