"""CUDA / distributed helpers for training scripts."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from training.utils.env import env_str
from training.utils.gpu_reservation import gpu_lock_enabled, wait_for_gpu_reservation


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


def ensure_device_ready(device: torch.device) -> None:
    """Initialize CUDA context for ``device`` when it is a GPU."""
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA required for {device} but unavailable")
    torch.cuda.init()
    torch.cuda.set_device(device)
    torch.zeros((), device=device)


def resolve_llm_device_str(*, train_device: torch.device, llm_device: str | None) -> str:
    """Frozen LLM device string; falls back to the primary training device."""
    if llm_device and llm_device.strip():
        return llm_device.strip()
    return str(train_device)


def _build_llm_max_memory(
    *,
    llm_max_memory: dict[int, str] | None,
    primary_gpu_idx: int,
    default_primary_cap: str = "14.5GiB",
) -> dict[int | str, str]:
    if llm_max_memory:
        out: dict[int | str, str] = dict(llm_max_memory)
        out.setdefault("cpu", "64GiB")
        out.setdefault(primary_gpu_idx, default_primary_cap)
        return out
    return {primary_gpu_idx: default_primary_cap, "cpu": "64GiB"}


def _llm_device_map_for(max_memory: dict[int | str, str] | None) -> str | None:
    if max_memory is None:
        return None
    gpu_keys = [k for k in max_memory if isinstance(k, int)]
    if len(gpu_keys) >= 2:
        return "sequential"
    return "auto"


def resolve_llm_load_plan(
    *,
    ctx: TrainingContext,
    train_device: torch.device,
    llm_device: str | None,
    llm_max_memory: dict[int, str] | None,
) -> tuple[str, str | dict | None, dict[int | str, str] | None]:
    """
    HuggingFace load plan for the frozen LLM.

    When ``LLM_MAX_MEMORY`` lists multiple GPUs, Qwen is sharded with
    ``device_map='sequential'`` (GPU 0 first, spill to GPU 1, then CPU).

    Single-GPU split layouts (e.g. ``DEVICE=cpu`` + ``LLM_DEVICE=cuda:0``) use
    ``device_map='auto'`` with a GPU cap and CPU spill.
    """
    llm_device_str = resolve_llm_device_str(
        train_device=train_device,
        llm_device=llm_device,
    )
    split = llm_device_str != str(train_device)
    llm_dev = torch.device(llm_device_str)

    if llm_dev.type != "cuda":
        return llm_device_str, None, None

    primary_gpu_idx = llm_dev.index if llm_dev.index is not None else 0
    gpu_entries = [k for k in (llm_max_memory or {}) if isinstance(k, int)]

    if llm_max_memory and ctx.num_cuda_devices >= 2 and len(gpu_entries) >= 2:
        max_mem = _build_llm_max_memory(
            llm_max_memory=llm_max_memory,
            primary_gpu_idx=primary_gpu_idx,
        )
        return llm_device_str, _llm_device_map_for(max_mem), max_mem

    if not split:
        qwen_map = llm_device_map(ctx, max_memory=llm_max_memory)
        return llm_device_str, qwen_map, llm_max_memory if qwen_map else None

    max_mem = _build_llm_max_memory(
        llm_max_memory=llm_max_memory,
        primary_gpu_idx=primary_gpu_idx,
    )
    return llm_device_str, _llm_device_map_for(max_mem), max_mem


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


def device_env_has_explicit_index() -> bool:
    """True when ``DEVICE`` names a specific GPU (e.g. ``cuda:1``), not bare ``cuda``."""
    raw = (env_str("DEVICE", "cuda") or "cuda").strip().lower()
    return raw.startswith("cuda:") and raw != "cuda"


def init_cuda_devices() -> int:
    """
    Initialize CUDA and touch only the primary device (``DEVICE`` → ``cuda:0``).

    Sibling GPUs are left alone at init; HuggingFace ``device_map`` loads spill
    weights on GPU 1 when ``LLM_MAX_MEMORY`` caps are set.
    """
    if not torch.cuda.is_available():
        return 0
    torch.cuda.init()
    n = torch.cuda.device_count()
    device = resolve_device()
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.zeros((), device=device)
    return n


def init_eval_device() -> tuple[torch.device, torch.dtype, int]:
    """
    Resolve ``DEVICE`` (default ``cuda:0``) and initialize only that GPU.

    Whisper/adapter run on the primary device; Qwen may spill to GPU 1 via
    ``device_map='sequential'`` and ``LLM_MAX_MEMORY`` when two GPUs are visible.

    GPU file locks are skipped when ``DEVICE=cpu`` (no CUDA reservation).
    """
    device = resolve_device()
    if gpu_lock_enabled() and device.type == "cuda":
        wait_for_gpu_reservation()
    num_cuda = visible_gpu_count()
    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.set_device(device)
        torch.zeros((), device=device)
        torch.cuda.empty_cache()
    return device, torch_dtype, num_cuda


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

    if gpu_lock_enabled():
        wait_for_gpu_reservation(local_rank=local_rank, world_size=world_size)

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
        model_parallel = (
            num_cuda >= 2 and device.type == "cuda" and not device_env_has_explicit_index()
        )

    return TrainingContext(
        device=device,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        is_main=rank == 0,
        num_cuda_devices=num_cuda,
        model_parallel=model_parallel,
    )


def llm_device_map(
    ctx: TrainingContext,
    *,
    max_memory: dict[int, str] | None = None,
) -> str | None:
    """
    HuggingFace ``device_map`` for the frozen LLM, or ``None`` for a single device.

    With ``max_memory`` set and 2+ GPUs, uses ``sequential`` so GPU 0 fills first
    and overflow spills to GPU 1 (subject to per-device caps). Otherwise ``auto``.
    """
    if ctx.device.type != "cuda":
        return None
    if ctx.model_parallel:
        return "sequential" if max_memory else "auto"
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
