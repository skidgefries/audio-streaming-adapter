"""CUDA / multi-GPU helpers for training scripts."""

from __future__ import annotations

import torch


def init_cuda_for_device_map() -> int:
    """
    Touch every visible GPU before ``device_map="auto"`` load.

    Avoids accelerate errors when CUDA was not initialized yet.
    """
    if not torch.cuda.is_available():
        return 0
    torch.cuda.init()
    n = torch.cuda.device_count()
    for i in range(n):
        torch.zeros((), device=f"cuda:{i}")
    return n


def default_train_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def qwen_sharded_max_memory(*, reserve_on_gpu0_gib: float) -> dict[int, str] | None:
    """
    Cap GPU 0 memory for Hugging Face sharding so Whisper + adapter + gate fit.

    Returns ``None`` when fewer than two GPUs are visible (single-GPU load unchanged).
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return None

    max_memory: dict[int, str] = {}
    for i in range(torch.cuda.device_count()):
        total_gib = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        usable = total_gib * 0.90
        if i == 0:
            cap = max(4.0, usable - reserve_on_gpu0_gib)
            max_memory[i] = f"{int(cap)}GiB"
        else:
            max_memory[i] = f"{int(usable)}GiB"
    return max_memory


def qwen_balanced_max_memory(*, reserve_on_gpu0_gib: float) -> dict[int, str] | None:
    """
    Equal per-GPU caps for ``device_map="auto"`` (~50/50 Qwen layers).

    GPU 0 cap subtracts ``reserve_on_gpu0_gib`` so Whisper + adapter + gate still fit.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return None

    usable_gib = [
        torch.cuda.get_device_properties(i).total_memory / (1024**3) * 0.90
        for i in range(torch.cuda.device_count())
    ]
    half_gib = min(usable_gib[0] - reserve_on_gpu0_gib, usable_gib[1])
    half_gib = max(4.0, half_gib)
    cap = f"{int(half_gib)}GiB"
    return {i: cap for i in range(torch.cuda.device_count())}


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
