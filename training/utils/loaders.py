"""
Frozen model loaders for training scripts.

These are thin wrappers over ``encoder`` and ``llm`` (same as inference in
``src/adapter_llm_pipeline.py``). Prefer importing from ``encoder`` / ``llm`` directly
in new code.
"""

from __future__ import annotations

import torch

from encoder import WhisperConfig, load_whisper_models
from llm import QwenConfig, load_qwen_models, QwenModels


def default_device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda:0", torch.bfloat16
    return "cpu", torch.float32


def load_frozen_whisper(*, model_id: str, device: str, torch_dtype: torch.dtype):
    cfg = WhisperConfig(model_id=model_id, device=device, torch_dtype=torch_dtype)
    return load_whisper_models(cfg=cfg)


def load_frozen_qwen_embeddings(
    *,
    model_id: str,
    device: str,
    torch_dtype: torch.dtype,
    device_map="auto",
    max_memory: dict[int, str] | None = None,
) -> QwenModels:
    cfg = QwenConfig(
        model_id=model_id,
        device=device,
        torch_dtype=torch_dtype,
        device_map=device_map,
        max_memory=max_memory,
        embeddings_only=True,
    )
    return load_qwen_models(cfg=cfg)


def load_frozen_qwen_causal_lm(
    *,
    model_id: str,
    device: str,
    torch_dtype: torch.dtype,
    device_map="auto",
    max_memory: dict[int, str] | None = None,
) -> QwenModels:
    cfg = QwenConfig(
        model_id=model_id,
        device=device,
        torch_dtype=torch_dtype,
        device_map=device_map,
        max_memory=max_memory,
        embeddings_only=False,
    )
    return load_qwen_models(cfg=cfg)
