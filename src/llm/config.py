from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from transformers import GenerationConfig


@dataclass(frozen=True)
class QwenConfig:
    model_id: str = "Qwen/Qwen3-8B"
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.float16
    device_map: str | dict | None = "auto"
    embeddings_only: bool = False


@dataclass(frozen=True)
class LlmGenerationParams:
    """
    High-level generation parameters for causal LMs.

    These map onto Hugging Face `GenerationConfig` and are intended to be passed around
    as a stable config object (e.g., notebooks + scripts).
    """

    max_new_tokens: int = 100
    do_sample: bool = True

    # Sampling-only params (used only when do_sample=True)
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None

    # Common decoding params
    repetition_penalty: float | None = None


def build_hf_generation_config(
    *,
    model: Any,
    tokenizer: Any,
    params: LlmGenerationParams,
) -> GenerationConfig:
    """
    Build a `GenerationConfig` consistent with the model + tokenizer.

    Important: pretrained models often ship a `generation_config.json` (e.g. temperature/top_p/top_k).
    For greedy decoding we *clear* sampling fields to avoid warnings and ambiguous behavior.
    """
    base = getattr(model, "generation_config", None)
    gen_cfg = copy.deepcopy(base) if base is not None else GenerationConfig()

    gen_cfg.max_new_tokens = int(params.max_new_tokens)
    gen_cfg.do_sample = bool(params.do_sample)

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        gen_cfg.eos_token_id = eos_id
        gen_cfg.pad_token_id = eos_id

    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None:
        gen_cfg.bos_token_id = bos_id

    if params.do_sample:
        # Defaults chosen to be reasonable; caller can override.
        gen_cfg.temperature = 0.7 if params.temperature is None else float(params.temperature)
        gen_cfg.top_p = 0.9 if params.top_p is None else float(params.top_p)
        gen_cfg.top_k = 50 if params.top_k is None else int(params.top_k)
    else:
        # Clear sampling-related fields to avoid HF warnings (model configs often set these).
        gen_cfg.temperature = None
        gen_cfg.top_p = None
        gen_cfg.top_k = None

    if params.repetition_penalty is not None:
        gen_cfg.repetition_penalty = float(params.repetition_penalty)

    return gen_cfg
