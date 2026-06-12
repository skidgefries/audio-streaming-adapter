from __future__ import annotations

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
    max_memory: dict[int, str] | None = None
    embeddings_only: bool = False


@dataclass(frozen=True)
class LlmGenerationParams:
    """
    High-level generation parameters for causal LMs.

    These map onto Hugging Face `GenerationConfig` and are intended to be passed around
    as a stable config object (e.g., notebooks + scripts).
    """

    max_new_tokens: int = 100
    min_new_tokens: int | None = None
    do_sample: bool = True
    num_beams: int = 1

    # Sampling-only params (used only when do_sample=True)
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None

    # Common decoding params
    repetition_penalty: float | None = None
    no_repeat_ngram_size: int | None = None


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
    eos_id = getattr(tokenizer, "eos_token_id", None)
    bos_id = getattr(tokenizer, "bos_token_id", None)

    cfg_kwargs: dict[str, Any] = {
        "max_new_tokens": int(params.max_new_tokens),
        "do_sample": bool(params.do_sample),
    }
    if params.min_new_tokens is not None:
        cfg_kwargs["min_new_tokens"] = int(params.min_new_tokens)
    if eos_id is not None:
        cfg_kwargs["eos_token_id"] = eos_id
        cfg_kwargs["pad_token_id"] = eos_id
    if bos_id is not None:
        cfg_kwargs["bos_token_id"] = bos_id

    if params.do_sample:
        cfg_kwargs["temperature"] = (
            0.7 if params.temperature is None else float(params.temperature)
        )
        cfg_kwargs["top_p"] = 0.9 if params.top_p is None else float(params.top_p)
        cfg_kwargs["top_k"] = 50 if params.top_k is None else int(params.top_k)
    else:
        beams = max(1, int(params.num_beams))
        cfg_kwargs["num_beams"] = beams
        if beams > 1:
            cfg_kwargs["num_return_sequences"] = 1

    if params.repetition_penalty is not None:
        cfg_kwargs["repetition_penalty"] = float(params.repetition_penalty)
    if params.no_repeat_ngram_size is not None and int(params.no_repeat_ngram_size) > 0:
        cfg_kwargs["no_repeat_ngram_size"] = int(params.no_repeat_ngram_size)

    # Fresh config avoids inheriting sampling fields from the model's generation_config.json.
    gen_cfg = GenerationConfig(**cfg_kwargs)
    return gen_cfg
