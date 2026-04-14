from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from .config import QwenConfig


@dataclass(frozen=True)
class QwenModels:
    model_id: str
    tokenizer: Any
    causal_lm: Any
    embedder: Any


def load_qwen_models(
    *,
    cfg: QwenConfig | None = None,
    model_id: str = "Qwen/Qwen3-8B",
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
    device_map: str | dict | None = "auto",
    embeddings_only: bool = False,
) -> QwenModels:
    """
    Load Qwen tokenizer and (optionally) the causal LM.

    - **embeddings_only=True**: loads an `AutoModel` for `get_input_embeddings()` only.
      This matches Stage 1 usage (contrastive alignment).
    - **embeddings_only=False**: loads full `AutoModelForCausalLM` for generation or LM loss.
    """
    if cfg is not None:
        model_id = cfg.model_id
        device = cfg.device
        torch_dtype = cfg.torch_dtype
        device_map = cfg.device_map
        embeddings_only = cfg.embeddings_only

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if embeddings_only:
        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        embedder = model.get_input_embeddings()
        return QwenModels(model_id=model_id, tokenizer=tokenizer, causal_lm=None, embedder=embedder)

    causal_lm = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        device_map=device_map,
    )
    causal_lm.eval()
    for p in causal_lm.parameters():
        p.requires_grad = False
    embedder = causal_lm.get_input_embeddings()
    return QwenModels(model_id=model_id, tokenizer=tokenizer, causal_lm=causal_lm, embedder=embedder)

