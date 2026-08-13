from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .config import QwenConfig


@dataclass(frozen=True)
class QwenModels:
    model_id: str
    tokenizer: Any
    causal_lm: Any
    embedder: Any


_EMBED_TOKENS_KEY = "model.embed_tokens.weight"


def _find_embed_tokens_key(weight_map_or_keys) -> str:
    keys = list(weight_map_or_keys)
    if _EMBED_TOKENS_KEY in keys:
        return _EMBED_TOKENS_KEY
    for key in keys:
        if key.endswith("embed_tokens.weight"):
            return key
    raise KeyError(f"No embed_tokens weight among keys: {keys[:20]}...")


def _load_embedder_only(*, model_id: str, torch_dtype: torch.dtype) -> torch.nn.Embedding:
    """
    Load only ``embed_tokens`` instead of the full base model.

    Works for Qwen, Vicuna/Llama, and other HF causal LMs that expose
    ``model.embed_tokens.weight`` (sharded index or single ``model.safetensors``).

    Stage 1 contrastive training needs embedding lookup only; loading the full
    LM routinely OOMs on 16 GiB GPUs that already host Whisper + adapter.
    """
    config = AutoConfig.from_pretrained(model_id)
    embedder = torch.nn.Embedding(config.vocab_size, config.hidden_size)

    try:
        index_path = hf_hub_download(model_id, "model.safetensors.index.json")
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        weight_map: dict[str, str] = index["weight_map"]
        weight_key = _find_embed_tokens_key(weight_map)
        shard_path = hf_hub_download(model_id, weight_map[weight_key])
        state = load_file(shard_path)
    except Exception:
        # Single-file safetensors (some Vicuna/Llama mirrors) or missing index.
        shard_path = hf_hub_download(model_id, "model.safetensors")
        state = load_file(shard_path)
        weight_key = _find_embed_tokens_key(state.keys())

    embedder.weight.data.copy_(state[weight_key].to(dtype=torch_dtype))
    embedder.eval()
    for param in embedder.parameters():
        param.requires_grad = False
    return embedder


def load_qwen_models(
    *,
    cfg: QwenConfig | None = None,
    model_id: str = "Qwen/Qwen3-8B",
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
    device_map: str | dict | None = "auto",
    max_memory: dict[int, str] | None = None,
    embeddings_only: bool = False,
) -> QwenModels:
    """
    Load Qwen tokenizer and (optionally) the causal LM.

    - **embeddings_only=True**: loads only ``embed_tokens`` weights (~1.2 GiB for
      Qwen3-8B). This matches Stage 1 usage (contrastive alignment).
    - **embeddings_only=False**: loads full `AutoModelForCausalLM` for generation or LM loss.
    """
    if cfg is not None:
        model_id = cfg.model_id
        device = cfg.device
        torch_dtype = cfg.torch_dtype
        device_map = cfg.device_map
        max_memory = cfg.max_memory
        embeddings_only = cfg.embeddings_only

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    load_kw: dict = dict(
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    if device_map is not None:
        load_kw["device_map"] = device_map
    if max_memory is not None:
        load_kw["max_memory"] = max_memory
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    target = torch.device(device)

    if embeddings_only:
        embedder = _load_embedder_only(model_id=model_id, torch_dtype=torch_dtype)
        embedder = embedder.to(target)
        return QwenModels(model_id=model_id, tokenizer=tokenizer, causal_lm=None, embedder=embedder)

    causal_lm = AutoModelForCausalLM.from_pretrained(model_id, use_safetensors=True, **load_kw)
    if device_map is None:
        causal_lm = causal_lm.to(target)
    causal_lm.eval()
    for p in causal_lm.parameters():
        p.requires_grad = False
    embedder = causal_lm.get_input_embeddings()
    return QwenModels(model_id=model_id, tokenizer=tokenizer, causal_lm=causal_lm, embedder=embedder)

