"""Vicuna embeddings-only loader for Stage 1 contrastive training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, LlamaTokenizer

_EMBED_TOKENS_KEY = "model.embed_tokens.weight"
DEFAULT_MODEL_ID = "lmsys/vicuna-7b-v1.5"


def _require_sentencepiece() -> None:
    try:
        import sentencepiece  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Vicuna tokenizer requires sentencepiece. Install with:\n"
            "  uv pip install sentencepiece protobuf\n"
            "then re-run."
        ) from exc


@dataclass(frozen=True)
class VicunaModels:
    model_id: str
    tokenizer: Any
    embedder: torch.nn.Embedding
    hidden_size: int
    causal_lm: Any = None


def load_vicuna_tokenizer(model_id: str = DEFAULT_MODEL_ID) -> LlamaTokenizer:
    """
    Load Vicuna/Llama SentencePiece tokenizer (slow tokenizer only).

    Avoids AutoTokenizer fast-path conversion that can mis-read tokenizer.model as tiktoken.
    """
    _require_sentencepiece()
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _resolve_embed_tokens_key(weight_map: dict[str, str]) -> str:
    if _EMBED_TOKENS_KEY in weight_map:
        return _EMBED_TOKENS_KEY
    for key in weight_map:
        if key.endswith("embed_tokens.weight"):
            return key
    raise KeyError("No embed_tokens weight found in model weight map")


def _load_tensor_from_shard(shard_path: str, weight_key: str) -> torch.Tensor:
    if shard_path.endswith(".safetensors"):
        state = load_file(shard_path)
        return state[weight_key]
    state = torch.load(shard_path, map_location="cpu", weights_only=True)
    return state[weight_key]


def _load_embedder_only(*, model_id: str, torch_dtype: torch.dtype) -> tuple[torch.nn.Embedding, int]:
    config = AutoConfig.from_pretrained(model_id)
    embedder = torch.nn.Embedding(config.vocab_size, config.hidden_size)

    index_candidates = (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    last_error: Exception | None = None
    for index_name in index_candidates:
        try:
            index_path = hf_hub_download(model_id, index_name)
            with open(index_path, encoding="utf-8") as f:
                index = json.load(f)
            weight_map: dict[str, str] = index["weight_map"]
            weight_key = _resolve_embed_tokens_key(weight_map)
            shard_name = weight_map[weight_key]
            shard_path = hf_hub_download(model_id, shard_name)
            tensor = _load_tensor_from_shard(shard_path, weight_key)
            embedder.weight.data.copy_(tensor.to(dtype=torch_dtype))
            embedder.eval()
            for param in embedder.parameters():
                param.requires_grad = False
            return embedder, int(config.hidden_size)
        except Exception as exc:  # noqa: BLE001 - try next weight format
            last_error = exc
            continue

    raise RuntimeError(
        f"Could not load embed_tokens for {model_id}. "
        "Expected model.safetensors.index.json or pytorch_model.bin.index.json in the HF cache."
    ) from last_error


def load_vicuna_embeddings(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> VicunaModels:
    """Load Vicuna tokenizer and frozen embed_tokens only (~vocab x hidden_size)."""
    tokenizer = load_vicuna_tokenizer(model_id)
    embedder, hidden_size = _load_embedder_only(model_id=model_id, torch_dtype=torch_dtype)
    embedder = embedder.to(torch.device(device))

    return VicunaModels(
        model_id=model_id,
        tokenizer=tokenizer,
        embedder=embedder,
        hidden_size=hidden_size,
    )


def load_vicuna_causal_lm(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict | None = None,
    max_memory: dict[int, str] | None = None,
) -> VicunaModels:
    """
    Load frozen Vicuna tokenizer + causal LM for Stage 1 NLL / ASR eval.

    Uses the Llama slow tokenizer (same as embeddings-only training). Does not
    force safetensors; Vicuna-7B v1.5 is often published as ``.bin`` shards.
    Prefers an existing Hugging Face cache snapshot (``local_files_only``) and
    downloads only when the weights are missing.
    """
    tokenizer = load_vicuna_tokenizer(model_id)
    load_kw: dict[str, Any] = dict(
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    if device_map is not None:
        load_kw["device_map"] = device_map
    if max_memory is not None:
        load_kw["max_memory"] = max_memory

    try:
        causal_lm = AutoModelForCausalLM.from_pretrained(
            model_id, local_files_only=True, **load_kw
        )
        print(f"Loaded {model_id} from local HF cache")
    except Exception:
        print(f"{model_id} not fully cached — downloading ...")
        causal_lm = AutoModelForCausalLM.from_pretrained(model_id, **load_kw)
    target = torch.device(device)
    if device_map is None:
        causal_lm = causal_lm.to(target)
    causal_lm.eval()
    for param in causal_lm.parameters():
        param.requires_grad = False

    embedder = causal_lm.get_input_embeddings()
    hidden_size = int(getattr(causal_lm.config, "hidden_size", embedder.embedding_dim))
    return VicunaModels(
        model_id=model_id,
        tokenizer=tokenizer,
        embedder=embedder,
        hidden_size=hidden_size,
        causal_lm=causal_lm,
    )
