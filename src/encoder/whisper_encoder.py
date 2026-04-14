from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from dataset.librispeech import load_mono_waveform_16k

from .config import WhisperConfig


@dataclass(frozen=True)
class WhisperModels:
    model_id: str
    processor: Any
    model: Any


def load_whisper_models(
    *,
    cfg: WhisperConfig | None = None,
    model_id: str = "openai/whisper-small",
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
) -> WhisperModels:
    if cfg is not None:
        model_id = cfg.model_id
        device = cfg.device
        torch_dtype = cfg.torch_dtype
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return WhisperModels(model_id=model_id, processor=processor, model=model)


def encode_waveform_to_hidden(
    waveform,
    *,
    whisper_processor: Any,
    whisper_model: Any,
    device: str,
    torch_dtype: torch.dtype,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """
    Encode a waveform into Whisper encoder hidden states `(1, T, D)`.

    `whisper_processor` expects CPU/NumPy-like audio. If a CUDA tensor is passed, it is
    moved to CPU first.
    """
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().float().cpu()

    inputs = whisper_processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
    input_features = inputs.input_features.to(device, dtype=torch_dtype)

    with torch.no_grad():
        encoder_outputs = whisper_model.model.encoder(input_features)
    return encoder_outputs.last_hidden_state


# --- File / dataset helpers (used by qwen_summarize_*.py demos) -----------------

_whisper_by_model_id: dict[str, WhisperModels] = {}


def _get_whisper_cached(*, model_id: str = "openai/whisper-small") -> WhisperModels:
    if model_id not in _whisper_by_model_id:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        if device == "cuda":
            torch.cuda.init()
        _whisper_by_model_id[model_id] = load_whisper_models(
            model_id=model_id, device=device, torch_dtype=torch_dtype
        )
    return _whisper_by_model_id[model_id]


def get_encoder_output(file_path: str, *, model_id: str = "openai/whisper-small") -> torch.Tensor:
    """
    Load an audio file and return Whisper encoder hidden states ``(1, T, D)`` (CPU float).

    Typical: ``T`` ~1500 frames, ``D=768`` for whisper-small.
    """
    waveform = load_mono_waveform_16k(file_path)
    wm = _get_whisper_cached(model_id=model_id)
    p = next(wm.model.parameters())
    enc = encode_waveform_to_hidden(
        waveform,
        whisper_processor=wm.processor,
        whisper_model=wm.model,
        device=str(p.device),
        torch_dtype=p.dtype,
    )
    return enc.detach().float().cpu()


def encode_dataset_stream(
    dataset_root: str,
    *,
    patterns: tuple[str, ...] = ("**/*.flac", "**/*.wav"),
    limit: int | None = None,
    model_id: str = "openai/whisper-small",
) -> Iterator[dict[str, Any]]:
    """
    Yield ``{"path", "encoded"}`` for each audio file under ``dataset_root``.

    ``encoded`` is ``(1, T, D)`` on CPU (float32).
    """
    files: list[str] = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(dataset_root, pat), recursive=True))
    files = sorted(set(files))
    for i, fp in enumerate(files):
        if limit is not None and i >= limit:
            break
        yield {"path": fp, "encoded": get_encoder_output(fp, model_id=model_id)}
