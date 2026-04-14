from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from .config import WhisperConfig


@dataclass(frozen=True)
class WhisperAsrModels:
    model_id: str
    processor: Any
    model: Any
    pipe: Any
    device: str
    torch_dtype: torch.dtype


def load_whisper_asr_models(cfg: WhisperConfig | None = None) -> WhisperAsrModels:
    """
    Load a Whisper ASR pipeline (HF `pipeline("automatic-speech-recognition")`).

    This replaces `utils/whisper_model_loader.py`.
    """
    if cfg is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        cfg = WhisperConfig(device=device, torch_dtype=torch_dtype)

    processor = AutoProcessor.from_pretrained(cfg.model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        cfg.model_id,
        torch_dtype=cfg.torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(cfg.device)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=cfg.torch_dtype,
        device=cfg.device,
    )

    return WhisperAsrModels(
        model_id=cfg.model_id,
        processor=processor,
        model=model,
        pipe=pipe,
        device=cfg.device,
        torch_dtype=cfg.torch_dtype,
    )

