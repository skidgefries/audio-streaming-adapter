from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WhisperConfig:
    model_id: str = "openai/whisper-small"
    sample_rate: int = 16000
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.float16

