"""
Audio encoder subpackage.

- `WhisperConfig`, `load_whisper_models`
- `encode_waveform_to_hidden`, `get_encoder_output`, `encode_dataset_stream` (``whisper_encoder.py``)
- `WhisperWindowFeatureExtractor` (full encode, then :class:`adapter.windowing.WhisperFrameWindowizer`)
- `load_whisper_asr_models` (HF ASR pipeline)
"""

from .config import WhisperConfig
from .whisper_encoder import (
    WhisperModels,
    encode_dataset_stream,
    encode_waveform_to_hidden,
    get_encoder_output,
    load_whisper_models,
)
from .asr_pipeline import WhisperAsrModels, load_whisper_asr_models
from .waveform_window_encoder import WhisperWindowFeatureExtractor

__all__ = [
    "WhisperConfig",
    "WhisperModels",
    "load_whisper_models",
    "encode_waveform_to_hidden",
    "get_encoder_output",
    "encode_dataset_stream",
    "WhisperAsrModels",
    "load_whisper_asr_models",
    "WhisperWindowFeatureExtractor",
]
