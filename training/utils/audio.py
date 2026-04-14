"""
Deprecated location: sliding-window Whisper lives under ``encoder``.

Prefer::

    from encoder import WhisperWindowFeatureExtractor

or the full path::

    from encoder.waveform_window_encoder import WhisperWindowFeatureExtractor
"""

from src.encoder import WhisperWindowFeatureExtractor

__all__ = ["WhisperWindowFeatureExtractor"]
