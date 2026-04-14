"""
Projector Module

This module provides projection networks for transforming Whisper encoder
embeddings into LLM embedding spaces.
"""

from .projector import WhisperToQwenProjector

__all__ = ["WhisperToQwenProjector"]
