"""
Notebook / legacy helpers: LibriSpeech paths, old checkpoint format.

Prefer `dataset.*` for data and `training.utils.checkpointing` for full training checkpoints.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from dataset import LibriSpeechConfig, LibriSpeechPairs, load_mono_waveform_16k
from training.utils.checkpointing import load_adapter_state_dict


def default_librispeech_root_from_training_dir(training_dir: str) -> str:
    """Resolve LibriSpeech root relative to a `training/` or `training/utils/` caller directory."""
    return LibriSpeechConfig.default_train_clean_100_from_training_dir(training_dir).root


@dataclass(frozen=True)
class AdapterCheckpoint:
    stage: int
    adapter_state_dict: dict
    gate_state_dict: dict | None = None
    avg_loss: float | None = None
    extra: dict | None = None


def save_checkpoint(path: str, ckpt: AdapterCheckpoint) -> None:
    """Legacy saver used by older notebooks."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "stage": ckpt.stage,
        "adapter_state_dict": ckpt.adapter_state_dict,
        "avg_loss": ckpt.avg_loss,
    }
    if ckpt.gate_state_dict is not None:
        payload["gate_state_dict"] = ckpt.gate_state_dict
    if ckpt.extra:
        payload.update(ckpt.extra)
    torch.save(payload, path)


__all__ = [
    "AdapterCheckpoint",
    "LibriSpeechConfig",
    "LibriSpeechPairs",
    "default_librispeech_root_from_training_dir",
    "load_adapter_state_dict",
    "load_mono_waveform_16k",
    "save_checkpoint",
]
