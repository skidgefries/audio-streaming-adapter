"""Smart Turn dataset for turn-end gate supervision."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


class SmartTurnGateDataset(Dataset):
    """
    HuggingFace Smart Turn clips with ``endpoint_bool`` labels for gate training.

    Each item is ``(waveform_tensor, endpoint_label)`` where waveform is 16 kHz mono
    float32 on CPU and endpoint_label is 1.0 (turn complete) or 0.0 (incomplete).
    """

    def __init__(
        self,
        dataset_id: str = "pipecat-ai/smart-turn-data-v3.2-train",
        *,
        split: str = "train",
        max_samples: int | None = None,
    ):
        from datasets import load_dataset

        loaded = load_dataset(dataset_id)
        if split not in loaded:
            available = list(loaded.keys())
            raise ValueError(
                f"Split {split!r} not in dataset {dataset_id!r}; available: {available}"
            )
        ds = loaded[split]
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        self.ds = ds
        print(f"SmartTurnGateDataset: {len(self.ds)} clips from {dataset_id} [{split}]")

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        sample = self.ds[idx]
        waveform = _audio_to_mono_16k_tensor(sample["audio"])
        endpoint = 1.0 if bool(sample["endpoint_bool"]) else 0.0
        return waveform, endpoint


def _audio_to_mono_16k_tensor(audio: dict[str, Any]) -> torch.Tensor:
    """Convert HF Audio dict to 1D float32 tensor at 16 kHz."""
    import numpy as np

    array = audio["array"]
    sample_rate = int(audio["sampling_rate"])

    if hasattr(array, "numpy"):
        waveform = array.numpy()
    else:
        waveform = np.asarray(array, dtype=np.float32)

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=0)

    waveform = waveform.astype(np.float32, copy=False)
    if sample_rate != 16000:
        import librosa

        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)

    return torch.tensor(waveform, dtype=torch.float32)


def smart_turn_collate(
    batch: list[tuple[torch.Tensor, float]],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Collate variable-length waveforms with endpoint labels."""
    waveforms = [item[0] for item in batch]
    labels = torch.tensor([[item[1]] for item in batch], dtype=torch.float32)
    return waveforms, labels
