from __future__ import annotations

import glob
import os

import torch
from torch.utils.data import Dataset


class LibriSpeechPairs(Dataset):
    """Index LibriSpeech (audio_path, transcription) pairs from `*.trans.txt` files."""

    def __init__(self, dataset_root: str):
        self.dataset_root = dataset_root
        self.pairs: list[tuple[str, str]] = []

        for trans_file in glob.glob(f"{dataset_root}/**/*.trans.txt", recursive=True):
            folder = os.path.dirname(trans_file)
            with open(trans_file) as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) != 2:
                        continue
                    file_id, transcription = parts
                    audio_path = os.path.join(folder, f"{file_id}.flac")
                    if os.path.exists(audio_path):
                        self.pairs.append((audio_path, transcription))

        print(f"Found {len(self.pairs)} audio-transcription pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.pairs[idx]


def load_mono_waveform_16k(audio_path: str) -> torch.Tensor:
    """Load audio with librosa and resample to 16k mono. Returns CPU float tensor."""
    import librosa

    waveform, sample_rate = librosa.load(audio_path, sr=None)
    if waveform.ndim > 1:
        waveform = librosa.to_mono(waveform)
    if sample_rate != 16000:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
    return torch.tensor(waveform, dtype=torch.float32)

