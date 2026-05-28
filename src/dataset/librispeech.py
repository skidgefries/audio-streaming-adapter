from __future__ import annotations

import glob
import os
from collections.abc import Sequence

import torch
from torch.utils.data import Dataset


def _index_librispeech_pairs(dataset_root: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
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
                    pairs.append((audio_path, transcription))
    return pairs


class LibriSpeechPairs(Dataset):
    """Index LibriSpeech (audio_path, transcription) pairs from `*.trans.txt` files."""

    def __init__(self, dataset_root: str | Sequence[str]):
        roots = [dataset_root] if isinstance(dataset_root, str) else list(dataset_root)
        self.dataset_roots = roots
        self.dataset_root = roots[0]
        self.pairs: list[tuple[str, str]] = []

        for root in roots:
            split_pairs = _index_librispeech_pairs(root)
            print(f"  {root}: {len(split_pairs)} pairs")
            self.pairs.extend(split_pairs)

        print(f"Found {len(self.pairs)} audio-transcription pairs across {len(roots)} split(s)")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.pairs[idx]



# tst if the loss is working with a subset of the dataset
class LibriSpeechPairsCustom(Dataset):
    """LibriSpeech dataset with specific audio files by file ID."""

    def __init__(self, dataset_root: str, file_ids: list[str]):
        self.dataset_root = dataset_root
        self.pairs: list[tuple[str, str]] = []
        
        # Build a lookup of all available pairs first
        all_pairs: dict[str, tuple[str, str]] = {}
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
                        all_pairs[file_id] = (audio_path, transcription)

        # Only keep the requested file IDs
        for fid in file_ids:
            if fid in all_pairs:
                self.pairs.append(all_pairs[fid])
            else:
                print(f"[WARN] file_id '{fid}' not found in dataset")

        print(f"Found {len(self.pairs)}/{len(file_ids)} requested pairs")

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

