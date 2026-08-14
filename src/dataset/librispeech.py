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
    """Decode audio to 16 kHz mono float32 on CPU.

    FLAC/WAV decode is CPU-only (no CUDA codec). Prefer this from DataLoader
    workers so the GPU is not stalled on I/O. Whisper log-mel + encode should
    run on CUDA after the tensor is moved.
    """
    import torchaudio

    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.reshape(-1).contiguous().float()
    if int(sample_rate) != 16000:
        waveform = torchaudio.functional.resample(waveform, int(sample_rate), 16000)
    return waveform


class LibriSpeechWaveformPairs(LibriSpeechPairs):
    """Same index as ``LibriSpeechPairs``, but ``__getitem__`` also decodes audio."""

    def __getitem__(self, idx: int) -> tuple[str, str, torch.Tensor]:
        audio_path, transcription = self.pairs[idx]
        return audio_path, transcription, load_mono_waveform_16k(audio_path)


def collate_librispeech_waveforms(
    batch: list[tuple[str, str, torch.Tensor]],
) -> tuple[list[str], list[str], list[torch.Tensor]]:
    paths, texts, waves = zip(*batch, strict=True)
    return list(paths), list(texts), list(waves)

