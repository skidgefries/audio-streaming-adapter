"""
Train the linear Whisper→Qwen projector (baseline). Run from repo root::

    cd audio-streaming-adapter && uv run python training/projector_trainer.py
"""

from __future__ import annotations

import glob
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_root = os.path.join(_pkg_root, "src")
sys.path.insert(0, _src_root)
sys.path.insert(0, _pkg_root)

from src.encoder import get_encoder_output
from src.projector.projector import WhisperToQwenProjector

torch.cuda.empty_cache()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_MODEL_ID = "Qwen/Qwen3-8B"

EPOCHS = 3
BATCH_SIZE = 1
LR = 1e-4
SAVE_PATH = os.path.join(_pkg_root, "checkpoints", "projector.pt")
DATASET_ROOT = os.path.join(
    _pkg_root, "datasets", "librispeech_data", "LibriSpeech", "train-clean-100"
)


class LibriSpeechDataset(Dataset):
    """Pairs LibriSpeech flac paths with transcriptions."""

    def __init__(self, dataset_root: str):
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

    def __getitem__(self, idx: int):
        audio_path, transcription = self.pairs[idx]
        encoder_output = get_encoder_output(audio_path)
        return encoder_output.squeeze(0), transcription


def train() -> None:
    if DEVICE == "cuda":
        torch.cuda.init()

    print(f"Loading Qwen ({QWEN_MODEL_ID}) ...")
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        device_map="auto" if DEVICE == "cuda" else None,
    )
    if DEVICE == "cpu":
        qwen_model = qwen_model.to(DEVICE)
    qwen_model.eval()
    for param in qwen_model.parameters():
        param.requires_grad = False
    print("Qwen loaded and frozen.\n")

    projector = WhisperToQwenProjector(in_dim=768, out_dim=4096).half().to(DEVICE)
    projector.train()
    optimizer = torch.optim.AdamW(projector.parameters(), lr=LR)

    dataset = LibriSpeechDataset(DATASET_ROOT)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Training projector for {EPOCHS} epochs ...\n")

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for step, (encoder_output, transcription) in enumerate(dataloader):
            encoder_output = encoder_output.to(DEVICE, dtype=torch.float16)
            projected = projector(encoder_output)

            tokens = qwen_tokenizer(
                transcription,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            ).to(DEVICE)

            label_ids = tokens.input_ids
            label_embeds = qwen_model.get_input_embeddings()(label_ids)

            inputs_embeds = torch.cat([projected, label_embeds], dim=1)

            audio_ignore = torch.full(
                (encoder_output.size(0), projected.size(1)),
                -100,
                dtype=torch.long,
                device=DEVICE,
            )
            labels = torch.cat([audio_ignore, label_ids], dim=1)

            if DEVICE == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = qwen_model(inputs_embeds=inputs_embeds, labels=labels).loss
            else:
                loss = qwen_model(inputs_embeds=inputs_embeds, labels=labels).loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

            total_loss += loss.item()

            if step % 100 == 0:
                print(f"Epoch {epoch + 1} | Step {step}/{len(dataloader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / max(len(dataloader), 1)
        print(f"\nEpoch {epoch + 1} complete | Avg Loss: {avg_loss:.4f}\n")

        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        torch.save(projector.state_dict(), SAVE_PATH)
        print(f"Projector saved to {SAVE_PATH}\n")


if __name__ == "__main__":
    train()
