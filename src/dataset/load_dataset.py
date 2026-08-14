import torchaudio
import os

# Set your desired download folder
# DOWNLOAD_DIR = "/home/ml/workspaces/supriya_adh/streaming_adapter/librispeech_data"
DOWNLOAD_DIR = "./datasets/librispeech_data"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

splits = ["train-clean-100", "dev-clean", "test-clean"]

for split in splits:
    print(f"Downloading split: {split} ...")
    dataset = torchaudio.datasets.LIBRISPEECH(
        root=DOWNLOAD_DIR,
        url=split,
        download=True
    )
    print(f"{split} downloaded — {len(dataset)} samples")

print("\nAll splits downloaded successfully!")

print(f"Data saved to: {os.path.abspath(DOWNLOAD_DIR)}")