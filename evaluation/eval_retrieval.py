"""
Retrieval evaluation for Stage 1 alignment.

Computes R@1, R@5, R@10 on a held-out set (test-clean).
For each audio embedding, ranks all candidate texts by cosine similarity
and checks if the correct text is in the top K.
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_root = os.path.join(_pkg_root, "src")
sys.path.insert(0, _src_root)
sys.path.insert(0, _pkg_root)

from src.adapter.streaming_adapter import StreamingAdapter
from src.dataset import LibriSpeechPairs, load_mono_waveform_16k
from src.encoder.waveform_window_encoder import WhisperWindowFeatureExtractor, unpack_encoder_window
from training.utils.loaders import default_device_and_dtype, load_frozen_qwen_embeddings

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE, TORCH_DTYPE = default_device_and_dtype()

WHISPER_DIM = 768
LLM_DIM = 4096
WHISPER_MODEL = "openai/whisper-small"
LLM_MODEL_ID = "Qwen/Qwen3-8B"

CHECKPOINT_PATH = "checkpoints/adapter_stage1_epoch7.pt"
TEST_CLEAN_ROOT = "datasets/librispeech_data/LibriSpeech/test-clean"
NUM_UTTERANCES = 100  # how many to evaluate on


# ── Load models ───────────────────────────────────────────────────────────────
def load_models():
    print("Loading Whisper...")
    audio = WhisperWindowFeatureExtractor(
        model_id=WHISPER_MODEL,
        device=DEVICE,
        torch_dtype=TORCH_DTYPE,
    )

    print("Loading Qwen embedder...")
    qwen_models = load_frozen_qwen_embeddings(
        model_id=LLM_MODEL_ID,
        device=DEVICE,
        torch_dtype=TORCH_DTYPE,
        device_map="auto",
    )

    print("Loading adapter from checkpoint...")
    adapter = StreamingAdapter(
        d_encoder=WHISPER_DIM,
        d_llm=LLM_DIM,
        num_queries=4,
        num_layers=2,
        num_heads=4,
        d_ffn=2048,
        dropout=0.0,  # no dropout at eval
        ema_alpha=0.8,
        learnable_ema=False,
        use_rate_controller=False,
    ).to(DEVICE, dtype=torch.bfloat16)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    adapter.eval()
    print(f"  Loaded checkpoint from epoch {ckpt['epoch']}, step {ckpt['global_step']}\n")

    return audio, qwen_models.tokenizer, qwen_models.embedder, adapter


# ── Compute embeddings ────────────────────────────────────────────────────────
@torch.no_grad()
def compute_embeddings(audio_extractor, tokenizer, text_embedder, adapter, pairs):
    audio_vecs = []
    text_vecs = []

    for i, (audio_path, transcription) in enumerate(pairs):
        if i % 50 == 0:
            print(f"  Processing {i}/{len(pairs)}...")

        # ── Audio embedding ──
        wave = load_mono_waveform_16k(audio_path)
        windows = audio_extractor.waveform_to_windows(wave)
        if len(windows) == 0:
            continue

        adapter.reset_streaming_state()
        chunks = []
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for w in windows:
                enc, enc_mask = unpack_encoder_window(w)
                out = adapter.forward_window(enc, encoder_attention_mask=enc_mask)
                chunks.append(out["tokens"])

        audio_tokens = torch.cat(chunks, dim=1)  # (1, T, 4096)
        audio_pooled = audio_tokens.float().mean(dim=1)  # (1, 4096)
        audio_vecs.append(audio_pooled.squeeze(0).cpu())

        # ── Text embedding ──
        text_tokens = tokenizer(
            transcription,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(DEVICE)
        label_embeds = text_embedder(text_tokens.input_ids).float()  # (1, T, 4096)
        text_pooled = label_embeds.mean(dim=1)  # (1, 4096)
        text_vecs.append(text_pooled.squeeze(0).cpu())

    audio_bank = torch.stack(audio_vecs)  # (N, 4096)
    text_bank = torch.stack(text_vecs)   # (N, 4096)

    # Center and normalize
    audio_bank = audio_bank - audio_bank.mean(dim=0, keepdim=True)
    text_bank = text_bank - text_bank.mean(dim=0, keepdim=True)
    audio_bank = F.normalize(audio_bank, dim=-1)
    text_bank = F.normalize(text_bank, dim=-1)

    return audio_bank, text_bank


# ── Compute recall ────────────────────────────────────────────────────────────
def compute_recall(audio_bank, text_bank, ks=(1, 5, 10)):
    n = audio_bank.shape[0]

    # (N, N) similarity matrix
    sim_matrix = audio_bank @ text_bank.T

    # For each audio, rank all texts
    ranked = sim_matrix.argsort(dim=1, descending=True)  # (N, N)

    results = {}
    for k in ks:
        top_k = ranked[:, :k]  # (N, k)
        correct = torch.arange(n).unsqueeze(1)  # (N, 1)
        hits = (top_k == correct).any(dim=1).float()
        results[f"R@{k}"] = hits.mean().item() * 100

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    torch.cuda.empty_cache()

    # Load dataset
    print(f"Loading test-clean from {TEST_CLEAN_ROOT}...")
    dataset = LibriSpeechPairs(TEST_CLEAN_ROOT)
    pairs = dataset.pairs[:NUM_UTTERANCES]
    print(f"Evaluating on {len(pairs)} utterances\n")

    # Load models
    audio, tokenizer, text_embedder, adapter = load_models()

    # Compute embeddings
    print("Computing embeddings...")
    audio_bank, text_bank = compute_embeddings(
        audio, tokenizer, text_embedder, adapter, pairs
    )
    print(f"  Audio bank: {audio_bank.shape}")
    print(f"  Text bank:  {text_bank.shape}\n")

    # Compute retrieval metrics
    print("Computing retrieval metrics...")
    results = compute_recall(audio_bank, text_bank)

    print("\n" + "="*50)
    print("RETRIEVAL RESULTS (Audio → Text)")
    print("="*50)
    for k, v in results.items():
        print(f"  {k}: {v:.2f}%")
    print("="*50)

    # Interpret results
    print("\nInterpretation:")
    r1 = results["R@1"]
    if r1 >= 30:
        print("  R@1 ≥ 30% — solid alignment, ready for Stage 2")
    elif r1 >= 10:
        print("  R@1 ≥ 10% — marginal, Stage 2 may refine")
    else:
        print("  R@1 < 10% — alignment too weak, continue Stage 1")


if __name__ == "__main__":
    main()