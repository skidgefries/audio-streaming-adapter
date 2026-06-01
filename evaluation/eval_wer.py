"""
WER Evaluation for StreamingAdapter.

Loads adapter checkpoint from HuggingFace or local path,
runs inference on test-clean, computes Word Error Rate
against ground truth transcriptions.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        uv run python evaluation/eval_wer.py

    # Use a local checkpoint instead:
    uv run python evaluation/eval_wer.py --checkpoint checkpoints/adapter_stage2.pt

    # Limit number of utterances:
    uv run python evaluation/eval_wer.py --num-utterances 200
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import sys

import torch

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_root = os.path.join(_pkg_root, "src")
sys.path.insert(0, _src_root)
sys.path.insert(0, _pkg_root)

from huggingface_hub import hf_hub_download

from src.adapter.streaming_adapter import StreamingAdapter
from src.dataset import LibriSpeechPairs, load_mono_waveform_16k
from src.encoder import WhisperWindowFeatureExtractor
from training.utils.loaders import default_device_and_dtype, load_frozen_qwen_causal_lm

# ── Config ────────────────────────────────────────────────────────────────────

WHISPER_DIM = 768
LLM_DIM = 4096
WHISPER_MODEL = "openai/whisper-small"
LLM_MODEL_ID = "Qwen/Qwen3-8B"

TEST_CLEAN_ROOT = "datasets/librispeech_data/LibriSpeech/test-clean"

# HuggingFace checkpoint
HF_REPO_ID = "skidgefries/streaming-adapter-stage1-may28"
HF_FILENAME = "checkpoints/adapter_stage1_epoch7.pt"

MAX_NEW_TOKENS = 128    # max tokens to generate per utterance
MAX_WINDOWS = 16        # cap windows per utterance for memory


# ── HuggingFace download ──────────────────────────────────────────────────────

def download_checkpoint(repo_id: str, filename: str) -> str:
    """Download checkpoint from HuggingFace, asking for token if needed."""
    print(f"Downloading checkpoint from HuggingFace...")
    print(f"  Repo: {repo_id}")
    print(f"  File: {filename}")

    # Try without token first (works if repo is public)
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"  Downloaded to: {path}\n")
        return path
    except Exception:
        pass

    # Ask for token if private
    token = os.environ.get("HF_TOKEN") or getpass.getpass(
        "Repo appears private. Enter HuggingFace token: "
    )
    path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    print(f"  Downloaded to: {path}\n")
    return path


# ── WER ───────────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Standard dynamic programming edit distance."""
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[m]


def compute_wer(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """Compute Word Error Rate."""
    total_words = 0
    total_edits = 0
    for ref, hyp in zip(references, hypotheses):
        ref_words = _normalise(ref).split()
        hyp_words = _normalise(hyp).split()
        total_words += len(ref_words)
        total_edits += _edit_distance(ref_words, hyp_words)
    wer = total_edits / max(1, total_words) * 100
    return {
        "wer": wer,
        "total_words": total_words,
        "total_edits": total_edits,
    }


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models(checkpoint_path: str, device: str, torch_dtype: torch.dtype):
    print(f"Loading Whisper ({WHISPER_MODEL})...")
    audio_extractor = WhisperWindowFeatureExtractor(
        model_id=WHISPER_MODEL,
        device=device,
        torch_dtype=torch_dtype,
    )

    print(f"Loading Qwen ({LLM_MODEL_ID})...")
    qwen_models = load_frozen_qwen_causal_lm(
        model_id=LLM_MODEL_ID,
        device=device,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    llm_tokenizer = qwen_models.tokenizer
    llm_model = qwen_models.causal_lm
    text_embedder = qwen_models.embedder
    llm_model.eval()

    print(f"Loading adapter from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)

    stage = ckpt.get("stage", "?")
    epoch = ckpt.get("epoch", "?")
    metrics = ckpt.get("metrics", {})
    print(f"  Stage {stage} | Epoch {epoch}")
    print(f"  Metrics: {metrics}")

    adapter = StreamingAdapter(
        d_encoder=WHISPER_DIM,
        d_llm=LLM_DIM,
        num_queries=4,
        num_layers=2,
        num_heads=4,
        d_ffn=2048,
        dropout=0.0,
        ema_alpha=0.8,
        learnable_ema=False,
        use_rate_controller=stage >= 2 if isinstance(stage, int) else False,
        rate_threshold=0.5,
        target_rate=2.0,
    ).to(device)
    adapter.load_state_dict(ckpt["adapter_state_dict"], strict=False)
    adapter.eval()
    print("  Adapter loaded.\n")

    return audio_extractor, llm_tokenizer, llm_model, text_embedder, adapter


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def transcribe(
    audio_path: str,
    audio_extractor: WhisperWindowFeatureExtractor,
    adapter: StreamingAdapter,
    llm_tokenizer,
    llm_model: torch.nn.Module,
    text_embedder: torch.nn.Module,
    device: str,
) -> str:
    """Transcribe one audio file using adapter + Qwen generation."""

    # 1. Audio → adapter tokens
    wave = load_mono_waveform_16k(audio_path)
    windows = audio_extractor.waveform_to_windows(wave)
    if not windows:
        return ""

    windows = windows[:MAX_WINDOWS]

    adapter.reset_streaming_state()
    chunks = []
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for w in windows:
            out = adapter.forward_window(w.to(device, dtype=torch.float32))
            chunks.append(out["tokens"])

    audio_tokens = torch.cat(chunks, dim=1)  # (1, T_audio, 4096)

    # 2. Build input: [audio_tokens] [BOS]
    bos_id = llm_tokenizer.bos_token_id or llm_tokenizer.eos_token_id
    llm_device = next(llm_model.parameters()).device
    bos_embed = text_embedder(
        torch.tensor([[bos_id]], device=llm_device)
    )  # (1, 1, 4096)

    inputs_embeds = torch.cat(
        [audio_tokens.to(llm_device), bos_embed], dim=1
    )  # (1, T_audio+1, 4096)

    # 3. Generate transcript
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output_ids = llm_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=llm_tokenizer.eos_token_id,
            eos_token_id=llm_tokenizer.eos_token_id,
        )

    # 4. Decode
    transcript = llm_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return transcript.strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WER evaluation for StreamingAdapter")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Local path to checkpoint (.pt). If not provided, downloads from HuggingFace.",
    )
    parser.add_argument(
        "--num-utterances",
        type=int,
        default=500,
        help="Number of test-clean utterances to evaluate (default: 500)",
    )
    parser.add_argument(
        "--test-root",
        default=TEST_CLEAN_ROOT,
        help="Path to test-clean LibriSpeech directory",
    )
    args = parser.parse_args()

    # Pick checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        print(f"Using local checkpoint: {checkpoint_path}\n")
    else:
        checkpoint_path = download_checkpoint(HF_REPO_ID, HF_FILENAME)

    device, torch_dtype = default_device_and_dtype()
    print(f"Device: {device} | Dtype: {torch_dtype}\n")

    # Load models
    audio_extractor, llm_tokenizer, llm_model, text_embedder, adapter = load_models(
        checkpoint_path, device, torch_dtype
    )

    # Load dataset
    print(f"Loading test-clean from {args.test_root}...")
    dataset = LibriSpeechPairs(args.test_root)
    pairs = dataset.pairs[: args.num_utterances]
    print(f"Evaluating on {len(pairs)} utterances\n")

    # Run inference
    references = []
    hypotheses = []
    failed = 0

    for i, (audio_path, transcription) in enumerate(pairs):
        if i % 50 == 0:
            print(f"  [{i}/{len(pairs)}] processing...")

        try:
            hypothesis = transcribe(
                audio_path,
                audio_extractor,
                adapter,
                llm_tokenizer,
                llm_model,
                text_embedder,
                device,
            )
            references.append(transcription)
            hypotheses.append(hypothesis)

            # Print first 5 examples
            if i < 5:
                print(f"\n  Example {i + 1}:")
                print(f"    REF: {_normalise(transcription)}")
                print(f"    HYP: {_normalise(hypothesis)}")

        except Exception as e:
            print(f"  [WARN] Failed on {audio_path}: {e}")
            failed += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute WER
    print(f"\n{'=' * 60}")
    print("WER RESULTS")
    print(f"{'=' * 60}")
    results = compute_wer(references, hypotheses)
    print(f"  WER:          {results['wer']:.2f}%")
    print(f"  Total words:  {results['total_words']}")
    print(f"  Total edits:  {results['total_edits']}")
    print(f"  Utterances:   {len(references)} evaluated, {failed} failed")
    print(f"{'=' * 60}")

    # Interpretation
    print("\nInterpretation:")
    wer = results["wer"]
    if wer < 8:
        print("  ✅ WER < 8%  — approaches Whisper baseline, excellent")
    elif wer < 15:
        print("  ✅ WER < 15% — solid ASR performance")
    elif wer < 30:
        print("  ⚠️  WER < 30% — decent, Stage 2 still improving")
    elif wer < 60:
        print("  ⚠️  WER < 60% — marginal, adapter partially working")
    else:
        print("  ❌ WER > 60% — poor, likely Stage 1 only or untrained Stage 2")


if __name__ == "__main__":
    main()