"""
Retrieval evaluation for Stage 1 alignment.

Computes R@1, R@5, R@10 on a held-out set (test-clean).
For each audio embedding, ranks all candidate texts by cosine similarity
and checks if the correct text is in the top K.
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext

import torch
import torch.nn.functional as F

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_root = os.path.join(_pkg_root, "src")
_training_dir = os.path.join(_pkg_root, "training")
sys.path.insert(0, _src_root)
sys.path.insert(0, _pkg_root)

from src.adapter.streaming_adapter import StreamingAdapter
from src.dataset import LibriSpeechConfig, LibriSpeechPairs, load_mono_waveform_16k
from src.encoder.waveform_window_encoder import WhisperWindowFeatureExtractor
from training.utils.config import CheckpointConfig, FrozenModelIdsConfig
from training.utils.devices import cleanup_distributed, init_training_context, llm_device_map, llm_input_device
from training.utils.env import env_int, load_project_env
from training.utils.loaders import default_device_and_dtype, load_frozen_qwen_embeddings
from training.utils.stage1_validation import compute_retrieval_recall

WHISPER_DIM = 768
LLM_DIM = 4096


def _maybe_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def load_models(
    *,
    checkpoint_path: str,
    whisper_model_id: str,
    llm_model_id: str,
    device: torch.device,
    torch_dtype: torch.dtype,
    llm_map: str | None,
):
    device_str = str(device)
    print("Loading Whisper...")
    audio = WhisperWindowFeatureExtractor(
        model_id=whisper_model_id,
        device=device_str,
        torch_dtype=torch_dtype,
    )

    print("Loading Qwen embedder...")
    qwen_models = load_frozen_qwen_embeddings(
        model_id=llm_model_id,
        device=device_str,
        torch_dtype=torch_dtype,
        device_map=llm_map,
    )
    llm_device = llm_input_device(qwen_models.embedder)

    print("Loading adapter from checkpoint...")
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
        use_rate_controller=False,
    ).to(device, dtype=torch.bfloat16)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    adapter.eval()
    print(
        f"  Loaded {checkpoint_path}\n"
        f"  epoch={ckpt.get('epoch', '?')} step={ckpt.get('global_step', '?')}\n"
    )

    return audio, qwen_models.tokenizer, qwen_models.embedder, adapter, llm_device


@torch.no_grad()
def compute_embeddings(
    audio_extractor,
    tokenizer,
    text_embedder,
    adapter,
    pairs,
    *,
    device: torch.device,
    llm_device: torch.device,
    max_text_tokens: int,
):
    audio_vecs = []
    text_vecs = []

    for i, (audio_path, transcription) in enumerate(pairs):
        if i % 50 == 0:
            print(f"  Processing {i}/{len(pairs)}...")

        wave = load_mono_waveform_16k(audio_path)
        windows = audio_extractor.waveform_to_windows(wave)
        if len(windows) == 0:
            continue

        adapter.reset_streaming_state()
        chunks = []
        with _maybe_autocast(device):
            for w in windows:
                out = adapter.forward_window(w)
                chunks.append(out["tokens"])

        audio_tokens = torch.cat(chunks, dim=1)
        audio_pooled = audio_tokens.float().mean(dim=1)
        audio_vecs.append(audio_pooled.squeeze(0).cpu())

        text_tokens = tokenizer(
            transcription,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_text_tokens,
        )
        label_embeds = text_embedder(text_tokens.input_ids.to(llm_device)).float()
        text_pooled = label_embeds.mean(dim=1)
        text_vecs.append(text_pooled.squeeze(0).cpu())

    audio_bank = torch.stack(audio_vecs)
    text_bank = torch.stack(text_vecs)

    audio_bank = audio_bank - audio_bank.mean(dim=0, keepdim=True)
    text_bank = text_bank - text_bank.mean(dim=0, keepdim=True)
    audio_bank = F.normalize(audio_bank, dim=-1)
    text_bank = F.normalize(text_bank, dim=-1)

    return audio_bank, text_bank


def interpret_r1(r1: float) -> str:
    if r1 >= 30:
        return "  R@1 ≥ 30% — solid alignment, ready for Stage 2"
    if r1 >= 10:
        return "  R@1 ≥ 10% — marginal, Stage 2 may refine"
    return "  R@1 < 10% — alignment too weak, continue Stage 1"


def main() -> None:
    load_project_env(_pkg_root)

    model_ids = FrozenModelIdsConfig.from_env()
    ckpt_cfg = CheckpointConfig.from_env(pkg_root=_pkg_root)
    default_checkpoint = os.path.join(ckpt_cfg.dir, "adapter_stage1.pt")
    default_test_root = LibriSpeechConfig.test_clean_root(_training_dir)

    ap = argparse.ArgumentParser(description="Stage 1 retrieval eval (R@1, R@5, R@10 on test-clean)")
    ap.add_argument("--checkpoint", type=str, default=default_checkpoint)
    ap.add_argument("--dataset-root", type=str, default=default_test_root)
    ap.add_argument(
        "--num-utterances",
        type=int,
        default=env_int("RETRIEVAL_NUM_UTTERANCES", 100),
    )
    ap.add_argument("--max-text-tokens", type=int, default=128)
    args = ap.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    ctx = init_training_context()
    _, torch_dtype = default_device_and_dtype()
    device = ctx.device
    llm_map = llm_device_map(ctx)

    if ctx.is_main:
        print(f"Device: {device} (visible CUDA devices: {ctx.num_cuda_devices})")
        print(f"Qwen device_map: {llm_map!r}")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Loading test-clean from {args.dataset_root}...")
    dataset = LibriSpeechPairs(args.dataset_root)
    pairs = dataset.pairs[: args.num_utterances]
    print(f"Evaluating on {len(pairs)} utterances\n")

    audio, tokenizer, text_embedder, adapter, llm_device = load_models(
        checkpoint_path=args.checkpoint,
        whisper_model_id=model_ids.whisper_model_id,
        llm_model_id=model_ids.llm_model_id,
        device=device,
        torch_dtype=torch_dtype,
        llm_map=llm_map,
    )

    print("Computing embeddings...")
    audio_bank, text_bank = compute_embeddings(
        audio,
        tokenizer,
        text_embedder,
        adapter,
        pairs,
        device=device,
        llm_device=llm_device,
        max_text_tokens=args.max_text_tokens,
    )
    print(f"  Audio bank: {audio_bank.shape}")
    print(f"  Text bank:  {text_bank.shape}\n")

    print("Computing retrieval metrics...")
    recall = compute_retrieval_recall(audio_bank, text_bank)

    print("\n" + "=" * 50)
    print("RETRIEVAL RESULTS (Audio → Text)")
    print("=" * 50)
    for k in (1, 5, 10):
        print(f"  R@{k}: {recall[f'recall_at_{k}']:.2f}%")
    print("=" * 50)

    print("\nInterpretation:")
    print(interpret_r1(recall["recall_at_1"]))

    cleanup_distributed()


if __name__ == "__main__":
    main()
