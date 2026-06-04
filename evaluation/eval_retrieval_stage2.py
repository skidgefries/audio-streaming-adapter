"""
Retrieval evaluation for Stage 2 (ASR distillation) checkpoints.

Same metrics and procedure as ``eval_retrieval.py`` (R@1, R@5, R@10 on test-clean),
but builds :class:`StreamingAdapter` with the Stage 2 options (rate controller, etc.)
so ``adapter_stage2.pt`` loads without state_dict key mismatches.

The early-commit gate in the checkpoint is not used for retrieval pooling.
"""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext

import torch
import torch.nn.functional as F

_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_root = os.path.join(_pkg_root, "src")
sys.path.insert(0, _src_root)
sys.path.insert(0, _pkg_root)

from src.adapter.streaming_adapter import StreamingAdapter
from src.dataset import LibriSpeechPairs, load_mono_waveform_16k
from src.encoder.waveform_window_encoder import WhisperWindowFeatureExtractor
from training.utils.config import Stage2Config
from training.utils.devices import TrainingContext, llm_device_map, llm_input_device, resolve_device
from training.utils.env import env_int, load_project_env
from training.utils.loaders import load_frozen_qwen_embeddings

STAGE2 = Stage2Config.from_env()


def _torch_dtype_for(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def _maybe_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()

WHISPER_DIM = 768
LLM_DIM = 4096
WHISPER_MODEL = "openai/whisper-small"
LLM_MODEL_ID = "Qwen/Qwen3-8B"

CHECKPOINT_PATH = os.path.join(_pkg_root, "checkpoints", "adapter_stage2.pt")
TEST_CLEAN_ROOT = os.path.join(
    _pkg_root, "datasets/librispeech_data/LibriSpeech/test-clean"
)
NUM_UTTERANCES = 2620


def _qwen_max_memory() -> dict[int, str] | None:
    """Cap GPU use for Qwen (e.g. ``QWEN_MAX_CUDA_GIB=6`` → ~5–7 GB on device 0)."""
    gib = env_int("QWEN_MAX_CUDA_GIB", 0)
    if gib <= 0:
        return None
    return {0: f"{gib}GiB"}


def load_models(
    checkpoint_path: str = CHECKPOINT_PATH,
    *,
    device: torch.device,
    torch_dtype: torch.dtype,
    llm_map: str | None,
):
    max_memory = _qwen_max_memory() if device.type == "cuda" else None
    if max_memory:
        print(f"  Qwen max CUDA memory: {max_memory[0]}")

    device_str = str(device)
    print("Loading Whisper...")
    audio = WhisperWindowFeatureExtractor(
        model_id=WHISPER_MODEL,
        device=device_str,
        torch_dtype=torch_dtype,
    )

    print("Loading Qwen embedder...")
    qwen_models = load_frozen_qwen_embeddings(
        model_id=LLM_MODEL_ID,
        device=device_str,
        torch_dtype=torch_dtype,
        device_map=llm_map,
        max_memory=max_memory,
    )
    llm_device = llm_input_device(qwen_models.embedder)

    adapter_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print("Loading Stage 2 adapter from checkpoint...")
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
        use_rate_controller=STAGE2.use_rate_controller,
        rate_threshold=0.5,
        target_rate=STAGE2.rate_target,
    ).to(device, dtype=adapter_dtype)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    adapter.eval()
    print(
        f"  Loaded {checkpoint_path}\n"
        f"  epoch={ckpt.get('epoch', '?')} step={ckpt.get('global_step', '?')} "
        f"rate_controller={STAGE2.use_rate_controller} target_rate={STAGE2.rate_target}\n"
    )

    return audio, qwen_models.tokenizer, qwen_models.embedder, adapter, llm_device


@torch.no_grad()
def compute_embeddings(
    audio_extractor, tokenizer, text_embedder, adapter, pairs, *, device: torch.device, llm_device
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
            max_length=STAGE2.max_text_tokens,
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


def compute_recall(audio_bank, text_bank, ks=(1, 5, 10)):
    n = audio_bank.shape[0]
    sim_matrix = audio_bank @ text_bank.T
    ranked = sim_matrix.argsort(dim=1, descending=True)

    results = {}
    for k in ks:
        top_k = ranked[:, :k]
        correct = torch.arange(n).unsqueeze(1)
        hits = (top_k == correct).any(dim=1).float()
        results[f"R@{k}"] = hits.mean().item() * 100

    return results


def main():
    load_project_env(_pkg_root)
    device = resolve_device()
    torch_dtype = _torch_dtype_for(device)
    ctx = TrainingContext(
        device=device,
        rank=0,
        world_size=1,
        local_rank=0,
        is_main=True,
        num_cuda_devices=0 if device.type == "cpu" else torch.cuda.device_count(),
        model_parallel=device.type == "cuda" and torch.cuda.device_count() >= 2,
    )
    llm_map = llm_device_map(ctx)

    print(f"Device: {device} (Qwen device_map: {llm_map!r})")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Loading test-clean from {TEST_CLEAN_ROOT}...")
    dataset = LibriSpeechPairs(TEST_CLEAN_ROOT)
    pairs = dataset.pairs[:NUM_UTTERANCES]
    print(f"Evaluating on {len(pairs)} utterances\n")

    audio, tokenizer, text_embedder, adapter, llm_device = load_models(
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
    )
    print(f"  Audio bank: {audio_bank.shape}")
    print(f"  Text bank:  {text_bank.shape}\n")

    print("Computing retrieval metrics...")
    results = compute_recall(audio_bank, text_bank)

    print("\n" + "=" * 50)
    print("RETRIEVAL RESULTS — Stage 2 checkpoint (Audio → Text)")
    print("=" * 50)
    for k, v in results.items():
        print(f"  {k}: {v:.2f}%")
    print("=" * 50)

    print("\nInterpretation:")
    r1 = results["R@1"]
    if r1 >= 30:
        print("  R@1 ≥ 30% — strong audio–text alignment after Stage 2")
    elif r1 >= 10:
        print("  R@1 ≥ 10% — moderate alignment; compare with Stage 1 eval")
    else:
        print("  R@1 < 10% — weak alignment; check checkpoint and rate-controller settings")


if __name__ == "__main__":
    main()
