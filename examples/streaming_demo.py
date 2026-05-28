#!/usr/bin/env env python3
"""
End-to-end streaming demo: per-window KV-cache + generate on turn-end commit.

Uses Qwen3-8B, Whisper-small, stage-2 adapter + TurnEndCommitGate checkpoint.

Example::

    cd audio-streaming-adapter
    source .venv/bin/activate
    PYTHONPATH=src:training python examples/streaming_demo.py

    # Shorter decode, CPU-only if GPUs are full:
    CUDA_VISIBLE_DEVICES= PYTHONPATH=src:training python examples/streaming_demo.py \\
        --device cpu --max-new-tokens 24 --duration-s 2.4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TRAINING = ROOT / "training"
for p in (SRC, TRAINING, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from adapter.streaming_adapter import StreamingAdapter
from adapter.turn_end_commit_gate import TurnEndCommitGate
from adapter.windowing import AudioWaveformWindowizer
from adapter_llm_pipeline import WhisperAdapterLLMCommitGatePipeline
from encoder import WhisperConfig, load_whisper_models
from llm import load_qwen_models
from training.utils.checkpointing import load_gate_state_dict_safe


def _default_checkpoint() -> Path:
    return ROOT / "checkpoints" / "adapter_stage2.pt"


def _make_demo_waveform(*, duration_s: float, sample_rate: int = 16000) -> torch.Tensor:
    """Speech-like synthetic mono waveform (random + slow envelope)."""
    n = int(duration_s * sample_rate)
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(n).astype(np.float32) * 0.02
    t = np.linspace(0, duration_s, n, dtype=np.float32)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t)
    wave = noise * envelope
    return torch.from_numpy(wave)


def _load_models(
    *,
    device: str,
    torch_dtype: torch.dtype,
    checkpoint: Path,
    use_rate_controller: bool,
):
    whisper = load_whisper_models(
        cfg=WhisperConfig(
            model_id="openai/whisper-small",
            device=device,
            torch_dtype=torch_dtype,
        )
    )

    llm_device_map = None if device.startswith("cuda") and ":" in device else ("auto" if device == "cuda" else None)
    qwen = load_qwen_models(
        model_id="Qwen/Qwen3-8B",
        device=device if llm_device_map is None else "cuda",
        torch_dtype=torch_dtype,
        device_map=llm_device_map,
    )

    adapter = StreamingAdapter(
        d_encoder=768,
        d_llm=4096,
        num_queries=4,
        num_layers=2,
        num_heads=4,
        d_ffn=2048,
        dropout=0.0,
        ema_alpha=0.8,
        learnable_ema=False,
        use_rate_controller=use_rate_controller,
        target_rate=2.0,
    ).to(device=device, dtype=torch_dtype)

    gate = TurnEndCommitGate(
        d_llm=4096,
        hidden_dim=256,
        threshold=0.5,
        latency_weight=0.1,
        min_silence_ms=200.0,
        require_silence_for_commit=True,
        token_activity_threshold=8.0,
    ).to(device=device, dtype=torch_dtype)

    ckpt = torch.load(checkpoint, map_location="cpu")
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    load_gate_state_dict_safe(gate, ckpt)
    adapter.eval()
    gate.eval()

    windowizer = AudioWaveformWindowizer(
        sample_rate=16000,
        window_seconds=0.8,
        stride_seconds=0.4,
    )

    pipeline = WhisperAdapterLLMCommitGatePipeline(
        whisper_processor=whisper.processor,
        whisper_model=whisper.model,
        windowizer=windowizer,
        streaming_adapter=adapter,
        early_commit_gate=gate,
        llm_model=qwen.causal_lm,
        llm_tokenizer=qwen.tokenizer,
        device=device,
        torch_dtype=torch_dtype,
    )
    return pipeline


def _print_streaming_trace(result: dict) -> None:
    print("\n--- Per-window trace ---")
    for step in result.get("window_steps", []):
        n_tok = int(step.window_tokens.shape[1])
        print(
            f"  window {step.window_index}: {n_tok} audio tokens | "
            f"commit_prob={step.commit_prob:.3f} should_commit={step.should_commit} "
            f"committed={step.committed}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming KV-cache demo (Qwen3-8B)")
    parser.add_argument("--checkpoint", type=Path, default=_default_checkpoint())
    parser.add_argument("--device", default=os.environ.get("DEMO_DEVICE", "cuda:0"))
    parser.add_argument("--duration-s", type=float, default=2.4, help="Synthetic audio length")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--no-rate-controller", action="store_true")
    parser.add_argument("--compare-batch", action="store_true", help="Also run batch generate()")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = args.device
    if device == "cuda":
        device = "cuda:0"
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

    print(f"Loading models on {device} ({torch_dtype}) ...")
    t_load = time.time()
    pipeline = _load_models(
        device=device,
        torch_dtype=torch_dtype,
        checkpoint=args.checkpoint,
        use_rate_controller=not args.no_rate_controller,
    )
    print(f"  loaded in {time.time() - t_load:.1f}s")

    waveform = _make_demo_waveform(duration_s=args.duration_s)
    print(f"\nDemo audio: {args.duration_s}s synthetic mono @ 16 kHz ({waveform.numel()} samples)")

    print("\n=== generate_streaming() — incremental KV-cache ===")
    t0 = time.time()
    with torch.no_grad():
        stream_out = pipeline.generate_streaming(
            waveform,
            n_windows=-1,
            train_style_asr=True,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            finalize_if_no_commit=True,
        )
    stream_s = time.time() - t0

    _print_streaming_trace(stream_out)
    print(f"\nStreaming result ({stream_s:.1f}s wall):")
    print(f"  windows processed: {stream_out['num_windows_used']}")
    print(f"  audio tokens in KV: {stream_out['num_audio_tokens']}")
    print(f"  committed on gate: {stream_out['committed_on_gate']}")
    print(f"  first_token_time_s: {stream_out.get('first_token_time_s')}")
    print(f"  text: {stream_out['text'][:500]!r}")

    if args.compare_batch:
        print("\n=== generate() — batch path (reference) ===")
        t0 = time.time()
        with torch.no_grad():
            batch_out = pipeline.generate(
                waveform,
                n_windows=-1,
                train_style_asr=True,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        batch_s = time.time() - t0
        print(f"  windows: {batch_out['num_windows_used']}")
        print(f"  text: {batch_out['text'][:500]!r}")
        print(f"  batch wall time: {batch_s:.1f}s")


if __name__ == "__main__":
    main()
