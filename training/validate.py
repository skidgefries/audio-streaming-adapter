#!/usr/bin/env python3
"""
Validation script for the streaming adapter training module.

This script performs a series of checks to ensure the training infrastructure
is properly set up and ready to use.

Usage:
    python validate.py
"""

import os
import sys
import torch
from pathlib import Path


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def print_status(message, success=True):
    """Print a status message with ✓ or ✗."""
    symbol = "✓" if success else "✗"
    print(f"{symbol} {message}")
    return success


def check_packages():
    """Check if required packages are installed."""
    print_header("CHECKING REQUIRED PACKAGES")

    all_ok = True

    packages = [
    ("torch", "PyTorch"),
    ("transformers", "Transformers"),
    ("librosa", "Librosa"),
    ("numpy", "NumPy"),
]

    for pkg_name, friendly_name in packages:
        try:
            __import__(pkg_name)
            print_status(f"{friendly_name} installed")
        except ImportError:
            print_status(f"{friendly_name} NOT installed", success=False)
            all_ok = False

    return all_ok


def check_imports():
    """Check if training module components can be imported."""
    print_header("CHECKING MODULE IMPORTS")

    all_ok = True

    # training/validate.py → package root is parent; src/ is sibling of training/
    pkg_root = Path(__file__).resolve().parents[1]
    src_dir = pkg_root / "src"
    sys.path.insert(0, str(src_dir))
    sys.path.insert(0, str(pkg_root))

    # Check adapter components
    adapter_components = [
        ("adapter.streaming_adapter", "StreamingAdapter"),
        ("adapter.early_commit_gate", "EarlyCommitGate"),
        ("adapter.cross_attention", "QFormerLayer"),
        ("adapter.stability_buffer", "StabilityBuffer"),
        ("adapter.rate_controller", "AdaptiveRateController"),
    ]

    for module_name, class_name in adapter_components:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print_status(f"{class_name} imported from {module_name}")
        except (ImportError, AttributeError) as e:
            print_status(f"{class_name} import failed: {e}", success=False)
            all_ok = False

    try:
        from encoder.whisper_encoder import encode_dataset_stream, get_encoder_output

        _ = (get_encoder_output, encode_dataset_stream)
        print_status("encoder file/stream helpers imported")
    except ImportError as e:
        print_status(f"encoder whisper_encoder helpers import failed: {e}", success=False)
        all_ok = False

    try:
        from adapter_llm_pipeline import WhisperAdapterLLMCommitGatePipeline, WhisperAdapterLLMPipeline

        _ = (WhisperAdapterLLMPipeline, WhisperAdapterLLMCommitGatePipeline)
        print_status("adapter_llm_pipeline classes imported")
    except ImportError as e:
        print_status(f"adapter_llm_pipeline import failed: {e}", success=False)
        all_ok = False

    try:
        from training.adapter_contrastive_trainer import train as train_stage1
        from training.adapter_asr_trainer import train as train_stage2
        from training.adapter_task_trainer import train as train_stage3
        from training.utils.config import Stage1Config, Stage2Config, Stage3Config

        _ = (Stage1Config(), Stage2Config(), Stage3Config())
        print_status("Training stage functions and utils.config imported")
    except ImportError as e:
        print_status(f"Training stage functions import failed: {e}", success=False)
        all_ok = False

    return all_ok


def check_model_init():
    """Check if models can be initialized."""
    print_header("CHECKING MODEL INITIALIZATION")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    all_ok = True

    try:
        pkg_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(pkg_root / "src"))
        sys.path.insert(0, str(pkg_root))
        from adapter.streaming_adapter import StreamingAdapter
        from adapter.early_commit_gate import EarlyCommitGate

        # Test StreamingAdapter initialization
        adapter = StreamingAdapter(
            d_encoder=1024,
            d_llm=2560,
            num_queries=4,
            num_layers=2,
            num_heads=4,
            d_ffn=2048,
            dropout=0.1,
            ema_alpha=0.8,
            use_rate_controller=False,
        ).to(device, dtype=torch.float16)
        print_status("StreamingAdapter initialized")

        # Test with rate controller
        adapter_rc = StreamingAdapter(
            d_encoder=1024,
            d_llm=2560,
            num_queries=4,
            use_rate_controller=True,
            target_rate=2.0,
        ).to(device, dtype=torch.float16)
        print_status("StreamingAdapter (with rate controller) initialized")

        # Test EarlyCommitGate initialization
        gate = EarlyCommitGate(
            d_llm=2560,
            hidden_dim=256,
            threshold=0.5,
            latency_weight=0.1,
        ).to(device, dtype=torch.float16)
        print_status("EarlyCommitGate initialized")

        # Test forward pass with dummy data
        batch_size = 2
        T = 50  # Time steps in audio window

        dummy_encoder_features = torch.randn(
            batch_size, T, 1024, device=device, dtype=torch.float16
        )

        # Test window-wise forward pass (single window)
        adapter.reset_streaming_state()
        result = adapter.forward_window(dummy_encoder_features)
        print_status(f"Adapter forward_window pass: output shape {result['tokens'].shape}")

        # Test streaming state management
        adapter.reset_streaming_state()
        print_status("Streaming state reset successful")

        # Accumulate tokens for gate
        accumulated = torch.cat([
            torch.randn(1, i*4, 2560, device=device, dtype=torch.float16)
            for i in range(1, 4)
        ], dim=1)

        gate_result = gate(accumulated, timestep=5, total_timesteps=10)
        print_status(f"Gate forward pass: commit_prob shape {gate_result['commit_prob'].shape}")

        # Check CUDA memory if available
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            print_status(f"CUDA memory allocated: {memory_allocated:.2f} GB")
            torch.cuda.empty_cache()

    except Exception as e:
        print_status(f"Model initialization failed: {e}", success=False)
        import traceback
        traceback.print_exc()
        all_ok = False
        return all_ok

    return all_ok


def check_dataset():
    """Check if dataset path is configured."""
    print_header("CHECKING DATASET CONFIGURATION")

    pkg_root = Path(__file__).resolve().parents[1]
    default_path = pkg_root / "datasets" / "librispeech_data" / "LibriSpeech" / "train-clean-100"

    if default_path.is_dir():
        print_status(f"Dataset path exists: {default_path}")

        # Count samples
        trans_files = list(default_path.glob("**/*.trans.txt"))
        print_status(f"Found {len(trans_files)} transcription files")

        return True
    else:
        print_status(f"Dataset path NOT found: {default_path}", success=False)
        print("  Note: Training will fail without a dataset.")
        print("  Please update DATASET_ROOT in trainer scripts to your dataset path.")
        return False


def check_config():
    """Check training configuration parameters."""
    print_header("CHECKING TRAINING CONFIGURATION")

    all_ok = True

    # Verify architecture dimensions match
    whisper_dim = 1024  # whisper-medium
    phi2_dim = 2560     # Phi-2 embedding

    print(f"Whisper encoder dim: {whisper_dim}")
    print(f"Phi-2 embedding dim: {phi2_dim}")

    if whisper_dim > 0 and phi2_dim > 0:
        print_status("Architecture dimensions valid")
    else:
        print_status("Invalid architecture dimensions", success=False)
        all_ok = False

    # Check window parameters
    window_size = 0.8  # seconds
    stride = 0.4      # seconds
    sample_rate = 16000  # Hz

    window_samples = int(window_size * sample_rate)
    stride_samples = int(stride * sample_rate)

    print(f"\nWindow parameters:")
    print(f"  Window size: {window_size}s ({window_samples} samples)")
    print(f"  Stride: {stride}s ({stride_samples} samples)")
    print(f"  Sample rate: {sample_rate} Hz")

    if stride_samples > 0 and stride_samples < window_samples:
        print_status("Window parameters valid")
    else:
        print_status("Invalid window parameters", success=False)
        all_ok = False

    # Compute expected compression ratio
    # Whisper: ~1500 frames per 30s = 50 frames/s
    # Adapter: 4 tokens per 0.8s window = 5 tokens/s
    whisper_fps = 50
    adapter_tps = 4 / 0.8  # tokens per second

    compression_ratio = whisper_fps / adapter_tps
    print(f"\nCompression ratio: {compression_ratio:.1f}x")
    print(f"  Whisper: {whisper_fps} frames/s")
    print(f"  Adapter: {adapter_tps:.1f} tokens/s")
    print_status(f"Compression ratio: {compression_ratio:.1f}x reduction")

    return all_ok


def check_filesystem():
    """Check filesystem structure."""
    print_header("CHECKING FILESYSTEM STRUCTURE")

    base_dir = Path(__file__).parent
    all_ok = True

    # Check for training files
    training_files = [
        "adapter_contrastive_trainer.py",
        "adapter_asr_trainer.py",
        "adapter_task_trainer.py",
        "utils/__init__.py",
        "utils/config.py",
        "utils/checkpointing.py",
    ]

    for filename in training_files:
        filepath = base_dir / filename
        if filepath.exists():
            print_status(f"{filename} exists")
        else:
            print_status(f"{filename} NOT found", success=False)
            all_ok = False

    # Create checkpoint and log directories if they don't exist
    checkpoints_dir = base_dir / "checkpoints"
    logs_dir = base_dir / "logs"

    for directory in [checkpoints_dir, logs_dir]:
        directory.mkdir(exist_ok=True)
        print_status(f"Directory ready: {directory.name}")

    return all_ok


def run_all_checks():
    """Run all validation checks."""
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + "  STREAMING ADAPTER TRAINING VALIDATION".center(58) + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60 + "\n")

    results = {
        "packages": check_packages(),
        "imports": check_imports(),
        "model_init": check_model_init(),
        "dataset": check_dataset(),
        "config": check_config(),
        "filesystem": check_filesystem(),
    }

    # Print summary
    print_header("VALIDATION SUMMARY")

    for check_name, passed in results.items():
        symbol = "✓" if passed else "✗"
        status = "PASS" if passed else "FAIL"
        print(f"{symbol} {check_name:20s} : {status}")

    # Overall result
    all_passed = all(results.values())

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Training is ready to run!")
    else:
        failed = [name for name, passed in results.items() if not passed]
        print("✗ SOME CHECKS FAILED")
        print(f"  Failed: {', '.join(failed)}")
        print("  Please fix the issues above before training.")
    print("=" * 60 + "\n")

    # Provide next steps
    if all_passed:
        print("Next steps:")
        print("  1. Run a stage from package root: uv run python training/adapter_contrastive_trainer.py")
        print("  2. Then stage 2 / 3 the same way under audio-streaming-adapter/")
    else:
        print("Next steps:")
        print("  1. Install missing packages (if any)")
        print("  2. Fix import errors (check adapter module)")
        print("  3. Configure dataset path in trainer scripts")
        print("  4. Run this validation script again")

    return all_passed


if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
