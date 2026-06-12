"""
Summarize audio with a trained :class:`StreamingAdapter`.

If you trained with **EarlyCommitGate** and need the same per-window + ``L_gate`` behavior as
``training/adapter_asr_trainer.py``, use ``WhisperAdapterLLMCommitGatePipeline`` from
``adapter_llm_pipeline.py`` instead of ad-hoc ``forward`` calls. This script uses a direct
encoder → windows → adapter path without the early-commit gate.
"""
import argparse
import glob
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# Add adapter module to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adapter.streaming_adapter import StreamingAdapter
from adapter.windowing import AudioWaveformWindowizer
from dataset.librispeech import load_mono_waveform_16k
from encoder.whisper_encoder import encode_waveform_to_hidden, load_whisper_models

# Qwen model setup
torch.cuda.empty_cache()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_MODEL_ID = "Qwen/Qwen3-8B"

print(f"Loading Qwen tokenizer and model ({QWEN_MODEL_ID}) on {DEVICE} ...")
qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
qwen_model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL_ID,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    device_map="auto",
)
qwen_model.eval()
print("Qwen model ready.\n")

# Streaming Adapter: Whisper hidden size → Qwen hidden size with compression
# Note: whisper-small gives 768-dim encoder features; adjust d_encoder if you change model
# We'll use 768 to match the actual encoder output
streaming_adapter = StreamingAdapter(
    d_encoder=768,      # openai/whisper-small encoder dim
    d_llm=4096,         # Qwen3-8B embedding dimension
    num_queries=4,       # Number of compressed tokens per window
    num_layers=2,       # Number of Q-Former layers
    num_heads=4,        # Attention heads
    d_ffn=2048,         # FFN hidden dimension
    dropout=0.1,        # Dropout rate
    ema_alpha=0.8,      # EMA smoothing factor for stability buffer
    learnable_ema=False,
    use_rate_controller=False,  # Disable rate controller for fixed compression
).half().to(DEVICE)

# Optional: load a trained adapter checkpoint (same convention as the notebooks / walkthrough)
_CKPT = os.environ.get("ADAPTER_CHECKPOINT_PATH", "").strip()
if _CKPT:
    ckpt_path = _CKPT
    if not os.path.isabs(ckpt_path):
        # Training scripts save under src/adapter/training/checkpoints/
        base = os.path.join(os.path.dirname(__file__), "adapter", "training")
        ckpt_path = os.path.normpath(os.path.join(base, ckpt_path))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("adapter_state_dict") or ckpt.get("model_state_dict") or ckpt.get("state_dict")
    if state is None:
        raise KeyError(f"No adapter state_dict found in checkpoint keys: {list(ckpt.keys())}")
    streaming_adapter.load_state_dict(state, strict=False)
    print(f"Loaded adapter checkpoint: {ckpt_path}\n")

# Generation parameters
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
DO_SAMPLE = True

WHISPER_MODEL_ID = "openai/whisper-small"
_whisper = load_whisper_models(model_id=WHISPER_MODEL_ID, device=DEVICE, torch_dtype=torch.float16)

# Windowing: raw audio → per-chunk Whisper encode (defaults 0.2s / 0.1s stride)
WINDOW_SECONDS = 0.2
STRIDE_SECONDS = 0.1


def waveform_to_adapter_windows(waveform: torch.Tensor) -> list[torch.Tensor]:
    """Raw audio windows → Whisper encode each chunk → list of ``(1, T, D)`` on DEVICE."""
    windowizer = AudioWaveformWindowizer(
        window_seconds=WINDOW_SECONDS,
        stride_seconds=STRIDE_SECONDS,
    )
    enc_windows: list[torch.Tensor] = []
    for chunk in windowizer(waveform):
        enc = encode_waveform_to_hidden(
            chunk,
            whisper_processor=_whisper.processor,
            whisper_model=_whisper.model,
            device=DEVICE,
            torch_dtype=torch.float16,
        )
        enc_windows.append(enc.to(DEVICE, dtype=torch.float16))
    return enc_windows


def qwen_summarize_single_file(file_path: str) -> dict[str, str]:
    """
    Process a single audio file using StreamingAdapter and generate its summary.

    Args:
        file_path: Path to audio file (.flac, .wav, etc.)

    Returns:
        dict with:
            file: input file path
            summary: generated summary text
            status: "success" or "error"
            error: error message if applicable
            num_windows: number of streaming windows processed
            num_tokens: total number of compressed tokens
    """
    try:
        # Step 1: Load waveform and build per-window Whisper encoder inputs
        print(f"⟳ Encoding: {file_path}")
        wave = load_mono_waveform_16k(file_path)
        windows = waveform_to_adapter_windows(wave)
        if not windows:
            raise ValueError("Audio shorter than one window; no adapter windows produced.")
        print(f"  ✓ {len(windows)} windows (Whisper encode per chunk, no 30s pad)")

        # Step 2: Process windows through StreamingAdapter
        print(f"  ⟳ Processing through StreamingAdapter...")
        streaming_adapter.reset_streaming_state()

        with torch.no_grad():
            adapter_output = streaming_adapter(windows)

        compressed_tokens = adapter_output["tokens"]  # (1, num_windows * num_queries, 4096)
        num_windows = len()
        num_tokens = compressed_tokens.shape[1]

        print(f"  ✓ Compressed to {num_tokens} tokens from {num_windows} windows")
        total_enc_frames = sum(w.shape[1] for w in windows)
        compression_ratio = (total_enc_frames * 768) / num_tokens if num_tokens > 0 else 0
        overlap_percent = (WINDOW_SECONDS - STRIDE_SECONDS) / WINDOW_SECONDS * 100
        print(f"  ✓ Windowing: {WINDOW_SECONDS}s window, {STRIDE_SECONDS}s stride ({overlap_percent:.0f}% overlap)")

        # Step 5: Build prompt embeddings
        filename = os.path.basename(file_path)
        prompt = (
            f"You are a helpful assistant. Below is a compressed audio encoding for file '{filename}'. "
            f"The audio has been processed through a streaming adapter that extracted {num_tokens} "
            f"compressed token representations from {num_windows} temporal windows.\n\n"
            f"Please provide a concise and accurate summary of the audio content.\n\n"
            f"Focus on the main topics, speakers (if identifiable), and key information conveyed.\n\n"
            f"Summary:"
        )

        if hasattr(qwen_tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted = qwen_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = qwen_tokenizer(formatted, return_tensors="pt").input_ids.to(DEVICE)
        else:
            prompt_ids = qwen_tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

        prompt_embeds = qwen_model.get_input_embeddings()(prompt_ids)  # (1, seq_len, 4096)

        # Step 6: Concatenate prompt and compressed audio embeddings
        inputs_embeds = torch.cat([prompt_embeds, compressed_tokens], dim=1)  # (1, seq_len + num_tokens, 4096)

        print(f"  ✓ Total input length: {inputs_embeds.shape[1]} tokens")

        # Step 7: Generate summary
        print(f"  ⟳ Generating summary...")
        with torch.no_grad():
            attention_mask = torch.ones(
                inputs_embeds.shape[0],
                inputs_embeds.shape[1],
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            output_ids = qwen_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=DO_SAMPLE,
                pad_token_id=qwen_tokenizer.eos_token_id,
            )

        summary = qwen_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        print(f"  ✓ Summary: {summary[:100]}...")

        return {
            "file": file_path,
            "summary": summary,
            "status": "success",
            "error": None,
            "num_windows": num_windows,
            "num_tokens": num_tokens,
            "compression_ratio": compression_ratio,
            "overlap_percent": overlap_percent,
        }

    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "file": file_path,
            "summary": None,
            "status": "error",
            "error": str(e),
            "num_windows": 0,
            "num_tokens": 0,
            "compression_ratio": 0.0,
            "overlap_percent": 0.0,
        }


def process_files(file_paths: list[str], output_file: str = None) -> list[dict]:
    """
    Process multiple files one at a time using StreamingAdapter and generate individual summaries.

    Args:
        file_paths: List of file paths to process
        output_file: Optional path to save results to JSON file

    Returns:
        List of result dictionaries
    """
    results = []

    print(f"\n{'='*70}")
    print(f"Processing {len(file_paths)} file(s) with StreamingAdapter")
    print(f"{'='*70}\n")

    for idx, file_path in enumerate(file_paths, 1):
        print(f"\n[{idx}/{len(file_paths)}] ─" * 35)
        result = qwen_summarize_single_file(file_path)
        results.append(result)

        if idx % 10 == 0:
            print(f"\n{'─'*70}")
            print(f"Processed {idx}/{len(file_paths)} files")
            print(f"{'─'*70}\n")

    # Print summary statistics
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE")
    print(f"{'='*70}")
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")
    print(f"Total files: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")

    # Aggregate compression statistics
    if success_count > 0:
        total_windows = sum(r["num_windows"] for r in results if r["status"] == "success")
        total_tokens = sum(r["num_tokens"] for r in results if r["status"] == "success")
        avg_windows = total_windows / success_count
        avg_tokens = total_tokens / success_count
        compression_ratio = (1500 * 768) / avg_tokens if avg_tokens > 0 else 0
        print(f"\nCompression Statistics:")
        print(f"  Average windows per file: {avg_windows:.1f}")
        print(f"  Average tokens per file: {avg_tokens:.1f}")
        avg_compression_ratio = sum(r["compression_ratio"] for r in results if r["status"] == "success") / success_count
        print(f"  Average compression ratio: {avg_compression_ratio:.1f}x (1500 frames → {avg_tokens:.1f} tokens)")

    # Save to file if requested
    if output_file:
        import json
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")

    # Print all summaries
    print(f"\n{'='*70}")
    print("ALL SUMMARIES")
    print(f"{'='*70}\n")
    for result in results:
        if result["status"] == "success":
            print(f"File: {result['file']}")
            print(f"Windows: {result['num_windows']}, Tokens: {result['num_tokens']}, Compression: {result['compression_ratio']:.1f}x, Overlap: {result['overlap_percent']:.0f}%")
            print(f"Summary: {result['summary']}\n")
            print(f"{'─'*70}\n")
        else:
            print(f"File: {result['file']}")
            print(f"Error: {result['error']}\n")
            print(f"{'─'*70}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Summarize individual audio files using Whisper encoder + StreamingAdapter + Qwen LLM"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to a single audio file or directory containing audio files"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save results as JSON file"
    )
    parser.add_argument(
        "--pattern", "-p",
        type=str,
        default="*.flac",
        help="File pattern to match (default: *.flac)"
    )
    parser.add_argument(
        "--max-files", "-n",
        type=int,
        default=None,
        help="Maximum number of files to process (default: all)"
    )
    parser.add_argument(
        "--num-queries", "-m",
        type=int,
        default=4,
        help="Number of compressed tokens per window (default: 4)"
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=0.2,
        help="Audio window length in seconds (default: 0.2)",
    )
    parser.add_argument(
        "--stride-seconds",
        type=float,
        default=0.1,
        help="Audio window stride in seconds (default: 0.1, 50%% overlap at 0.2s window)",
    )

    args = parser.parse_args()

    global WINDOW_SECONDS, STRIDE_SECONDS
    WINDOW_SECONDS = args.window_seconds
    STRIDE_SECONDS = args.stride_seconds

    print(f"StreamingAdapter Configuration:")
    print(f"  Window: {WINDOW_SECONDS}s")
    print(f"  Stride: {STRIDE_SECONDS}s")
    overlap_percent = (WINDOW_SECONDS - STRIDE_SECONDS) / WINDOW_SECONDS * 100 if WINDOW_SECONDS > 0 else 0
    print(f"  Overlap: {overlap_percent:.0f}%")
    print(f"  Num queries: {args.num_queries}\n")

    # Determine input files
    if os.path.isfile(args.input):
        file_paths = [args.input]
    elif os.path.isdir(args.input):
        pattern = os.path.join(args.input, "**", args.pattern)
        file_paths = sorted(glob.glob(pattern, recursive=True))
        print(f"Found {len(file_paths)} files matching '{args.pattern}'")
    else:
        print(f"Error: Input path does not exist: {args.input}")
        return

    # Limit files if specified
    if args.max_files is not None and args.max_files < len(file_paths):
        file_paths = file_paths[:args.max_files]
        print(f"Limited to {len(file_paths)} files as requested")

    if not file_paths:
        print("No files to process.")
        return

    # Process files
    process_files(file_paths, output_file=args.output)


if __name__ == "__main__":
    main()
