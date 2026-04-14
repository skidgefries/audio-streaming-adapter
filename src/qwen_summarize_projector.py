import argparse
import glob
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for proper imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encoder import get_encoder_output
from projector.projector import WhisperToQwenProjector

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

# Projector: Whisper hidden size → Qwen hidden size
projector = WhisperToQwenProjector(in_dim=768, out_dim=4096).half().to(DEVICE)

# Generation parameters
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
DO_SAMPLE = True


def qwen_summarize_single_file(file_path: str) -> dict[str, str]:
    """
    Process a single audio file and generate its summary.

    Args:
        file_path: Path to audio file (.flac, .wav, etc.)

    Returns:
        dict with:
            file: input file path
            summary: generated summary text
            status: "success" or "error"
            error: error message if applicable
    """
    try:
        # Step 1: Get Whisper encoder output
        print(f"⟳ Encoding: {file_path}")
        encoder_output = get_encoder_output(file_path)  # (1, 1500, 768)
        print(f"  ✓ Shape: {encoder_output.shape}")

        # Step 2: Project to Qwen embedding space
        encoder_output = encoder_output.to(DEVICE, dtype=torch.float16)
        with torch.no_grad():
            projected = projector(encoder_output)  # (1, 1500, 4096)

        # Step 3: Build prompt embeddings
        filename = os.path.basename(file_path)
        prompt = (
            f"You are a helpful assistant. Below is an audio encoding for file '{filename}'. "
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

        # Step 4: Concatenate prompt and audio embeddings
        inputs_embeds = torch.cat([prompt_embeds, projected], dim=1)  # (1, seq_len+1500, 4096)

        # Step 5: Generate summary
        print(f"  ⟳ Generating summary...")
        with torch.no_grad():
            output_ids = qwen_model.generate(
                inputs_embeds=inputs_embeds,
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
        }

    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return {
            "file": file_path,
            "summary": None,
            "status": "error",
            "error": str(e),
        }


def process_files(file_paths: list[str], output_file: str = None) -> list[dict]:
    """
    Process multiple files one at a time and generate individual summaries.

    Args:
        file_paths: List of file paths to process
        output_file: Optional path to save results to JSON file

    Returns:
        List of result dictionaries
    """
    results = []

    print(f"\n{'='*70}")
    print(f"Processing {len(file_paths)} file(s)")
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
            print(f"Summary: {result['summary']}\n")
            print(f"{'─'*70}\n")
        else:
            print(f"File: {result['file']}")
            print(f"Error: {result['error']}\n")
            print(f"{'─'*70}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Summarize individual audio files using Whisper encoder + Qwen LLM"
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

    args = parser.parse_args()

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
