import glob
import sys
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for proper imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encoder import encode_dataset_stream
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

# Configuration
CHUNK_FILE_LIMIT = 1000      # how many files to accumulate before summarizing
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
DO_SAMPLE = True


def qwen_summarize_tensor(encoder_output: torch.Tensor, context: str = "audio chunk") -> str:
    """
    Projects encoder_output (1, 1500, 768) into Qwen embedding space,
    prepends a prompt, and generates a summary.
    """
    encoder_output = encoder_output.to(DEVICE, dtype=torch.float16)

    # Project: (1, 1500, 768) → (1, 1500, 4096)
    with torch.no_grad():
        projected = projector(encoder_output)

    # Build prompt embeddings
    prompt = (
        f"You are a helpful assistant. Below is a segment of {context}. "
        f"Please provide a concise and accurate summary of the audio content.\n\nSummary:"
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

    # Concat: [prompt embeddings | projected audio]
    inputs_embeds = torch.cat([prompt_embeds, projected], dim=1)   # (1, seq_len+1500, 4096)

    with torch.no_grad():
        output_ids = qwen_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=DO_SAMPLE,
            pad_token_id=qwen_tokenizer.eos_token_id,
        )

    summary = qwen_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return summary


def summarize_encoded_dataset(dataset_root: str) -> str:
    buffer: list[torch.Tensor] = []   # rolling tensor buffer
    chunk_summaries: list[str] = []
    chunk_index: int = 0

    def flush_buffer(buf: list[torch.Tensor], idx: int) -> str:
        print(f"{'─'*60}")
        print(f"[SUMMARIZING] Chunk {idx + 1} ({len(buf)} files) ...")
        # Mean-pool all tensors in the chunk → (1, 1500, 768)
        stacked = torch.stack([t.squeeze(0) for t in buf], dim=0)  # (N, 1500, 768)
        pooled = stacked.mean(dim=0, keepdim=True)                 # (1, 1500, 768)
        summary = qwen_summarize_tensor(pooled, context="LibriSpeech audio chunk")
        print(f"[CHUNK {idx + 1} SUMMARY]\n{summary}\n")
        return summary

    # Step 1 & 2: stream encodings, accumulate, flush on limit
    for entry in encode_dataset_stream(dataset_root):
        buffer.append(entry["encoded"])

        if len(buffer) >= CHUNK_FILE_LIMIT:
            summary = flush_buffer(buffer, chunk_index)
            chunk_summaries.append(summary)
            chunk_index += 1
            buffer = []

    # Step 3: flush remaining
    if buffer:
        summary = flush_buffer(buffer, chunk_index)
        chunk_summaries.append(summary)

    if not chunk_summaries:
        print("No encodings were produced — nothing to summarize.")
        return ""

    # Step 4: combine chunk summaries into one final summary
    print(f"\n{'═'*60}")
    print(f"[FINAL SUMMARY] Combining {len(chunk_summaries)} chunk summaries ...")

    combined_text = "\n\n".join(
        f"Chunk {i + 1} summary:\n{s}" for i, s in enumerate(chunk_summaries)
    )

    # Final summary is text-based (chunk summaries are already strings)
    prompt = (
        f"You are a helpful assistant. Below are summaries of consecutive audio chunks. "
        f"Please provide one final combined summary.\n\n{combined_text}\n\nFinal Summary:"
    )
    if hasattr(qwen_tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        formatted = qwen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = qwen_tokenizer(formatted, return_tensors="pt").to(DEVICE)
    else:
        inputs = qwen_tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=DO_SAMPLE,
            pad_token_id=qwen_tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    final_summary = qwen_tokenizer.decode(generated, skip_special_tokens=True).strip()

    print(f"\n{'═'*60}")
    print("[FINAL COMBINED SUMMARY]")
    print(final_summary)
    print(f"{'═'*60}\n")

    return final_summary


if __name__ == "__main__":
    # Use relative path from this script location
    # File is at: src/qwen_summarize_batched.py
    # Datasets are at: ../datasets/
    DATASET_ROOT = os.path.join(
        os.path.dirname(__file__),
        "../datasets/librispeech_data/LibriSpeech/train-clean-100"
    )
    final = summarize_encoded_dataset(DATASET_ROOT)
