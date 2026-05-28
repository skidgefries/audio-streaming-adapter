# Audio Streaming Adapter

A modular audio-to-text processing pipeline that uses Whisper encoder embeddings with a learnable StreamingAdapter network to efficiently compress audio features for downstream LLM generation tasks like summarization, ASR, and question answering.

## Overview

This project implements a streaming audio adapter bridge between audio encoders (Whisper) and Large Language Models (Qwen). It provides three main processing approaches:

1. **Batch Chunked Summarization** - Process large datasets with hierarchical summarization
2. **Per-File Summarization** - Generate individual summaries for each audio file using simple projector
3. **StreamingAdapter Summarization** - Efficient per-file processing with learnable compression, temporal awareness, and early-commit gating

### Key Features

- **Efficient Compression**: Reduces 30-second audio from ~3,000 frames to ~240 tokens (12.5x compression)
- **Streaming Architecture**: Processes audio in overlapping windows (0.8s window, 0.4s stride) with EMA smoothing
- **Three-Stage Curriculum Learning**: Progressive training from alignment → ASR → task-specific generation
- **Adaptive Rate Control**: Dynamically adjusts token allocation based on audio complexity
- **Early-Commit Gate**: Learns when to trigger generation for optimal latency-accuracy tradeoff

### Key Components

- **Whisper Encoder** (Component 1): Extracts audio features into 768-dimensional embeddings per frame
- **StreamingAdapter** (Component 2): Q-Former style cross-attention network for learnable compression
- **Early-Commit Gate** (Component 3): Trainable gate for latency optimization
- **Qwen LLM** (Component 4): Generates text summaries from compressed representations

## Installation

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (recommended)
- 16GB+ system RAM

### Setup with uv

```bash
# Clone the repository
git clone <repository-url>
cd audio-streaming-adapter

# Install dependencies
uv pip install -e .

# or install from the parent directory
cd /home/ml/workspaces/kristina/audio-stream
uv pip install -e audio-streaming-adapter/
```

### Remote server bootstrap (Stage 2 training)

1. Copy and edit environment variables:

```bash
cd audio-streaming-adapter
cp .env.example .env
# set CUDA_VISIBLE_DEVICES, WANDB_API_KEY, HF_TOKEN, etc.
```

2. Run the setup script (sources **`.env`** first, then pyenv 3.12, **`uv sync`**, **PyTorch CUDA compatibility check** with automatic wheel reinstall, LibriSpeech download, Stage 1 checkpoint fetch, and training):

```bash
bash scripts/setup_remote_training.sh
```

Setup only (no training): set `SKIP_TRAINING=1` in `.env` or export it before running the script.

**Launchers:** one visible GPU → `uv run training/adapter_asr_trainer.py`; two or more → `torchrun --nproc_per_node=1` (frozen Qwen auto-shards across GPUs). Set `DEVICE=cpu` to force CPU. See `training/utils/config.py` (`DeviceConfig`, `TrainingLaunchConfig`).

Requires **pyenv**, **uv**, and **wget** or **curl**.

### Dependencies

```
accelerate>=1.13.0
datasets>=4.8.4
librosa>=0.11.0
torch>=2.11.0
torchaudio>=2.11.0
torchcodec>=0.11.0
transformers>=5.5.0
```

## Project Structure

```
audio-streaming-adapter/
├── training/                         # CLI curriculum + shared helpers under utils/
│   ├── utils/                        # Configs, checkpointing, losses, metrics, logging (see below)
│   │   ├── config.py                 # Stage configs + windowing / adapter / tuning dataclasses
│   │   ├── losses.py                 # Contrastive InfoNCE, KL / prefix / revision helpers
│   │   ├── optimization.py           # TrainingPipeline (backward, clip, optim, scheduler)
│   │   ├── checkpointing.py          # Checkpoints (weights + optional metrics + hyperparams dict)
│   │   ├── logging.py                # Optional Weights & Biases wrapper
│   │   ├── metrics.py                # RunningMean, BLEU (transcript vs response, compressed-audio path)
│   │   ├── audio.py                  # Deprecated shim → `encoder.WhisperWindowFeatureExtractor`
│   │   ├── loaders.py                # Thin re-exports → prefer `encoder` / `llm` directly
│   │   ├── common.py                 # Legacy notebook helpers + old checkpoint format
│   │   └── __init__.py               # Re-exports common symbols
│   ├── adapter_contrastive_trainer.py
│   ├── adapter_asr_trainer.py
│   ├── adapter_task_trainer.py
│   ├── validate.py
│   └── projector_trainer.py
├── src/
│   ├── adapter/                      # StreamingAdapter, gate, windowing, Q-Former layers
│   ├── encoder/                      # WhisperConfig, whisper encoder + ASR helpers
│   ├── llm/                          # QwenConfig, generation helpers, Qwen loaders
│   ├── dataset/                      # LibriSpeechConfig, LibriSpeechPairs, waveform I/O
│   ├── adapter_llm_pipeline.py       # End-to-end inference pipeline
│   └── …                             # CLI demos (summarize_*, projector, etc.)
├── notebooks/
├── docs/
├── datasets/                         # e.g. LibriSpeech train-clean-100 (local)
└── README.md
```

## Notebooks (`notebooks/`)

| Notebook | Purpose |
|----------|---------|
| `adapter_llm_walkthrough.ipynb` | Inference: Whisper → StreamingAdapter → Qwen (`WhisperAdapterLLMPipeline`) |
| `adapter_llm_commit_gate_walkthrough.ipynb` | Same as above + **EarlyCommitGate** (`WhisperAdapterLLMCommitGatePipeline`), aligned with stage 2/3 `L_gate` |
| `adapter_walkthrough.ipynb` | Component walkthrough (no full `adapter_llm_pipeline` by default) |
| `training_stage1_contrastive.ipynb` | **Stage 1:** same training loop as `training/adapter_contrastive_trainer.py` (optional `MAX_STEPS_DEBUG`) |
| `training_stage2_asr.ipynb` | **Stage 2:** ASR + rate controller + gate (see `training/adapter_asr_trainer.py`) |
| `training_stage3_task.ipynb` | **Stage 3:** teacher–student distillation (see `training/adapter_task_trainer.py`) |

**Expectations:** These notebooks are documentation and experimentation. They can be **large** on disk, and saved outputs may be **stale** from older runs. **Logic is aligned** with `src/` and `training/` where the repo is updated, but notebooks are **not executed in CI** the way `tests/` and the library code are—treat outputs as illustrative and **re-run** cells when you need a trustworthy trace.

Run training notebooks with the kernel’s **current working directory** set to `notebooks/` (or set env **`AUDIO_STREAM_ADAPTER_ROOT`**). Each notebook adds `src/` to `sys.path` as needed.

**CLI training** (full epochs, LibriSpeech): from the `audio-streaming-adapter/` directory, run `uv run python training/adapter_contrastive_trainer.py` (and similarly for stage 2/3). Scripts prepend `src/` to `sys.path` so `adapter`, `dataset`, `encoder`, and `llm` import correctly.

**Hyperparameters** live in `training/utils/config.py` (`Stage1Config`, `Stage2Config`, `Stage3Config`, `OptimConfig`, `DataConfig`, plus optional `WhisperWaveformWindowingConfig`, `StreamingAdapterTrainConfig`, `TuningConfig`, …). Edit those dataclasses (or duplicate fields near the top of a stage script) to compare runs.

**Models and inference** use `src/encoder`, `src/llm`, `src/adapter`, and `src/adapter_llm_pipeline.py` — not copies under `training/utils/`.

**Checkpoints** use `training.utils.checkpointing.save_checkpoint` (keys: `adapter_state_dict`, optional `gate_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `metrics`, `hyperparams`). Serialize tuning bundles with `dataclasses.asdict` into `hyperparams`. Resume via `load_adapter_state_dict` or `torch.load` as in the stage trainers.

**Metrics** (e.g. BLEU for transcript vs. model response) live in `training/utils/metrics.py` (requires `sacrebleu`).

**Weights & Biases:** install `wandb`, authenticate, then set `WANDB = WandbConfig(enabled=True, …)` in the stage script you run (see `training/utils/logging.py`).

## Quick Start

### 1. Basic Audio Summarization with Simple Projector

```bash
cd audio-streaming-adapter/src

# Configure audio file path
AUDIO_FILE="/path/to/your/audio.wav"

# Run summarization
uv run python qwen_summarize_projector.py --input "$AUDIO_FILE"
```

### 2. Advanced Summarization with StreamingAdapter

```bash
# Run with learned adapter (requires trained checkpoint)
uv run python qwen_summarize_adapter.py --input "$AUDIO_FILE" \
    --checkpoint "checkpoints/adapter_adapter.pt"
```

### 3. Batch Processing

```bash
# Process entire dataset
DATASET_ROOT="/path/to/audio/files"
OUTPUT_DIR="./summaries"

uv run python qwen_summarize_batched.py \
    --dataset "$DATASET_ROOT" \
    --output "$OUTPUT_DIR" \
    --chunk_limit 1000
```

## Training

The StreamingAdapter is trained with a **three-stage curriculum**. Each stage has:

1. **Canonical Python entrypoint** under `training/` (full LibriSpeech loops, checkpointing, logging).
2. **Notebook walkthrough** under `notebooks/` (same ideas; Stage 1’s final cell matches the script line-for-line).

Work from the **`audio-streaming-adapter/`** directory so `training/` and `checkpoints/` resolve correctly. Install deps from the repo root (`uv sync` or `uv pip install -e audio-streaming-adapter/` as in [Installation](#installation)).

### Notebook setup (all stages)

1. Open Jupyter or VS Code with the project interpreter (same env as `uv run`).
2. Set the notebook kernel’s **working directory** to `notebooks/` (or set env **`AUDIO_STREAM_ADAPTER_ROOT`** to the absolute path of `audio-streaming-adapter/`).
3. Run the **second code cell** in each training notebook: it adds `src/` and the package root to `sys.path` and `chdir`s to the package root (same layout the CLI scripts expect).

### Stage 1: Contrastive audio–text alignment

**What it does:** Aligns adapter token embeddings with frozen LLM text embeddings; loss `L = L_align + λ_stability · L_stability` using `training.utils.losses.contrastive_infonce_loss` and mean stability from `StreamingAdapter.forward_window`. Audio features use **`WhisperWindowFeatureExtractor`**: **`AudioWaveformWindowizer`** (0.8s / 0.4s on raw waveform) → one Whisper encode per chunk — same as `adapter_contrastive_trainer.py`.

**CLI (full training):**

```bash
cd audio-streaming-adapter
uv run python training/adapter_contrastive_trainer.py
```

**Walkthrough:** open `notebooks/training_stage1_contrastive.ipynb`. Earlier cells dissect components; the section **“Full training (script-equivalent)”** runs the same loop as the CLI (set `MAX_STEPS_DEBUG` to a small integer for a smoke test).

**Defaults (see `training/utils/config.py` + script top):** epochs 5, batch size 4, LR `1e-4`, SGD + linear warmup (`warmup_steps=100`), `λ_stability=0.1`, temperature `0.2`, `checkpoints/adapter_adapter.pt`.

### Stage 2: ASR distillation

**What it does:** Keeps speech content while training compression / optional rate controller and gate losses (`training/adapter_asr_trainer.py`).

**CLI (two GPUs recommended for Qwen3-8B):** All trainable modules and Whisper use CUDA (`DEVICE=cuda` by default). With two or more visible GPUs, the frozen Qwen model is sharded automatically via HuggingFace `device_map="auto"` — no manual memory fractions.

```bash
cd audio-streaming-adapter
# single GPU
uv run training/adapter_asr_trainer.py

# multi-GPU (2+ visible devices)
torchrun --standalone --nnodes=1 --nproc_per_node=1 training/adapter_asr_trainer.py
```

On a single GPU, set `CUDA_VISIBLE_DEVICES=0` or leave one device visible. If training OOMs, enable `ENABLE_LLM_GRADIENT_CHECKPOINTING=true`, lower `MAX_WINDOWS_PER_UTT`, or reduce `BATCH_SIZE`.

**Upload checkpoints to Hugging Face Hub** (after each epoch, requires `HF_TOKEN` with write access):

```bash
HF_CHECKPOINT_REPO=your-org/audio-streaming-adapter-checkpoints
HF_UPLOAD_CHECKPOINTS=true
```

Each stage uploads only its own epoch files (`adapter_stage1_epoch{N}.pt`, `adapter_stage2_epoch{N}.pt`, `adapter_stage3_epoch{N}.pt`, …). Uploads are **additive** — existing files from other stages stay in the repo. Local resume checkpoints (`adapter_stage{N}.pt`) are not uploaded.

**Walkthrough:** `notebooks/training_stage2_asr.ipynb` — follow cells top-to-bottom; align hyperparameters with `Stage2Config`, `DeviceConfig`, `OptimConfig`, and constants at the top of `adapter_asr_trainer.py`.

### Evaluation: LibriSpeech ASR (WER / BLEU-4)

`evaluation/eval_librispeech_asr_metrics.py` runs end-to-end transcription on test-clean and reports **avg WER** and **corpus BLEU-4** (same style as SALMONN `eval_librispeech_full_metrics.py`, ASR section only).

- **Stage 1:** `WhisperAdapterLLMPipeline`, checkpoint from `adapter_contrastive_trainer.py` (`use_rate_controller=False`).
- **Stage 2:** `WhisperAdapterLLMCommitGatePipeline`, checkpoint from `adapter_asr_trainer.py` (rate controller + early-commit gate). Training CE uses **`[audio tokens | im_end | teacher-forced transcript]`**; default ASR eval for stages **1–2** uses **`audio tokens → generate`** (do not append im_end at inference — Qwen3 treats it as chat). Chat tails after a blank line are trimmed. **`--prompt-asr`** switches to prompt+audio (`--asr-prompt`, chat template, `enable_thinking=False`) — the Stage 3 target path. Optional `--early-commit-truncation` for stage 2.

```bash
cd audio-streaming-adapter
# Compare stage 1 vs stage 2 (writes outputs/asr_eval/asr_comparison.json)
CUDA_VISIBLE_DEVICES=1 uv run evaluation/eval_librispeech_asr_metrics.py --compare-stages

# Single checkpoint
uv run evaluation/eval_librispeech_asr_metrics.py \
  --checkpoint checkpoints/adapter_stage2.pt --stage 2 --num-samples 50
```

Retrieval-only scripts remain under `evaluation/eval_retrieval*.py`.

### Stage 3: Task distillation

**What it does:** Teacher vs student frozen LLMs with adapter + optional gate (`training/adapter_task_trainer.py`); device split via `Stage3DeviceConfig`.

**CLI:**

```bash
cd audio-streaming-adapter
uv run python training/adapter_task_trainer.py
```

**Walkthrough:** `notebooks/training_stage3_task.ipynb` — same pattern: path cell first, then match `Stage3Config` / `adapter_task_trainer.py`.

### Validation

Before long runs:

```bash
cd audio-streaming-adapter
uv run python training/validate.py
```

**Turn-end gate silence modes** (`GATE_SILENCE_MODE` in `.env`):

| Mode | Behavior |
|------|----------|
| `rule` (default) | Fixed token-norm heuristics (`SilenceTracker`) |
| `learned` | Trainable MLP on per-window tokens `Z_t` (`LearnedSilenceHead`) |
| `both` | Run rule + learned paths in parallel; trains both classifiers; compare `gate_loss_rule` vs `gate_loss_learned` in logs |

When `both`, set `GATE_ACTIVE_SILENCE_PATH=rule|learned` to pick which path drives `should_commit` at inference. Pipeline diagnostics include `early_commit_commit_probs_rule` and `early_commit_commit_probs_learned` when both are enabled.

## Architecture

### StreamingAdapter Network (Component 2)

The StreamingAdapter implements a Q-Former style neural network:

```
Whisper frames F ∈ R^{T × D_enc} (T=1500, D_enc=768)
    ↓
Learnable queries Q ∈ R^{m × D_q} (m=4)
    ↓
[Q-Former layers (self-attn + cross-attn + FFN)] × 2 layers
    ↓
[AdaptiveRateController] (optional, enables dynamic m)
    ↓
[Output projection to LLM dimension]
    ↓
[StabilityBuffer (EMA smoothing)]
    ↓
Tokens ready for LLM ∈ R^{m × D_llm} (D_llm=4096)
```

**Key Parameters:**
- Input dimension: 768 (Whisper-small encoder)
- Output dimension: 4096 (Qwen LLM embedding)
- Max tokens per window: 4
- Q-Former layers: 2
- `cross_layer_in_between` (default 1): With `K>0`, each block of `K+1` layers ends with cross-attention; earlier layers in the block are self-attention-only (so the stack does not apply cross-attention on layer 0). With `K=0`, every layer has cross-attention.
- Attention heads: 4
- FFN hidden dimension: 2048
- EMA alpha: 0.8

### WhisperAdapterLLMPipeline (notebook / inference)

`WhisperAdapterLLMPipeline` (in `src/adapter_llm_pipeline.py`) runs the full path: **waveform → `AudioWaveformWindowizer` → Whisper encode per chunk → `StreamingAdapter` → causal LM**. Use `generate(waveform, n_windows=k, ...)` where `k` is how many overlapping windows feed the adapter for one LLM call; **`n_windows=-1`** uses every window. Import from `adapter` with `PYTHONPATH` including `src` (see notebooks).

LLM decoding uses a **deep copy** of the model’s `GenerationConfig`. Default **`do_sample=True`**, with optional **`temperature`**, **`top_p`**, and **`top_k`** (defaults 0.7, 0.9, 50 when unset). With **`do_sample=False`**, those three are set to **`None`** so greedy decoding does not conflict with sampling fields. Prompt embeddings are concatenated with adapter tokens and an **`attention_mask`** of all ones is passed so `pad_token_id == eos_token_id` does not break masking.

For extra detail on generation behavior, set the environment variable **`TRANSFORMERS_VERBOSITY=info`** before importing `transformers` (e.g. in the notebook: `os.environ.setdefault("TRANSFORMERS_VERBOSITY", "info")`).

### Streaming KV-cache inference (Qwen3-8B)

`WhisperAdapterLLMCommitGatePipeline.generate_streaming()` and `WhisperAdapterStreamingSession` (`src/adapter_llm_streaming.py`) implement the live path documented in `docs/EARLY_COMMIT.md`:

1. Each 0.8s / 0.4s window → adapter compressed tokens → **append to Qwen3-8B KV-cache** (`LlmKvCacheSession`)
2. `TurnEndCommitGate` on accumulated tokens each window
3. On `should_commit` → decode from cache (no full-prefix re-encode)
4. If no commit by end-of-audio → optional `finalize()` generation

```python
from adapter_llm_pipeline import WhisperAdapterLLMCommitGatePipeline
from llm import load_qwen_models

qwen = load_qwen_models(model_id="Qwen/Qwen3-8B", device="cuda", torch_dtype=torch.float16)
pipeline = WhisperAdapterLLMCommitGatePipeline(..., llm_model=qwen.causal_lm, llm_tokenizer=qwen.tokenizer, ...)

# Batch path (legacy): all windows → one generate()
out = pipeline.generate(waveform, train_style_asr=True)

# Streaming path: per-window KV-cache + generate on commit
out = pipeline.generate_streaming(waveform, train_style_asr=True)
print(out["first_token_time_s"], out["committed_on_gate"], out["text"])
```

For microphone-style ingestion, reuse one session:

```python
session = pipeline.create_streaming_session()
session.begin(train_style_asr=True)
for chunk in live_audio_chunks:  # each ≥ 0.8s window of 16 kHz mono
    step = session.push_waveform(chunk)
    if session.committed:
        print(step.generated_text)
        break
result = session.finalize(force_generate=True)
```

Runnable demo (synthetic 2.4s audio, stage-2 checkpoint, Qwen3-8B):

```bash
cd audio-streaming-adapter && source .venv/bin/activate
PYTHONPATH=src:training python examples/streaming_demo.py

# If GPUs are full:
CUDA_VISIBLE_DEVICES= python examples/streaming_demo.py --device cpu --max-new-tokens 16
```

**Other log noise:** set **`TOKENIZERS_PARALLELISM=false`** when using a multithreaded `DataLoader` (see notebook cell 0). For Whisper ASR in the walkthrough, call the pipeline with **`generate_kwargs={"language": "en", "task": "transcribe"}`** to reduce deprecated `forced_decoder_ids` / multilingual default messages (some logits-processor messages may still appear depending on `transformers` version).

### Windowing Strategy

- Window size: 0.8 seconds (40 frames at 20ms stride)
- Stride: 0.4 seconds (20 frames, 50% overlap)
- Compression: 300 words → 240 tokens (with 50% overlap)

### Loss Functions

**Stage 1:**
```python
L = L_align + λ_stability · L_stability
```

**Stage 2:**
```python
L = L_asr + λ_align · L_align + λ_stability · L_stability 
    + λ_sparse · L_sparse + λ_rate · L_rate
```

**Stage 3:**
```python
L = L_task + λ_asr · L_asr + λ_stability · L_stability 
    + λ_rate · L_rate + λ_gate · L_gate 
    + λ_prefix_consistency · L_prefix + λ_revision · L_revision
```

## Workflows

### Workflow 1: Batch Chunked Summarization

Process large datasets with hierarchical summarization:

```python
from encoder import encode_dataset_stream
from qwen_summarize_batched import qwen_summarize_tensor

for item in encode_dataset_stream("/path/to/dataset"):
    summary = qwen_summarize_tensor(item["encoded"])
    # Process summary
```

### Workflow 2: Per-File Simple Summarization

Generate individual summaries with frame-by-frame projection:

```python
from encoder import get_encoder_output
from projector import WhisperToQwenProjector

encoder_output = get_encoder_output("/path/to/audio.wav")
projected = projector(encoder_output)
summary = qwen_model.generate(projected)
```

### Workflow 3: Per-File StreamingAdapter Summarization

Efficient per-file processing with temporal compression:

```python
from adapter import StreamingAdapter
from encoder import get_encoder_output

# Load adapter
adapter = StreamingAdapter(d_encoder=768, d_llm=4096, num_queries=4)
adapter.load_state_dict(torch.load("checkpoints/adapter_adapter.pt"))

# Process audio
encoder_output = get_encoder_output("/path/to/audio.wav")
windows = split_into_windows(encoder_output)

# Stream through adapter
adapter.reset_streaming_state()
result = adapter(windows)
summary = qwen_model.generate(result["tokens"])
```

## Performance

### Compression Ratios

- **Whisper without adapter**: ~50 frames/second × embedding_dim
- **With StreamingAdapter**: 5 tokens/second × embedding_dim
- **Compression ratio**: ~10x reduction in sequence length
- **Memory reduction**: From ~30MB to ~3MB per 30s clip

### Processing Time (approximate)

- Whisper encoding (30s audio): ~0.5s on A100
- StreamingAdapter transformation: ~0.1s on A100
- Qwen generation: ~1-2s on A100
- **Total**: ~2-3s for end-to-end summarization

### Memory Usage

- Whisper encoder: ~1.5GB (fp16)
- StreamingAdapter: ~2MB (trainable parameters)
- Qwen 8B: ~16GB (fp16)
- **Total**: ~18GB GPU memory

## Configuration

### Adaptive Rate Controller

Control token allocation based on audio complexity:

```python
streaming_adapter = StreamingAdapter(
    d_encoder=768,
    d_llm=4096,
    num_queries=4,
    use_rate_controller=True,  # Enable adaptive control
    target_rate=2.0,           # Target: 2 tokens/window average
    rate_threshold=0.5,        # Hard gate threshold for inference
)
```

### Early-Commit Gate

Optimize latency for streaming generation:

```python
from adapter import EarlyCommitGate

gate = EarlyCommitGate(
    d_llm=4096,
    hidden_dim=256,
    threshold=0.5,
    latency_weight=0.1,  # Higher = more aggressive early commit
)
```

### Windowing Parameters

Adjust based on your use case:

```python
# Higher overlap (75%) = better temporal continuity = more computation
WINDOW_SIZE_FRAMES = 40   # 0.8s at 20ms stride
STRIDE_FRAMES = 10        # 0.2s (75% overlap)

# Lower overlap (0%) = faster but more jitter
WINDOW_SIZE_FRAMES = 40   # 0.8s
STRIDE_FRAMES = 40        # 0.8s (no overlap)
```

## Troubleshooting

### Common Issues

**Out of Memory Errors:**
- Reduce batch size in training scripts
- Use gradient accumulation instead of larger batches
- Process fewer files in batch mode

**Poor Summarization Quality:**
- Ensure Whisper encoder is correctly loaded
- Verify the adapter checkpoint corresponds to your audio type
- Try increasing num_queries or adjusting target_rate

**Import Errors:**
- Make sure you're in the correct directory (`audio-streaming-adapter/src`)
- Install with `uv pip install -e .` from the project root
- Check that all dependencies are installed

**Slow Training:**
- Enable mixed precision (already enabled in training scripts)
- Reduce number of workers in dataloader
- Use smaller window size (e.g., 0.6s instead of 0.8s)

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Testing

Run the test suite:

```bash
cd audio-streaming-adapter
uv run python tests/test_adapter.py
```

Tests cover:
- Q-Former layer functionality
- Stability buffer EMA smoothing
- Rate controller adaptive gating
- Early-commit gate latency optimization
- Full streaming adapter pipeline
- Four-component integration

## Contributing

We welcome contributions! Areas of interest:

- Additional adapter architectures
- Support for different LLM backends
- Optimization for edge deployment
- New training tasks beyond ASR and summarization

## License

[Add your license here]

## References

- [BLIP-2 Paper](https://arxiv.org/abs/2301.12597) (Q-Former architecture)
- [Whisper Model](https://arxiv.org/abs/2212.04356)
- [Streaming Audio Processing](../../docs/phase2_cross_attention_adapter.md)

## Contact

For questions and issues, please open a GitHub issue or contact the maintainers.