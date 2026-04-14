# Audio Streaming Adapter

A modular audio summarization pipeline that processes audio files using Whisper encoder embeddings and generates text summaries with Qwen LLM. The project provides multiple workflow options for different use cases, from batch processing of large datasets to individual file analysis with varying compression strategies.

## Overview

This project implements flexible audio-to-text summarization with three main processing approaches:

1. **Batch Chunked Summarization** - Process large datasets with hierarchical summarization
2. **Per-File Summarization** - Generate individual summaries for each audio file
3. **StreamingAdapter Summarization** - Efficient per-file processing with learnable compression and temporal awareness

### Key Components

- **Whisper Encoder**: Extracts audio features into 768-dimensional embeddings per frame
- **Projector/StreamingAdapter**: Transforms embeddings to Qwen's 4096-dimensional space
- **Qwen LLM**: Generates text summaries from encoded representations

---

## File Descriptions

### Core Modules

#### `encoder/whisper_encoder.py`
Whisper loading, `encode_waveform_to_hidden`, and file/dataset helpers.

**Key APIs:**
- `load_whisper_models`, `encode_waveform_to_hidden` — core tensor path
- `get_encoder_output(file_path)` — load file (16 kHz mono via librosa) → `(1, T, 768)` CPU tensor
- `encode_dataset_stream(dataset_root)` — glob `.flac` / `.wav`, yield `{"path", "encoded"}`

Import with `from encoder import get_encoder_output, encode_dataset_stream` (with `src` on `PYTHONPATH`).

#### `projector.py`
Simple linear projection network for Whisper-to-LLM dimension mapping.

**Class: `WhisperToQwenProjector`**
- Architecture: `Linear(768→2048) → ReLU → Linear(2048→4096)`
- Purpose: Maps Whisper embeddings (768-dim) to Qwen embedding space (4096-dim)
- Processing: Frame-by-frame projection (no temporal compression)
- Output shape: `(1, 1500, 4096)` - preserves full sequence length

### Adaptation Note: The StreamingAdapter Module

The `StreamingAdapter` is a sophisticated compression module that replaces the simple `WhisperToQwenProjector` for more efficient and temporally-aware audio processing. It is implemented as a separate package to maintain a clean separation between the base audio streaming utilities and the trainable adapter network.

#### Package Structure

The `StreamingAdapter` is located in the `audio-streaming-adapter` package:
```
audio-streaming-adapter/
└── src/adapter/
    ├── streaming_adapter.py       # Main adapter class (Component 2 of 4)
    ├── rate_controller.py         # Adaptive token rate controller (optional)
    ├── stability_buffer.py        # EMA smoothing for temporal consistency
    ├── cross_attention.py         # Q-Former attention mechanisms
    └── early_commit_gate.py       # Latency optimization gate (Component 3)
```

#### Architecture Overview

The `StreamingAdapter` implements a Q-Former style neural network that compresses Whisper encoder outputs temporally before passing them to the LLM. This approach is based on the research paper's 4-component architecture:

1. **Frozen Audio Encoder (Whisper)** - Not in this module (base package)
2. **StreamingAdapter Network** - **THIS MODULE (trainable)**
3. **Early-Commit Gate** - Separate module (available but optional)
4. **Frozen LLM** - Not in this module (Qwen)

#### Key Components

**1. `streaming_adapter.py` - Main Adapter Network**

The core `StreamingAdapter` class performs:
- **Input**: Whisper encoder features `(batch, T, d_encoder)` where T=1500 frames
- **Learnable Queries**: m query vectors (default m=4) that learn to extract salient information
- **Q-Former Layers**: Stack of layers implementing:
  - Self-attention between query tokens
  - Cross-attention: queries attend to audio frames
  - Feed-Forward Networks
- **Output Projection**: Maps from encoder dimension to LLM embedding space
- **Stability Buffer**: EMA smoothing across streaming windows for temporal consistency

**Pipeline for a single window:**
```
Whisper frames F ∈ R^{T × d_enc}
    ↓
[Q-Former layers (self-attn + cross-attn + FFN)] × num_layers
    ↓
[Optional: AdaptiveRateController]
    ↓
[Output projection to LLM dim]
    ↓
[StabilityBuffer (EMA smoothing)]
    ↓
Tokens ready for LLM ∈ R^{m × d_llm}
```

**2. `rate_controller.py` - Adaptive Token Rate Control**

An optional module that dynamically adjusts how many tokens to emit per window based on audio complexity:
- **Silent/simple segments**: Emit fewer tokens (1-2)
- **Dense/noisy segments**: Emit more tokens (up to m_max=4)
- **Training mode**: Uses soft differentiable gates
- **Inference mode**: Uses hard threshold gating
- **Loss functions**:
  - `L_sparse`: Encourages using fewer tokens when possible
  - `L_rate`: Penalizes deviation from target token rate

**3. `stability_buffer.py` - Temporal Consistency**

Implements Exponential Moving Average (EMA) smoothing:
- **Purpose**: Prevents abrupt changes between overlapping windows
- **Mechanism**: `z_smooth = α × z_smooth + (1-α) × z_current`
- **Alpha (α)**: Smoothing factor (default 0.8, higher = more smoothing, lower = more responsive)
- **Learnable**: Can make alpha trainable for data-driven adaptation

**4. `cross_attention.py` - Attention Mechanisms**

Implements the Q-Former layer components:
- **Multi-head self-attention**: Query tokens interact with each other
- **Cross-attention**: Query tokens attend to audio frames
- **Pre/post-layer normalization** and residual connections

#### Windowing Strategy

The `StreamingAdapter` is designed for **chunked streaming** with overlapping windows:

**Default Configuration:**
- **Window size**: 40 frames (0.8 seconds at 20ms stride)
- **Stride**: 20 frames (0.4 seconds, 50% overlap)
- **Queries per window**: m=4 tokens
- **Result**: 1500 frames → 75 windows → 300 compressed tokens

**Testing Configuration** (as used in `qwen_summarize_single_adapter.py`):
- **Window size**: 10 frames (0.2 seconds)
- **Stride**: 5 frames (0.1 seconds, 50% overlap)
- **Queries per window**: m=4 tokens
- **Result**: 1500 frames → 300 windows → 1200 compressed tokens

**Why Overlap?**
- Prevents loss of information at window boundaries
- Provides better temporal continuity
- Enables effective EMA smoothing via stability buffer
- Standard practice in streaming audio processing

#### Integrating the StreamingAdapter

The `StreamingAdapter` is used in `qwen_summarize_single_adapter.py` as follows:

```python
import sys
sys.path.insert(0, "/path/to/audio-streaming-adapter/src")
from adapter.streaming_adapter import StreamingAdapter

# Initialize adapter
streaming_adapter = StreamingAdapter(
    d_encoder=768,              # Whisper output dimension
    d_llm=4096,                 # Qwen embedding dimension
    num_queries=4,              # Tokens per window
    num_layers=2,               # Q-Former stacks
    num_heads=4,                # Attention heads
    d_ffn=2048,                 # FFN hidden size
    dropout=0.1,                # Regularization
    ema_alpha=0.8,              # Stability buffer smoothing
    learnable_ema=False,         # Fixed alpha
    use_rate_controller=False,   # Fixed compression (or True for adaptive)
)

# Split encoder output into windows
windows = split_into_windows(encoder_output, window_size=10, stride=5)

# Process through adapter
streaming_adapter.reset_streaming_state()  # Clear previous state
output = streaming_adapter(windows)
compressed_tokens = output["tokens"]       # (batch, num_windows*m, d_llm)

# Feed to LLM for summarization
inputs_embeds = torch.cat([prompt_embeds, compressed_tokens], dim=1)
summary = qwen_model.generate(inputs_embeds=inputs_embeds)
```

#### StreamingAdapter vs. Simple Projector

| Aspect | WhisperToQwenProjector | StreamingAdapter |
|--------|------------------------|------------------|
| **Architecture** | 2-layer MLP | Q-Former with attention |
| **Compression** | None (frame-by-frame) | Configurable (5x - 0.25x) |
| **Temporal Info** | No cross-frame processing | Captures via attention |
| **Adaptability** | Fixed weights | Learnable queries |
| **Stability** | No smoothing | EMA buffer |
| **Statefulness** | Stateless | Maintains buffer state |
| **Trainable** | No | Yes (can be fine-tuned) |
| **Token Count** | 1500 per file | 300-1200 per file |
| **Computation** | Fast | Slower (attention) |

#### When to Use StreamingAdapter

**Use StreamingAdapter when:**
- Processing many files and need efficiency
- Token budget is constrained (e.g., longer context limits)
- Want temporal awareness without frame-level detail
- Need adaptive compression for varied audio content
- Streaming real-time audio with buffering

**Use Simple Projector when:**
- Maximum temporal detail is required
- Processing individual files in isolation
- LLM has ample capacity (long context)
- Need faster inference speed
- Simple baseline for comparison

#### Loss Functions (Training Mode)

When training the `StreamingAdapter`, the following losses are used:

1. **L_stability** (from stability buffer):
   - Regularizes temporal consistency
   - Penalizes sudden changes between window outputs
   - Computed as L2 difference between consecutive smoothed outputs

2. **L_sparse** (from rate controller, if enabled):
   - Encourages using fewer tokens
   - L1 regularization on gate scores
   - Promotes sparse token usage

3. **L_rate** (from rate controller, if enabled):
   - Maintains target average token rate
   - Mean squared error: (effective_tokens - target_rate)²
   - Balances efficiency vs. information retention

**Total loss** (example):
```python
total_loss = stability_loss + λ_sparse * sparse_loss + λ_rate * rate_loss
```

#### Future Enhancements

The `StreamingAdapter` module is designed to support:
- **Early-Commit Gate**: Decide when to start LLM generation for latency optimization
- **Rate Controller Training**: Learn optimal token allocation per window
- **Multi-head Variants**: Different query types for different audio features
- **Hierarchical Compression**: Multi-stage compression for very long audio

For implementation details, see the source code in the `audio-streaming-adapter/src/adapter/` directory.

---

## Workflows

### Workflow 1: Batch Chunked Summarization

**Script:** `qwen_encoded_summarize.py`

**Use Case:** Summarizing large datasets (thousands of files) with hierarchical aggregation.

**Pipeline:**
```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Stream Audio Encodings                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │   File 1 │ → Encode →│  (1,1500, ││   File 2 │ → Encode  │          │
│  │ .flac    │           │    768)   ││ .flac    │           │          │
│  └──────────┘ └──────────┘ └──────────┘           │          │
│                    ↓                        │          │
│  Accumulate 1000 files → buffer [(1,1500,768) × 1000]        │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: Mean Pool Chunk                                         │
│  stacked = torch.stack(buffer) → (1000, 1500, 768)              │
│  pooled = stacked.mean(dim=0) → (1, 1500, 768)                 │
│                                                             │
│  Result: ONE averaged embedding representing 1000 files         │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: Project & Summarize                                     │
│  pooled(1,1500,768) → projector → (1,1500,4096)                │
│  concat([prompt_embeds, projected]) → Qwen → chunk_summary     │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: Hierarchical Combination                                │
│  [Chunk 1 Summary] + [Chunk 2 Summary] + ...                  │
│           ↓                                                     │
│  Qwen (text-to-text) → Final Combined Summary                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- **Memory efficient**: Streams files, accumulates in batches of 1000
- **Hierarchical**: Chunk summaries → Final summary
- **Aggregation**: Mean pooling captures collective content
- **Output:** One final summary text

**Usage:**
```bash
# Edit DATASET_ROOT in script, then run:
python qwen_encoded_summarize.py
```

**Configuration:**
- `CHUNK_FILE_LIMIT = 1000`: Files per chunk
- `MAX_NEW_TOKENS = 512`: Max summary length
- `TEMPERATURE = 0.7`: Generation randomness

---

### Workflow 2: Per-File Simple Summarization

**Script:** `qwen_summarize_single.py`

**Use Case:** Analyzing individual audio files with frame-level detail preserved.

**Pipeline:**
```
┌─────────────────────────────────────────────────────────────────┐
│  File → Encode → Project → Qwen → Summary                       │
│                                                             │
│  Input:  One audio file                                         │
│    ↓                                                           │
│  Encode: (1, 1500, 768) - Whisper encoder output                │
│    ↓                                                           │
│  Project: (1, 1500, 4096) - Frame-by-frame projection          │
│    ↓                                                           │
│  Concat: [prompt_embeds | projected_audio]                     │
│    ↓                                                           │
│  Qwen Generate: Summary text                                    │
└─────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- **No compression**: All 1500 frames preserved for LLM
- **Individual outputs**: One summary per file
- **Simple setup**: No additional configuration needed
- **Output:** JSON with file paths and summaries + console output

**Usage:**
```bash
# Single file
python qwen_summarize_single.py audio.flac

# Directory of files
python qwen_summarize_single.py /path/to/dataset/

# With options
python qwen_summarize_single.py /dataset/ \
  --pattern "*.wav" \
  --max-files 50 \
  --output results.json
```

**Command-line Options:**
- `--output, -o`: Save results to JSON file
- `--pattern, -p`: File pattern (default: `*.flac`)
- `--max-files, -n`: Limit number of files to process

---

### Workflow 3: Per-File StreamingAdapter Summarization

**Script:** `qwen_summarize_single_adapter.py`

**Use Case:** Efficient per-file processing with learnable compression and temporal awareness.

**Pipeline:**
```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Encode & Split into Windows                             │
│  File → Encode → (1, 1500, 768)                                 │
│                                                                  │
│  Windowing (default: 50% overlap):                              │
│  ├── Window 1: frames [0-10)    (10 frames @ 20ms = 0.2s)       │
│  ├── Window 2: frames [5-15)    ← 50% overlap                  │
│  ├── Window 3: frames [10-20)   ← 50% overlap                  │
│  └── ... (~300 windows total)                                   │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: StreamingAdapter Compression                            │
│  For each window:                                                │
│    Input: (1, 10, 768) - Whisper frames                         │
│      ↓                                                           │
│    Q-Former (self-attn + cross-attn + FFN) × 2 layers           │
│      ↓                                                           │
│    Compress: (1, 10, 768) → (1, 4, 4096) - 4 learnable tokens  │
│      ↓                                                           │
│    Stability Buffer (EMA smoothing)                             │
│                                                                  │
  Concatenate all windows: (1, 300×4, 4096) = (1, 1200, 4096)    │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: LLM Summarization                                       │
│  concat([prompt_embeds | compressed_audio])                     │
│  Qwen Generate → Summary text                                    │
└─────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- **Compression**: 1500 frames → ~1200 tokens (configurable)
- **Temporal awareness**: 50% overlapping windows preserve edge information
- **Learnable queries**: Q-Former extracts salient representations
- **Stability buffer**: EMA smoothing between windows for consistency
- **Adaptive**: Adjustable compression via window parameters

**Default Configuration (Testing Mode):**
- **Window size**: 10 frames (0.2s at 20ms stride)
- **Stride**: 5 frames (0.1s, 50% overlap)
- **Queries per window**: 4 tokens
- **Output**: ~300 windows × 4 tokens = 1200 tokens

**Usage:**
```bash
# Default settings (1500 → ~1200 tokens)
python qwen_summarize_single_adapter.py audio.flac

# Adjustable compression
python qwen_summarize_single_adapter.py audio.flac \
  --window-size 10 \
  --stride 5 \
  --num-queries 4

# More compression (stride=10 → ~600 tokens)
python qwen_summarize_single_adapter.py audio.flac --stride 10

# Higher fidelity (stride=2 → ~3000 tokens)
python qwen_summarize_single_adapter.py audio.flac --stride 2
```

**Command-line Options:**
- `--output, -o`: Save results to JSON
- `--pattern, -p`: File pattern (default: `*.flac`)
- `--max-files, -n`: Limit file count
- `--num-queries, -m`: Tokens per window (default: 4)
- `--window-size`: Frames per window (default: 10)
- `--stride`: Stride in frames (default: 5 → 50% overlap)

**Output Enhancement:**
```json
{
  "file": "audio.flac",
  "summary": "The audio contains...",
  "status": "success",
  "num_windows": 300,
  "num_tokens": 1200,
  "compression_ratio": 960.0,
  "overlap_percent": 50.0
}
```

---

## Workflow Comparison

| Aspect | Batch Chunked | Per-File Simple | Per-File Adapter |
|--------|---------------|-----------------|------------------|
| **Input** | 1000+ files | 1-N files | 1-N files |
| **Output** | 1 final summary | N summaries | N summaries |
| **Compression** | Mean pooling | None (100%) | Configurable (~80%) |
| **Tokens to LLM** | 1500 per chunk | 1500 per file | 300-3000 per file |
| **Temporal info** | Aggregated | Preserved | Preserved (with overlap) |
| **Memory usage** | Low (streaming) | Medium | Medium |
| **Use case** | Dataset overview | Detailed analysis | Efficient analysis |
| **Components** | ` projector.py` | `projector.py` | `StreamingAdapter` |

---

## Installation

### Requirements

```bash
# Python 3.9+
pip install torch transformers librosa numpy
```

### Project Structure

```
audio-streaming-base/  # legacy layout; use encoder/whisper_encoder.py in this repo
├── encoder/whisper_encoder.py         # Audio encoding (or `from encoder import ...`)
├── projector.py                       # Simple projection
├── qwen_encoded_summarize.py          # Batch workflow
├── qwen_summarize_single.py           # Per-file simple
├── qwen_summarize_single_adapter.py   # Per-file adapter
├── utils/
│   └── whisper_model_loader.py        # Whisper model loading
└── README.md                          # This file

../audio-streaming-adapter/
└── src/adapter/
    ├── streaming_adapter.py           # Q-Former adapter
    ├── rate_controller.py             # Adaptive rate control
    ├── stability_buffer.py            # Temporal smoothing
    └── cross_attention.py             # Attention mechanisms
```

### Models

- **Whisper Encoder**: Automatically loaded by `utils/whisper_model_loader.py`
- **Qwen3-8B**: Downloaded automatically by Hugging Face

---

## Usage Examples

### Example 1: Summarize Large Dataset

```bash
# Edit qwen_encoded_summarize.py DATASET_ROOT
python qwen_encoded_summarize.py
```

Output:
```
Found 2500 files

[SUMMARIZING] Chunk 1 (1000 files) ...
[CHUNK 1 SUMMARY]
This segment contains various spoken content including...

[SUMMARIZING] Chunk 2 (1000 files) ...
[CHUNK 2 SUMMARY]
Continues with additional speech about...

[SUMMARIZING] Chunk 3 (500 files) ...
[CHUNK 3 SUMMARY]
Final segment contains...

[FINAL SUMMARY] Combining 3 chunk summaries ...
The entire dataset contains a rich collection of...
```

### Example 2: Individual File Analysis

```bash
# Simple projector (full detail)
python qwen_summarize_single.py /data/test_audio/file_001.flac \
  --output simple_results.json

# StreamingAdapter (efficient)
python qwen_summarize_single_adapter.py /data/test_audio/file_001.flac \
  --stride 5 \
  --output adapter_results.json
```

### Example 3: Batch Processing

```bash
# Process first 50 files in a directory
python qwen_summarize_single_adapter.py /data/librispeech/ \
  --max-files 50 \
  --pattern "*.flac" \
  --output batch_50.json
```

Output statistics:
```
======================================================================
PROCESSING COMPLETE
======================================================================
Total files: 50
Successful: 50
Errors: 0

Compression Statistics:
  Average windows per file: 300.0
  Average tokens per file: 1200.0
  Average compression ratio: 960.0x (1500 frames → 1200.0 tokens)
```

---

## Compression Configuration Guide

### Choosing Compression Strategy

**Use Simple Projector (No Compression) when:**
- Maximum temporal detail required
- Processing individual files
- LLM has sufficient capacity

**Use StreamingAdapter (With Compression) when:**
- Processing many files efficiently
- Token budget constrained
- Need temporal awareness but not frame-level detail

### Compression Ratios

| Stride | Windows | Tokens | Ratio | Use Case |
|--------|---------|--------|-------|----------|
| 1 | 1500 | 6000 | 0.25x | Maximum detail |
| 2 | 750 | 3000 | 0.5x | High fidelity |
| 5 | 300 | 1200 | 1.25x | **Default (testing)** |
| 10 | 150 | 600 | 2.5x | Balanced |
| 20 | 75 | 300 | 5x | High compression |

*Note: Ratio = frames/tokens (lower = more tokens preserved)*

### Overlap Configuration

```bash
# No overlap (stride = window_size)
--window-size 10 --stride 10

# 25% overlap (stride = window_size * 0.75)
--window-size 10 --stride 8

# 50% overlap (stride = window_size * 0.5) ← Recommended
--window-size 10 --stride 5

# 75% overlap (stride = window_size * 0.25)
--window-size 10 --stride 2
```

**Higher overlap** → Better temporal continuity, more windows/tokens
**Lower overlap** → Less windows/tokens, possible edge information loss

---

## Architecture Details

### Encoding Process

```python
# Whisper Encoder Pipeline
waveform (audio) 
  → librosa.load (16kHz, mono)
  → Whisper processor (Mel spectrogram)
  → Whisper encoder
  → last_hidden_state (1, 1500, 768)
```

- **Frame rate**: 20ms per frame
- **Total duration**: 30 seconds per file (1500 frames × 20ms)
- **Dimensionality**: 768 features per frame (Whisper-tiny)

### Projection vs. Adapter

**Simple Projector:**
```
(1, 1500, 768)
  → Linear(768→2048) → ReLU → Linear(2048→4096)
  → (1, 1500, 4096)
```
- Independent per-frame transformation
- No cross-frame information flow
- Fast computation

**StreamingAdapter:**
```
(1, 10, 768) [single window]
  → Q-Former × 2 layers
    * Self-attention on queries (4 tokens)
    * Cross-attention to frames
    * Feed-forward network
  → Stability Buffer (EMA smoothing)
  → (1, 4, 4096) [compressed tokens]
```
- Learnable query tokens capture salient features
- Cross-attention enables frame-query interaction
- Temporal smoothing via EMA buffer

---

## Performance Considerations

### Memory Usage

| Workflow | Peak Memory | Scale Factor |
|----------|-------------|--------------|
| Batch chunked | 1000 × encoder | Low (streaming) |
| Per-file simple | 1 file + LLM | Medium |
| Per-file adapter | 1 file + windows + LLM | Medium |

### Processing Time (approximate)

| Workflow | Per File | 100 Files |
|----------|----------|-----------|
| Batch chunked | N/A (batched) | ~10-15s |
| Per-file simple | 0.5-1s | ~50-100s |
| Per-file adapter | 1-2s | ~100-200s |

*Times vary based on hardware and configuration*

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size or max-files
python qwen_summarize_single.py dataset/ --max-files 10

# Use CPU (slower)
CUDA_VISIBLE_DEVICES="" python script.py
```

**2. Module Not Found for StreamingAdapter**
```bash
# Ensure path is correct in script:
sys.path.insert(0, "/home/ml/workspaces/kristina/audio-stream/audio-streaming-adapter/src")
```

**3. Whisper Model Loading Errors**
```bash
# Check utils/whisper_model_loader.py
# Ensure model cache directory exists
# Try downloading manually via Hugging Face CLI
```

**4. Audio File Format Issues**
```bash
# Convert to FLAC (recommended):
ffmpeg -i input.wav -ac 1 -ar 16000 output.flac

# Or use librosa directly (supports wav, mp3, flac, etc.)
```

### Debug Mode

Edit scripts to add verbosity:
```python
# Add to encoder/whisper_encoder.py (e.g. inside get_encoder_output)
print(f"Processing {file_path}, sr={sample_rate}, shape={waveform.shape}")

# Add to main scripts
import traceback
try:
    # processing
except Exception as e:
    traceback.print_exc()
```

---

## Advanced Configuration

### StreamingAdapter Parameters

```python
streaming_adapter = StreamingAdapter(
    d_encoder=768,          # Match Whisper output dim
    d_llm=4096,             # Match Qwen embedding dim
    num_queries=4,          # Tokens per window (1-32)
    num_layers=2,           # Q-Former stacks
    num_heads=4,            # Attention heads
    d_ffn=2048,             # FFN hidden size
    dropout=0.1,            # Regularization
    ema_alpha=0.8,          # Smoothing factor (0-1)
    learnable_ema=False,
    use_rate_controller=False,  # Enable adaptive compression
    rate_threshold=0.5,     # Hard gate threshold
    target_rate=2.0,        # Target tokens/window
)
```

### Generation Parameters

```python
# In all scripts:
MAX_NEW_TOKENS = 512       # Max summary length
TEMPERATURE = 0.7          # 0.0 = deterministic, 1.0 = random
DO_SAMPLE = True           # Use sampling
TOP_P = 0.9               # Nucleus sampling (add if needed)
TOP_K = 50                # Top-k sampling (add if needed)
```

---

## Future Enhancements

- [ ] Add Early-Commit Gate integration for latency optimization
- [ ] Support for streaming real-time audio processing
- [ ] Multi-language support (currently English-focused)
- [ ] Batch processing with parallel execution
- [ ] Web UI for interactive summarization
- [ ] Evaluation metrics (ROUGE, BLEU) for summary quality

---

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Submit pull request

---

## License

[Specify your license here]

---

## References

- **Whisper**: OpenAI's speech recognition model
- **Qwen**: Alibaba's family of large language models
- **BLIP-2**: Bootstrapped Language-Image Pre-training (Q-Former architecture)
- **Research Paper**: [Link to relevant paper if applicable]

---

## Contact

For questions or issues, please open an issue on the repository or contact [your-email@example.com].

---

**Last Updated:** 2024-01