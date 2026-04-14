# Phase 2: Streaming Adapter — Design Rationale

## Architecture (4 Components)

From the research summary, the system has 4 distinct components:

```
┌──────────────────────────────────────────────────────────────┐
│  Component 1: Frozen Audio Encoder (Whisper)                 │
│    Raw audio → frame features F ∈ R^{T × D_enc}             │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│  Component 2: Streaming Adapter Network (TRAINABLE)          │
│    Q-Former style cross-attention resampler                  │
│    • Learnable queries Q ∈ R^{m × D_q} (m=1-4)              │
│    • Self-Attention + Cross-Attention + FFN (BLIP-2 style)   │
│    • Stability buffer (EMA) for temporal consistency         │
│    • Optional: adaptive token rate controller                │
│    Output: compressed tokens Z ∈ R^{m × D_llm}              │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│  Component 3: Early-Commit Gate (TRAINABLE, optional)        │
│    Decides: "Start generating NOW?" vs "Keep listening?"     │
│    g_t = σ(MLP(pool(Z_{1:t})))                               │
│    Learns latency-accuracy tradeoff                          │
└───────────────────────┬──────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────┐
│  Component 4: Frozen LLM (small LLaMA/Phi/Mistral)           │
│    Consumes accumulated tokens via KV-cache                  │
│    Generates response incrementally                          │
└──────────────────────────────────────────────────────────────┘
```

**Key**: Only Components 2 and 3 are trainable. Encoder and LLM remain frozen.

---

## Chunked Streaming Pipeline

```
Window params: 0.8s window / 0.4s stride (50% overlap)
Target: ~1-3 tokens/sec (2 minutes ≈ 240 tokens vs. 3000+ frames)
```

1. Audio split into overlapping windows (0.8s window, 0.4s stride)
2. Frozen Whisper encoder → frame features F ∈ R^{T × D_enc}
3. Trainable adapter compresses each window → m tokens (1-4 tokens/window)
4. Tokens appended to LLM context (KV-cache friendly)
5. LLM generates early and continues as more tokens arrive

**Implementation note (current codebase):**
- Encoder outputs are windowed using `adapter/windowing.py::WhisperFrameWindowizer` over the encoder time axis.
- For training notebooks, `adapter/adapter_llm_pipeline.py::whisper_waveform_to_encoder_windows(...)` runs the full-utterance Whisper encoder pass and then applies the `WhisperFrameWindowizer`.

---

## Component 2: Streaming Adapter Network

### Q-Former Layer (BLIP-2 Style)

Each layer has three sub-layers:

1. **Self-Attention** (Q ↔ Q): Queries attend to each other — coordination to avoid redundancy
2. **Cross-Attention** (Q → F): Queries attend to encoder frames — information extraction
3. **FFN**: Per-token nonlinear transformation

**Implementation note (current codebase):**
- The per-layer module is implemented in `src/adapter/cross_attention.py` as `QFormerLayer`.
- `src/adapter/streaming_adapter.py` stacks these layers and can insert self-only layers between cross-attention layers (see below).

### Why Cross-Attention with Learnable Queries?

**Option 1: Average pooling** — Destroys temporal structure. "Hello world" and "World hello" produce the same token.

**Option 2: Strided CNN / downsampling** — Rigid, content-independent. Silence gets same attention as critical phonemes.

**Option 3: Cross-attention with learnable queries (Q-Former)** — Content-adaptive, decoupled from input length, queries specialize via self-attention.

### Stability Buffer (EMA)

**Mechanism**: `Z'_t = alpha * Z_t + (1 - alpha) * Z'_{t-1}` with alpha ≈ 0.8

Why EMA:
1. **Recency bias**: Old windows decay exponentially — correct for streaming
2. **No extra parameters** (or just one scalar alpha)
3. **Causal**: Only depends on past, not future
4. **Proven**: Used in batch normalization

### Mechanism vs. Loss

- The **EMA** is the *mechanism* that smooths tokens at inference time.
- The **stability loss** `L_stability = sum_t ||Z_t - Z_{t-1}||^2` is the *training signal* that penalizes large jumps between adjacent windows.

Together they ensure smooth token streams.

### Adaptive Token Rate Controller (Optional)

Dynamically adjusts how many of the m query slots to use per window:
- Silence: 1 token. Dense speech: 3-4 tokens.
- Uses soft gating (training) / hard thresholding (inference)
- Produces two losses:
  - **L_sparse**: L1 on gate scores — encourages using fewer tokens
  - **L_rate**: MSE between effective token count and target rate

### Cross-attention placement across layers (current implementation)

`StreamingAdapter` supports inserting **self-attention-only** layers between cross-attention layers.

- Parameter: `cross_layer_in_between = K`
- Period: `P = K + 1`
- **Cross-attention runs at the end of each block**: layer `i` uses cross-attention iff `i % P == P - 1`
  - Example `K=1` → cross-attn on layers `1, 3, 5, ...` (layer 0 is self-only)
  - `K=0` → every layer uses cross-attention

---

## Component 3: Early-Commit Gate (Separate Module)

**This is NOT the rate controller.** They solve different problems:

| | Rate Controller | Early-Commit Gate |
|---|---|---|
| Question | "How many tokens for THIS window?" | "Should the LLM START generating?" |
| Scope | Per-window decision | Per-stream decision |
| Part of | Component 2 (adapter, optional) | Component 3 (separate) |
| Loss | L_sparse + L_rate | L_gate |

The gate operates on accumulated tokens Z_{1:t} and outputs commit probability:
```
g_t = σ(MLP(mean_pool(Z_{1:t})))
```

L_gate balances:
- Committing too early → accuracy penalty (not enough context)
- Committing too late → latency penalty (unnecessary waiting)

---

## Loss Functions (All Stages)

### Stage 1: Audio-Text Alignment
```
L = L_align + λ_stability · L_stability
```

### Stage 2: Content Preservation
```
L = L_asr + λ_align · L_align + λ_stability · L_stability + λ_sparse · L_sparse
```

### Stage 3: Task Distillation
```
L = L_task + λ_asr · L_asr + λ_stability · L_stability + λ_rate · L_rate + λ_gate · L_gate
```

### Loss Inventory

| Loss | Source | What it does |
|------|--------|-------------|
| L_align | Contrastive | Audio-text embedding similarity |
| L_stability | StabilityBuffer | Temporal consistency: MSE(Z_t, Z_{t-1}) |
| L_sparse | RateController | Sparsity: fewer active tokens when possible |
| L_rate | RateController | Token rate: stay near target R = m/t |
| L_gate | EarlyCommitGate | Latency-accuracy tradeoff |
| L_asr | Training Stage 2 | ASR distillation (content preservation) |
| L_task | Training Stage 3 | KL divergence from teacher LLM |
| Prefix consistency | Training | P(y \| Z_{1:t}) ≈ P(y \| Z_{1:t+k}) |
| Revision penalty | Training | Penalize LLM changing earlier output |

**Note**: Prefix consistency and revision penalty require the LLM in the loop,
so they are implemented in the training pipeline, not in the adapter itself.

---

## Checkpoint + inference wiring (current notebooks)

- Training checkpoints are written under `audio-streaming-adapter/checkpoints/` when you run trainers from the `audio-streaming-adapter/` directory (see `training/utils/config.py` / each stage script).
- `notebooks/adapter_llm_walkthrough.ipynb` can load a trained adapter checkpoint via the env var:
  - `ADAPTER_CHECKPOINT_PATH="checkpoints/adapter_adapter.pt"`
  - The notebook loads `adapter_state_dict` / `model_state_dict` / `state_dict` (first one present) into the `StreamingAdapter` before generation.

---

## Architecture Dimensions

For Whisper-medium + small LLM (Phi-2):

| Component | Dimension | Rationale |
|-----------|-----------|-----------|
| Whisper encoder dim (D_enc) | 1024 | Fixed by Whisper-medium |
| Number of queries (m) | 4 | Max tokens per window; rate controller picks 1-3 |
| Query dim (D_q) | 1024 | Match encoder dim |
| Number of attention heads | 4 | 4 heads × 256 dim/head = 1024 |
| FFN hidden dim | 2048 | Standard 2x expansion |
| Output dim (D_llm) | 2560 | Must match LLM embedding dim (Phi-2) |
| Number of Q-Former layers | 2 | Shallow enough to be fast, deep enough to be expressive |
| Window size | 0.8s | Research spec |
| Window stride | 0.4s | 50% overlap |

---

## Data Flow Summary

```
Raw Audio (16kHz waveform)
  ↓ [Overlapping windows: 0.8s window, 0.4s stride]
Audio Window (12800 samples)
  ↓ [Component 1: Frozen Whisper Encoder]
Frame Features F ∈ R^{T × 1024}       (T ≈ 40 for 0.8s)
  ↓ [Component 2: Streaming Adapter]
  │   Q-Former layers (self-attn + cross-attn + FFN)
  │   Optional rate controller (soft gating)
  │   Output projection (1024 → 2560)
  │   Stability buffer (EMA smoothing)
Compressed Tokens Z ∈ R^{m × 2560}    (m = 1-4, optionally adaptive)
  ↓ [Component 3: Early-Commit Gate]
  │   Should LLM start generating?
  ↓ [Component 4: Frozen Small LLM]
  │   Append tokens to KV-cache
  │   Incremental generation
Streaming Output
```
