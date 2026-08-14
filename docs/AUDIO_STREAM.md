# Audio Streaming Adapter — Research & Implementation

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Solution Overview](#2-solution-overview)
3. [System Architecture (4 Components)](#3-system-architecture-4-components)
4. [Complete Inference Pipeline](#4-complete-inference-pipeline)
5. [Complete Training Pipeline](#5-complete-training-pipeline)
6. [Component 2: Streaming Adapter Network](#6-component-2-streaming-adapter-network)
7. [Component 3: Turn-End Commit Gate](#7-component-3-turn-end-commit-gate)
  - [VAD vs Turn Detection](#71-vad-vs-turn-detection)
  - [Combining VAD + Turn Detection](#combining-vad--turn-detection)
  - [Gate Training and Configuration](#72-gate-training-and-configuration)
  - [Checkpoint Migration](#73-checkpoint-migration)
  - [Gate Design Comparison](#74-gate-design-comparison)
8. [Loss Functions by Training Stage](#8-loss-functions-by-training-stage)
9. [Stage 1: Contrastive Audio–Text Alignment](#9-stage-1-contrastive-audiotext-alignment)
  - [The Cone Collapse Problem](#91-the-cone-collapse-problem)
  - [The Fix: Batch Centering](#92-the-fix-batch-centering)
10. [Stage 2: Content Preservation (ASR Distillation)](#10-stage-2-content-preservation-asr-distillation)
11. [Stage 3: Task Distillation (Streaming)](#11-stage-3-task-distillation-streaming)
12. [Evaluation Strategy](#12-evaluation-strategy)
13. [Baseline: SALMONN-7B](#13-baseline-salmonn-7b)
  - [What SALMONN Does Well](#131-what-salmonn-does-well)
    - [Why Cosine Retrieval is Misleading for SALMONN](#132-why-cosine-retrieval-is-misleading-for-salmonn)
    - [Evaluation Script](#133-evaluation-script)
14. [Our System vs. SALMONN Baseline](#14-our-system-vs-salmonn-baseline)
15. [Key Differences Summary](#15-key-differences-summary)
16. [Theoretical Foundation](#16-theoretical-foundation)
17. [Related Work](#17-related-work)

## 1. Problem Statement

Audio encoders (e.g. Whisper) produce dense frame sequences at tens of frames per second. Feeding these frames directly into small language models (LLMs) causes two fundamental problems:

1. **Context explosion**: 2 minutes of audio produces 3,000+ encoder frames — far beyond what a small LLM can hold in context.
2. **Latency**: The LLM must wait for the entire audio stream before generating any response.

Neither batch speech-to-text (ASR pipeline → LLM) nor direct frame injection is suitable for real-time, low-latency audio-LLM interaction.

## 2. Solution Overview

A **streaming adapter network** that compresses continuous audio into a low-bitrate, append-only token stream (1–3 tokens/sec) that a frozen small LLM can consume in real-time.

```
Target compression: ~3000 frames → ~240 tokens over 2 minutes (≈ 12.5× reduction)
Target latency: sub-window first-token (< 0.8s from speech onset)
```

Only the adapter (and optionally the early-commit gate) are trained. The audio encoder and LLM remain completely frozen throughout all stages.

## 3. System Architecture (4 Components)


| #   | Component            | Module (Trainable)                          | Role                                                 |
| --- | -------------------- | ------------------------------------------- | ---------------------------------------------------- |
| 1   | Audio encoder        | `whisper_encoder.py` (Frozen)               | Mono 16 kHz waveform → Whisper encoder hidden states |
| 2   | Streaming adapter    | `streaming_adapter.py` (Yes)                | Encoder frames → compressed LLM-space tokens         |
| 3   | Turn-end commit gate | `turn_end_commit_gate.py` (Yes) (stage 2/3) | When to start LLM generation                         |
| 4   | Causal LLM           | `qwen.py` (Frozen)                          | Text generation (Qwen3-8B)                           |


Module paths:

- `src/encoder/whisper_encoder.py`
- `src/adapter/streaming_adapter.py`
- `src/adapter/turn_end_commit_gate.py`
- `src/llm/qwen.py`

**Window geometry (training and inference):** 0.8 s window, 0.4 s stride (50% overlap) @ 16 kHz → 12 800 samples per chunk, new window every 0.4 s.

**Whisper canvas:** Each chunk is mel-padded to 3000 frames (30 s). The encoder always outputs **T = 1500** frames per chunk. **No post-encode trimming** is applied — short windows still produce `(1, 1500, 768)`.

**Current LLM:** Qwen3-8B (`D_llm = 4096`). Swapping LLMs requires only changing the adapter output projection.

## 4. Complete Inference Pipeline

Two inference paths exist. Both share the same per-window encode → adapter steps; they differ in how tokens reach the LLM.

### 4.1 Streaming inference (target path, unified)

**Classes:** `WhisperAdapterLLMCommitGatePipeline.generate_streaming()`, `WhisperAdapterStreamingSession` (`src/adapter_llm_streaming.py`), `LlmKvCacheSession` (`src/llm/kv_cache.py`).

```
For each window t = 0, 1, 2, …
  │
  ├─ AudioWaveformWindowizer                src/adapter/windowing.py
  │    slice 0.8 s chunk (12 800 samples) every 0.4 s hop
  │
  ├─ encode_waveform_to_hidden()            src/encoder/whisper_encoder.py
  │    mel → pad to 3000 frames (30 s canvas)
  │    Whisper-small encoder (frozen)
  │    output: F ∈ R^{1500 × 768}           no trimming
  │
  ├─ StreamingAdapter.forward_window()      src/adapter/streaming_adapter.py
  │    learnable queries Q ∈ R^{m × 768}    m = 4 (num_queries)
  │    Q-Former × 2 (self-attn + cross-attn + FFN)
  │    optional AdaptiveRateController      stage 2+; hard gate at inference
  │    output_proj: 768 → 4096
  │    StabilityBuffer (EMA, α ≈ 0.8)
  │    output: Z_t ∈ R^{m × 4096}           typically (1, 4, 4096)
  │
  ├─ Append Z_t to accumulated sequence Z_{1:t}
  │
  ├─ TurnEndCommitGate                      src/adapter/turn_end_commit_gate.py
  │    SilenceTracker on Z_t (token activity)
  │    attention-pool + classifier on Z_{1:t}
  │    commit_prob, should_commit
  │
  ├─ LlmKvCacheSession.append_embeddings(Z_t)
  │     incrementally extend Qwen3-8B KV-cache (no re-encoding prior windows)
  │
  └─ if should_commit:
        generate from cache → stream text tokens
        (stop consuming audio for this turn)
     else:
        wait for next window

End of audio (no commit):
  optional finalize() → generate from cache
```

**Rate controller (optional, stage 2+):** scales or zeroes query slots per window based on complexity. Does not remove slots from the tensor — inactive slots become zeros.

**Gate rule (default):** `should_commit = (commit_prob > τ) AND silence_ready`.

**Prompt modes:**


| Mode                   | LLM prefix                  | When                    |
| ---------------------- | --------------------------- | ----------------------- |
| `train_style_asr=True` | audio tokens only           | Stage 1–2 ASR eval      |
| Chat prompt            | `[prompt_embeds | Z_{1:t}]` | Stage 3 / summarization |


**Demo:** `examples/streaming_demo.py`

### 4.2 Batch inference (legacy path)

**Classes:** `WhisperAdapterLLMPipeline.generate()`, `WhisperAdapterLLMCommitGatePipeline.generate()`.

```
Full waveform
  │
  ▼
Windowize all chunks upfront
  │
  ▼
Whisper encode each chunk → list of (1, 1500, 768)
  │
  ▼
Adapter over all windows (forward or forward_window loop)
  │
  ▼
Concatenate all Z_t → (1, num_windows × m, 4096)
  │
  ├─ [Commit-gate pipeline] gate diagnostics per window;
  │    optional early window truncation (does not use KV-cache streaming)
  │
  ▼
Single model.generate() on full prefix
  │
  ▼
Text output
```

**Difference from §4.1:** all windows are processed first; the LLM receives one batched prefix and decodes once. Gate records probabilities but does not trigger incremental decode.

### 4.3 Inference checklist by checkpoint


| Checkpoint | Pipeline class                        | Rate ctrl | Gate |
| ---------- | ------------------------------------- | --------- | ---- |
| Stage 1    | `WhisperAdapterLLMPipeline`           | Off       | Off  |
| Stage 2    | `WhisperAdapterLLMCommitGatePipeline` | On        | On   |
| Stage 3    | `WhisperAdapterLLMCommitGatePipeline` | On        | On   |


## 5. Complete Training Pipeline

Only the **adapter** (all stages) and **turn-end gate** (stages 2–3) receive gradients. Whisper and Qwen remain frozen.

### 5.1 Shared per-utterance window loop (all stages)

LibriSpeech (or Smart Turn) utterance:

```
audio file → load_mono_waveform_16k()
  │
  ▼
AudioWaveformWindowizer (0.8 s / 0.4 s)
  │
  ▼
for each window chunk:
  │
  ├─ Whisper encode (frozen)              → (1, 1500, 768)
  ├─ adapter.forward_window(F)            → Z_t ∈ (1, 4, 4096)   [gradients]
  ├─ optional rate controller             → L_sparse, L_rate      [stage 2/3]
  ├─ StabilityBuffer                      → L_stability
  └─ append Z_t to Z_{1:t}
  │
  ▼
utterance tokens: concat all Z_t → (1, T_utt, 4096)
  where T_utt = num_windows × 4
```

**Trainers:** `WhisperWindowFeatureExtractor` / `encode_waveform_to_hidden` + `forward_window` (same geometry as inference).

**Notebooks:** set `ADAPTER_CHECKPOINT_PATH` (e.g. `checkpoints/adapter_stage1.pt`) when running
`notebooks/adapter_llm_walkthrough.ipynb` or other training walkthroughs. Checkpoints load
`adapter_state_dict` / `model_state_dict` / `state_dict` (first key present).

### 5.2 Stage 1 — Contrastive alignment

**Script:** `training/adapter_contrastive_trainer.py`  
**Trains:** adapter only | **Rate controller:** off | **Gate:** off | **LLM CE:** off

```
Per batch of utterances:
  │
  ├─ For each utterance: window loop → Z_{1:T}
  ├─ Pad batch → audio_tokens (B, T_max, 4096)
  │
  ├─ Transcript → tokenizer → frozen Qwen embedder → label_embeds
  │
  ├─ L_align = InfoNCE(mean_pool(Z), mean_pool(label_embeds))
  │            with batch-centering fix (see §9.2)
  │
  └─ L_stability = mean of per-window stability losses

L = L_align + λ_stability · L_stability
```

**Checkpoint:** `checkpoints/adapter_stage1.pt`

### 5.3 Stage 2 — ASR distillation

**Script:** `training/adapter_asr_trainer.py`  
**Trains:** adapter + `TurnEndCommitGate` | **Rate controller:** on (target ~2 tokens/window)

```
Per utterance:
  │
  ├─ Window loop:
  │     forward_window → Z_t
  │     gate(Z_{1:t}, t, T) → L_gate (BCE; synthetic or Smart Turn labels)
  │
  ├─ Stack all audio tokens
  │
  ├─ Frozen Qwen teacher forcing:
  │     inputs_embeds = [audio_tokens | im_end/BOS | transcript_embeds]
  │     labels: -100 on audio/BOS positions, transcript tokens on text
  │     → L_asr (cross-entropy)
  │
  ├─ L_align (contrastive, weighted)
  ├─ L_stability, L_sparse, L_rate (from adapter + rate controller)
  │
  └─ L = L_asr + λ_align·L_align + λ_stability·L_stability
            + λ_sparse·L_sparse + λ_rate·L_rate + λ_gate·L_gate
```

**Inference note:** stage 2 eval uses `audio_tokens → generate` without appending im_end (Qwen3 treats im_end as chat mode).

**Checkpoint:** `checkpoints/adapter_stage2.pt` (includes `gate_state_dict`)

### 5.4 Stage 3 — Task distillation

**Script:** `training/adapter_task_trainer.py`  
**Trains:** adapter + gate | **Teacher & student:** both frozen Qwen3-8B

```
Per utterance:
  │
  ├─ Teacher (text-only, frozen):
  │     prompt + generate → teacher logits
  │
  ├─ Window loop (same as stage 2):
  │     adapter → Z_t, gate → L_gate
  │     at commit points: student forward on [Z_{1:t} | BOS]
  │
  ├─ L_task = KL(student_logits ∥ teacher_logits)
  ├─ L_asr, L_stability, L_rate, L_gate (optional weights)
  ├─ prefix consistency: P(y|Z_{1:t}) ≈ P(y|Z_{1:t+k})
  └─ revision penalty on contradicting earlier tokens

L = L_task + λ_asr·L_asr + λ_stability·L_stability + λ_rate·L_rate
      + λ_gate·L_gate + λ_prefix·L_prefix + λ_revision·L_revision
```

**Target inference:** chat prompt + audio tokens (`--prompt-asr` path).

### 5.5 Training vs inference


| Aspect          | Training                             | Inference (streaming)            |
| --------------- | ------------------------------------ | -------------------------------- |
| Audio scope     | Full utterance, all windows          | Chunk-by-chunk over time         |
| Whisper output  | `(1, 1500, 768)` per window, no trim | Same                             |
| Adapter         | Gradients on; EMA state updated      | Eval; EMA carried across windows |
| Rate controller | Soft gating + losses                 | Hard zero inactive slots         |
| Gate            | L_gate BCE                           | `should_commit` triggers decode  |
| LLM             | Frozen; CE or KL for loss only       | KV-cache append + `generate`     |
| Token delivery  | Concat all windows, one LM forward   | Incremental append per window    |


## 6. Component 2: Streaming Adapter Network

**Implementation**: `src/adapter/streaming_adapter.py`

### Q-Former Layer (BLIP-2 Style)

Each layer (`src/adapter/cross_attention.py::QFormerLayer`) has three sub-layers:


| Sub-layer       | Operation           | Purpose                                         |
| --------------- | ------------------- | ----------------------------------------------- |
| Self-Attention  | Q ↔ Q               | Queries coordinate to avoid redundancy          |
| Cross-Attention | Q → F               | Queries extract information from encoder frames |
| FFN             | Per-token nonlinear | Expressiveness / feature transformation         |


### Why Cross-Attention with Learnable Queries?


| Option                                      | Problem                                                           |
| ------------------------------------------- | ----------------------------------------------------------------- |
| Average pooling                             | Destroys temporal order — "hello world" = "world hello"           |
| Strided CNN                                 | Rigid, content-independent — silence gets same weight as phonemes |
| Q-Former (chosen)                           | Content-adaptive; queries specialize via                          |
| self-attention; decoupled from input length |                                                                   |


### Cross-Attention Layer Placement

`StreamingAdapter` supports interleaving self-attention-only layers between cross-attention layers via `cross_layer_in_between = K`:

- `K = 0`: Every layer uses cross-attention (full Q-Former stack)
- `K = 1`: Period P = 2; cross-attention on layers 1, 3, 5, … (layer 0 is self-only)
- Pattern: layer `i` uses cross-attention iff `i % (K+1) == K`

### Stability Buffer (EMA)

**Mechanism**: `Z'_t = α · Z_t + (1 − α) · Z'_{t-1}`,  α ≈ 0.8

**Why EMA**:

- Recency bias: Old windows decay — correct for causal streaming
- No added parameters (or one scalar α if learnable)
- Causal: Depends only on past windows, never future
- Proven: Analogous to batch norm running stats

**Mechanism vs. Loss** (these are separate):

- **EMA** is the inference-time smoothing mechanism
- **Stability loss** `L_stability = Σ_t ||Z_t − Z_{t-1}||²` is the training signal that penalises large jumps between adjacent windows

### Adaptive Token Rate Controller (Optional)

`src/adapter/rate_controller.py`

Dynamically selects how many of the m query slots to activate per window:

- Silence → 1 token
- Dense speech → 3–4 tokens

Losses produced:

- `L_sparse`: L1 on gate scores — encourages using fewer tokens
- `L_rate`: MSE between effective token count and target rate R

Inference uses hard thresholding; training uses soft gating.

## 7. Component 3: Turn-End Commit Gate

**Implementation**: `src/adapter/turn_end_commit_gate.py`

### 7.1 VAD vs Turn Detection

These are often confused because both affect *when the agent speaks*, but they answer
**different questions** at **different levels**.


|                    | VAD (Voice Activity Detection)                | Turn detection (e.g. Smart Turn)                                            |
| ------------------ | --------------------------------------------- | --------------------------------------------------------------------------- |
| **Question**       | Is there **speech** or **silence** right now? | Has the user **finished their turn** (or will they continue)?               |
| **Input**          | Raw audio (energy, lightweight ML)            | Raw audio or encoded tokens (prosody, phrasing, context)                    |
| **Output**         | Speech / non-speech                           | Turn **complete** vs **incomplete**                                         |
| **Typical model**  | Silero VAD                                    | Smart Turn V3, our `TurnEndCommitGate`                                      |
| **When it runs**   | Continuously, cheap                           | After a candidate pause (often post-VAD) or every streaming window          |
| **Knows content?** | No — silence after any speech looks the same  | Yes — distinguishes real turn ends from backchannels and mid-thought pauses |


**VAD** segments the waveform into “someone is talking” vs “nobody is talking.” It does
not know *why* there is silence or whether the user is done.

**Turn detection** decides whether the conversational floor has changed — i.e. whether
the agent should **take a turn** and start responding.

#### Example: why VAD alone is not enough

Agent is explaining something; the user listens and backchannels:

```
Agent:  "...and that's why we use streaming windows."
User:   "ok"          ← short ack, still listening
        [silence]
```


| Stage                         | VAD says         | Turn detection says                         | Agent should respond?   |
| ----------------------------- | ---------------- | ------------------------------------------- | ----------------------- |
| After user says "ok"          | Silence detected | **Incomplete** — backchannel, not a handoff | **No** — keep listening |
| User finishes a real question | Silence detected | **Complete** — turn ended                   | **Yes** — generate      |


Smart Turn’s training data explicitly includes **midfiller** / **endfiller** clips
(short utterances like “ok”, “yes”, “mm-hmm”) labeled complete vs incomplete so the model
learns not to treat every pause as turn-end.

#### Where our pipeline fits

```
Audio stream
  ↓  [Whisper + Adapter]        ← runs while user speaks; prefills LLM KV-cache
  ↓  [TurnEndCommitGate]        ← turn detection on adapter tokens Z_{1:t}
  ↓  should_commit → LLM generate
```

- `**TurnEndCommitGate**` combines turn detection + token-based silence tracking (no separate VAD model)
- To reject unwanted commits on “ok / yes / uh-huh”, train the gate on **Smart Turn**
`endpoint_bool` labels (`GATE_LABEL_SOURCE=smart_turn`). Synthetic LibriSpeech labels
(last window = complete) do not teach backchannel behavior.

#### Combining VAD + turn detection

VAD and turn detection are **complementary** — the gate combines both:


| Layer             | Mechanism                                           | Blocks commit when…                                       |
| ----------------- | --------------------------------------------------- | --------------------------------------------------------- |
| **Silence (VAD)** | :class:`SilenceTracker` — token activity from `Z_t` | User is **actively speaking** (low token activity window) |
| **Turn-end**      | Smart Turn-style head on `Z_{1:t}`                  | Pause is a **backchannel** ("ok", "yes") not a handoff    |


**Inference rule** (when `require_silence_for_commit=True`, default):

```python
should_commit = (commit_prob > threshold) AND silence_tracker.silence_ready
```

`silence_ready` means: not in speech **and** trailing silence ≥ `min_silence_ms` (default 200 ms).

**Classifier joint input** — silence features are concatenated with pooled adapter tokens:

```python
silence_features = [silence_indicator, silence_duration_norm, activity_prob]  # dim=3
logit = classifier(concat(attention_pool(Z_{1:t}), silence_features))
```

Use `SilenceTracker.update_from_window_tokens(window_tokens=Z_t)` each step, or
`gate.make_silence_tracker()`.

Env: `GATE_MIN_SILENCE_MS`, `GATE_REQUIRE_SILENCE`, `GATE_TOKEN_ACTIVITY_THRESHOLD`.

> **This is NOT the rate controller.** They solve different problems.


|          | Rate Controller                    | Turn-End Commit Gate                           |
| -------- | ---------------------------------- | ---------------------------------------------- |
| Question | "How many tokens for THIS window?" | "Has the user stopped speaking → start LLM?"   |
| Scope    | Per-window                         | Per-stream                                     |
| Part of  | Component 2 (adapter, optional)    | Component 3 (separate module)                  |
| Loss     | L_sparse + L_rate                  | L_gate (BCE + optional latency)                |
| Replaces | —                                  | Separate VAD + turn detection (e.g. SmartTurn) |


The gate operates on accumulated adapter tokens `Z_{1:t}` **plus silence features**
from :class:`SilenceTracker` using a **Smart Turn-style** attention pool + classifier:

```python
# Attention pool over token positions (Smart Turn V3 pattern on adapter tokens)
weights = softmax(MLP(Z_{1:t}), dim=1)
pooled = sum(Z_{1:t} * weights, dim=1)
silence = SilenceTracker.features()   # [silence_indicator, silence_duration_norm, activity_prob]
g_t = σ(classifier(concat(pooled, silence)))   # turn-end probability ∈ [0, 1]
should_commit = (g_t > threshold) AND silence_tracker.silence_ready   # when enabled
```

**Unified pipeline latency win**: Whisper encoder + adapter run **while the user speaks**,
appending tokens to the LLM KV-cache. When `should_commit` fires, only gate inference +
LLM decode remain — no separate encode-at-turn-end step.

**L_gate**:

- **Primary**: BCE on turn-end labels (`endpoint_bool` from Smart Turn, or synthetic
last-window labels on LibriSpeech)
- **Optional latency penalty** (Stage 3 streaming tradeoffs):

```python
position = timestep / max(total_timesteps - 1, 1)
latency_penalty = latency_weight * position * (1.0 - commit_prob).mean()
gate_loss = bce_loss + latency_penalty
```

> **Previous design (reference only)**: mean-pool MLP gate in `src/adapter/early_commit_gate.py`
> (commented out). `EarlyCommitGate` is a backward-compatible alias for `TurnEndCommitGate`.
> See [§7.3 Checkpoint migration](#73-checkpoint-migration).

### 7.2 Gate training and configuration

**Latency motivation.** A separate VAD + SmartTurn stack encodes audio only after turn-end.
In the unified pipeline, Whisper + adapter run **while the user speaks**, prefilling the LLM
KV-cache. Latency after turn-end ≈ gate inference + first LLM token.


|                  | Separate VAD + SmartTurn | TurnEndCommitGate                    |
| ---------------- | ------------------------ | ------------------------------------ |
| Input            | Raw audio / silence      | Accumulated adapter tokens `Z_{1:t}` |
| Cost at turn-end | Full encode + classify   | Attention pool + classifier          |


**Loss (`L_gate`)**:

```python
gate_loss = bce_loss(endpoint_label, logits) + latency_penalty
```

- **BCE**: Smart Turn-style, batch-balanced `pos_weight`
- **Latency penalty** (optional): penalizes low `commit_prob` late in the utterance

**Label sources** (`GATE_LABEL_SOURCE` in `training/utils/config.py` → `GateConfig`):


| Mode                  | Behavior                                                                                                                                        |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `synthetic` (default) | LibriSpeech full utterances: all windows except the last → `0`; final window → `1`                                                              |
| `smart_turn`          | `SmartTurnGateDataset` (`src/dataset/smart_turn_gate.py`); HF id default `pipecat-ai/smart-turn-data-v3.2-train`; each clip has `endpoint_bool` |


**Trainers**:


| Script                             | Gate integration                                                 |
| ---------------------------------- | ---------------------------------------------------------------- |
| `training/adapter_asr_trainer.py`  | Joint adapter + gate (synthetic); or Smart Turn gate-only branch |
| `training/adapter_task_trainer.py` | Joint training (synthetic); or Smart Turn gate-only branch       |


Gate is evaluated on full `accumulated_tokens` including the current window at every timestep `t`.

**Environment variables**:


| Variable                        | Default                                 | Description                              |
| ------------------------------- | --------------------------------------- | ---------------------------------------- |
| `GATE_LABEL_SOURCE`             | `synthetic`                             | `synthetic` or `smart_turn`              |
| `SMART_TURN_DATASET`            | `pipecat-ai/smart-turn-data-v3.2-train` | HF dataset id                            |
| `SMART_TURN_SPLIT`              | `train`                                 | Dataset split                            |
| `SMART_TURN_MAX_SAMPLES`        | *(none)*                                | Cap clips for debugging                  |
| `GATE_HIDDEN_DIM`               | `256`                                   | Pool/classifier hidden size              |
| `GATE_THRESHOLD`                | `0.5`                                   | Inference commit threshold               |
| `GATE_LATENCY_WEIGHT`           | `0.1`                                   | Latency penalty weight                   |
| `GATE_MIN_SILENCE_MS`           | `200`                                   | Min trailing silence for `silence_ready` |
| `GATE_REQUIRE_SILENCE`          | `1`                                     | Require silence before `should_commit`   |
| `GATE_TOKEN_ACTIVITY_THRESHOLD` | `8.0`                                   | Mean token L2 norm for speech activity   |


**Streaming inference example**:

```python
pipeline = WhisperAdapterLLMCommitGatePipeline(..., early_commit_gate=gate)
result = pipeline.generate_streaming(waveform, train_style_asr=True)
# result["first_token_time_s"], result["early_commit_commit_probs"], result["committed_on_gate"]
```

Or per-window with `SilenceTracker`:

```python
gate = TurnEndCommitGate(d_llm=4096, require_silence_for_commit=True)
tracker = gate.make_silence_tracker()
for t, step in enumerate(adapter_windows):
    accumulated = torch.cat(all_tokens[: t + 1], dim=1)
    result = gate(accumulated, t, total_windows, silence_tracker=tracker, window_tokens=step["tokens"])
    if result["should_commit"].item():
        break  # → LLM generate
```

**Token-only design**: all silence and turn-end signals come from adapter tokens —
`accumulated_tokens` (`Z_{1:t}`) for the turn-end classifier; `window_tokens` (`Z_t`) for
`SilenceTracker` activity. No raw waveforms or external VAD in this component.

### 7.3 Checkpoint migration

Checkpoint key remains `gate_state_dict`.

**Old mean-pool gate weights are incompatible** with `TurnEndCommitGate` (different parameter
names/shapes). Trainers use `load_gate_state_dict_safe()` and warn on mismatch, leaving the
gate randomly initialized until re-trained. Re-train Stage 2 (or run Smart Turn gate fine-tune)
after upgrading.

### 7.4 Gate design comparison


|                 | Smart Turn V3          | Old EarlyCommitGate                 | TurnEndCommitGate                    |
| --------------- | ---------------------- | ----------------------------------- | ------------------------------------ |
| Input           | Raw audio (mel)        | Adapter tokens                      | Adapter tokens                       |
| Architecture    | Whisper + pool + MLP   | Mean pool + MLP                     | Attention pool + MLP                 |
| Training labels | `endpoint_bool`        | Latency penalty only                | BCE on endpoint + optional latency   |
| Role in stack   | Separate turn detector | Generic early commit                | Integrated turn-end + silence gating |
| Implementation  | `VAD/smart-turn/`      | `early_commit_gate.py` (deprecated) | `turn_end_commit_gate.py`            |


## 8. Loss Functions by Training Stage

### Loss inventory


| Loss               | Stage introduced | What it does                                              |
| ------------------ | ---------------- | --------------------------------------------------------- |
| `L_align`          | Stage 1          | InfoNCE contrastive — audio tokens vs. text embeddings    |
| `L_stability`      | Stage 1          | MSE(Z_t, Z_{t-1}) — temporal smoothness                   |
| `L_asr`            | Stage 2          | Frozen causal LM loss with teacher-forced text            |
| `L_sparse`         | Stage 2          | L1 on rate controller gates — fewer tokens when possible  |
| `L_rate`           | Stage 2          | MSE between effective token count and target rate R       |
| `L_gate`           | Stage 2/3        | Turn-end BCE (+ optional latency penalty) for commit gate |
| `L_task`           | Stage 3          | KL divergence from teacher LLM distribution               |
| Prefix consistency | Stage 3          | P(y | Z_{1:t}) ≈ P(y | Z_{1:t+k})                         |
| Revision penalty   | Stage 3          | Penalise changing earlier generated output                |


**Note**: Prefix consistency and revision penalty require the LLM in the forward pass; they are implemented in the training loop, not in the adapter module.

## 9. Stage 1: Contrastive Audio–Text Alignment

**Status: Completed.**

**Goal**: Align adapter-produced audio tokens to frozen LLM text embeddings using contrastive learning. No text transformer is run inside the adapter — only the LLM embedding layer.

**Loss**:

```
L = L_align + λ_stability · L_stability
```

**Inputs**: LibriSpeech `(audio_path, transcription)` pairs.

**Core forward pass** (see §5.2 for full training flow):

```
waveform
  → AudioWaveformWindowizer (0.8s / 0.4s)
  → Whisper encode per chunk → (1, 1500, 768) per window
  → StreamingAdapter.forward_window() per window → Z_t (1, 4, 4096)
  → concatenate Z_{1:T}
  → mean pool → (batch-center) → L2-normalize → audio_vec

transcription
  → tokenizer → frozen Qwen embedder → mean pool → (batch-center) → L2-normalize → text_vec

L_align = InfoNCE(audio_vec, text_vec)
```

**Checkpoint:** `checkpoints/adapter_stage1.pt`

### 9.1 The Cone Collapse Problem

During initial Stage 1 training, the contrastive loss was not learning. Diagnostics showed:


| Metric             | Value                 | Problem                                                          |
| ------------------ | --------------------- | ---------------------------------------------------------------- |
| `neg_sim`          | 0.40                  | Random unrelated pairs had 40% cosine similarity before training |
| `pos_sim`          | flat at 0.65          | Positives could not separate from negatives                      |
| `pos_minus_neg`    | flat at 0.25          | No real contrastive signal                                       |
| `train/align` loss | stuck at ln(4) ≈ 1.38 | Loss was not decreasing                                          |
| `text_std`         | 0.008                 | Embedding cloud was heavily squashed                             |


**Root cause: anisotropy of LLM embeddings.**

Transformer language models trained on next-token prediction produce highly anisotropic token embeddings. All word vectors cluster inside a narrow cone — the "common direction of language":

```
         ___
        /   \     ← all word vectors live in here
       / ••• \
      / ••••• \
     /•••••••••\
    /___________\
          |
          | "common direction" (anisotropy bias)
          ↓
```

When you mean-pool any sentence, the average lands near the cone's centre — regardless of content:

```
mean("the cat sat") ≈ cone centre
mean("the dog ran") ≈ cone centre
mean("hello world") ≈ cone centre
```

The model was asked "which audio matches which text?" but every text vector was already 40% identical to every other one. The actual content differences were drowned in shared "language-ness."

**Why the cone forms:** during LM pretraining, the model is rewarded for predicting frequent tokens,
which biases embedding geometry — high-frequency directions dominate and rare directions stay
underused.

Contrastive learning cares about the **gap** between matching and non-matching pairs, not absolute
similarity. A high baseline similarity (the cone) caps that gap; centering resets the baseline so the
gap can grow as the model learns.

This is a well-documented property of transformer LM embeddings:

- **Ethayarajh (2019)** — first showed BERT/GPT embeddings are highly anisotropic
- **Gao et al. (2021)** — SimCSE: showed mean-pooled LM embeddings cluster too tightly for similarity without correction
- **Su et al. (2021)** — proposed full whitening (centering + decorrelation) as a stronger variant

### 9.2 The Fix: Batch Centering

**Four lines of code. Zero new parameters.**

Before computing contrastive loss, subtract the batch mean from both the audio and text pooled vectors:

```python
# File: training/utils/losses.py
# Function: contrastive_infonce_loss

# Before (broken):
a = F.normalize(audio_tokens.float().mean(dim=1), dim=-1)
t = F.normalize(text_embeddings.float().mean(dim=1), dim=-1)

# After (fixed):
a_pooled = audio_tokens.float().mean(dim=1)
t_pooled = text_embeddings.float().mean(dim=1)
a_pooled = a_pooled - a_pooled.mean(dim=0, keepdim=True)   # remove common direction
t_pooled = t_pooled - t_pooled.mean(dim=0, keepdim=True)
a = F.normalize(a_pooled, dim=-1)
t = F.normalize(t_pooled, dim=-1)
```

**Effect**: Moves every point so the cloud is centered at the origin. Random pairs now average to near-zero similarity; the meaningful pair-by-pair differences become visible.


| Metric             | Before centering | After centering         |
| ------------------ | ---------------- | ----------------------- |
| `text_std`         | 0.008            | 0.015                   |
| `audio_std`        | 0.012            | 0.0156                  |
| `neg_sim`          | 0.40             | ~0                      |
| `pos_sim`          | flat at 0.65     | climbing 0 → 0.45       |
| `pos_minus_neg`    | flat at 0.25     | climbing 0 → 0.5        |
| `train/align` loss | stuck at 1.38    | 2.05 → 1.4 and dropping |


**The same centering fix is applied in retrieval eval (`eval_stage1.py` / `eval_stage2.py --metric retrieval-cosine`)** to ensure training and evaluation metrics are consistent.

**What to watch in future contrastive runs**: If `neg_sim` sits well above zero, suspect anisotropy.
Check `pooled.std(dim=0).mean()` — it should be near `1/sqrt(D)` (≈ 0.0156 for D=4096). Log
`pos_sim` and `neg_sim` separately, not just the loss — the loss can decrease for the wrong reason
(e.g. stability collapse) while the gap does not improve. If centering is insufficient at scale, next
steps are: projection heads (SimCLR-style MLP), full whitening, or learned attention pooling.

## 10. Stage 2: Content Preservation (ASR Distillation)

**Status:** Training scripts and checkpoints available (`checkpoints/adapter_stage2.pt`). See §5.3 for the full pipeline.

**Goal:** Make adapter tokens sufficient for accurate ASR when fed into the frozen LLM. Adds ASR distillation on top of the Stage 1 alignment objective.

**Loss**:

```
L = L_asr + λ_align · L_align + λ_stability · L_stability + λ_sparse · L_sparse
```

**New in Stage 2**:

- `L_asr`: Frozen causal LM loss. Audio tokens are prepended to the LLM context; the LLM predicts the correct transcript in teacher-forced mode. Gradients flow back through the adapter only (LLM stays frozen).
- Optional `AdaptiveRateController` introduced for token efficiency (`L_sparse`).
- `TurnEndCommitGate` trained jointly (gate loss contribution is small; primary focus is content fidelity).

## 11. Stage 3: Task Distillation (Streaming)

**Status:** Trainer implemented (`training/adapter_task_trainer.py`); depends on stage 2 checkpoint. See §5.4 for the full pipeline.

**Goal:** Full streaming training with early commitment. Distil knowledge from a frozen text-only teacher LLM into the streaming audio system.

**Loss**:

```
L = L_task + λ_asr · L_asr + λ_stability · L_stability + λ_rate · L_rate + λ_gate · L_gate
```

**New in Stage 3**:

- `L_task`: KL divergence between the audio-conditioned LLM distribution and the teacher text-only LLM distribution.
- Prefix consistency: P(y | Z_{1:t}) ≈ P(y | Z_{1:t+k}) — earlier predictions should not change as more audio arrives.
- Revision penalty: Penalises the LLM for retracting or contradicting earlier generated tokens.
- `TurnEndCommitGate` is trained to full effect with the task loss backpropagating through the commit decision.

## 12. Evaluation Strategy

We evaluate on **three axes** to separately measure what the adapter has learned:

### Axis 1: Retrieval (audio → text alignment)

Given N audio clips and their N transcripts, rank all transcripts for each audio by similarity. The correct transcript should rank first.

**Metrics**: R@1, R@3, R@5, R@10, MRR, Median Rank

**Scoring modes**:


| Mode   | Description                                                                 | When to use                              |
| ------ | --------------------------------------------------------------------------- | ---------------------------------------- |
| Cosine | Dot product of pooled + centered + L2-normalized embeddings                 | Our system (Stage 1+)                    |
| NLL    | Negative log-likelihood of transcript under LLM conditioned on audio prefix | After Stage 2; also for SALMONN baseline |


Cosine retrieval is O(N) per query after embedding; NLL retrieval is O(N²) over the query set — use smaller N for NLL.

**Implementation**:

- Our system: `evaluation/eval_stage1.py` / `evaluation/eval_stage2.py` with `--metric retrieval-cosine` or `--metric retrieval-nll`
- SALMONN baseline: `salmonn/SALMONN-7B/eval_librispeech_full_metrics.py`

### Axis 2: ASR quality (WER / BLEU)

Generate transcript end-to-end (audio → adapter tokens → LLM → text). Compute word error rate and BLEU-4 against the LibriSpeech reference.

**Metrics**: avg WER, corpus BLEU-4

### Axis 3: Streaming / latency

- First-token latency (time from audio onset to first generated token)
- Stability: revision rate, prefix-consistency score

## 13. Baseline: SALMONN-7B

We use [SALMONN-7B](https://github.com/bytedance/SALMONN) as our primary baseline for LibriSpeech test-clean evaluation.

**Evaluation script**: `salmonn/SALMONN-7B/eval_librispeech_full_metrics.py`

SALMONN architecture:

- **Audio encoder 1**: Whisper **Large-v2** (frozen) — speech / ASR-style features
- **Audio encoder 2**: **BEATs** (frozen) — general audio event / non-speech features
- **Bridge**: Window-level Q-Former (BLIP-2 style) that receives the **concatenation** of both encoder outputs — batch processing, not streaming
- **LLM**: Vicuna **7B** or **13B** (frozen)

SALMONN's Q-Former input dimension is `Whisper_d_model + BEATs_encoder_embed_dim` (see `model.py::init_speech_Qformer`). This dual-encoder design is intended to give the model complementary representations: Whisper captures fine-grained phonetic/prosodic structure, while BEATs captures event-level acoustic features useful for non-speech audio tasks.

### 13.1 What SALMONN Does Well

SALMONN was designed for audio question-answering, not streaming retrieval. Given an audio clip and a prompt, it produces semantically correct, contextually appropriate text responses. Its LLM is a full Vicuna model with strong language understanding, and its Q-Former effectively bridges Whisper + BEATs and Vicuna for content-level tasks.

The dual-encoder input (Whisper + BEATs) is a key strength for general audio tasks — BEATs was pretrained on AudioSet with a masked audio modelling objective and provides rich non-speech representations that Whisper alone misses (e.g. environmental sounds, music, speaker emotion).

### 13.2 Why Cosine Retrieval is Misleading for SALMONN

SALMONN uses two separate representation spaces:

- **Audio side**: speech_llama_proj(QFormer(audio)) → Vicuna hidden space
- **Text side**: Vicuna embed_tokens(transcript) → the same Vicuna hidden space

Despite sharing the LLM embedding dimension, these two representations are **not aligned for cosine similarity**. The audio Q-Former projection was trained with a task-completion objective, not a contrastive embedding objective. The resulting audio vectors and text token embedding vectors do not form a joint metric space — cosine similarity between them is not meaningful.

**Expected behaviour on cosine retrieval**: Low R@1 even if SALMONN correctly understands the audio content. This is not a failure of the model — it is a measurement artefact from using the wrong retrieval mode.

### 13.3 Evaluation Script

`eval_librispeech_full_metrics.py` runs both retrieval modes and full ASR in one pass:

```bash
CUDA_VISIBLE_DEVICES=0,1 python eval_librispeech_full_metrics.py \
  --extract_dir /path/to/librispeech_test_clean \
  --ckpt_path ./checkpoints/salmonn_7b_v0.pth \
  --whisper_path openai/whisper-large-v2 \
  --beats_path ./checkpoints/beats.pt \
  --vicuna_path lmsys/vicuna-7b-v1.5 \
  --vicuna_device_map auto \
  --output_dir ./outputs/librispeech_baseline \
  --num_samples 500 \
  --retrieval_mode nll \
  --retrieval_nll_batch_size 8 \
  --partial_save_every 10
```

**Outputs** (all JSON, no `.pt` files):


| File                                  | Contents                                                      |
| ------------------------------------- | ------------------------------------------------------------- |
| `retrieval_metrics_{cosine,nll}.json` | R@1/3/5/10, MRR, median rank (standard and centered variants) |
| `retrieval_pairs_{cosine,nll}.json`   | Per-utterance top-k ranked transcripts with scores            |
| `asr_metrics.json`                    | avg WER, BLEU-4                                               |
| `asr_predictions.json`                | Per-utterance reference, prediction, WER                      |
| `combined_metrics.json`               | All metrics in one file                                       |
| `partial_retrieval/*.json`            | Rolling partial checkpoints (metrics + sim rows as lists)     |


**For SALMONN, use `--retrieval_mode nll`** as the primary retrieval metric. NLL measures how well the model's audio representation supports predicting the correct transcript under the LLM — this reflects the model's actual audio understanding regardless of whether audio and text embeddings are aligned in cosine space.

---

## 14. Our System vs. SALMONN Baseline

The key architectural and methodological differences:

### Audio Encoder

| | SALMONN | Ours |
| |---------|------|
| Encoder 1 | Whisper **Large-v2** | Whisper **small** |
| Encoder 2 | **BEATs** (audio event encoder) | **None — removed** |
| Q-Former input | Whisper_dim + BEATs_dim (concatenated) | Whisper_dim only |
| Encoder parameters | ~1.5B (Whisper-L) + ~90M (BEATs) | ~244M (Whisper-small) |
| Whisper output dim | 1280 | 768 |
| Latency | Higher (two encoders, larger Whisper) | Lower (one small encoder) |
| Rationale | Rich multi-modal audio features, offline QA | Low latency, speech-focused, streaming |

**Removal of BEATs**: Our system deliberately drops the BEATs encoder. SALMONN feeds the concatenation of Whisper and BEATs outputs into its Q-Former (`input_dim = Whisper_d_model + BEATs_encoder_embed_dim`). BEATs was pretrained on AudioSet with a masked audio modelling objective and provides rich non-speech representations (environmental sounds, music, speaker emotion) that Whisper's speech-optimised encoder misses. For SALMONN's broad audio-QA scope this dual-encoder design is beneficial. For our speech-only evaluation on LibriSpeech, BEATs does not contribute content that Whisper does not already capture — including it would add ~90M frozen parameters, a second encoder pass per window, and a hard dependency on the `beats.pt` checkpoint, with no expected gain on speech tasks. Removing BEATs simplifies the pipeline, reduces inference cost, and is consistent with the low-latency goal.

The trade-off is that our system is currently speech-specialised. If the scope expands to general audio tasks (sound events, music, etc.), a second encoder analogous to BEATs could be reintroduced as a second input branch to the adapter Q-Former.

Using Whisper-small rather than Large-v2 is a further deliberate latency choice. If the adapter + smaller single encoder achieves competitive downstream performance, this provides strong justification for the overall low-latency design.

### Language Model

| | SALMONN | Ours |
| |---------|------|
| Model | Vicuna **7B / 13B** | **Qwen3-8B** (current) |
| Context | Full utterance | Incremental (KV-cache streaming) |
| Design goal | Rich semantic QA | Low-latency streaming response |
| Flexibility | Fixed | LLM-agnostic by design |

We are evaluating smaller LLMs (e.g. Phi, Qwen3-4B) for further latency reduction. A smaller LLM that performs comparably to Vicuna-7B on audio tasks would provide additional justification for the streaming / low-latency design.

### Attention Mechanism

| | SALMONN | Ours |
| |---------|------|
| Type | Full bidirectional attention (batch Q-Former) | Q-Former + considering **monotonic attention** |
| Streaming | No — requires full audio upfront | Yes — window-by-window |
| Temporal constraint | None (attends over all frames) | Causal; future frames unavailable |

**Monotonic attention** (MoChA / monotonic chunkwise attention) is a candidate upgrade for Stage 2/3. It enforces a left-to-right alignment constraint between the input sequence (audio frames) and the query sequence — directly matching the causal structure of streaming audio processing.

### Training Objective

| | SALMONN | Ours |
| |---|---|
| Primary alignment | Task-completion (instruction following) | **Contrastive** (explicit audio–text metric space) |
| Retrieval | NLL only (not designed for cosine) | Both cosine (Stage 1+) and NLL (Stage 2+) |
| Streaming support | None | Yes, by design (stability loss, EMA, commit gate) |
| Latency objective | None | Early-commit gate (Stage 3) |

### Key Improvements Over SALMONN

1. **Contrastive learning**: Explicit audio–text alignment loss creates a proper joint metric space. Unlike SALMONN's task-completion objective, our Stage 1 model is specifically optimised to produce audio vectors that are close to their matching text vectors and far from non-matching ones. This enables meaningful cosine retrieval.
2. **Removal of BEATs encoder**: SALMONN uses two encoders (Whisper Large-v2 + BEATs) whose outputs are concatenated before the Q-Former. We remove BEATs entirely. For speech-only tasks on LibriSpeech, BEATs adds ~90M parameters and a second encoder pass without contributing features that Whisper does not already provide. The single-encoder design reduces model size, removes the `beats.pt` dependency, and lowers per-window inference cost — all directly supporting the low-latency goal.
3. **Streaming support with monotonic attention** (planned): SALMONN requires complete audio before processing. Our adapter processes each 0.8s window causally, accumulating tokens in the LLM's KV-cache without re-encoding prior windows.
4. **Low latency**: The entire system (Whisper-small, single encoder, 2-layer Q-Former, Qwen3-8B) is designed around the sub-second first-token latency target. Whisper-small, no BEATs pass, shallow adapter, and KV-cache streaming all contribute.
5. **Early-commit gate**: SALMONN always waits for the full audio. Our gate learns to trigger generation as soon as sufficient context is accumulated, trading off latency against accuracy in a learned, data-driven way.

## 15. Key Differences Summary


| Dimension          | SALMONN (baseline)       | Our System                          |
| ------------------ | ------------------------ | ----------------------------------- |
| Audio encoder 1    | Whisper Large-v2         | Whisper **small**                   |
| Audio encoder 2    | **BEATs** (audio events) | **None — removed**                  |
| Encoder dim        | 1280 + BEATs_dim         | **768**                             |
| LLM                | Vicuna 7B / 13B          | **Qwen3-8B** (swappable)            |
| Processing mode    | Full-utterance batch     | **Window streaming**                |
| Attention          | Full bidirectional       | Q-Former, considering **monotonic** |
| Alignment training | Task-completion only     | **Contrastive** (Stage 1)           |
| Streaming          | ✗                        | ✓                                   |
| Early-commit       | ✗                        | ✓                                   |
| Cosine retrieval   | Not meaningful           | ✓ (by design)                       |
| NLL retrieval      | ✓ (meaningful)           | ✓ (Stage 2+)                        |
| Latency objective  | ✗                        | ✓                                   |
| BEATs dependency   | Required (`beats.pt`)    | **Removed**                         |


## 16. Theoretical Foundation

**Information Bottleneck**: The adapter minimises `I(Audio; Tokens)` while maximising `I(Tokens; Task)` — compress aggressively but preserve task-relevant information.

**Compression ratio**: per window, 1500 encoder frames → m adapter tokens (m = 4 default, or fewer active slots with rate controller). Overlapping windows reuse audio context at 0.4 s stride.

**Prefix Consistency**: `P(y | Z_{1:t}) ≈ P(y | Z_{1:t+k})` — LLM predictions should not flip as new audio tokens arrive.

**Rate-Distortion**: Target R = m/t tokens/window while maintaining `I(Audio; Tokens) ≈ I(Audio; Text)`.

## 17. Related Work


| System                             | Relationship                                                                                   |
| ---------------------------------- | ---------------------------------------------------------------------------------------------- |
| **SALMONN** (Tang et al. 2023)     | Our primary baseline. Window-level Q-Former; we extend to streaming with stability objectives. |
| **BLIP-2 / Flamingo**              | Q-Former / Perceiver resampler concept; we adapt to causal streaming audio.                    |
| **MoChA** (Chiu & Raffel 2018)     | Monotonic chunkwise attention — candidate for Stage 2/3 attention replacement.                 |
| **Moshi / Mini-Omni / LLaMA-Omni** | Low-latency speech interaction systems; serve as latency benchmarks.                           |
| **SimCSE** (Gao et al. 2021)       | Contrastive sentence embeddings; motivates our centering fix for anisotropy.                   |
| **Ethayarajh (2019)**              | Characterised LM embedding anisotropy (cone collapse).                                         |


SALMONN NLL numbers serve as the baseline for NLL retrieval and ASR. SALMONN cosine numbers are expected to be low and are reported only for completeness — they do not reflect SALMONN's audio understanding capability.