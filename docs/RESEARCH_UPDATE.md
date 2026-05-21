# Streaming Adapter

---

## Problem

Audio encoders (e.g. Whisper) produce **dense frame sequences** (tens of frames/sec). Feeding them directly into small LLMs causes:

| Issue | Impact |
|-------|--------|
| Context explosion | 2 min audio → 3 000+ frames; exceeds small LLM context |
| No streaming | Must wait for full audio before generation |
| High latency | Unsuitable for real-time interaction |

---

## Solution

A **trainable streaming adapter** that compresses continuous audio into a low-bitrate, append-only token stream (1–3 tokens/sec) consumed incrementally by a frozen small LLM.

- **25× compression**: ~3 000 frames → ~240 tokens over 2 minutes
- **Window-by-window**: 0.8 s window · 0.4 s stride (50 % overlap)
- **Only the adapter trains** — Whisper encoder and LLM stay frozen

---

## Architecture

![System Architecture](assets/architecture.png)

**Trainable components**: Component 2 (Streaming Adapter) + Component 3 (Early-Commit Gate, optional).
**Frozen**: Whisper-small encoder, Qwen3-8B LLM.

---

## Comparison with SALMONN Baseline

| Dimension | SALMONN-7B (baseline) | Our System |
|-----------|----------------------|------------|
| Audio encoder 1 | Whisper **Large-v2** | Whisper **small** |
| Audio encoder 2 | **BEATs** (audio events) | **Removed** |
| Q-Former input | Whisper_dim + BEATs_dim | Whisper_dim only |
| LLM | Vicuna **7B / 13B** | **Qwen3-8B** (swappable) |
| Processing | Full-utterance batch | **Window streaming** |
| Attention | Full bidirectional | Q-Former → **monotonic** (planned) |
| Alignment objective | Task-completion only | **Contrastive** (Stage 1) |
| Streaming | ✗ | ✓ |
| Early-commit gate | ✗ | ✓ |
| Cosine retrieval | Not meaningful | ✓ meaningful (by design) |
| NLL retrieval | ✓ | ✓ (Stage 2+) |
| BEATs checkpoint | Required | **Not needed** |

**Why NLL for SALMONN**: SALMONN's audio and text embeddings are not aligned for cosine similarity (task-completion objective, not contrastive). Even if SALMONN perfectly understands the audio, cosine R@1 will be low — NLL is the correct retrieval mode for this baseline.

---

## Training Stages

![Training Flow](assets/training_flow.png)

### Stage 1 — Contrastive Audio–Text Alignment ✅

- **Goal**: Align adapter audio tokens to frozen LLM text embeddings via InfoNCE.
- **Input**: LibriSpeech `(audio, transcript)` pairs.
- **Key problem solved**: Anisotropy / cone collapse in LLM embeddings.
  - LM embeddings cluster in a narrow cone → all mean-pooled sentences look ~40% similar before training.
  - **Fix**: subtract batch mean before L2-normalise (4 lines, zero new parameters).
  - Result: `neg_sim` 0.40 → ~0; `train/align` loss started decreasing.
- **Checkpoint**: `checkpoints/adapter_adapter.pt`

### Stage 2 — ASR Distillation ⏳

- **Goal**: Make adapter tokens sufficient for accurate ASR inside the frozen LLM.
- **New**: Frozen causal LM loss with teacher-forced transcripts (`L_asr`); optional `AdaptiveRateController` (`L_sparse`).
- **Not started.**

### Stage 3 — Task Distillation ⏳

- **Goal**: Full streaming training with early commitment. KL distillation from a text-only teacher LLM.
- **New**: prefix consistency, revision penalty, full `EarlyCommitGate` training.
- **Not started.**

---

## Current Status

| Work item | Status |
|-----------|--------|
| Stage 1 training | ✅ Complete |
| Stage 1 evaluation — cosine retrieval on LibriSpeech test-clean | 🔄 In progress |
| SALMONN baseline eval — NLL retrieval + WER + BLEU on LibriSpeech test-clean | 🔄 In progress |
| Stage 2 training | ⏳ Not started |
| Stage 3 training | ⏳ Not started |

---

## Doing Now

1. **SALMONN-7B baseline evaluation** (`salmonn/SALMONN-7B/eval_librispeech_full_metrics.py`):
   - NLL retrieval (R@1/5/10, MRR) on LibriSpeech test-clean
   - ASR generation: avg WER, corpus BLEU-4
   - Results will be used as the baseline to compare against our system.

2. **Stage 1 retrieval evaluation** (`evaluation/eval_retrieval.py`):
   - Cosine retrieval on LibriSpeech test-clean using the trained Stage 1 adapter.
   - Batch centering applied at eval time (consistent with training).

---

## To Do

### Immediate
- [ ] Finish SALMONN NLL + ASR baseline numbers → record in results table
- [ ] Run Stage 1 cosine retrieval eval → compare R@1/5/10, MRR against SALMONN NLL baseline
- [ ] Run Stage 1 NLL retrieval eval → establish Stage 1 NLL numbers

### Short-term
- [ ] Start Stage 2 training (ASR distillation)
- [ ] Evaluate Stage 2: NLL retrieval + WER/BLEU — compare against SALMONN and Stage 1
- [ ] Decide on monotonic attention integration for streaming (Stage 2 or Stage 3)

### Medium-term
- [ ] Start Stage 3 training (task distillation + full gate)
- [ ] Evaluate Stage 3: latency (first-token), stability (revision rate, prefix-consistency), task accuracy
- [ ] Experiment with smaller LLMs (Qwen3-4B, Phi) for further latency reduction
- [ ] Evaluate with / without BEATs on non-speech audio (if scope expands beyond LibriSpeech)

---

## Key Files

| File | Purpose |
|------|---------|
| `training/adapter_contrastive_trainer.py` | Stage 1 training script |
| `training/utils/losses.py::contrastive_infonce_loss` | InfoNCE with batch centering fix |
| `src/adapter/streaming_adapter.py` | Component 2: Q-Former + stability buffer |
| `src/adapter/early_commit_gate.py` | Component 3: commit gate |
| `evaluation/eval_retrieval.py` | Cosine retrieval eval (our system) |
| `evaluation/eval_retrieval_nll.py` | NLL retrieval eval (our system) |
| `salmonn/SALMONN-7B/eval_librispeech_full_metrics.py` | Full baseline eval (cosine + NLL + WER + BLEU) |
| `checkpoints/adapter_adapter.pt` | Stage 1 adapter checkpoint |
| `docs/AUDIO_STREAM.md` | Full documentation |
