## Training stages (implementation-aligned)

This project trains only **Component 2** (`StreamingAdapter`) and optionally **Component 3** (`EarlyCommitGate`). Whisper + LLM stay frozen.

### Stage 1 — Contrastive audio–text alignment

- **Goal**: align adapter-produced audio tokens to **frozen LLM text embeddings** (LLM embedding layer only; no text transformer inside the adapter).
- **Inputs**: LibriSpeech `(audio, transcription)`.
- **Core flow**:
  - `waveform → WhisperWindowFeatureExtractor` (0.8s / 0.4s **raw waveform** segments, each segment → Whisper encoder) → list of `(1, T, D)` → `StreamingAdapter.forward_window` per segment → tokens concatenated per utterance
  - `transcription → tokenizer → embedding layer → text_embeddings`
- **Loss**: \(L = L_{align} + \lambda_{stability} \cdot L_{stability}\)
  - `L_align`: InfoNCE-style contrastive loss (pooled audio tokens vs pooled text embeddings)
  - `L_stability`: from `StabilityBuffer` output across consecutive windows
- **Checkpoint**: `checkpoints/adapter_adapter.pt`

### Stage 2 — ASR distillation (content preservation)

- **Goal**: make adapter tokens sufficient for accurate ASR when fed into the frozen LLM.
- **Adds**:
  - optional `AdaptiveRateController` (inside `StreamingAdapter`) → `L_sparse` + `L_rate`
  - `EarlyCommitGate` is trained jointly (gate loss is small in stage 2, larger in stage 3)
- **Loss** (script structure): \(L = L_{asr} + \lambda_{align} L_{align} + \lambda_{stability} L_{stability} + \lambda_{sparse} L_{sparse} + \lambda_{rate} L_{rate} + \epsilon L_{gate}\)
  - `L_asr`: frozen causal LM loss using `inputs_embeds` (audio tokens + BOS + teacher-forced text)
- **Checkpoint**: `checkpoints/adapter_adapter.pt` (also stores `gate_state_dict`)

### Stage 3 — Task distillation (streaming)

- **Goal**: full streaming training with early commitment; compare student outputs to a frozen teacher (distillation).
- **Adds**: KL distillation + prefix consistency + revision-style objectives (implemented in the stage 3 trainer loop).
- **Checkpoint**: `checkpoints/adapter_adapter.pt` (also stores `gate_state_dict`)

### Notebooks

Step-by-step notebooks live in `notebooks/`:
- `training_stage1_contrastive.ipynb`, `training_stage2_asr.ipynb`, `training_stage3_task.ipynb`
- Stage 1’s final cell matches `training/adapter_contrastive_trainer.py`; other stages follow the same pattern vs their CLI scripts.