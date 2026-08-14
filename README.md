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

2. Run the setup script (sources **`.env`** first, then pyenv 3.12, **`uv sync`**, **PyTorch CUDA compatibility check**, LibriSpeech download (`src/dataset/load_dataset.py`), Stage 1 checkpoint fetch, parallel prefetch of **Whisper small** + **Qwen3-8B**, then `uv run python training/adapter_asr_trainer.py`):

```bash
bash scripts/setup_remote_training.sh
```

Setup only (no training): set `SKIP_TRAINING=1` in `.env`. Skip model cache warmup with `SKIP_MODEL_PREFETCH=1`.

**GPUs:** with two visible GPUs (`CUDA_VISIBLE_DEVICES=0,1`), Stage 2 loads Whisper on GPU 0, then Qwen (`device_map="auto"` by default, or `sequential` when `LLM_MAX_MEMORY` is set so GPU 0 fills first and overflow spills to GPU 1). Set `LLM_MAX_MEMORY=0:14GiB,1:5GiB` to cap spill on a shared GPU. Set `DEVICE=cpu` to force CPU. See `training/utils/config.py` (`DeviceConfig`).

**GPU reservation:** training/eval entry points use exclusive file locks on visible GPUs (`training/utils/gpu_reservation.py`). While a run holds a lock, other processes that use `init_training_context()` or `init_eval_device()` **block** until it exits. `adapter_asr_only_trainer.py` enables this by default (`GPU_LOCK=true`). Disable with `GPU_LOCK=off`. Lock files live under `.gpu_locks/` (override with `GPU_LOCK_DIR`).

Requires **uv** and **wget** or **curl**. **pyenv** is optional — if missing, the script uses system Python 3.12+ automatically (or set `SKIP_PYENV=1`). If no `.env` exists, the script copies `.env.example` → `.env` on first run.

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

**Checkpoints** use `training.utils.checkpointing.save_checkpoint` (keys: `adapter_state_dict`, optional `gate_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `metrics`, `hyperparams`). Serialize tuning bundles with `dataclasses.asdict` into `hyperparams`. Resume via `load_adapter_state_dict` or `torch.load` as in the stage trainers. Stage 2 also writes **`checkpoints/adapter_stage2_step{N}.pt`** every **`SAVE_EVERY_STEPS`** (and updates `adapter_stage2.pt` for resume).

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

**Softmax InfoNCE + GradCache** (`training/adapter_contrastive_trainer_softmax.py`): InfoNCE uses a true macro-batch of in-batch negatives (`MACRO_BATCH_SIZE`, default **128**) while encoding only `BATCH_SIZE` utterances at a time (default **8**) so peak VRAM stays at the micro-batch. This is **not** ordinary gradient accumulation — the similarity matrix is `128×128`. Implementation: `training/utils/grad_cache.py`.

GradCache runs the adapter in **`eval()`** for both the cache and recompute passes (disables dropout so representation grads match the surrogate; no BatchNorm in the adapter), then restores `train()`. Softmax defaults when env is unset: `λ_stability=0.01`, `grad_clip_norm=2.0` (so Align is not dominated by stability + hard clip early). W&B / console log `diag/pos_minus_neg` and per-term adapter grad norms `train/grad_norm_align` vs `train/grad_norm_stab`.

Set `RANDOM_SEED` (default `42`) to reproduce runs; the seed is stored in checkpoint `hyperparams`.

**2-GPU training (recommended):**

```bash
cd audio-streaming-adapter
# Data-parallel: each rank encodes 64, all-gather → InfoNCE batch 128
RANDOM_SEED=42 BATCH_SIZE=8 MACRO_BATCH_SIZE=128 \
  uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  training/adapter_contrastive_trainer_softmax.py
```

**Single process (still uses both GPUs when visible):** Whisper/adapter on `cuda:0`, text embeddings on `cuda:1`.

```bash
cd audio-streaming-adapter
# Optional: BATCH_SIZE=8 MACRO_BATCH_SIZE=128 RANDOM_SEED=42
uv run python training/adapter_contrastive_trainer_softmax.py
```

**CLI (full training):**

```bash
cd audio-streaming-adapter
uv run python training/adapter_contrastive_trainer.py
```

**Walkthrough:** open `notebooks/training_stage1_contrastive.ipynb`. Earlier cells dissect components; the section **“Full training (script-equivalent)”** runs the same loop as the CLI (set `MAX_STEPS_DEBUG` to a small integer for a smoke test).

**Defaults (see `training/utils/config.py` + script top):** epochs 5, batch size 4, LR `1e-4`, SGD + linear warmup (`warmup_steps=100`), `λ_stability=0.1`, temperature `0.2`, `checkpoints/adapter_adapter.pt`.

#### Stage 1 Vicuna Optuna (Case 1)

Self-contained experiment at **`../experiments/stage1-optuna-vicuna/`** (workspace sibling to `audio-streaming-adapter/`). Trains a new **`StreamingAdapterAttention`** (Q-Former only, no rate controller) with frozen **Whisper-small** + **Vicuna-7B** embeddings on **LibriSpeech train-clean-100**, validates **dev-clean** and evaluates **test-clean** after each epoch, and sweeps adapter architecture with **Optuna** + **WandB**.

**Layout:**

```
experiments/stage1-optuna-vicuna/
  pyproject.toml                  # separate uv project (own .venv)
  deps/                           # vendored adapter, dataset, encoder (self-contained)
  stage1_optuna_case1.py          # CLI entry point (recommended)
  stage1_optuna_case1.ipynb
  streaming_adapter_attention.py
  vicuna_loader.py
  losses.py
  validation.py
  wandb_logging.py
  train.py
  checkpoints/trial_{n}/
  optuna_case1.db
  datasets/librispeech_data/LibriSpeech/   # optional local data copy
```

**Remote deployment:** copy the entire `stage1-optuna-vicuna/` directory to the server (includes `deps/` — no separate `audio-streaming-adapter/src/` required). Point LibriSpeech data via env or place under `datasets/` inside the experiment dir:

```bash
export LIBRISPEECH_BASE=/path/to/LibriSpeech   # contains train-clean-100/, dev-clean/, test-clean/
# or: datasets/librispeech_data/LibriSpeech/ inside the experiment directory
```

**Setup:**

```bash
cd stage1-optuna-vicuna   # on remote: e.g. /workspace/audio-streaming-adapter
uv sync
# optional: WANDB_API_KEY, HF_TOKEN, LIBRISPEECH_BASE in .env
```

`paths.py` loads `stage1-optuna-vicuna/.env` on import, so the CLI script, the notebook, and direct `train.py` use all pick up the same credentials. Real environment variables take precedence over `.env`, and `.env` is gitignored — keep secrets out of commits.

The experiment `pyproject.toml` has its own `.venv`. All Python dependencies for adapter/dataset/encoder are vendored under `deps/`.

**Run (recommended — Python script):**

```bash
cd experiments/stage1-optuna-vicuna

# Full Optuna study (20 trials)
uv run python stage1_optuna_case1.py

# Smoke test (single fixed architecture, 3 epochs)
uv run python stage1_optuna_case1.py --smoke-test

# Custom trial count
uv run python stage1_optuna_case1.py --n-trials 5
```

**Run (Jupyter notebook):**

```bash
cd experiments/stage1-optuna-vicuna
uv run python -m ipykernel install --user --name stage1-optuna-vicuna --display-name "stage1-optuna-vicuna"
uv run jupyter lab stage1_optuna_case1.ipynb
```

In Jupyter, select kernel **stage1-optuna-vicuna** (uses `experiments/stage1-optuna-vicuna/.venv`).

**Fixed hyperparameters:** `lr=1e-4`, `num_queries=2`, `epochs=3`, Softmax InfoNCE + learnable temperature.

**Reproducibility:** `--seed` (default `42`) sets the base seed. Trial *N* runs with **`seed + N`**, so trials differ from one another while each stays a pure function of `(base_seed, trial_number)` — any single trial can be replayed in isolation. The seed is applied before adapter init and covers Python, NumPy, and Torch RNGs, the shuffle generator, and DataLoader workers; the same seed also drives the `TPESampler`, so the search sequence itself repeats. Each trial records its seed in the WandB config (`trial_seed`), a `seed=<n>` tag, the run summary, and every saved checkpoint.

```bash
# reproduce trial 7 of a seed-42 study on its own
uv run python stage1_optuna_case1.py --smoke-test --seed 49   # 42 + 7
```

Add `--deterministic` to force deterministic cuDNN kernels for bit-exact reruns — slower, and only needed when comparing weights exactly rather than reproducing the setup.

**Optuna Case 1 search space:**

| Param | Values |
|---|---|
| `num_layers` | `{2, 4, 6, 8, 10}` |
| `cross_layer_in_between` | `{1, 2, 4, 8}` |
| `num_heads` | `{4, 8, 12}` |
| `d_ffn` | `{1024, 2048, 4096}` |

Invalid combos (`cross_layer_in_between >= num_layers`) are pruned. Objective: maximize final-epoch **val/recall_at_1**.

**Data:** uses `audio-streaming-adapter/datasets/librispeech_data/LibriSpeech/{train-clean-100,dev-clean,test-clean}`.

**WandB logging** (`wandb_logging.py`): logs to project `--wandb-project` (default `stage1-optuna-vicuna`, separate from the main `audio-streaming-adapter` project) — one run per Optuna trial, named `L{num_layers}_C{cross_layer}_H{num_heads}_F{d_ffn}_t{trial}` and grouped under `--wandb-group` (default `stage1-vicuna-optuna-case1`), tagged with each sampled hyperparameter. Logged per run:

| Scope | Keys |
|---|---|
| Config | full `ExperimentConfig` + sampled hyperparameters + `trial_number` + `trial_seed` |
| Train (every 10 steps) | `train/{loss,align,stability,lr,temperature}`, `diag/{pos_sim,neg_sim,pos_minus_neg}` |
| Epoch end | `val/*` and `test/*` (loss, align, stability, similarity diagnostics, `recall_at_{1,5,10}`) |
| Summary | `best_val_recall_at_1`, `final_val_recall_at_1`, `checkpoint_dir` |

Credentials resolve in this order: `WANDB_MODE=offline` (no credentials needed) → `WANDB_API_KEY` (env or experiment `.env`) → `wandb login` / `~/.netrc`. If none are found, the run continues with console-only output and prints a warning at trial start. Pass `--no-wandb` to disable logging intentionally.

### Stage 2: ASR distillation

**What it does:** Keeps speech content while training compression / optional rate controller and gate losses (`training/adapter_asr_trainer.py`).

**CLI (two GPUs recommended for Qwen3-8B):** Whisper, adapter, and gate use GPU 0 (`DEVICE=cuda` by default). With two or more visible GPUs, the frozen Qwen model is loaded via HuggingFace `device_map="auto"` (balanced shard). Set `LLM_MAX_MEMORY` (e.g. `0:14GiB,1:5GiB`) to use `device_map="sequential"`: fill GPU 0 first, spill overflow to GPU 1 with a per-GPU cap — useful when GPU 1 is shared with another process.

```bash
cd audio-streaming-adapter
# single GPU
uv run training/adapter_asr_trainer.py

# multi-GPU (2+ visible devices)
torchrun --standalone --nnodes=1 --nproc_per_node=1 training/adapter_asr_trainer.py

# shared GPU 1 (5GiB cap for training spill)
LLM_MAX_MEMORY=0:14GiB,1:5GiB uv run training/adapter_asr_trainer.py
```

On a single GPU, set `CUDA_VISIBLE_DEVICES=0` or leave one device visible. If training OOMs, enable `ENABLE_LLM_GRADIENT_CHECKPOINTING=true`, lower `MAX_WINDOWS_PER_UTT`, or reduce `BATCH_SIZE`.

**Upload checkpoints to Hugging Face Hub** (after each epoch, requires `HF_TOKEN` with write access):

```bash
HF_CHECKPOINT_REPO=your-org/audio-streaming-adapter-checkpoints
HF_UPLOAD_CHECKPOINTS=true
```

Each stage uploads only its own epoch files (`adapter_stage1_epoch{N}.pt`, `adapter_stage2_epoch{N}.pt`, `adapter_stage3_epoch{N}.pt`, …). Uploads are **additive** — existing files from other stages stay in the repo. Local resume checkpoints (`adapter_stage{N}.pt`) are not uploaded.

**Walkthrough:** `notebooks/training_stage2_asr.ipynb` — follow cells top-to-bottom; align hyperparameters with `Stage2Config`, `DeviceConfig`, `OptimConfig`, and constants at the top of `adapter_asr_trainer.py`.

**Vicuna ASR + align** (`training/adapter_asr_align_vicuna_trainer.py`): same Stage 2 simplified loss (ASR + InfoNCE align + stability) as `adapter_asr_align_trainer.py`, but the frozen LM is **Vicuna-7B** (`lmsys/vicuna-7b-v1.5`, override with `VICUNA_MODEL_ID`). Weights load from the Hugging Face cache when present. Micro-batch is `BATCH_SIZE` (default **16**); effective optimizer batch is `MACRO_BATCH_SIZE` (default **128**, 8 accumulation steps). Warm-start adapter weights from `STAGE1_CHECKPOINT`; resume Stage 2 from `RESUME_CHECKPOINT` when set. Writes `checkpoints/adapter_asr_align_vicuna.pt` (override with `CHECKPOINT_BASENAME`).

```bash
cd audio-streaming-adapter
uv run training/adapter_asr_align_vicuna_trainer.py
```

### Evaluation (LibriSpeech test-clean)

Two entry scripts — pick the stage, then pick the metric with `--metric`:

| Script | Stage | Metrics (`--metric`) |
|--------|-------|----------------------|
| `evaluation/eval_stage1.py` | 1 (contrastive) | `retrieval-cosine`, `retrieval-nll`, `asr` |
| `evaluation/eval_stage2.py` | 2 (ASR distillation) | `retrieval-cosine`, `retrieval-nll`, `asr` |

Shared flags: `--checkpoint`, `--dataset-root`, `--num-samples` (integer or `all`), `--run-name`, `--output-dir`. Outputs go under `outputs/experiments/{RUN_NAME}/`.

**ASR** (`--metric asr`): avg WER + corpus BLEU-4. Stage 1 uses `WhisperAdapterLLMPipeline`; stage 2 uses `WhisperAdapterLLMCommitGatePipeline` (rate controller + gate). Default prefix is train-style **`[audio | im_end/BOS] → generate`** (`--append-im-end`). Use `--no-append-im-end`, `--compare-im-end`, or `--prompt-asr` (Stage 3 path). Stage 2: optional `--early-commit-truncation`.

**Retrieval cosine** (`--metric retrieval-cosine`): R@1, R@5, R@10 via centered cosine similarity → `retrieval_cosine.json`.

**Retrieval NLL** (`--metric retrieval-nll`): R@1, R@5, R@10 via frozen Qwen NLL (slow; `--resume` / `--rank-only` supported) → `retrieval_nll.json`.

```bash
cd audio-streaming-adapter

# Stage 1 retrieval (quick)
uv run evaluation/eval_stage1.py --metric retrieval-cosine --num-samples 100

# Stage 2 ASR on 50 utterances
uv run evaluation/eval_stage2.py --metric asr --num-samples 50

# Stage 2 full test-clean NLL (resumable)
uv run evaluation/eval_stage2.py --metric retrieval-nll --num-samples all

# Full experiment suite (all metrics × all checkpoints)
bash scripts/run_experiment_suite.sh

# Stage 2 im_end ablation
bash scripts/run_im_end_ablation.sh
```

### Experiment suite (eval + ablation)

Run the full pipeline (baseline evals → gate training → ASR ablations → ablation evals):

```bash
cd audio-streaming-adapter
bash scripts/run_experiment_suite.sh
```

**GPU layout:** `CUDA_VISIBLE_DEVICES=0,1`, `LLM_MAX_MEMORY=0:14GiB,1:5GiB` (GPU 0 full, GPU 1 capped at 5GiB for Qwen spill).

| Step | `RUN_NAME` | Checkpoint | Eval `--stage` |
|------|------------|------------|------------------|
| 1 | `stage1_baseline` | `checkpoints/adapter_stage1.pt` | 1 |
| 2a | `stage2_epoch1` | `checkpoints/adapter_stage2_epoch1.pt` | 2 |
| 2b | `stage2_last` | `checkpoints/adapter_stage2.pt` | 2 |
| 3 | `gate_smart_turn` | `checkpoints/gate_smart_turn.pt` | (train only) |
| 4a | `asr_only_1ep` | `checkpoints/adapter_asr_only_1ep.pt` | 1 |
| 4b | `asr_plus_gate_1ep` | `checkpoints/adapter_asr_plus_gate_1ep.pt` | 2 |

**Outputs per run:** `outputs/experiments/{RUN_NAME}/asr_metrics.json`, `retrieval_cosine.json`, `retrieval_nll.json`

**Logs:** `logs/experiments/{TIMESTAMP}/` plus `summary.json`

**Gate training** (standalone, Smart Turn labels):

```bash
GATE_LABEL_SOURCE=smart_turn EPOCHS=5 CHECKPOINT_BASENAME=gate_smart_turn \
  ADAPTER_CHECKPOINT=checkpoints/adapter_stage1.pt \
  uv run training/gate_training.py
```

**ASR ablation env vars** (used by `adapter_asr_trainer.py`):

- `LOAD_STAGE1_CHECKPOINT=false` — random-init adapter
- `TRAIN_GATE=false` — skip gate in training loop
- `GATE_CHECKPOINT=checkpoints/gate_smart_turn.pt` — load frozen gate for checkpoint save / Stage 2 eval
- `CHECKPOINT_BASENAME=adapter_asr_only_1ep` — custom checkpoint filename
- Set `LAMBDA_ALIGN/STABILITY/RATE/GATE=0` for ASR-only loss

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

**Step-by-step logging:** pass `verbose=True` (or a custom `PipelineStepLogger(enabled=True)`) when constructing `WhisperAdapterLLMPipeline` / `WhisperAdapterLLMCommitGatePipeline`. Each `generate()` call logs encode → window selection → adapter → gate (stage 2) → LLM decode with tensor shapes and timings. The result dict includes `pipeline_trace` when logging is enabled.

**Stage-2 smoke test (one dataset sample, all checkpoints):** runs the full commit-gate pipeline on one LibriSpeech utterance for every `checkpoints/adapter_stage2*.pt`:

```bash
cd audio-streaming-adapter
uv run python src/adapter_llm_pipeline.py \
  --checkpoints-dir checkpoints \
  --dataset-root datasets/librispeech_data/LibriSpeech/test-clean \
  --sample-index 0 \
  --output-json outputs/experiments/stage2_smoke_test.json
```

Defaults to **`--device cpu`**. For GPU: `--device cuda` or `--device cuda:0`. Use `--n-windows 4` for a faster smoke test; default `--n-windows -1` uses all adapter windows (slow on long utterances). Each checkpoint reports **WER** (decode vs. reference) and **NLL** (teacher-forcing loss on the reference transcript, same path as Stage-2 training).

### Streaming KV-cache inference (Qwen3-8B)

`WhisperAdapterLLMCommitGatePipeline.generate_streaming()` and `WhisperAdapterStreamingSession` (`src/adapter_llm_streaming.py`) implement the live path documented in `docs/AUDIO_STREAM.md` §4.1 and §7:

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
- Reduce `BATCH_SIZE` (micro-batch per forward pass) in `.env`
- Stage 2 / ASR trainers: set `MACRO_BATCH_SIZE` for gradient accumulation
  (`MACRO_BATCH_SIZE / BATCH_SIZE` micro-batches per optimizer step)
- Stage 1 softmax InfoNCE: `MACRO_BATCH_SIZE` is the **InfoNCE** batch (GradCache);
  keep `BATCH_SIZE` small for VRAM (e.g. `BATCH_SIZE=8`, `MACRO_BATCH_SIZE=128`).
  `MACRO_BATCH_SIZE` must be a multiple of `BATCH_SIZE`
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
- [Audio Streaming Adapter — Research & Implementation](docs/AUDIO_STREAM.md)

## Contact

For questions and issues, please open a GitHub issue or contact the maintainers.