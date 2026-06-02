#!/usr/bin/env bash
# Bootstrap a remote GPU server for Stage 2 ASR training.
#
# Usage:
#   cp .env.example .env   # edit values
#   bash scripts/setup_remote_training.sh
#
# Order: source .env → pyenv/uv → PyTorch compat → LibriSpeech → checkpoint → models (parallel) → train
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

log() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

# --- Source .env first (all later steps see these variables) --------------------

source_env() {
  local env_file="${ROOT}/.env"
  if [[ -f "$env_file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
    log "Sourced ${env_file}"
  else
    log "No ${env_file} — copy .env.example to .env and fill required values"
  fi
}

setup_pyenv() {
  if [[ "${SKIP_PYENV:-0}" == "1" ]]; then
    log "Skipping pyenv (SKIP_PYENV=1)"
    return
  fi
  if ! command -v pyenv >/dev/null 2>&1; then
    die "pyenv not found. Install pyenv or set SKIP_PYENV=1 with Python 3.12+ on PATH."
  fi
  local version="${PYENV_VERSION:-3.12}"
  log "pyenv: install/use Python ${version}"
  pyenv install -s "$version"
  pyenv local "$version"
  log "Active interpreter: $(pyenv which python) ($(pyenv which python | xargs -I{} {} --version))"
}

run_uv_sync() {
  if [[ "${SKIP_UV_SYNC:-0}" == "1" ]]; then
    log "Skipping uv sync (SKIP_UV_SYNC=1)"
    return
  fi
  require_cmd uv
  log "uv sync"
  uv sync
}

ensure_pytorch_compatible() {
  if [[ "${SKIP_TORCH_COMPAT:-0}" == "1" ]]; then
    log "Skipping PyTorch compatibility check (SKIP_TORCH_COMPAT=1)"
    return
  fi
  require_cmd uv
  log "Checking PyTorch CUDA compatibility (auto-reinstall on failure)"
  uv run python -m training.utils.torch_compat
}

count_training_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local IFS=,
    local -a ids=(${CUDA_VISIBLE_DEVICES})
    echo "${#ids[@]}"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
    return
  fi
  echo "0"
}

prepare_dataset_dir() {
  log "Creating datasets/librispeech_data"
  mkdir -p datasets/librispeech_data
}

download_librispeech() {
  if [[ "${SKIP_DATASET:-0}" == "1" ]]; then
    log "Skipping LibriSpeech download (SKIP_DATASET=1)"
    return
  fi
  log "Downloading LibriSpeech (train-clean-100, train-clean-360, dev-clean)"
  uv run python src/dataset/load_dataset.py
}

download_stage1_checkpoint() {
  if [[ "${SKIP_CHECKPOINT:-0}" == "1" ]]; then
    log "Skipping checkpoint download (SKIP_CHECKPOINT=1)"
    return
  fi

  local ckpt_path="${STAGE1_CHECKPOINT:-checkpoints/adapter_stage1.pt}"
  if [[ "$ckpt_path" != /* ]]; then
    ckpt_path="${ROOT}/${ckpt_path}"
  fi
  local ckpt_dir
  ckpt_dir="$(dirname "$ckpt_path")"
  mkdir -p "$ckpt_dir"

  if [[ -f "$ckpt_path" ]]; then
    log "Checkpoint already present: ${ckpt_path}"
    return
  fi

  local repo="${HF_CHECKPOINT_REPO:-vaghawan/audio-streaming-adapter-checkpoints}"
  local url="https://huggingface.co/${repo}/resolve/main/adapter_stage1.pt"
  log "Downloading adapter_stage1.pt from ${repo}"
  if command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "$ckpt_path" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -fL --progress-bar -o "$ckpt_path" "$url"
  else
    die "Need wget or curl to download the checkpoint"
  fi
  log "Saved checkpoint to ${ckpt_path}"
}

prefetch_frozen_models() {
  if [[ "${SKIP_MODEL_PREFETCH:-0}" == "1" ]]; then
    log "Skipping model prefetch (SKIP_MODEL_PREFETCH=1)"
    return
  fi
  local whisper_id="${WHISPER_MODEL_ID:-openai/whisper-small}"
  local llm_id="${LLM_MODEL_ID:-Qwen/Qwen3-8B}"
  log "Prefetching frozen models in parallel: ${whisper_id}, ${llm_id}"
  uv run python scripts/prefetch_frozen_models.py
}

run_training() {
  if [[ "${SKIP_TRAINING:-0}" == "1" ]]; then
    log "Skipping training (SKIP_TRAINING=1)"
    return
  fi

  local gpus
  gpus="$(count_training_gpus)"
  log "Training GPUs (effective): ${gpus}"
  log "Launch: uv run python training/adapter_asr_trainer.py"
  uv run python training/adapter_asr_trainer.py
}

main() {
  source_env
  log "Project root: ${ROOT}"
  setup_pyenv
  run_uv_sync
  ensure_pytorch_compatible
  prepare_dataset_dir
  download_librispeech
  download_stage1_checkpoint
  prefetch_frozen_models
  run_training
  log "Done."
}

main "$@"
