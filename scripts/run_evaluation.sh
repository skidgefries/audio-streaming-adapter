#!/usr/bin/env bash
# Run Stage 2 ASR eval on full LibriSpeech test-clean for selected checkpoints,
# then bin predictions by WER / BLEU-4 and write aggregate metrics.
#
# Usage:
#   bash scripts/run_evaluation.sh
#
# Logs:   logs/evaluation/{TIMESTAMP}/
# Outputs: outputs/experiments/{RUN_NAME}/
#          outputs/experiments/{RUN_NAME}/analysis/
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

log() { printf '\n==> [%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# --- Source .env -------------------------------------------------------------

ENV_FILE="${ROOT}/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
  log "Sourced ${ENV_FILE}"
fi

# --- GPU layout: Whisper/adapter on GPU 0, Qwen split across 0+1 ------------

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export DEVICE="${DEVICE:-cuda:0}"
if [[ -z "${LLM_MAX_MEMORY:-}" ]]; then
  die "LLM_MAX_MEMORY is not set; add it to ${ENV_FILE} (e.g. LLM_MAX_MEMORY=0:15GiB,1:3GiB)"
fi
export LLM_MAX_MEMORY
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="${ROOT}/logs/evaluation/${TIMESTAMP}"
OUTPUT_DIR="${ROOT}/outputs/experiments"
CKPT_DIR="${CHECKPOINT_DIR:-${ROOT}/checkpoints}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# Checkpoints to evaluate (full test-clean)
declare -a CHECKPOINTS=(
  "${CKPT_DIR}/adapter_stage2_epoch1.pt"
  "${CKPT_DIR}/adapter_stage2_epoch2.pt"
  "${CKPT_DIR}/adapter_stage2_step36500.pt"
)

log "Log directory: ${LOG_DIR}"
log "Output directory: ${OUTPUT_DIR}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
log "DEVICE=${DEVICE}"
log "LLM_MAX_MEMORY=${LLM_MAX_MEMORY}"

cuda_cleanup() {
  uv run python - <<'PY' || true
import gc
try:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except Exception:
    pass
PY
}

run_cmd() {
  local log_file="$1"
  shift
  log "Running: $*"
  "$@" 2>&1 | tee -a "$log_file"
  local rc="${PIPESTATUS[0]}"
  if [[ "$rc" -ne 0 ]]; then
    die "Command failed (exit ${rc}): $* — see ${log_file}"
  fi
}

run_name_for_checkpoint() {
  local ckpt="$1"
  basename "${ckpt%.pt}"
}

run_eval() {
  local checkpoint="$1"
  local run_name
  run_name="$(run_name_for_checkpoint "$checkpoint")"
  local log_file="${LOG_DIR}/${run_name}_eval.log"

  if [[ ! -f "$checkpoint" ]]; then
    log "SKIP ${run_name}: checkpoint not found at ${checkpoint}"
    return 0
  fi

  log "Evaluating ${run_name} on full test-clean (${checkpoint})"

  run_cmd "$log_file" uv run evaluation/eval_stage2.py \
    --device cuda \
    --metric asr \
    --checkpoint "$checkpoint" \
    --run-name "$run_name" \
    --output-dir "$OUTPUT_DIR" \
    --num-samples all

  cuda_cleanup

  local run_dir="${OUTPUT_DIR}/${run_name}"
  if [[ ! -f "${run_dir}/asr_predictions.json" ]]; then
    die "Expected ${run_dir}/asr_predictions.json after eval"
  fi

  log "Analyzing predictions for ${run_name}"
  run_cmd "${LOG_DIR}/${run_name}_analysis.log" \
    uv run scripts/analyze_asr_predictions.py "$run_dir"
}

write_batch_summary() {
  local summary_file="${LOG_DIR}/batch_summary.json"
  uv run python - <<'PY' "$OUTPUT_DIR" "$summary_file" "${CHECKPOINTS[@]}"
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
checkpoints = sys.argv[3:]

runs = {}
for ckpt in checkpoints:
    run_name = Path(ckpt).stem
    run_dir = output_dir / run_name
    entry = {"checkpoint": ckpt, "run_name": run_name}
    for rel in (
        "asr_metrics.json",
        "analysis/aggregate_metrics.json",
        "analysis/wer_bins_summary.json",
        "analysis/bleu_bins_summary.json",
    ):
        path = run_dir / rel
        if path.is_file():
            entry[rel.replace("/", "_").replace(".json", "")] = json.loads(
                path.read_text(encoding="utf-8")
            )
    if len(entry) > 2:
        runs[run_name] = entry

summary = {"runs": runs}
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"Wrote batch summary -> {summary_path}")
PY
}

# =============================================================================
# Run eval + analysis for each checkpoint
# =============================================================================

for ckpt in "${CHECKPOINTS[@]}"; do
  run_eval "$ckpt"
done

write_batch_summary

log "Evaluation complete."
log "Logs: ${LOG_DIR}"
log "Outputs: ${OUTPUT_DIR}"
log "Per-run analysis: ${OUTPUT_DIR}/<run_name>/analysis/"
log "Batch summary: ${LOG_DIR}/batch_summary.json"
