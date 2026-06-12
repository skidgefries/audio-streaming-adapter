# Evaluation

Two entry scripts for LibriSpeech test-clean. Pick the stage, then choose a metric with `--metric`.

| Script | Metrics |
|--------|---------|
| `eval_stage1.py` | `retrieval-cosine`, `retrieval-nll`, `asr` |
| `eval_stage2.py` | `retrieval-cosine`, `retrieval-nll`, `asr` |
| `eval_asr_only.py` | `retrieval-cosine`, `retrieval-nll`, `asr` |

Run from the package root (`audio-streaming-adapter/`):

```bash
# Stage 1
uv run evaluation/eval_stage1.py --metric retrieval-cosine
uv run evaluation/eval_stage1.py --metric retrieval-nll --num-samples 100
uv run evaluation/eval_stage1.py --metric asr

# Stage 2
uv run evaluation/eval_stage2.py --metric retrieval-cosine --num-samples all
uv run evaluation/eval_stage2.py --metric retrieval-nll --num-samples all
uv run evaluation/eval_stage2.py --metric asr --compare-im-end

# ASR-only (adapter_asr_only_trainer.py)
uv run evaluation/eval_asr_only.py --metric asr --num-samples 100
uv run evaluation/eval_asr_only.py --metric retrieval-cosine --num-samples all
```

Shared flags: `--checkpoint`, `--dataset-root`, `--num-samples` (integer or `all`), `--run-name`, `--output-dir`.

Results:

- `eval_stage1.py` / `eval_stage2.py` → `outputs/experiments/{RUN_NAME}/`
- `eval_asr_only.py` → `outputs/asr_eval_only/{RUN_NAME}/` (`asr_metrics.json`, `asr_predictions.json`, …)
