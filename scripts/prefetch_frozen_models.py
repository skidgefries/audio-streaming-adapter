#!/usr/bin/env python3
"""Download frozen Whisper + Qwen weights to the Hugging Face cache (parallel)."""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_pkg_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_pkg_root))

from training.utils.env import env_str, load_project_env

load_project_env(str(_pkg_root))

WHISPER_MODEL_ID = env_str("WHISPER_MODEL_ID", "openai/whisper-small") or "openai/whisper-small"
LLM_MODEL_ID = env_str("LLM_MODEL_ID", "Qwen/Qwen3-8B") or "Qwen/Qwen3-8B"


def _hf_token() -> str | None:
    return env_str("HF_TOKEN") or env_str("HUGGINGFACE_HUB_TOKEN")


def prefetch_repo(repo_id: str) -> str:
    from huggingface_hub import snapshot_download

    token = _hf_token()
    path = snapshot_download(repo_id, token=token)
    print(f"Cached {repo_id} -> {path}", flush=True)
    return repo_id


def main() -> None:
    repos = (WHISPER_MODEL_ID, LLM_MODEL_ID)
    print(f"Prefetching in parallel: {repos[0]!r}, {repos[1]!r}", flush=True)

    errors: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(prefetch_repo, repo_id): repo_id for repo_id in repos}
        for fut in as_completed(futures):
            repo_id = futures[fut]
            try:
                fut.result()
            except BaseException as exc:
                errors.append(exc)
                print(f"FAILED {repo_id}: {exc}", file=sys.stderr, flush=True)

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
