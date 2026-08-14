#!/usr/bin/env python3
"""Download frozen Vicuna weights to the Hugging Face cache."""

from __future__ import annotations

import sys
from pathlib import Path

_pkg_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_pkg_root))

from training.utils.env import apply_hf_hub_endpoint, env_str, load_project_env

load_project_env(str(_pkg_root))
apply_hf_hub_endpoint(str(_pkg_root))

VICUNA_MODEL_ID = env_str("VICUNA_MODEL_ID", "lmsys/vicuna-7b-v1.5") or "lmsys/vicuna-7b-v1.5"


def _hf_token() -> str | None:
    return env_str("HF_TOKEN") or env_str("HUGGINGFACE_HUB_TOKEN")


def prefetch_repo(repo_id: str) -> str:
    from huggingface_hub import snapshot_download

    token = _hf_token()
    path = snapshot_download(repo_id, token=token)
    print(f"Cached {repo_id} -> {path}", flush=True)
    return repo_id


def main() -> None:
    print(f"Prefetching Vicuna: {VICUNA_MODEL_ID!r}", flush=True)
    try:
        prefetch_repo(VICUNA_MODEL_ID)
    except BaseException as exc:
        print(f"FAILED {VICUNA_MODEL_ID}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
