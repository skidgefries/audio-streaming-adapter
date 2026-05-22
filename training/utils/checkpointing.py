from __future__ import annotations

import os
import re
from dataclasses import dataclass

import torch

STAGE_EPOCH_FILENAME = re.compile(r"^adapter_stage(\d+)_epoch(\d+)\.pt$")


@dataclass(frozen=True)
class TrainingCheckpoint:
    stage: int
    epoch: int
    global_step: int
    adapter_state_dict: dict
    gate_state_dict: dict | None = None
    optimizer_state_dict: dict | None = None
    scheduler_state_dict: dict | None = None
    metrics: dict | None = None
    hyperparams: dict | None = None


def save_checkpoint(path: str, ckpt: TrainingCheckpoint) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "stage": ckpt.stage,
        "epoch": ckpt.epoch,
        "global_step": ckpt.global_step,
        "adapter_state_dict": ckpt.adapter_state_dict,
        "metrics": ckpt.metrics or {},
        "hyperparams": ckpt.hyperparams or {},
    }
    if ckpt.gate_state_dict is not None:
        payload["gate_state_dict"] = ckpt.gate_state_dict
    if ckpt.optimizer_state_dict is not None:
        payload["optimizer_state_dict"] = ckpt.optimizer_state_dict
    if ckpt.scheduler_state_dict is not None:
        payload["scheduler_state_dict"] = ckpt.scheduler_state_dict

    torch.save(payload, path)


def load_adapter_state_dict(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("adapter_state_dict") or ckpt.get("model_state_dict") or ckpt.get("state_dict")
    if state is None:
        raise KeyError(f"No adapter state_dict found in checkpoint keys: {list(ckpt.keys())}")
    return state


def hub_path_for_stage_epoch_upload(local_path: str, *, stage: int) -> str:
    """
    Hub path for one epoch checkpoint of ``stage`` only (repo root, no prefix).

    Only ``adapter_stage{N}_epoch{M}.pt`` is allowed. Resume files such as
    ``adapter_stage{N}.pt`` are never uploaded.
    """
    filename = os.path.basename(local_path)
    match = STAGE_EPOCH_FILENAME.match(filename)
    if not match:
        raise ValueError(
            f"Only adapter_stage{{N}}_epoch{{M}}.pt checkpoints are uploaded; got: {filename}"
        )
    file_stage = int(match.group(1))
    if file_stage != stage:
        raise ValueError(
            f"Checkpoint is stage {file_stage} but upload requested for stage {stage}: {filename}"
        )
    return filename


def upload_checkpoint_to_hub(
    local_path: str,
    *,
    repo_id: str,
    path_in_repo: str | None = None,
    revision: str = "main",
    private: bool = False,
    token: str | None = None,
) -> str:
    """
    Upload one checkpoint file to the Hugging Face Hub.

    Uses ``upload_file`` (additive): other repo files are left unchanged.
    """
    if not os.path.isfile(local_path):
        raise FileNotFoundError(local_path)

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError("huggingface_hub is required for checkpoint upload") from exc

    hub_path = path_in_repo or os.path.basename(local_path)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=hub_path,
        repo_id=repo_id,
        revision=revision,
        commit_message=f"Upload {hub_path}",
    )
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{hub_path}"


def maybe_upload_stage_epoch_checkpoint(
    local_path: str,
    *,
    stage: int,
    repo_id: str,
    revision: str = "main",
    private: bool = False,
    token: str | None = None,
    enabled: bool = False,
) -> None:
    """Upload ``adapter_stage{stage}_epoch{N}.pt`` when enabled; never touches other stages' files."""
    if not enabled:
        return
    try:
        hub_path = hub_path_for_stage_epoch_upload(local_path, stage=stage)
        url = upload_checkpoint_to_hub(
            local_path,
            repo_id=repo_id,
            path_in_repo=hub_path,
            revision=revision,
            private=private,
            token=token,
        )
        print(f"Uploaded checkpoint to {url}")
    except ValueError:
        return
    except Exception as exc:
        print(f"[WARN] Hugging Face upload failed for {local_path}: {exc}")
