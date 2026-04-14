from __future__ import annotations

import os
from dataclasses import dataclass

import torch


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
