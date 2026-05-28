"""Gate training helpers shared by Stage 2/3 trainers."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.adapter.turn_end_commit_gate import (
    LearnedSilenceTracker,
    SilenceTracker,
    TurnEndCommitGate,
    synthetic_endpoint_label,
)
from training.utils.checkpointing import load_gate_state_dict_safe


def build_turn_end_gate(
    *,
    d_llm: int,
    hidden_dim: int,
    threshold: float,
    latency_weight: float,
    min_silence_ms: float = 200.0,
    require_silence_for_commit: bool = True,
    token_activity_threshold: float = 8.0,
    window_duration_sec: float = 0.8,
    silence_mode: str = "rule",
    active_silence_path: str = "rule",
    learned_silence_hidden_dim: int = 64,
    device: torch.device | str,
    dtype: torch.dtype,
) -> TurnEndCommitGate:
    gate = TurnEndCommitGate(
        d_llm=d_llm,
        hidden_dim=hidden_dim,
        threshold=threshold,
        latency_weight=latency_weight,
        min_silence_ms=min_silence_ms,
        require_silence_for_commit=require_silence_for_commit,
        token_activity_threshold=token_activity_threshold,
        window_duration_sec=window_duration_sec,
        silence_mode=silence_mode,
        active_silence_path=active_silence_path,
        learned_silence_hidden_dim=learned_silence_hidden_dim,
    ).to(device, dtype=dtype)
    gate.train()
    return gate


def make_silence_trackers(
    gate_module: TurnEndCommitGate,
) -> tuple[SilenceTracker | None, LearnedSilenceTracker | None]:
    """Create fresh rule and/or learned trackers for one utterance stream."""
    mode = gate_module.silence_mode
    rule = gate_module.make_silence_tracker() if mode in ("rule", "both") else None
    learned = (
        gate_module.make_learned_silence_tracker() if mode in ("learned", "both") else None
    )
    return rule, learned


def gate_forward_step(
    gate_module: nn.Module,
    accumulated: torch.Tensor,
    *,
    timestep: int,
    total_timesteps: int,
    endpoint_label: torch.Tensor | float | None = None,
    silence_tracker: SilenceTracker | None = None,
    learned_silence_tracker: LearnedSilenceTracker | None = None,
    window_tokens: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    return gate_module(
        accumulated,
        timestep,
        total_timesteps,
        endpoint_label=endpoint_label,
        silence_tracker=silence_tracker,
        learned_silence_tracker=learned_silence_tracker,
        window_tokens=window_tokens,
    )


def endpoint_label_for_timestep(
    timestep: int,
    total_timesteps: int,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return synthetic_endpoint_label(
        timestep,
        total_timesteps,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )


__all__ = [
    "SilenceTracker",
    "LearnedSilenceTracker",
    "build_turn_end_gate",
    "make_silence_trackers",
    "gate_forward_step",
    "endpoint_label_for_timestep",
    "load_gate_state_dict_safe",
]
