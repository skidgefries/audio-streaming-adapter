"""
Turn-End Commit Gate (Component 3 — turn detection + token-based silence gating).

Operates on accumulated adapter tokens Z_{1:t}. Silence / activity is inferred from
per-window adapter tokens Z_t (already encoded audio).

Silence modes (``silence_mode``):
  - ``rule``: fixed token-norm heuristics (:class:`SilenceTracker`)
  - ``learned``: trainable :class:`LearnedSilenceHead` on Z_t
  - ``both``: run both paths in parallel for comparison / dual training

See ``docs/AUDIO_STREAM.md`` §7.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch.nn.functional import softmax


SILENCE_FEATURE_DIM = 3  # [silence_indicator, silence_duration_norm, activity_prob]
SilenceMode = Literal["rule", "learned", "both"]


def synthetic_endpoint_label(
    timestep: int,
    total_timesteps: int,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Label=1 on the final window, 0 otherwise (LibriSpeech full utterances)."""
    value = 1.0 if timestep == total_timesteps - 1 else 0.0
    return torch.full((batch_size, 1), value, device=device, dtype=dtype)


def _build_classifier(d_llm: int, hidden_dim: int, silence_feature_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_llm + silence_feature_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_dim, 64),
        nn.GELU(),
        nn.Linear(64, 1),
    )


def _init_linear_modules(*modules: nn.Module) -> None:
    for module in modules:
        for sub in module.modules():
            if isinstance(sub, nn.Linear):
                sub.weight.data.normal_(mean=0.0, std=0.1)
                if sub.bias is not None:
                    sub.bias.data.zero_()


class SilenceTracker:
    """
    Rule-based speech vs silence from **adapter token activity** per window.

    Call ``update_from_window_tokens(window_tokens)`` with this window's ``Z_t``.
    Feature vector (dim=3):
        [silence_indicator, silence_duration_norm, activity_prob]
    """

    def __init__(
        self,
        token_activity_threshold: float = 8.0,
        min_silence_ms: float = 200.0,
        silence_norm_ms: float = 2000.0,
        token_activity_scale: float = 0.5,
    ):
        self.token_activity_threshold = token_activity_threshold
        self.min_silence_ms = min_silence_ms
        self.silence_norm_ms = silence_norm_ms
        self.token_activity_scale = token_activity_scale
        self.reset()

    def reset(self) -> None:
        self.is_speech: bool = False
        self.silence_duration_ms: float = 0.0
        self.last_activity_prob: float = 0.0

    def _token_activity(self, window_tokens: torch.Tensor) -> float:
        tokens = window_tokens.float()
        if tokens.ndim == 3:
            tokens = tokens.reshape(-1, tokens.shape[-1])
        elif tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.numel() == 0:
            return 0.0
        return float(tokens.norm(dim=-1).mean().item())

    def _activity_to_prob(self, activity: float) -> float:
        return float(
            torch.sigmoid(
                torch.tensor(
                    (activity - self.token_activity_threshold) * self.token_activity_scale
                )
            ).item()
        )

    def update_from_window_tokens(
        self,
        window_tokens: torch.Tensor,
        *,
        window_duration_sec: float = 0.8,
    ) -> dict[str, float]:
        activity = self._token_activity(window_tokens)
        activity_prob = self._activity_to_prob(activity)
        self.last_activity_prob = activity_prob
        self.is_speech = activity_prob >= 0.5
        if self.is_speech:
            self.silence_duration_ms = 0.0
        else:
            self.silence_duration_ms += window_duration_sec * 1000.0
        return self.snapshot()

    def snapshot(self) -> dict[str, float]:
        return {
            "is_speech": float(self.is_speech),
            "silence_duration_ms": self.silence_duration_ms,
            "activity_prob": self.last_activity_prob,
            "silence_ready": float(self.silence_ready),
        }

    @property
    def silence_ready(self) -> bool:
        return (not self.is_speech) and (self.silence_duration_ms >= self.min_silence_ms)

    def feature_vector(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        silence_indicator = 1.0 - float(self.is_speech)
        silence_norm = min(self.silence_duration_ms / max(self.silence_norm_ms, 1.0), 1.0)
        vec = torch.tensor(
            [[silence_indicator, silence_norm, self.last_activity_prob]],
            device=device,
            dtype=dtype,
        )
        return vec.expand(batch_size, -1)


class LearnedSilenceHead(nn.Module):
    """Trainable speech-activity head on per-window adapter tokens ``Z_t``."""

    def __init__(self, d_llm: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_llm, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        _init_linear_modules(self)

    def forward(self, window_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            window_tokens: (batch, m, d_llm) or (m, d_llm)

        Returns:
            speech_prob: (batch, 1) in [0, 1]
        """
        if window_tokens.ndim == 3:
            x = window_tokens.mean(dim=1)
        elif window_tokens.ndim == 2:
            x = window_tokens.mean(dim=0, keepdim=True)
        else:
            raise ValueError(
                f"Expected window_tokens (B, m, d) or (m, d), got {tuple(window_tokens.shape)}"
            )
        return torch.sigmoid(self.net(x))


class LearnedSilenceTracker:
    """
    Stateful silence tracker driven by :class:`LearnedSilenceHead`.

    Duration gating uses a hard threshold on speech_prob (detached); feature vectors
    use soft speech_prob so gradients flow into the head during training.
    """

    def __init__(
        self,
        head: LearnedSilenceHead,
        min_silence_ms: float = 200.0,
        silence_norm_ms: float = 2000.0,
    ):
        self.head = head
        self.min_silence_ms = min_silence_ms
        self.silence_norm_ms = silence_norm_ms
        self.reset()
        self._last_speech_prob: torch.Tensor | None = None

    def reset(self) -> None:
        self.is_speech: bool = False
        self.silence_duration_ms: float = 0.0
        self._last_speech_prob = None

    def update_from_window_tokens(
        self,
        window_tokens: torch.Tensor,
        *,
        window_duration_sec: float = 0.8,
    ) -> dict[str, float]:
        speech_prob = self.head(window_tokens)
        self._last_speech_prob = speech_prob
        prob_scalar = float(speech_prob.detach().mean().item())
        self.is_speech = prob_scalar >= 0.5
        if self.is_speech:
            self.silence_duration_ms = 0.0
        else:
            self.silence_duration_ms += window_duration_sec * 1000.0
        return self.snapshot()

    def snapshot(self) -> dict[str, float]:
        activity = (
            float(self._last_speech_prob.detach().mean().item())
            if self._last_speech_prob is not None
            else 0.0
        )
        return {
            "is_speech": float(self.is_speech),
            "silence_duration_ms": self.silence_duration_ms,
            "activity_prob": activity,
            "silence_ready": float(self.silence_ready),
        }

    @property
    def silence_ready(self) -> bool:
        return (not self.is_speech) and (self.silence_duration_ms >= self.min_silence_ms)

    def feature_vector(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self._last_speech_prob is None:
            speech_prob = torch.full((batch_size, 1), 0.5, device=device, dtype=dtype)
        else:
            speech_prob = self._last_speech_prob.to(device=device, dtype=dtype)
            if speech_prob.shape[0] == 1 and batch_size > 1:
                speech_prob = speech_prob.expand(batch_size, -1)
            elif speech_prob.shape[0] != batch_size:
                speech_prob = speech_prob.mean(dim=0, keepdim=True).expand(batch_size, -1)

        silence_indicator = 1.0 - speech_prob
        silence_norm = min(self.silence_duration_ms / max(self.silence_norm_ms, 1.0), 1.0)
        norm_t = torch.full((batch_size, 1), silence_norm, device=device, dtype=dtype)
        return torch.cat([silence_indicator, norm_t, speech_prob], dim=-1)


class TurnEndCommitGate(nn.Module):
    """
    Turn-end detector on accumulated adapter tokens + silence features.

    Architecture (per path):
        Z_{1:t} → attention pool ─┐
                                  ├→ concat → classifier MLP → σ → commit probability
        silence features ─────────┘
    """

    def __init__(
        self,
        d_llm: int = 2560,
        hidden_dim: int = 256,
        threshold: float = 0.5,
        latency_weight: float = 0.1,
        min_silence_ms: float = 200.0,
        require_silence_for_commit: bool = True,
        token_activity_threshold: float = 8.0,
        window_duration_sec: float = 0.8,
        silence_mode: SilenceMode | str = "rule",
        active_silence_path: SilenceMode | str = "rule",
        learned_silence_hidden_dim: int = 64,
    ):
        super().__init__()
        self.d_llm = d_llm
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        self.latency_weight = latency_weight
        self.min_silence_ms = min_silence_ms
        self.require_silence_for_commit = require_silence_for_commit
        self.token_activity_threshold = token_activity_threshold
        self.window_duration_sec = window_duration_sec
        self.silence_feature_dim = SILENCE_FEATURE_DIM

        silence_mode = str(silence_mode).lower()
        if silence_mode not in ("rule", "learned", "both"):
            raise ValueError(f"silence_mode must be rule|learned|both, got {silence_mode!r}")
        self.silence_mode: SilenceMode = silence_mode  # type: ignore[assignment]

        active = str(active_silence_path).lower()
        if active not in ("rule", "learned"):
            raise ValueError(f"active_silence_path must be rule|learned, got {active!r}")
        if silence_mode != "both" and active != silence_mode:
            active = silence_mode
        self.active_silence_path = active

        self.pool_attention = nn.Sequential(
            nn.Linear(d_llm, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.classifier = _build_classifier(d_llm, hidden_dim, SILENCE_FEATURE_DIM)
        _init_linear_modules(self.pool_attention, self.classifier)

        self.learned_silence_head: LearnedSilenceHead | None = None
        self.classifier_learned: nn.Sequential | None = None
        if silence_mode in ("learned", "both"):
            self.learned_silence_head = LearnedSilenceHead(d_llm, learned_silence_hidden_dim)
            self.classifier_learned = _build_classifier(d_llm, hidden_dim, SILENCE_FEATURE_DIM)
            _init_linear_modules(self.learned_silence_head, self.classifier_learned)

    def make_silence_tracker(self) -> SilenceTracker:
        return SilenceTracker(
            token_activity_threshold=self.token_activity_threshold,
            min_silence_ms=self.min_silence_ms,
        )

    def make_learned_silence_tracker(self) -> LearnedSilenceTracker:
        if self.learned_silence_head is None:
            raise RuntimeError("Learned silence is not enabled (silence_mode='rule').")
        return LearnedSilenceTracker(
            head=self.learned_silence_head,
            min_silence_ms=self.min_silence_ms,
        )

    def _resolve_rule_silence_features(
        self,
        accumulated_tokens: torch.Tensor,
        *,
        silence_tracker: SilenceTracker | None,
        window_tokens: torch.Tensor | None,
        window_duration_sec: float,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        batch_size = accumulated_tokens.shape[0]
        device = accumulated_tokens.device
        dtype = accumulated_tokens.dtype

        if silence_tracker is not None:
            if window_tokens is not None:
                snap = silence_tracker.update_from_window_tokens(
                    window_tokens,
                    window_duration_sec=window_duration_sec,
                )
            else:
                snap = silence_tracker.snapshot()
            return silence_tracker.feature_vector(batch_size, device, dtype), snap

        return (
            torch.tensor([[1.0, 0.0, 0.5]], device=device, dtype=dtype).expand(batch_size, -1),
            None,
        )

    def _resolve_learned_silence_features(
        self,
        accumulated_tokens: torch.Tensor,
        *,
        learned_silence_tracker: LearnedSilenceTracker | None,
        window_tokens: torch.Tensor | None,
        window_duration_sec: float,
    ) -> tuple[torch.Tensor, dict[str, float] | None]:
        batch_size = accumulated_tokens.shape[0]
        device = accumulated_tokens.device
        dtype = accumulated_tokens.dtype

        if learned_silence_tracker is not None:
            if window_tokens is not None:
                snap = learned_silence_tracker.update_from_window_tokens(
                    window_tokens,
                    window_duration_sec=window_duration_sec,
                )
            else:
                snap = learned_silence_tracker.snapshot()
            return learned_silence_tracker.feature_vector(batch_size, device, dtype), snap

        return (
            torch.tensor([[1.0, 0.0, 0.5]], device=device, dtype=dtype).expand(batch_size, -1),
            None,
        )

    def _resolve_endpoint_label(
        self,
        accumulated_tokens: torch.Tensor,
        timestep: int,
        total_timesteps: int,
        endpoint_label: torch.Tensor | float | None,
    ) -> torch.Tensor:
        batch_size = accumulated_tokens.shape[0]
        device = accumulated_tokens.device
        dtype = accumulated_tokens.dtype

        if endpoint_label is None:
            return synthetic_endpoint_label(
                timestep,
                total_timesteps,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

        if isinstance(endpoint_label, (int, float)):
            value = float(endpoint_label)
            return torch.full((batch_size, 1), value, device=device, dtype=dtype)

        label = endpoint_label.to(device=device, dtype=dtype)
        if label.ndim == 0:
            label = label.view(1, 1).expand(batch_size, 1)
        elif label.ndim == 1:
            label = label.view(batch_size, 1)
        return label

    def _forward_path(
        self,
        pooled: torch.Tensor,
        silence_feats: torch.Tensor,
        classifier: nn.Sequential,
        *,
        labels: torch.Tensor,
        timestep: int,
        total_timesteps: int,
        silence_ready: bool | None,
    ) -> dict[str, torch.Tensor]:
        logits = classifier(torch.cat([pooled, silence_feats], dim=-1))
        commit_prob = torch.sigmoid(logits)

        turn_ready = commit_prob > self.threshold
        if self.require_silence_for_commit and silence_ready is not None:
            silence_ok = torch.full_like(turn_ready, float(silence_ready))
            should_commit = (turn_ready & silence_ok).float()
        else:
            should_commit = turn_ready.float()

        pos_weight = ((labels == 0).sum() / (labels == 1).sum().clamp(min=1)).clamp(
            min=0.1, max=10.0
        )
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits.view(-1),
            labels.view(-1),
            pos_weight=pos_weight,
        )

        position = timestep / max(total_timesteps - 1, 1)
        latency_penalty = self.latency_weight * position * (1.0 - commit_prob).mean()
        gate_loss = bce_loss + latency_penalty

        return {
            "commit_prob": commit_prob,
            "should_commit": should_commit,
            "gate_loss": gate_loss,
            "logits": logits,
            "bce_loss": bce_loss,
            "latency_penalty": latency_penalty,
            "silence_features": silence_feats,
        }

    def _attach_silence_snap(
        self,
        out: dict[str, torch.Tensor],
        snap: dict[str, float] | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if snap is None:
            return
        out["silence_duration_ms"] = torch.tensor(
            [snap["silence_duration_ms"]], device=device, dtype=dtype
        )
        out["silence_ready"] = torch.tensor([snap["silence_ready"]], device=device, dtype=dtype)
        out["is_speech"] = torch.tensor([snap["is_speech"]], device=device, dtype=dtype)

    def forward(
        self,
        accumulated_tokens: torch.Tensor,
        timestep: int,
        total_timesteps: int,
        endpoint_label: torch.Tensor | float | None = None,
        *,
        silence_tracker: SilenceTracker | None = None,
        learned_silence_tracker: LearnedSilenceTracker | None = None,
        window_tokens: torch.Tensor | None = None,
        window_duration_sec: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            accumulated_tokens: (batch, N, d_llm) — prefix Z_{1:t}
            window_tokens: (batch, m, d_llm) — this window's Z_t for silence tracking
            silence_tracker: rule-based tracker (required for ``rule`` / ``both``)
            learned_silence_tracker: learned tracker (required for ``learned`` / ``both``)
        """
        if accumulated_tokens.ndim != 3:
            raise ValueError(
                f"Expected accumulated_tokens (B, N, d_llm), got {tuple(accumulated_tokens.shape)}"
            )

        dur = self.window_duration_sec if window_duration_sec is None else window_duration_sec
        labels = self._resolve_endpoint_label(
            accumulated_tokens, timestep, total_timesteps, endpoint_label
        )

        hidden_states = accumulated_tokens
        attention_weights = self.pool_attention(hidden_states)
        attention_weights = softmax(attention_weights, dim=1)
        pooled = torch.sum(hidden_states * attention_weights, dim=1)

        device = accumulated_tokens.device
        dtype = accumulated_tokens.dtype

        if self.silence_mode == "rule":
            silence_feats, snap = self._resolve_rule_silence_features(
                accumulated_tokens,
                silence_tracker=silence_tracker,
                window_tokens=window_tokens,
                window_duration_sec=dur,
            )
            silence_ready = snap["silence_ready"] >= 0.5 if snap is not None else None
            out = self._forward_path(
                pooled,
                silence_feats,
                self.classifier,
                labels=labels,
                timestep=timestep,
                total_timesteps=total_timesteps,
                silence_ready=bool(silence_ready) if silence_ready is not None else None,
            )
            self._attach_silence_snap(out, snap, device, dtype)
            out["silence_path"] = torch.tensor([0.0], device=device, dtype=dtype)
            return out

        if self.silence_mode == "learned":
            assert self.classifier_learned is not None
            silence_feats, snap = self._resolve_learned_silence_features(
                accumulated_tokens,
                learned_silence_tracker=learned_silence_tracker,
                window_tokens=window_tokens,
                window_duration_sec=dur,
            )
            silence_ready = snap["silence_ready"] >= 0.5 if snap is not None else None
            out = self._forward_path(
                pooled,
                silence_feats,
                self.classifier_learned,
                labels=labels,
                timestep=timestep,
                total_timesteps=total_timesteps,
                silence_ready=bool(silence_ready) if silence_ready is not None else None,
            )
            self._attach_silence_snap(out, snap, device, dtype)
            out["silence_path"] = torch.tensor([1.0], device=device, dtype=dtype)
            return out

        # both — parallel rule + learned paths
        assert self.classifier_learned is not None

        rule_feats, rule_snap = self._resolve_rule_silence_features(
            accumulated_tokens,
            silence_tracker=silence_tracker,
            window_tokens=window_tokens,
            window_duration_sec=dur,
        )
        learned_feats, learned_snap = self._resolve_learned_silence_features(
            accumulated_tokens,
            learned_silence_tracker=learned_silence_tracker,
            window_tokens=window_tokens,
            window_duration_sec=dur,
        )

        rule_ready = bool(rule_snap["silence_ready"] >= 0.5) if rule_snap is not None else None
        learned_ready = (
            bool(learned_snap["silence_ready"] >= 0.5) if learned_snap is not None else None
        )

        rule_out = self._forward_path(
            pooled,
            rule_feats,
            self.classifier,
            labels=labels,
            timestep=timestep,
            total_timesteps=total_timesteps,
            silence_ready=rule_ready,
        )
        learned_out = self._forward_path(
            pooled,
            learned_feats,
            self.classifier_learned,
            labels=labels,
            timestep=timestep,
            total_timesteps=total_timesteps,
            silence_ready=learned_ready,
        )

        active = rule_out if self.active_silence_path == "rule" else learned_out
        combined: dict[str, torch.Tensor] = {
            "commit_prob": active["commit_prob"],
            "should_commit": active["should_commit"],
            "gate_loss": rule_out["gate_loss"] + learned_out["gate_loss"],
            "logits": active["logits"],
            "bce_loss": active["bce_loss"],
            "latency_penalty": active["latency_penalty"],
            "silence_features": active["silence_features"],
            "commit_prob_rule": rule_out["commit_prob"],
            "should_commit_rule": rule_out["should_commit"],
            "gate_loss_rule": rule_out["gate_loss"],
            "bce_loss_rule": rule_out["bce_loss"],
            "silence_features_rule": rule_feats,
            "commit_prob_learned": learned_out["commit_prob"],
            "should_commit_learned": learned_out["should_commit"],
            "gate_loss_learned": learned_out["gate_loss"],
            "bce_loss_learned": learned_out["bce_loss"],
            "silence_features_learned": learned_feats,
            "silence_path": torch.tensor(
                [0.0 if self.active_silence_path == "rule" else 1.0], device=device, dtype=dtype
            ),
        }

        if rule_snap is not None:
            combined["silence_ready_rule"] = torch.tensor(
                [rule_snap["silence_ready"]], device=device, dtype=dtype
            )
            combined["is_speech_rule"] = torch.tensor(
                [rule_snap["is_speech"]], device=device, dtype=dtype
            )
        if learned_snap is not None:
            combined["silence_ready_learned"] = torch.tensor(
                [learned_snap["silence_ready"]], device=device, dtype=dtype
            )
            combined["is_speech_learned"] = torch.tensor(
                [learned_snap["is_speech"]], device=device, dtype=dtype
            )

        snap = rule_snap if self.active_silence_path == "rule" else learned_snap
        self._attach_silence_snap(combined, snap, device, dtype)
        return combined
