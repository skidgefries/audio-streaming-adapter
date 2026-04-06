"""
Early-Commit Gate (Component 3 — separate from the Streaming Adapter).

Decides WHEN the LLM should start generating a response vs. keep listening
for more audio tokens. This is a latency-accuracy tradeoff:

  - Commit too early → insufficient context → wrong/incomplete answer
  - Commit too late  → unnecessary latency → poor user experience

The gate operates on the accumulated token sequence Z_{1:t} and outputs
a scalar commit probability g_t ∈ [0, 1].

This is DIFFERENT from the Rate Controller:
  - Rate Controller: "How many tokens for THIS window?" (inside adapter)
  - Early-Commit Gate: "Should the LLM START generating NOW?" (outside adapter)

From the paper: g_t = σ(W · Z_t)

Associated loss:
  - L_gate: Balances early commitment (low latency) against accuracy.
    Penalizes both:
    (a) committing too early (before enough info) → accuracy penalty
    (b) committing too late (after sufficient info) → latency penalty
"""

import torch
import torch.nn as nn


class EarlyCommitGate(nn.Module):
    """
    Trainable gate that learns when to trigger LLM generation.

    At each timestep t (after receiving window t's tokens), the gate
    estimates whether enough information has accumulated to generate
    a good response.

    Architecture:
        Accumulated tokens Z_{1:t} → pool → MLP → σ → commit probability

    Args:
        d_llm: Dimension of adapter output tokens (LLM embedding dim).
        hidden_dim: Hidden dimension of the gating MLP.
        threshold: Commit threshold for inference (generate if g_t > threshold).
        latency_weight: Weight for latency penalty in L_gate.
                        Higher = more aggressive early commitment.
    """

    def __init__(
        self,
        d_llm: int = 2560,
        hidden_dim: int = 256,
        threshold: float = 0.5,
        latency_weight: float = 0.1,
    ):
        super().__init__()
        self.threshold = threshold
        self.latency_weight = latency_weight

        self.gate_mlp = nn.Sequential(
            nn.Linear(d_llm, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        accumulated_tokens: torch.Tensor,
        timestep: int,
        total_timesteps: int,
    ) -> dict[str, torch.Tensor]:
        """
        Compute commit probability for current timestep.

        Args:
            accumulated_tokens: (batch, num_tokens_so_far, d_llm) -- all tokens
                received up to and including current window
            timestep: Current window index (0-based)
            total_timesteps: Total number of windows in the utterance (for loss)

        Returns:
            dict with:
                commit_prob: (batch, 1) -- probability of committing now
                should_commit: (batch, 1) -- binary decision (inference)
                gate_loss: scalar -- L_gate balancing latency vs accuracy
        """
        # Pool accumulated tokens to get a summary vector
        pooled = accumulated_tokens.mean(dim=1)  # (batch, d_llm)

        # Commit probability
        commit_logit = self.gate_mlp(pooled)  # (batch, 1)
        commit_prob = torch.sigmoid(commit_logit)  # (batch, 1)

        # Binary decision for inference
        should_commit = (commit_prob > self.threshold).float()

        # L_gate: encourage committing at the right time
        # Normalized position in the utterance [0, 1]
        position = timestep / max(total_timesteps - 1, 1)

        # Latency penalty: grows with time — penalizes NOT committing late
        # If commit_prob is low (not committing) and position is high (late),
        # this penalty is large
        latency_penalty = self.latency_weight * position * (1.0 - commit_prob).mean()

        # The gate loss encourages the gate to produce high commit_prob
        # when there's enough info (later timesteps) and low commit_prob
        # when there isn't (early timesteps).
        # Full training requires the task loss to backprop through this:
        # if committing early produces bad task loss, the gate learns to wait.
        gate_loss = latency_penalty

        return {
            "commit_prob": commit_prob,
            "should_commit": should_commit,
            "gate_loss": gate_loss,
        }
