"""
Adaptive Token Rate Controller (optional, part of Streaming Adapter).

Dynamically adjusts how many tokens to emit per window based on
audio complexity. Silent/simple segments get fewer tokens (1),
dense/noisy segments get more (up to m_max).

Uses soft gating during training (differentiable) and hard
thresholding during inference.

This is DIFFERENT from the Early-Commit Gate:
  - Rate Controller: "How many tokens for THIS window?" (token efficiency)
  - Early-Commit Gate: "Should the LLM START generating?" (latency/accuracy)

Associated losses:
  - L_sparse: Encourages using fewer tokens when possible (sparsity regularization)
  - L_rate: Penalizes exceeding target token rate R = m/t
"""

import torch
import torch.nn as nn


class AdaptiveRateController(nn.Module):
    """
    Produces per-query gate scores based on input complexity.

    Architecture:
        mean_pool(F) -> Linear -> ReLU -> Linear -> sigmoid -> gate per query

    During training: soft gates (multiply token embeddings by gate scores)
    During inference: hard gates (drop tokens with gate < threshold)

    Args:
        d_encoder: Dimension of encoder features (Whisper output dim).
        m_max: Maximum number of query tokens.
        hidden_dim: Hidden dimension of the gating MLP.
        threshold: Hard gate threshold for inference.
        target_rate: Target average number of active tokens per window.
                     Used for L_rate computation (e.g., 2.0 means aim for 2 tokens/window).
    """

    def __init__(
        self,
        d_encoder: int = 1024,
        m_max: int = 4,
        hidden_dim: int = 256,
        threshold: float = 0.5,
        target_rate: float = 2.0,
    ):
        super().__init__()
        self.m_max = m_max
        self.threshold = threshold
        self.target_rate = target_rate

        self.gate_mlp = nn.Sequential(
            nn.Linear(d_encoder, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, m_max),
        )

    def forward(
        self,
        encoder_features: torch.Tensor,
        tokens: torch.Tensor,
        *,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute gated tokens based on input complexity.

        Args:
            encoder_features: (batch, T, d_encoder) -- Whisper frame features
            tokens: (batch, m, d) -- adapter output tokens

        Returns:
            dict with:
                tokens: (batch, m, d) -- gated tokens
                gate_scores: (batch, m) -- gate values per query slot
                sparse_loss: scalar -- L_sparse (encourages fewer active tokens)
                rate_loss: scalar -- L_rate (penalizes deviation from target rate)
        """
        # Pool encoder features to get a single complexity vector (masked mean when provided)
        if encoder_attention_mask is not None:
            weights = encoder_attention_mask.unsqueeze(-1).to(
                dtype=encoder_features.dtype, device=encoder_features.device
            )
            pooled = (encoder_features * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        else:
            pooled = encoder_features.mean(dim=1)

        # Compute per-query gate scores
        gate_scores = torch.sigmoid(self.gate_mlp(pooled))  # (batch, m_max)

        if self.training:
            # Soft gating: multiply tokens by gate scores (keeps gradients flowing)
            gated = tokens * gate_scores.unsqueeze(-1)  # (batch, m, d)
        else:
            # Hard gating: zero out tokens below threshold
            mask = (gate_scores > self.threshold).unsqueeze(-1)  # (batch, m, 1)
            gated = tokens * mask.to(tokens.dtype)

        # L_sparse: encourage sparsity (L1 on gate scores)
        # Lower gate scores = fewer active tokens = more compression
        sparse_loss = gate_scores.mean()

        # L_rate: penalize deviation from target token rate
        # Effective token count = sum of gate scores (soft count)
        effective_count = gate_scores.sum(dim=-1)  # (batch,)
        rate_loss = torch.mean((effective_count - self.target_rate) ** 2)

        return {
            "tokens": gated,
            "gate_scores": gate_scores,
            "sparse_loss": sparse_loss,
            "rate_loss": rate_loss,
        }
