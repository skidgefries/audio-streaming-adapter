"""
Stability Buffer for temporal smoothing across streaming windows.

Uses Exponential Moving Average (EMA) to smooth token representations
from adjacent overlapping windows, preventing jittery outputs that
would confuse the downstream LLM.

During training, also provides L_stability = sum_t ||Z_t - Z_{t-1}||^2
as an auxiliary loss signal.
"""

import torch
import torch.nn as nn


class StabilityBuffer(nn.Module):
    """
    EMA-based temporal smoothing for streaming token sequences.

    For each new window's tokens Z_t, the smoothed output is:
        Z'_t = alpha * Z_t + (1 - alpha) * Z'_{t-1}

    Args:
        alpha: EMA smoothing factor in (0, 1]. Higher = more weight on current window.
               0.8 means current window contributes 80%, history contributes 20%.
        learnable: If True, alpha is a trainable parameter (initialized to init value).
    """

    def __init__(self, alpha: float = 0.8, learnable: bool = False):
        super().__init__()
        if learnable:
            # Store in logit space so sigmoid keeps it in (0, 1)
            alpha_logit = torch.log(torch.tensor(alpha / (1.0 - alpha)))
            self._alpha_logit = nn.Parameter(alpha_logit)
        else:
            self.register_buffer("_alpha", torch.tensor(alpha))

        self.learnable = learnable
        # Buffer state for streaming inference (not a parameter, not saved in state_dict)
        self._prev_tokens: torch.Tensor | None = None

    @property
    def alpha(self) -> torch.Tensor:
        if self.learnable:
            return torch.sigmoid(self._alpha_logit)
        return self._alpha

    def reset(self):
        """Reset buffer state. Call at the start of each new audio stream."""
        self._prev_tokens = None

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply EMA smoothing and compute stability loss.

        Args:
            tokens: (batch, m, d) — adapter output for current window

        Returns:
            smoothed: (batch, m, d) — EMA-smoothed tokens
            stability_loss: scalar — ||Z_t - Z_{t-1}||^2, zero for first window
        """
        if self._prev_tokens is None:
            # First window: no history to smooth against
            self._prev_tokens = tokens.detach()
            zero_loss = torch.tensor(0.0, device=tokens.device, dtype=tokens.dtype)
            return tokens, zero_loss

        # Stability loss: penalize large jumps between adjacent windows
        stability_loss = torch.mean((tokens - self._prev_tokens) ** 2)

        # EMA smoothing
        alpha = self.alpha
        smoothed = alpha * tokens + (1.0 - alpha) * self._prev_tokens

        # Update buffer (detach to prevent backprop through time)
        self._prev_tokens = smoothed.detach()

        return smoothed, stability_loss

    def forward_sequence(
        self,
        token_sequence: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Process a full sequence of windows (for training on complete utterances).

        Args:
            token_sequence: List of (batch, m, d) tensors, one per window

        Returns:
            smoothed_sequence: List of smoothed tensors
            total_stability_loss: Sum of per-step stability losses
        """
        self.reset()
        smoothed = []
        total_loss = torch.tensor(0.0, device=token_sequence[0].device)

        for tokens in token_sequence:
            s, loss = self.forward(tokens)
            smoothed.append(s)
            total_loss = total_loss + loss

        return smoothed, total_loss
