"""
Streaming Adapter Network (Component 2 of 4).

The 4 components from the research:
  1. Frozen Audio Encoder (Whisper) — not in this module
  2. Streaming Adapter Network (THIS MODULE, trainable)
  3. Early-Commit Gate — separate module (early_commit_gate.py)
  4. Frozen LLM — not in this module

Pipeline per window:
    Whisper frames F ∈ R^{T × D_enc}
        → [Q-Former layers (self-attn + cross-attn + FFN)]
        → [Optional: AdaptiveRateController]
        → [Output projection to LLM dim]
        → [StabilityBuffer (EMA smoothing)]
        → tokens ready for LLM

Chunked streaming: 0.8s window / 0.4s stride (50% overlap)
Target: 1-3 tokens/sec (2 min ≈ 240 tokens vs. 3000+ frames)

When cross_layer_in_between > 0, cross-attention is placed at the end of each
period block (not on layer 0), so the first layer is self-attention-only.
"""

import torch
import torch.nn as nn

from .cross_attention import QFormerLayer
from .stability_buffer import StabilityBuffer
from .rate_controller import AdaptiveRateController


class StreamingAdapter(nn.Module):
    """
    Component 2: Trainable streaming adapter network.

    Q-Former style cross-attention resampler with stability buffer
    and optional adaptive token rate controller.

    Args:
        d_encoder: Whisper encoder output dimension (1024 for whisper-medium).
        d_llm: Target LLM embedding dimension.
        num_queries: Maximum number of compressed tokens per window (m=1-4).
        num_layers: Number of stacked Q-Former layers.
        num_heads: Number of attention heads per Q-Former layer.
        d_ffn: FFN hidden dimension in Q-Former layers.
        dropout: Dropout rate.
        ema_alpha: EMA smoothing factor for stability buffer.
        learnable_ema: Whether EMA alpha is trainable.
        use_rate_controller: Whether to use adaptive token rate control.
                             Set False for initial experiments (fixed m tokens/window).
        rate_threshold: Hard gate threshold for inference (if rate controller enabled).
        target_rate: Target average tokens per window (if rate controller enabled).
        cross_layer_in_between: Number of self-attention-only layers (self + FFN, no
            cross-attention to audio) between successive cross-attention layers.
            0 means every layer includes cross-attention (full Q-Former stack).
            For K > 0, let P = K + 1. Cross-attention runs at the *end* of each block of
            P layers: layer index i uses cross-attention iff i % P == P - 1 (e.g. K=1 →
            cross on layers 1, 3, 5, … and self-only on 0, 2, 4, … so the stack does not
            start with cross-attention).
    """

    def __init__(
        self,
        d_encoder: int = 1024,
        d_llm: int = 2560,
        num_queries: int = 4,
        num_layers: int = 2,
        num_heads: int = 4,
        d_ffn: int = 2048,
        dropout: float = 0.1,
        ema_alpha: float = 0.8,
        learnable_ema: bool = False,
        use_rate_controller: bool = False,
        rate_threshold: float = 0.5,
        target_rate: float = 2.0,
        cross_layer_in_between: int = 1,
    ):
        super().__init__()
        if cross_layer_in_between < 0:
            raise ValueError("cross_layer_in_between must be >= 0")
        self.d_encoder = d_encoder
        self.d_llm = d_llm
        self.num_queries = num_queries
        self.use_rate_controller = use_rate_controller
        self.cross_layer_in_between = cross_layer_in_between
        
        # Learnable query vectors Q ∈ R^{m × D_q}
        self.queries = nn.Parameter(torch.randn(1, num_queries, d_encoder) * 0.02)

        # # Stack of Q-Former layers (self-attn + cross-attn + FFN, BLIP-2 style)
        # self.layers = nn.ModuleList([
        #     QFormerLayer(
        #         d_model=d_encoder,
        #         num_heads=num_heads,
        #         d_ffn=d_ffn,
        #         dropout=dropout,
        #     )
        #     for _ in range(num_layers)
        # ])

        # Stacked layers: optional self-only layers between cross-attention layers.
        # period P = K+1: cross at i ≡ P-1 (mod P) — last slot in each block, so layer 0
        # is self-only when K>0 (e.g. K=1 → cross on 1,3,5,... not 0,2,4,...).
        period = cross_layer_in_between + 1
        self.layers = nn.ModuleList([
            QFormerLayer(
                d_model=d_encoder,
                num_heads=num_heads,
                d_ffn=d_ffn,
                dropout=dropout,
                use_cross_attention=(i % period == period - 1),
            )
            for i in range(num_layers)
        ])


        # Project from encoder space to LLM embedding space
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_encoder),
            nn.Linear(d_encoder, d_llm),
        )

        # Optional: Adaptive rate controller
        self.rate_controller = None
        if use_rate_controller:
            self.rate_controller = AdaptiveRateController(
                d_encoder=d_encoder,
                m_max=num_queries,
                threshold=rate_threshold,
                target_rate=target_rate,
            )

        # Stability buffer for temporal consistency
        self.stability_buffer = StabilityBuffer(
            alpha=ema_alpha,
            learnable=learnable_ema,
        )

    def reset_streaming_state(self):
        """Reset buffer state. Call at the start of each new audio stream."""
        self.stability_buffer.reset()

    def forward_window(
        self,
        encoder_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Process a single audio window (streaming inference mode).

        Args:
            encoder_features: (batch, T, d_encoder) -- Whisper encoder output
                              for one overlapping window (0.8s of audio)

        Returns:
            dict with:
                tokens: (batch, m, d_llm) -- compressed, smoothed tokens
                stability_loss: scalar -- L_stability (temporal consistency)
                gate_scores: (batch, m) -- **rate-controller** gate values only
                    (None if no rate controller). This is unrelated to
                    :class:`~adapter.early_commit_gate.EarlyCommitGate`; training
                    ``L_gate`` uses the early-commit gate, not these scores.
                sparse_loss: scalar -- L_sparse (None if no rate controller)
                rate_loss: scalar -- L_rate (None if no rate controller)
        """
        batch_size = encoder_features.shape[0]

        # Expand learnable queries to batch size
        q = self.queries.expand(batch_size, -1, -1)  # (batch, m, d_encoder)

        # Pass through Q-Former layers (self-attn → cross-attn → FFN per layer)
        for layer in self.layers:
            q = layer(q, encoder_features)

        # Optional: adaptive rate control
        gate_scores = None
        sparse_loss = None
        rate_loss = None
        if self.rate_controller is not None:
            rc_result = self.rate_controller(encoder_features, q)
            q = rc_result["tokens"]
            gate_scores = rc_result["gate_scores"]
            sparse_loss = rc_result["sparse_loss"]
            rate_loss = rc_result["rate_loss"]

        # Project to LLM dimension
        z = self.output_proj(q)  # (batch, m, d_llm)

        # Temporal smoothing via stability buffer
        z_smooth, stability_loss = self.stability_buffer(z)

        return {
            "tokens": z_smooth,
            "stability_loss": stability_loss,
            "gate_scores": gate_scores,
            "sparse_loss": sparse_loss,
            "rate_loss": rate_loss,
        }

    def forward(
        self,
        encoder_features_sequence: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Process a full sequence of windows (training mode).

        Args:
            encoder_features_sequence: List of (batch, T, d_encoder) tensors,
                one per overlapping window from a complete utterance.

        Returns:
            dict with:
                tokens: (batch, total_tokens, d_llm) -- all tokens concatenated
                stability_loss: scalar -- total L_stability across all windows
                gate_scores: (batch, num_windows, m) or None
                sparse_loss: scalar or None -- total L_sparse
                rate_loss: scalar or None -- total L_rate
        """
        self.reset_streaming_state()
        device = encoder_features_sequence[0].device
        dtype = encoder_features_sequence[0].dtype

        all_tokens = []
        all_gates = []
        total_stability = torch.tensor(0.0, device=device, dtype=dtype)
        total_sparse = torch.tensor(0.0, device=device, dtype=dtype)
        total_rate = torch.tensor(0.0, device=device, dtype=dtype)

        for window_features in encoder_features_sequence:
            result = self.forward_window(window_features)
            all_tokens.append(result["tokens"])
            total_stability = total_stability + result["stability_loss"]

            if result["gate_scores"] is not None:
                all_gates.append(result["gate_scores"])
                total_sparse = total_sparse + result["sparse_loss"]
                total_rate = total_rate + result["rate_loss"]

        tokens = torch.cat(all_tokens, dim=1)  # (batch, num_windows * m, d_llm)

        gate_scores = None
        sparse_loss = None
        rate_loss = None
        if all_gates:
            gate_scores = torch.stack(all_gates, dim=1)  # (batch, num_windows, m)
            sparse_loss = total_sparse
            rate_loss = total_rate

        return {
            "tokens": tokens,
            "stability_loss": total_stability,
            "gate_scores": gate_scores,
            "sparse_loss": sparse_loss,
            "rate_loss": rate_loss,
        }
