"""
Q-Former Layer for the Streaming Adapter (BLIP-2 style).

Each layer has three sub-layers, matching the original Q-Former from
Salesforce's BLIP-2 (Li et al., 2023):

    1. Self-Attention:   queries attend to EACH OTHER
       → lets queries coordinate and avoid redundant extraction

    2. Cross-Attention:  queries attend to encoder features
       → extracts information from Whisper frames

    3. Feed-Forward:     per-token nonlinear transformation
       → enriches token representations

All three use pre-norm + residual connections.

Architecture per layer:
    Q ──→ [Self-Attn(Q, Q)] ──→ [Cross-Attn(Q, F)] ──→ [FFN] ──→ Z
           queries talk to         queries extract        refine
           each other              from audio frames      representations
"""

import torch
import torch.nn as nn
import math


class QFormerLayer(nn.Module):
    """
    Single Q-Former layer following BLIP-2 architecture.

    Sub-layer 1 — Self-Attention:
        Queries attend to each other. This is what makes Q-Former different
        from plain cross-attention. Without it, each query works independently
        and they may extract redundant information. With self-attention,
        Query 0 can see what Query 1 is capturing and focus elsewhere.

    Sub-layer 2 — Cross-Attention:
        Queries attend to encoder frames (Whisper output). This is where
        the actual information extraction happens.

    Sub-layer 3 — Feed-Forward Network:
        Standard transformer FFN for per-token nonlinear transformation.

    Args:
        d_model: Dimension of queries and output.
        num_heads: Number of attention heads (shared across self and cross attn).
        d_ffn: Hidden dimension of the feed-forward network.
        dropout: Dropout rate.
        use_cross_attention: If False, only self-attention + FFN (no audio cross-attn).
    """

    def __init__(
        self,
        d_model: int = 1024,
        num_heads: int = 4,
        d_ffn: int = 2048,
        dropout: float = 0.1,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = math.sqrt(self.d_head)
        self.use_cross_attention = use_cross_attention

        # ---- Sub-layer 1: Self-Attention (queries ↔ queries) ----
        self.self_attn_q = nn.Linear(d_model, d_model)
        self.self_attn_k = nn.Linear(d_model, d_model)
        self.self_attn_v = nn.Linear(d_model, d_model)
        self.self_attn_o = nn.Linear(d_model, d_model)
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn_dropout = nn.Dropout(dropout)

        # ---- Sub-layer 2: Cross-Attention (queries → encoder frames) ----
        if use_cross_attention:
            self.cross_attn_q = nn.Linear(d_model, d_model)
            self.cross_attn_k = nn.Linear(d_model, d_model)
            self.cross_attn_v = nn.Linear(d_model, d_model)
            self.cross_attn_o = nn.Linear(d_model, d_model)
            self.norm_cross_q = nn.LayerNorm(d_model)
            self.norm_cross_kv = nn.LayerNorm(d_model)
            self.cross_attn_dropout = nn.Dropout(dropout)

        # ---- Sub-layer 3: Feed-Forward Network ----
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def _multihead_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout: nn.Dropout,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Shared multi-head attention logic for both self and cross attention.

        Args:
            q: (batch, seq_q, d_model) — already projected queries
            k: (batch, seq_k, d_model) — already projected keys
            v: (batch, seq_k, d_model) — already projected values
            key_padding_mask: (batch, seq_k) bool, True = ignore key position
            dropout: dropout module for attention weights

        Returns:
            (batch, seq_q, d_model)
        """
        batch_size, seq_q, _ = q.shape
        seq_k = k.shape[1]

        q = q.view(batch_size, seq_q, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_k, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_k, self.num_heads, self.d_head).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                torch.finfo(attn_scores.dtype).min,
            )
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_q, self.d_model)

        return output

    def forward(
        self,
        queries: torch.Tensor,
        encoder_features: torch.Tensor,
        *,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            queries: (batch, m, d_model) — learnable query vectors
            encoder_features: (batch, T, d_model) — Whisper encoder output
            encoder_attention_mask: (batch, T) with 1 on real frames, 0 on padded tail

        Returns:
            (batch, m, d_model) — updated query representations
        """
        key_padding_mask = None
        if encoder_attention_mask is not None:
            key_padding_mask = encoder_attention_mask == 0

        q_norm = self.norm_self(queries)
        sa_out = self._multihead_attention(
            q=self.self_attn_q(q_norm),
            k=self.self_attn_k(q_norm),
            v=self.self_attn_v(q_norm),
            dropout=self.self_attn_dropout,
        )
        sa_out = self.self_attn_o(sa_out)
        queries = queries + sa_out

        if self.use_cross_attention:
            q_norm = self.norm_cross_q(queries)
            kv_norm = self.norm_cross_kv(encoder_features)
            ca_out = self._multihead_attention(
                q=self.cross_attn_q(q_norm),
                k=self.cross_attn_k(kv_norm),
                v=self.cross_attn_v(kv_norm),
                dropout=self.cross_attn_dropout,
                key_padding_mask=key_padding_mask,
            )
            ca_out = self.cross_attn_o(ca_out)
            queries = queries + ca_out

        queries = queries + self.ffn(self.norm_ffn(queries))

        return queries

# Keep backward-compatible name
CrossAttentionLayer = QFormerLayer
