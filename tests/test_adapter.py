"""Smoke tests for the streaming adapter components."""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adapter import (
    QFormerLayer,
    StabilityBuffer,
    AdaptiveRateController,
    EarlyCommitGate,
    StreamingAdapter,
)


def test_qformer_layer():
    layer = QFormerLayer(d_model=1024, num_heads=4, d_ffn=2048)
    queries = torch.randn(2, 4, 1024)        # batch=2, m=4 queries
    encoder_out = torch.randn(2, 100, 1024)   # batch=2, T=100 frames

    z = layer(queries, encoder_out)
    assert z.shape == (2, 4, 1024), f"Expected (2, 4, 1024), got {z.shape}"
    print(f"[PASS] QFormerLayer: {z.shape}")


def test_stability_buffer():
    buf = StabilityBuffer(alpha=0.8)

    t1 = torch.randn(2, 4, 2560)
    t2 = torch.randn(2, 4, 2560)

    s1, loss1 = buf(t1)
    assert loss1.item() == 0.0, "First window should have zero loss"
    assert torch.equal(s1, t1), "First window should pass through unchanged"

    s2, loss2 = buf(t2)
    assert loss2.item() > 0.0, "Second window should have non-zero loss"
    expected = 0.8 * t2 + 0.2 * t1
    assert torch.allclose(s2, expected, atol=1e-6), "EMA not applied correctly"
    print(f"[PASS] StabilityBuffer: loss1={loss1.item():.4f}, loss2={loss2.item():.4f}")


def test_rate_controller():
    rc = AdaptiveRateController(d_encoder=1024, m_max=4, target_rate=2.0)

    encoder_out = torch.randn(2, 100, 1024)
    tokens = torch.randn(2, 4, 1024)

    result = rc(encoder_out, tokens)
    assert result["tokens"].shape == (2, 4, 1024)
    assert result["gate_scores"].shape == (2, 4)
    assert result["sparse_loss"].item() >= 0
    assert result["rate_loss"].item() >= 0
    scores = result["gate_scores"]
    assert (scores >= 0).all() and (scores <= 1).all(), "Scores should be in [0, 1]"
    print(f"[PASS] AdaptiveRateController: sparse_loss={result['sparse_loss'].item():.4f}, "
          f"rate_loss={result['rate_loss'].item():.4f}")


def test_early_commit_gate():
    gate = EarlyCommitGate(d_llm=2560, hidden_dim=256)

    # Simulate accumulated tokens at timestep 3 of 5
    accumulated = torch.randn(2, 12, 2560)  # 3 windows * 4 tokens
    result = gate(accumulated, timestep=3, total_timesteps=5)

    assert result["commit_prob"].shape == (2, 1)
    assert result["should_commit"].shape == (2, 1)
    assert (result["commit_prob"] >= 0).all() and (result["commit_prob"] <= 1).all()
    print(f"[PASS] EarlyCommitGate: commit_prob={result['commit_prob'][:, 0].tolist()}, "
          f"gate_loss={result['gate_loss'].item():.4f}")


def test_cross_layer_in_between_pattern():
    """With K>0, first layer is self-only; cross runs at 1,3,… for K=1."""
    adapter = StreamingAdapter(
        d_encoder=1024,
        d_llm=2560,
        num_queries=4,
        num_layers=4,
        cross_layer_in_between=1,
        use_rate_controller=False,
    )
    period = adapter.cross_layer_in_between + 1
    for i, layer in enumerate(adapter.layers):
        assert layer.use_cross_attention == (i % period == period - 1)


def test_adapter_without_rate_controller():
    """Component 2 without optional rate controller (default)."""
    adapter = StreamingAdapter(
        d_encoder=1024, d_llm=2560, num_queries=4, num_layers=2,
        use_rate_controller=False,
    )

    encoder_out = torch.randn(2, 40, 1024)  # ~0.8s window → 40 frames
    result = adapter.forward_window(encoder_out)

    assert result["tokens"].shape == (2, 4, 2560)
    assert result["gate_scores"] is None, "No rate controller → no gate scores"
    assert result["sparse_loss"] is None
    assert result["rate_loss"] is None
    print(f"[PASS] Adapter (no rate ctrl): tokens={result['tokens'].shape}")


def test_adapter_with_rate_controller():
    """Component 2 with optional rate controller enabled."""
    adapter = StreamingAdapter(
        d_encoder=1024, d_llm=2560, num_queries=4, num_layers=2,
        use_rate_controller=True, target_rate=2.0,
    )

    encoder_out = torch.randn(2, 40, 1024)
    result = adapter.forward_window(encoder_out)

    assert result["tokens"].shape == (2, 4, 2560)
    assert result["gate_scores"] is not None
    assert result["sparse_loss"] is not None
    assert result["rate_loss"] is not None
    print(f"[PASS] Adapter (with rate ctrl): sparse={result['sparse_loss'].item():.4f}, "
          f"rate={result['rate_loss'].item():.4f}")


def test_adapter_full_sequence():
    """Process a full utterance as sequence of overlapping windows."""
    adapter = StreamingAdapter(
        d_encoder=1024, d_llm=2560, num_queries=4, num_layers=2,
        use_rate_controller=False,
    )

    # 5 overlapping windows (0.8s window, 0.4s stride) from ~2.8s utterance
    windows = [torch.randn(2, 40, 1024) for _ in range(5)]
    result = adapter(windows)

    assert result["tokens"].shape == (2, 20, 2560)  # 5 windows * 4 queries
    assert result["stability_loss"].item() > 0
    assert result["gate_scores"] is None  # no rate controller
    print(f"[PASS] Adapter sequence: tokens={result['tokens'].shape}, "
          f"stability_loss={result['stability_loss'].item():.4f}")


def test_four_component_pipeline():
    """Test all 4 components together (minus actual Whisper and LLM)."""
    # Component 2: Streaming Adapter
    adapter = StreamingAdapter(
        d_encoder=1024, d_llm=2560, num_queries=4, num_layers=2,
        use_rate_controller=True, target_rate=2.0,
    )
    # Component 3: Early-Commit Gate
    gate = EarlyCommitGate(d_llm=2560)

    # Simulate streaming: process windows one by one
    adapter.reset_streaming_state()
    accumulated_tokens = []
    num_windows = 4

    for t in range(num_windows):
        # Component 1 output (simulated Whisper encoder)
        encoder_out = torch.randn(1, 40, 1024)

        # Component 2: compress
        result = adapter.forward_window(encoder_out)
        accumulated_tokens.append(result["tokens"])

        # Component 3: should we start generating?
        all_tokens = torch.cat(accumulated_tokens, dim=1)
        gate_result = gate(all_tokens, timestep=t, total_timesteps=num_windows)

        commit = gate_result["should_commit"].item()
        prob = gate_result["commit_prob"].item()
        print(f"  Window {t}: {result['tokens'].shape[1]} tokens, "
              f"commit_prob={prob:.3f}, commit={bool(commit)}")

    total_tokens = torch.cat(accumulated_tokens, dim=1)
    print(f"[PASS] 4-component pipeline: {total_tokens.shape[1]} total tokens "
          f"(would feed to Component 4: frozen LLM)")


def test_parameter_count():
    # Without rate controller (default)
    adapter = StreamingAdapter(
        d_encoder=1024, d_llm=2560, num_queries=4, num_layers=2,
        use_rate_controller=False,
    )
    total = sum(p.numel() for p in adapter.parameters())
    print(f"[INFO] Adapter (no rate ctrl): {total:,} params ({total/1e6:.1f}M)")

    # With rate controller
    adapter_rc = StreamingAdapter(
        d_encoder=1024, d_llm=2560, num_queries=4, num_layers=2,
        use_rate_controller=True,
    )
    total_rc = sum(p.numel() for p in adapter_rc.parameters())
    print(f"[INFO] Adapter (with rate ctrl): {total_rc:,} params ({total_rc/1e6:.1f}M)")

    # Early-commit gate
    gate = EarlyCommitGate(d_llm=2560)
    gate_params = sum(p.numel() for p in gate.parameters())
    print(f"[INFO] Early-Commit Gate: {gate_params:,} params ({gate_params/1e6:.2f}M)")


if __name__ == "__main__":
    test_qformer_layer()
    test_cross_layer_in_between_pattern()
    test_stability_buffer()
    test_rate_controller()
    test_early_commit_gate()
    test_adapter_without_rate_controller()
    test_adapter_with_rate_controller()
    test_adapter_full_sequence()
    test_four_component_pipeline()
    test_parameter_count()
    print("\nAll tests passed!")
