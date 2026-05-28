"""Smoke tests for the streaming adapter components."""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adapter import (
    QFormerLayer,
    StabilityBuffer,
    AdaptiveRateController,
    TurnEndCommitGate,
    SilenceTracker,
    LearnedSilenceHead,
    LearnedSilenceTracker,
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
    gate = TurnEndCommitGate(
        d_llm=2560, hidden_dim=256, require_silence_for_commit=False
    )

    # Simulate accumulated tokens at timestep 3 of 5
    accumulated = torch.randn(2, 12, 2560)  # 3 windows * 4 tokens
    result = gate(accumulated, timestep=3, total_timesteps=5)

    assert result["commit_prob"].shape == (2, 1)
    assert result["should_commit"].shape == (2, 1)
    assert (result["commit_prob"] >= 0).all() and (result["commit_prob"] <= 1).all()
    assert "bce_loss" in result
    print(f"[PASS] TurnEndCommitGate: commit_prob={result['commit_prob'][:, 0].tolist()}, "
          f"gate_loss={result['gate_loss'].item():.4f}")


def test_silence_tracker_and_combined_gate():
    tracker = SilenceTracker(min_silence_ms=200.0, token_activity_threshold=5.0)
    speech_tokens = torch.randn(1, 4, 2560) * 2.0
    silence_tokens = torch.randn(1, 4, 2560) * 0.01

    tracker.update_from_window_tokens(speech_tokens)
    assert tracker.is_speech
    assert not tracker.silence_ready

    tracker.update_from_window_tokens(silence_tokens)
    assert not tracker.is_speech

    tracker.update_from_window_tokens(silence_tokens)
    assert tracker.silence_ready

    gate = TurnEndCommitGate(d_llm=2560, threshold=0.0, require_silence_for_commit=True)
    accumulated = torch.randn(1, 8, 2560)
    during_speech = gate(
        accumulated, timestep=0, total_timesteps=3,
        silence_tracker=tracker, window_tokens=speech_tokens,
    )
    assert during_speech["should_commit"].item() == 0.0

    after_silence = gate(
        accumulated, timestep=2, total_timesteps=3,
        silence_tracker=tracker, window_tokens=silence_tokens,
    )
    assert after_silence["silence_ready"].item() == 1.0
    assert after_silence["should_commit"].item() == 1.0
    print("[PASS] SilenceTracker (rule-based) + combined gate gating")


def test_learned_silence_head_and_gate():
    gate = TurnEndCommitGate(
        d_llm=2560,
        silence_mode="learned",
        require_silence_for_commit=False,
    )
    tracker = gate.make_learned_silence_tracker()
    window_tokens = torch.randn(1, 4, 2560, requires_grad=True)
    accumulated = torch.randn(1, 8, 2560)

    result = gate(
        accumulated,
        timestep=1,
        total_timesteps=3,
        learned_silence_tracker=tracker,
        window_tokens=window_tokens,
    )
    assert result["commit_prob"].shape == (1, 1)
    loss = result["gate_loss"]
    loss.backward()
    assert window_tokens.grad is not None
    assert window_tokens.grad.abs().sum().item() > 0
    print("[PASS] LearnedSilenceHead receives gradients through gate loss")


def test_gate_silence_mode_both():
    gate = TurnEndCommitGate(
        d_llm=2560,
        silence_mode="both",
        threshold=0.0,
        require_silence_for_commit=True,
        token_activity_threshold=5.0,
    )
    rule_tracker = gate.make_silence_tracker()
    learned_tracker = gate.make_learned_silence_tracker()

    speech_tokens = torch.randn(1, 4, 2560) * 2.0
    silence_tokens = torch.randn(1, 4, 2560) * 0.01
    accumulated = torch.randn(1, 8, 2560)

    for _ in range(2):
        rule_tracker.update_from_window_tokens(speech_tokens)
        learned_tracker.update_from_window_tokens(speech_tokens)
    rule_tracker.update_from_window_tokens(silence_tokens)
    learned_tracker.update_from_window_tokens(silence_tokens)
    rule_tracker.update_from_window_tokens(silence_tokens)
    learned_tracker.update_from_window_tokens(silence_tokens)

    result = gate(
        accumulated,
        timestep=2,
        total_timesteps=3,
        silence_tracker=rule_tracker,
        learned_silence_tracker=learned_tracker,
        window_tokens=silence_tokens,
    )
    assert "commit_prob_rule" in result
    assert "commit_prob_learned" in result
    assert "gate_loss_rule" in result
    assert "gate_loss_learned" in result
    assert abs(
        result["gate_loss"].item()
        - (result["gate_loss_rule"] + result["gate_loss_learned"]).item()
    ) < 1e-5
    print(
        "[PASS] both silence modes: "
        f"rule={result['commit_prob_rule'].item():.3f}, "
        f"learned={result['commit_prob_learned'].item():.3f}"
    )


def test_early_commit_gate_alias():
    """EarlyCommitGate is a backward-compatible alias for TurnEndCommitGate."""
    gate = EarlyCommitGate(d_llm=2560)
    accumulated = torch.randn(1, 4, 2560)
    result = gate(accumulated, timestep=0, total_timesteps=3)
    assert result["commit_prob"].shape == (1, 1)
    print("[PASS] EarlyCommitGate alias works")


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
    # Component 3: Turn-End Commit Gate
    gate = TurnEndCommitGate(d_llm=2560, require_silence_for_commit=False)

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


def test_audio_waveform_windowizer():
    sr = 16000
    windowizer = __import__("adapter.windowing", fromlist=["AudioWaveformWindowizer"]).AudioWaveformWindowizer(
        sample_rate=sr, window_seconds=0.8, stride_seconds=0.4
    )
    # 2.4s audio → windows at 0, 0.4, 0.8, 1.2, 1.6s → 5 windows
    wave = torch.randn(int(2.4 * sr))
    chunks = windowizer(wave)
    assert len(chunks) == 5
    assert all(c.shape == (int(0.8 * sr),) for c in chunks)
    short = torch.randn(int(0.5 * sr))
    assert windowizer(short) == []
    print(f"[PASS] AudioWaveformWindowizer: {len(chunks)} windows from 2.0s audio")


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

    # Turn-end commit gate
    gate = TurnEndCommitGate(d_llm=2560, require_silence_for_commit=False)
    gate_params = sum(p.numel() for p in gate.parameters())
    print(f"[INFO] Turn-End Commit Gate: {gate_params:,} params ({gate_params/1e6:.2f}M)")


if __name__ == "__main__":
    test_qformer_layer()
    test_cross_layer_in_between_pattern()
    test_stability_buffer()
    test_rate_controller()
    test_early_commit_gate()
    test_silence_tracker_and_combined_gate()
    test_learned_silence_head_and_gate()
    test_gate_silence_mode_both()
    test_audio_waveform_windowizer()
    test_early_commit_gate_alias()
    test_adapter_without_rate_controller()
    test_adapter_with_rate_controller()
    test_adapter_full_sequence()
    test_four_component_pipeline()
    test_parameter_count()
    print("\nAll tests passed!")
