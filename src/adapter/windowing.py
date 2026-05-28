"""
Overlapping windows for the streaming adapter stack.

Slice **raw mono waveform** into fixed-time chunks, then run Whisper encode **per chunk**
(:class:`AudioWaveformWindowizer`). Chunks shorter than one full window are skipped (no padding).
"""

from __future__ import annotations

import torch


class AudioWaveformWindowizer:
    """Convert raw mono waveform into overlapping audio windows for Whisper encoding.

    Default geometry: 0.8s window, 0.4s stride @ 16 kHz → 12 800 samples / 6 400 hop.

    Input:
        waveform: 1D ``(samples,)`` or ``(1, samples)``

    Output:
        List of 1D tensors, each ``(window_samples,)`` on CPU float32.
        Returns ``[]`` when audio is shorter than one window (no zero-padding).
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        window_seconds: float = 0.8,
        stride_seconds: float = 0.4,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.window_seconds = float(window_seconds)
        self.stride_seconds = float(stride_seconds)
        self.window_samples = max(1, int(round(self.window_seconds * self.sample_rate)))
        self.stride_samples = max(1, int(round(self.stride_seconds * self.sample_rate)))

    def __call__(self, waveform: torch.Tensor) -> list[torch.Tensor]:
        if waveform.ndim == 2:
            if waveform.shape[0] != 1:
                raise ValueError(
                    f"Expected mono waveform (1, samples) or (samples,), got {tuple(waveform.shape)}"
                )
            wave = waveform[0]
        elif waveform.ndim == 1:
            wave = waveform
        else:
            raise ValueError(f"Expected 1D or (1, samples) waveform, got {tuple(waveform.shape)}")

        wave = wave.detach().float().cpu().reshape(-1)
        if wave.numel() < self.window_samples:
            return []

        if wave.numel() == self.window_samples:
            return [wave.clone()]

        unfolded = wave.unfold(0, self.window_samples, self.stride_samples)
        return [unfolded[i].clone() for i in range(unfolded.shape[0])]


def stack_encoder_windows(
    enc_windows: list[torch.Tensor],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Stack per-chunk encoder outputs to ``(1, N, T_max, D)`` for batching.

    Shorter sequences leave trailing positions at zero (layout convenience only —
    not Whisper 30s padding).

    Each element of ``enc_windows`` should be ``(1, T_i, D)``.
    """
    if not enc_windows:
        raise ValueError("enc_windows must be non-empty")

    dev = torch.device(device)
    max_t = max(int(e.shape[1]) for e in enc_windows)
    d = int(enc_windows[0].shape[2])
    n = len(enc_windows)
    stacked = torch.zeros(1, n, max_t, d, device=dev, dtype=dtype)
    for i, enc in enumerate(enc_windows):
        t_i = int(enc.shape[1])
        stacked[0, i, :t_i] = enc[0].to(device=dev, dtype=dtype)
    return stacked
