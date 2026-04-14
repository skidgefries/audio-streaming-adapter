"""
Overlapping frame windows over Whisper encoder sequences (time axis).
"""

from __future__ import annotations

import torch


class WhisperFrameWindowizer:
    """Convert Whisper encoder frame sequences into overlapping windows.

    Input:
        enc: (B, T, D)

    Output:
        windows: (B, N, W, D) where N is the number of windows, W is frames per window.
    """

    def __init__(
        self,
        *,
        chunk_seconds: float = 30.0,
        window_seconds: float = 0.8,
        stride_seconds: float = 0.4,
    ) -> None:
        self.chunk_seconds = float(chunk_seconds)
        self.window_seconds = float(window_seconds)
        self.stride_seconds = float(stride_seconds)

    def _fps(self, *, num_frames: int) -> float:
        return float(num_frames) / self.chunk_seconds

    @staticmethod
    def _seconds_to_frames(*, seconds: float, fps: float) -> int:
        return max(1, int(round(float(seconds) * float(fps))))

    def __call__(self, enc: torch.Tensor) -> torch.Tensor:
        if enc.ndim != 3:
            raise ValueError(f"Expected enc with shape (B, T, D), got {tuple(enc.shape)}")

        _b, t, _d = enc.shape
        fps = self._fps(num_frames=t)

        win_frames = self._seconds_to_frames(seconds=self.window_seconds, fps=fps)
        hop_frames = self._seconds_to_frames(seconds=self.stride_seconds, fps=fps)

        if t < win_frames:
            raise ValueError(
                f"Encoder sequence too short for one window: T={t} < win_frames={win_frames}. "
                "If you expect shorter-than-window inputs, add a padding policy."
            )

        windows = enc.unfold(dimension=1, size=win_frames, step=hop_frames)
        return windows.permute(0, 1, 3, 2)
