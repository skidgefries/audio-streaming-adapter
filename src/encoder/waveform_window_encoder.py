"""
Overlapping windows for the adapter stack, using :class:`adapter.windowing.WhisperFrameWindowizer`.

Flow: **one** Whisper encode for the full utterance → slice encoder frames into
``(1, W, D)`` windows (default 0.8s / 0.4s in **time**, mapped to frame counts via
``chunk_seconds`` = utterance length).

This shares window math with :class:`adapter_llm_pipeline.WhisperAdapterLLMPipeline`
(full encode + frame windowing). It is **not** the same as running Whisper separately
on each raw waveform slice.
"""

from __future__ import annotations

import torch

from adapter.windowing import WhisperFrameWindowizer

from .whisper_encoder import encode_waveform_to_hidden, load_whisper_models


class WhisperWindowFeatureExtractor:
    """
    Full-utterance Whisper encode, then overlapping **frame** windows for each adapter step.

    Default time geometry matches the research spec: 0.8s windows, 0.4s stride (see
    :class:`WhisperFrameWindowizer`). ``chunk_seconds`` is set to the utterance duration
    so frames-per-second matches that clip (``fps = T / duration``).
    """

    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        torch_dtype: torch.dtype,
        window_seconds: float = 0.8,
        stride_seconds: float = 0.4,
        sample_rate: int = 16000,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.window_seconds = float(window_seconds)
        self.stride_seconds = float(stride_seconds)
        self.sample_rate = int(sample_rate)
        self.chunk_seconds_override = None

        wm = load_whisper_models(model_id=model_id, device=device, torch_dtype=torch_dtype)
        self.processor = wm.processor
        self.whisper = wm.model

    def waveform_to_windows(self, waveform_16k_mono: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            waveform_16k_mono: 1D CPU or CUDA tensor (audio samples at ``sample_rate``)

        Returns:
            List of encoder windows, each ``(1, W, D_enc)`` on ``device``.
        """
        if isinstance(waveform_16k_mono, torch.Tensor):
            wave = waveform_16k_mono.detach().float().cpu()
        else:
            wave = torch.tensor(waveform_16k_mono, dtype=torch.float32)

        if wave.numel() == 0:
            return []

        duration_s = float(wave.numel()) / float(self.sample_rate)
        chunk_s = self.chunk_seconds_override if self.chunk_seconds_override is not None else duration_s
        if chunk_s <= 0:
            return []

        dev = torch.device(self.device)
        enc = encode_waveform_to_hidden(
            wave,
            whisper_processor=self.processor,
            whisper_model=self.whisper,
            device=self.device,
            torch_dtype=self.torch_dtype,
            sample_rate=self.sample_rate,
        ).to(device=dev, dtype=self.torch_dtype)

        windowizer = WhisperFrameWindowizer(
            window_seconds=self.window_seconds,
            stride_seconds=self.stride_seconds,
        )
        try:
            windows = windowizer(enc)
        except ValueError:
            return []

        n = windows.shape[1]
        return [
            windows[0, i].unsqueeze(0).to(device=dev, dtype=self.torch_dtype).contiguous()
            for i in range(n)
        ]
