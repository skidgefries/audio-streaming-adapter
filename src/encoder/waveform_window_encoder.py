"""
Per-window Whisper encoding for the adapter stack.

Flow: raw waveform → :class:`adapter.windowing.AudioWaveformWindowizer` (0.8s / 0.4s)
→ **one Whisper encode per audio chunk** → ``(1, 1500, D_enc)`` adapter inputs (30 s padded canvas).

This matches true streaming: each adapter step only sees the encoder output for its
own audio window, not a slice of a full-utterance encode.
"""

from __future__ import annotations

import torch

from adapter.windowing import AudioWaveformWindowizer

from .whisper_encoder import encode_waveform_to_hidden, load_whisper_models


class WhisperWindowFeatureExtractor:
    """
    Overlapping **raw-audio** windows, then Whisper encode each chunk for adapter steps.

    Default time geometry: 0.8s windows, 0.4s stride @ 16 kHz.
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

        wm = load_whisper_models(model_id=model_id, device=device, torch_dtype=torch_dtype)
        self.processor = wm.processor
        self.whisper = wm.model

        self._audio_windowizer = AudioWaveformWindowizer(
            sample_rate=self.sample_rate,
            window_seconds=self.window_seconds,
            stride_seconds=self.stride_seconds,
        )

    def waveform_to_windows(self, waveform_16k_mono: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            waveform_16k_mono: 1D CPU or CUDA tensor (audio samples at ``sample_rate``)

        Returns:
            List of encoder windows, each ``(1, 1500, D_enc)`` on ``device``.
        """
        if isinstance(waveform_16k_mono, torch.Tensor):
            wave = waveform_16k_mono.detach().float().cpu()
        else:
            wave = torch.tensor(waveform_16k_mono, dtype=torch.float32)

        if wave.numel() == 0:
            return []

        dev = torch.device(self.device)
        audio_chunks = self._audio_windowizer(wave)
        enc_windows: list[torch.Tensor] = []
        for chunk in audio_chunks:
            enc = encode_waveform_to_hidden(
                chunk,
                whisper_processor=self.processor,
                whisper_model=self.whisper,
                device=self.device,
                torch_dtype=self.torch_dtype,
                sample_rate=self.sample_rate,
            ).to(device=dev, dtype=self.torch_dtype)
            enc_windows.append(enc.contiguous())
        return enc_windows
