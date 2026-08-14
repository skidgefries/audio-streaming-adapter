from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from dataset.librispeech import load_mono_waveform_16k

from .config import WhisperConfig


@dataclass(frozen=True)
class WhisperModels:
    model_id: str
    processor: Any
    model: Any


def load_whisper_models(
    *,
    cfg: WhisperConfig | None = None,
    model_id: str = "openai/whisper-small",
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
) -> WhisperModels:
    if cfg is not None:
        model_id = cfg.model_id
        device = cfg.device
        torch_dtype = cfg.torch_dtype
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return WhisperModels(model_id=model_id, processor=processor, model=model)


_WHISPER_ENCODE_BATCH = 32


def _log_mel_gpu(
    waveforms: list[torch.Tensor] | torch.Tensor,
    *,
    feature_extractor: Any,
    device: torch.device,
) -> torch.Tensor:
    """Whisper log-mel on ``device`` (STFT stays on GPU). Shape ``(B, n_mels, 3000)``."""
    n_samples = int(feature_extractor.n_samples)
    if isinstance(waveforms, torch.Tensor):
        if waveforms.ndim == 1:
            items = [waveforms]
        elif waveforms.ndim == 2:
            items = [waveforms[i] for i in range(waveforms.shape[0])]
        else:
            raise ValueError(f"Expected 1D or 2D waveforms, got {tuple(waveforms.shape)}")
    else:
        items = list(waveforms)

    padded: list[torch.Tensor] = []
    for wave in items:
        w = wave.detach().float().reshape(-1)
        if w.device != device:
            w = w.to(device, non_blocking=True)
        if w.numel() > n_samples:
            w = w[:n_samples]
        if w.numel() < n_samples:
            w = torch.nn.functional.pad(w, (0, n_samples - w.numel()))
        padded.append(w)
    batch = torch.stack(padded, dim=0)

    window = torch.hann_window(feature_extractor.n_fft, device=device)
    stft = torch.stft(
        batch,
        feature_extractor.n_fft,
        feature_extractor.hop_length,
        window=window,
        return_complex=True,
    )
    magnitudes = stft[..., :-1].abs() ** 2
    mel_filters = torch.as_tensor(
        feature_extractor.mel_filters, device=device, dtype=torch.float32
    )
    mel_spec = mel_filters.T @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    max_val = log_spec.amax(dim=(1, 2), keepdim=True)
    log_spec = torch.maximum(log_spec, max_val - 8.0)
    return (log_spec + 4.0) / 4.0


def encode_waveforms_to_hidden(
    waveforms: list[torch.Tensor] | torch.Tensor,
    *,
    whisper_processor: Any,
    whisper_model: Any,
    device: str,
    torch_dtype: torch.dtype,
    sample_rate: int = 16000,
    encode_batch_size: int = _WHISPER_ENCODE_BATCH,
) -> torch.Tensor:
    """
    Encode one or more waveforms to Whisper hidden states ``(B, 1500, D)`` on ``device``.

    Log-mel uses GPU STFT when ``device`` is CUDA. Encoder runs in chunks of
    ``encode_batch_size`` to bound activation memory.
    """
    del sample_rate  # Whisper feature extractor is fixed at 16 kHz
    if isinstance(waveforms, torch.Tensor) and waveforms.ndim == 1:
        items = [waveforms]
    elif isinstance(waveforms, torch.Tensor):
        items = [waveforms[i] for i in range(waveforms.shape[0])]
    else:
        items = list(waveforms)
    if not items:
        raise ValueError("waveforms must be non-empty")

    target = next(whisper_model.parameters()).device
    feat_device = torch.device(device) if str(device) != "cpu" else target
    feature_extractor = whisper_processor.feature_extractor
    hidden_chunks: list[torch.Tensor] = []
    bs = max(1, int(encode_batch_size))
    for start in range(0, len(items), bs):
        chunk = items[start : start + bs]
        input_features = _log_mel_gpu(
            chunk, feature_extractor=feature_extractor, device=feat_device
        ).to(device=target, dtype=torch_dtype)
        with torch.no_grad():
            hidden = whisper_model.model.encoder(input_features).last_hidden_state
        if hidden.device != target or hidden.dtype != torch_dtype:
            hidden = hidden.to(device=target, dtype=torch_dtype)
        hidden_chunks.append(hidden)
    return torch.cat(hidden_chunks, dim=0)


def encode_waveform_to_hidden(
    waveform,
    *,
    whisper_processor: Any,
    whisper_model: Any,
    device: str,
    torch_dtype: torch.dtype,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """
    Encode a waveform into Whisper encoder hidden states ``(1, T, D)``.

    Whisper's encoder requires mel length 3000 (30 s). Shorter clips are zero-padded
    to 3000 mel frames; the encoder always returns **T=1500** downsampled frames.
    No post-encode trimming is applied.
    """
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform, dtype=torch.float32)
    return encode_waveforms_to_hidden(
        [waveform.detach().float().reshape(-1)],
        whisper_processor=whisper_processor,
        whisper_model=whisper_model,
        device=device,
        torch_dtype=torch_dtype,
        sample_rate=sample_rate,
    )



# --- File / dataset helpers (used by qwen_summarize_*.py demos) -----------------

_whisper_by_model_id: dict[str, WhisperModels] = {}


def _get_whisper_cached(*, model_id: str = "openai/whisper-small") -> WhisperModels:
    if model_id not in _whisper_by_model_id:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        if device == "cuda":
            torch.cuda.init()
        _whisper_by_model_id[model_id] = load_whisper_models(
            model_id=model_id, device=device, torch_dtype=torch_dtype
        )
    return _whisper_by_model_id[model_id]


def get_encoder_output(file_path: str, *, model_id: str = "openai/whisper-small") -> torch.Tensor:
    """
    Load an audio file and return Whisper encoder hidden states ``(1, T, D)`` (CPU float).

    Typical: ``T=1500`` frames (Whisper 30 s canvas), ``D=768`` for whisper-small.
    """
    waveform = load_mono_waveform_16k(file_path)
    wm = _get_whisper_cached(model_id=model_id)
    p = next(wm.model.parameters())
    enc = encode_waveform_to_hidden(
        waveform,
        whisper_processor=wm.processor,
        whisper_model=wm.model,
        device=str(p.device),
        torch_dtype=p.dtype,
    )
    return enc.detach().float().cpu()


def encode_dataset_stream(
    dataset_root: str,
    *,
    patterns: tuple[str, ...] = ("**/*.flac", "**/*.wav"),
    limit: int | None = None,
    model_id: str = "openai/whisper-small",
) -> Iterator[dict[str, Any]]:
    """
    Yield ``{"path", "encoded"}`` for each audio file under ``dataset_root``.

    ``encoded`` is ``(1, T, D)`` on CPU (float32).
    """
    files: list[str] = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(dataset_root, pat), recursive=True))
    files = sorted(set(files))
    for i, fp in enumerate(files):
        if limit is not None and i >= limit:
            break
        yield {"path": fp, "encoded": get_encoder_output(fp, model_id=model_id)}
