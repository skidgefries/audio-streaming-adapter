from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = False
    project: str = "audio-streaming-adapter"
    entity: str | None = None
    run_name: str | None = None
    tags: list[str] | None = None


@dataclass(frozen=True)
class CheckpointConfig:
    dir: str = "checkpoints"
    save_every_steps: int | None = None
    save_every_epochs: int = 1


@dataclass(frozen=True)
class OptimConfig:
    lr: float
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    warmup_steps: int = 0


@dataclass(frozen=True)
class DataConfig:
    dataset_root: str
    batch_size: int
    num_workers: int = 2
    max_windows_per_utt: int | None = None


@dataclass(frozen=True)
class Stage1Config:
    """Contrastive audio–text alignment."""

    epochs: int = 5
    lambda_stability: float = 0.1
    temperature: float = 0.2


@dataclass(frozen=True)
class Stage2Config:
    """ASR distillation."""

    epochs: int = 10
    lambda_align: float = 0.1
    lambda_stability: float = 0.05
    lambda_rate: float = 0.001
    lambda_gate: float = 0.1
    temperature: float = 0.07

    use_rate_controller: bool = True
    rate_target: float = 2.0


@dataclass(frozen=True)
class Stage3Config:
    """Task distillation (teacher/student)."""

    epochs: int = 15
    kl_temperature: float = 2.0
    lambda_asr: float = 0.1
    lambda_stability: float = 0.05
    lambda_rate: float = 0.001
    lambda_gate: float = 0.5

    use_rate_controller: bool = True
    rate_target: float = 2.0


@dataclass(frozen=True)
class Stage3DeviceConfig:
    """Optional second GPU for student while teacher stays on CPU or another device."""

    student: str | None = None  # default: cuda if available else cpu
    teacher: str = "cpu"


@dataclass(frozen=True)
class WhisperFrameWindowingConfig:
    """Frame-level windowing after a full Whisper encoder pass (see ``adapter.windowing`` / pipeline)."""

    chunk_seconds: float = 30.0
    window_seconds: float = 0.8
    stride_seconds: float = 0.4


@dataclass(frozen=True)
class WhisperWaveformWindowingConfig:
    """Time geometry for ``encoder.WhisperWindowFeatureExtractor`` (full Whisper encode + ``WhisperFrameWindowizer``)."""

    window_seconds: float = 0.8
    stride_seconds: float = 0.4
    sample_rate: int = 16000


@dataclass(frozen=True)
class FrozenModelIdsConfig:
    """Typical HF ids for curriculum stages (tuning / logging)."""

    whisper_model_id: str = "openai/whisper-small"
    llm_model_id: str = "Qwen/Qwen3-8B"


@dataclass(frozen=True)
class StreamingAdapterTrainConfig:
    """Hyperparameters for :class:`adapter.streaming_adapter.StreamingAdapter` (serialize into checkpoints)."""

    d_encoder: int = 768
    d_llm: int = 4096
    num_queries: int = 4
    num_layers: int = 8
    num_heads: int = 12
    d_ffn: int = 2048
    dropout: float = 0.1
    ema_alpha: float = 0.8
    learnable_ema: bool = False
    use_rate_controller: bool = True
    rate_threshold: float = 0.5
    target_rate: float = 2.0
    cross_layer_in_between: int = 0


@dataclass(frozen=True)
class TuningConfig:
    """Optional sweep-friendly bundle (subset of knobs often tuned together)."""

    lr: float = 3e-5
    batch_size: int = 1
    grad_clip_norm: float = 1.0
    whisper_windowing: WhisperWaveformWindowingConfig = field(
        default_factory=WhisperWaveformWindowingConfig
    )
    encoder_frame_windowing: WhisperFrameWindowingConfig = field(
        default_factory=WhisperFrameWindowingConfig
    )
    adapter: StreamingAdapterTrainConfig = field(default_factory=StreamingAdapterTrainConfig)
    model_ids: FrozenModelIdsConfig = field(default_factory=FrozenModelIdsConfig)
