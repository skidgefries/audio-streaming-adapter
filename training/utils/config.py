from __future__ import annotations

import os
from dataclasses import dataclass, field

from training.utils.env import env_bool, env_float, env_int, env_optional_int, env_str


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = False
    project: str = "audio-streaming-adapter"
    entity: str | None = None
    run_name: str | None = None
    tags: list[str] | None = None

    @classmethod
    def from_env(cls) -> WandbConfig:
        entity = env_str("WANDB_ENTITY")
        run_name = env_str("WANDB_RUN_NAME")
        return cls(
            enabled=env_bool("WANDB_ENABLED", default=False),
            project=env_str("WANDB_PROJECT", "audio-streaming-adapter") or "audio-streaming-adapter",
            entity=entity,
            run_name=run_name,
            tags=None,
        )


@dataclass(frozen=True)
class CheckpointConfig:
    dir: str = "checkpoints"
    save_every_steps: int | None = None
    save_every_epochs: int = 1

    @classmethod
    def from_env(cls, *, pkg_root: str | None = None) -> CheckpointConfig:
        ckpt_dir = env_str("CHECKPOINT_DIR", "checkpoints") or "checkpoints"
        if pkg_root and not os.path.isabs(ckpt_dir):
            ckpt_dir = os.path.join(pkg_root, ckpt_dir)
        return cls(
            dir=ckpt_dir,
            save_every_steps=env_optional_int("CHECKPOINT_SAVE_EVERY_STEPS"),
            save_every_epochs=env_int("CHECKPOINT_SAVE_EVERY_EPOCHS", 1),
        )


@dataclass(frozen=True)
class HfCheckpointConfig:
    """Upload epoch checkpoints for the active training stage only (other repo files unchanged)."""

    repo_id: str = "vaghawan/audio-streaming-adapter-checkpoints"
    upload_enabled: bool = False
    private: bool = False
    revision: str = "main"

    @classmethod
    def from_env(cls) -> HfCheckpointConfig:
        return cls(
            repo_id=env_str("HF_CHECKPOINT_REPO", "vaghawan/audio-streaming-adapter-checkpoints")
            or "vaghawan/audio-streaming-adapter-checkpoints",
            upload_enabled=env_bool("HF_UPLOAD_CHECKPOINTS", False),
            private=env_bool("HF_CHECKPOINT_PRIVATE", False),
            revision=env_str("HF_CHECKPOINT_REVISION", "main") or "main",
        )


@dataclass(frozen=True)
class OptimConfig:
    lr: float
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    warmup_steps: int = 0

    @classmethod
    def from_env(cls) -> OptimConfig:
        return cls(
            lr=env_float("LEARNING_RATE", 5e-5),
            weight_decay=env_float("WEIGHT_DECAY", 0.01),
            grad_clip_norm=env_float("GRAD_CLIP_NORM", 1.0),
            warmup_steps=env_int("WARMUP_STEPS", 500),
        )


@dataclass(frozen=True)
class DataConfig:
    dataset_root: str
    batch_size: int
    num_workers: int = 2
    max_windows_per_utt: int | None = None

    @classmethod
    def from_env(cls, *, default_dataset_root: str) -> DataConfig:
        root = env_str("DATASET_ROOT") or default_dataset_root
        max_win = env_str("MAX_WINDOWS_PER_UTT")
        max_windows_per_utt: int | None = None
        if max_win:
            normalized = max_win.strip().lower()
            if normalized not in {"all", "none", "unlimited"}:
                parsed = int(max_win)
                max_windows_per_utt = parsed if parsed > 0 else None
        return cls(
            dataset_root=root,
            batch_size=env_int("BATCH_SIZE", 8),
            num_workers=env_int("NUM_WORKERS", 2),
            max_windows_per_utt=max_windows_per_utt,
        )


@dataclass(frozen=True)
class GateConfig:
    """Turn-end commit gate (Component 3)."""

    label_source: str = "synthetic"  # synthetic | smart_turn
    smart_turn_dataset: str = "pipecat-ai/smart-turn-data-v3.2-train"
    smart_turn_split: str = "train"
    smart_turn_max_samples: int | None = None
    hidden_dim: int = 256
    threshold: float = 0.5
    latency_weight: float = 0.1
    min_silence_ms: float = 200.0
    require_silence_for_commit: bool = True
    token_activity_threshold: float = 8.0
    window_seconds: float = 0.8
    stride_seconds: float = 0.4
    silence_mode: str = "rule"  # rule | learned | both
    active_silence_path: str = "rule"  # rule | learned (when silence_mode=both)
    learned_silence_hidden_dim: int = 64

    @classmethod
    def from_env(cls) -> GateConfig:
        max_samples = env_optional_int("SMART_TURN_MAX_SAMPLES")
        require_silence = env_str("GATE_REQUIRE_SILENCE", "1") or "1"
        return cls(
            label_source=env_str("GATE_LABEL_SOURCE", "synthetic") or "synthetic",
            smart_turn_dataset=env_str(
                "SMART_TURN_DATASET", "pipecat-ai/smart-turn-data-v3.2-train"
            )
            or "pipecat-ai/smart-turn-data-v3.2-train",
            smart_turn_split=env_str("SMART_TURN_SPLIT", "train") or "train",
            smart_turn_max_samples=max_samples,
            hidden_dim=env_int("GATE_HIDDEN_DIM", 256),
            threshold=env_float("GATE_THRESHOLD", 0.5),
            latency_weight=env_float("GATE_LATENCY_WEIGHT", 0.1),
            min_silence_ms=env_float("GATE_MIN_SILENCE_MS", 200.0),
            require_silence_for_commit=require_silence.strip().lower() not in (
                "0",
                "false",
                "no",
                "off",
            ),
            token_activity_threshold=env_float("GATE_TOKEN_ACTIVITY_THRESHOLD", 8.0),
            window_seconds=env_float("GATE_WINDOW_SECONDS", 0.8),
            stride_seconds=env_float("GATE_STRIDE_SECONDS", 0.4),
            silence_mode=env_str("GATE_SILENCE_MODE", "rule") or "rule",
            active_silence_path=env_str("GATE_ACTIVE_SILENCE_PATH", "rule") or "rule",
            learned_silence_hidden_dim=env_int("GATE_LEARNED_SILENCE_HIDDEN_DIM", 64),
        )


@dataclass(frozen=True)
class Stage1Config:
    """Contrastive audio–text alignment."""

    epochs: int = 10
    lambda_stability: float = 0.1
    # lambda_stability = 0.0
    temperature: float = 0.2


@dataclass(frozen=True)
class DeviceConfig:
    """Primary compute device. Set ``DEVICE=cpu`` to force CPU; default is CUDA when available."""

    device: str = "cuda"

    @classmethod
    def from_env(cls) -> DeviceConfig:
        return cls(device=env_str("DEVICE", "cuda") or "cuda")


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
    max_text_tokens: int = 128
    asr_micro_batch_size: int = 1
    enable_llm_gradient_checkpointing: bool = False

    @classmethod
    def from_env(cls) -> Stage2Config:
        return cls(
            epochs=env_int("EPOCHS", 10),
            lambda_align=env_float("LAMBDA_ALIGN", 0.1),
            lambda_stability=env_float("LAMBDA_STABILITY", 0.05),
            lambda_rate=env_float("LAMBDA_RATE", 0.001),
            lambda_gate=env_float("LAMBDA_GATE", 0.1),
            temperature=env_float("TEMPERATURE", 0.07),
            use_rate_controller=env_bool("USE_RATE_CONTROLLER", True),
            rate_target=env_float("RATE_TARGET", 2.0),
            max_text_tokens=env_int("MAX_TEXT_TOKENS", 128),
            asr_micro_batch_size=env_int("ASR_MICRO_BATCH_SIZE", 8),
            enable_llm_gradient_checkpointing=env_bool(
                "ENABLE_LLM_GRADIENT_CHECKPOINTING", False
            ),
        )


@dataclass(frozen=True)
class TrainingLaunchConfig:
    """
    How Stage 2 is launched from ``setup_remote_training.sh``.

    One visible GPU → ``uv run``; two or more → ``torchrun``.

    Stage 2 keeps ``nproc_per_node=1`` so a single process can shard the frozen LLM
    across all visible GPUs via HuggingFace ``device_map="auto"``. Override
    ``TORCHRUN_NPROC_PER_NODE`` only for custom distributed training.
    """

    use_torchrun: str = "auto"  # auto | true | false | 1 | 0
    torchrun_nproc_per_node: int = 1
    torchrun_master_port: int = 29500

    @classmethod
    def from_env(cls) -> TrainingLaunchConfig:
        return cls(
            use_torchrun=env_str("USE_TORCHRUN", "auto") or "auto",
            torchrun_nproc_per_node=env_int("TORCHRUN_NPROC_PER_NODE", 1),
            torchrun_master_port=env_int("TORCHRUN_MASTER_PORT", 29500),
        )

    def should_use_torchrun(self, visible_gpus: int) -> bool:
        mode = self.use_torchrun.strip().lower()
        if mode in ("1", "true", "yes", "on"):
            return True
        if mode in ("0", "false", "no", "off"):
            return False
        return visible_gpus >= 2


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
class WhisperWaveformWindowingConfig:
    """Time geometry for ``encoder.WhisperWindowFeatureExtractor`` (``AudioWaveformWindowizer`` + per-chunk encode)."""

    window_seconds: float = 0.8
    stride_seconds: float = 0.4
    sample_rate: int = 16000


@dataclass(frozen=True)
class FrozenModelIdsConfig:
    """Typical HF ids for curriculum stages (tuning / logging)."""

    whisper_model_id: str = "openai/whisper-small"
    llm_model_id: str = "Qwen/Qwen3-8B"

    @classmethod
    def from_env(cls) -> FrozenModelIdsConfig:
        return cls(
            whisper_model_id=env_str("WHISPER_MODEL_ID", "openai/whisper-small")
            or "openai/whisper-small",
            llm_model_id=env_str("LLM_MODEL_ID", "Qwen/Qwen3-8B") or "Qwen/Qwen3-8B",
        )


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
    batch_size: int = 16
    grad_clip_norm: float = 1.0
    whisper_windowing: WhisperWaveformWindowingConfig = field(
        default_factory=WhisperWaveformWindowingConfig
    )
    adapter: StreamingAdapterTrainConfig = field(default_factory=StreamingAdapterTrainConfig)
    model_ids: FrozenModelIdsConfig = field(default_factory=FrozenModelIdsConfig)
