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
    resume_checkpoint: str | None = None

    @classmethod
    def from_env(cls, *, pkg_root: str | None = None) -> CheckpointConfig:
        ckpt_dir = env_str("CHECKPOINT_DIR", "checkpoints") or "checkpoints"
        if pkg_root and not os.path.isabs(ckpt_dir):
            ckpt_dir = os.path.join(pkg_root, ckpt_dir)
        save_every_steps = env_optional_int("SAVE_EVERY_STEPS")
        if save_every_steps is None:
            save_every_steps = env_optional_int("CHECKPOINT_SAVE_EVERY_STEPS")
        resume = env_str("RESUME_CHECKPOINT")
        if resume and pkg_root and not os.path.isabs(resume):
            resume = os.path.join(pkg_root, resume)
        return cls(
            dir=ckpt_dir,
            save_every_steps=save_every_steps,
            save_every_epochs=env_int("CHECKPOINT_SAVE_EVERY_EPOCHS", 1),
            resume_checkpoint=resume,
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
            warmup_steps=env_int("WARMUP_STEPS", 1000),
        )


@dataclass(frozen=True)
class DataConfig:
    dataset_root: str
    batch_size: int
    num_workers: int = 2
    max_windows_per_utt: int | None = None
    macro_batch_size: int | None = None

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
            batch_size=env_int("BATCH_SIZE", 24),
            num_workers=env_int("NUM_WORKERS", 2),
            max_windows_per_utt=max_windows_per_utt,
            macro_batch_size=env_optional_int("MACRO_BATCH_SIZE"),
        )

    def gradient_accumulation_steps(self) -> int:
        """
        Optimizer steps per ``MACRO_BATCH_SIZE / BATCH_SIZE`` micro-batches.

        When ``MACRO_BATCH_SIZE`` is unset or equals ``batch_size``, returns 1.
        """
        if self.macro_batch_size is None or self.macro_batch_size <= self.batch_size:
            return 1
        if self.macro_batch_size % self.batch_size != 0:
            raise ValueError(
                f"MACRO_BATCH_SIZE ({self.macro_batch_size}) must be a multiple of "
                f"BATCH_SIZE ({self.batch_size})"
            )
        return self.macro_batch_size // self.batch_size


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
    temperature: float = 0.07
    val_enabled: bool = True
    val_every_steps: int = 1000
    val_max_utterances: int | None = None  # None = full dev-clean

def _parse_llm_max_memory(raw: str | None) -> dict[int | str, str] | None:
    """
    Parse ``LLM_MAX_MEMORY`` (e.g. ``0:10GiB,1:2GiB,cpu:64GiB``) for HuggingFace ``max_memory``.

    Indices are logical CUDA device ids (``0``, ``1``, …) after ``CUDA_VISIBLE_DEVICES``,
    plus optional ``cpu`` for CPU spill after GPU caps are filled.
    """
    if not raw or not raw.strip():
        return None
    result: dict[int | str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        idx_str, _, size = part.partition(":")
        idx_str = idx_str.strip().lower()
        size = size.strip()
        if not idx_str or not size:
            raise ValueError(f"Invalid LLM_MAX_MEMORY entry: {part!r}")
        key: int | str = "cpu" if idx_str == "cpu" else int(idx_str)
        result[key] = size
    return result if result else None


@dataclass(frozen=True)
class DeviceConfig:
    """
    Primary compute device for Whisper/adapter (``DEVICE``).

    Optional ``LLM_DEVICE`` places the frozen Qwen on a different device
    (e.g. ``DEVICE=cpu`` + ``LLM_DEVICE=cuda:1``).
    """

    device: str = "cuda"
    llm_device: str | None = None
    llm_max_memory: dict[int | str, str] | None = None

    @classmethod
    def from_env(cls) -> DeviceConfig:
        return cls(
            device=env_str("DEVICE", "cuda") or "cuda",
            llm_device=env_str("LLM_DEVICE"),
            llm_max_memory=_parse_llm_max_memory(env_str("LLM_MAX_MEMORY")),
        )


@dataclass(frozen=True)
class AsrExperimentConfig:
    """Ablation / experiment overrides for Stage 2 ASR training."""

    load_stage1_checkpoint: bool = True
    train_gate: bool = True
    checkpoint_basename: str = "adapter_stage2"
    gate_checkpoint: str | None = None
    adapter_checkpoint: str | None = None

    @classmethod
    def from_env(cls, *, pkg_root: str | None = None) -> AsrExperimentConfig:
        basename = env_str("CHECKPOINT_BASENAME", "adapter_stage2") or "adapter_stage2"
        gate_ckpt = env_str("GATE_CHECKPOINT")
        if gate_ckpt and pkg_root and not os.path.isabs(gate_ckpt):
            gate_ckpt = os.path.join(pkg_root, gate_ckpt)
        adapter_ckpt = env_str("ADAPTER_CHECKPOINT")
        if adapter_ckpt and pkg_root and not os.path.isabs(adapter_ckpt):
            adapter_ckpt = os.path.join(pkg_root, adapter_ckpt)
        return cls(
            load_stage1_checkpoint=env_bool("LOAD_STAGE1_CHECKPOINT", True),
            train_gate=env_bool("TRAIN_GATE", True),
            checkpoint_basename=basename,
            gate_checkpoint=gate_ckpt,
            adapter_checkpoint=adapter_ckpt,
        )


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
    val_enabled: bool = True
    val_every_steps: int = 1000
    val_max_utterances: int | None = None  # None = full dev-clean
    val_max_new_tokens: int = 496
    val_num_beams: int = 1
    val_repetition_penalty: float = 1.25
    val_log_every: int = 50

    @classmethod
    def from_env(cls) -> Stage2Config:
        val_max_raw = env_str("VAL_MAX_UTTERANCES")
        val_max_utterances: int | None = None
        if val_max_raw:
            normalized = val_max_raw.strip().lower()
            if normalized in {"all", "none", "unlimited"}:
                val_max_utterances = None
            else:
                val_max_utterances = int(val_max_raw)
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
            val_enabled=env_bool("VAL_ENABLED", True),
            val_every_steps=env_int("VAL_EVERY_STEPS", 1000),
            val_max_utterances=val_max_utterances,
            val_max_new_tokens=env_int("VAL_MAX_NEW_TOKENS", 496),
            val_num_beams=env_int("VAL_NUM_BEAMS", 1),
            val_repetition_penalty=env_float("VAL_REPETITION_PENALTY", 1.25),
            val_log_every=env_int("VAL_LOG_EVERY", 50),
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
    num_queries: int = 2
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
