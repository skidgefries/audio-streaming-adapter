"""
Training-side utilities: **configs**, **checkpointing**, **losses**, **metrics**, **logging**.

- **Models, frozen loaders, end-to-end inference**: use ``src/encoder``, ``src/llm``,
  ``src/adapter``, and ``src/adapter_llm_pipeline`` (not duplicated here).
- ``training.utils.audio`` / ``training.utils.loaders`` only re-export for backward
  compatibility; prefer ``encoder`` / ``llm`` imports in new code.
"""

from training.utils.audio import WhisperWindowFeatureExtractor
from training.utils.checkpointing import (
    TrainingCheckpoint,
    load_adapter_state_dict,
    maybe_upload_stage_epoch_checkpoint,
    save_checkpoint,
)
from training.utils.common import AdapterCheckpoint, default_librispeech_root_from_training_dir
from training.utils.common import save_checkpoint as save_legacy_adapter_checkpoint
from training.utils.config import (
    CheckpointConfig,
    DataConfig,
    FrozenModelIdsConfig,
    HfCheckpointConfig,
    LLM_CHOICES,
    LLM_PRESETS,
    LlmChoice,
    OptimConfig,
    Stage1Config,
    Stage2Config,
    DeviceConfig,
    Stage3Config,
    Stage3DeviceConfig,
    TrainingLaunchConfig,
    StreamingAdapterTrainConfig,
    TuningConfig,
    WandbConfig,
    WhisperWaveformWindowingConfig,
)
from training.utils.loaders import (
    default_device_and_dtype,
    load_frozen_llm_embeddings,
    load_frozen_qwen_causal_lm,
    load_frozen_qwen_embeddings,
    load_frozen_whisper,
)
from training.utils.logging import WandbLogger
from training.utils.losses import (
    contrastive_infonce_loss,
    kl_distill_loss,
    prefix_consistency_loss,
    revision_penalty_loss,
)
from training.utils.metrics import (
    RunningMean,
    bleu_sentence_0_1,
    bleu_sentence_0_100,
    metrics_reference_vs_response_after_compressed_audio,
    metrics_transcription_vs_response,
)
from training.utils.optimization import TrainingPipeline

__all__ = [
    "AdapterCheckpoint",
    "CheckpointConfig",
    "DataConfig",
    "FrozenModelIdsConfig",
    "HfCheckpointConfig",
    "LLM_CHOICES",
    "LLM_PRESETS",
    "LlmChoice",
    "OptimConfig",
    "RunningMean",
    "Stage1Config",
    "Stage2Config",
    "DeviceConfig",
    "Stage3Config",
    "TrainingLaunchConfig",
    "Stage3DeviceConfig",
    "StreamingAdapterTrainConfig",
    "TrainingCheckpoint",
    "TrainingPipeline",
    "TuningConfig",
    "WandbConfig",
    "WhisperWaveformWindowingConfig",
    "WandbLogger",
    "WhisperWindowFeatureExtractor",
    "bleu_sentence_0_1",
    "bleu_sentence_0_100",
    "contrastive_infonce_loss",
    "default_device_and_dtype",
    "default_librispeech_root_from_training_dir",
    "kl_distill_loss",
    "load_adapter_state_dict",
    "maybe_upload_stage_epoch_checkpoint",
    "load_frozen_llm_embeddings",
    "load_frozen_qwen_causal_lm",
    "load_frozen_qwen_embeddings",
    "load_frozen_whisper",
    "metrics_reference_vs_response_after_compressed_audio",
    "metrics_transcription_vs_response",
    "prefix_consistency_loss",
    "revision_penalty_loss",
    "save_checkpoint",
    "save_legacy_adapter_checkpoint",
]
