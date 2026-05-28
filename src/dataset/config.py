from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class LibriSpeechConfig:
    """
    Dataset config for LibriSpeech-style training pairs.

    `root` should point at the LibriSpeech split folder containing subfolders with
    `*.flac` and `*.trans.txt`.
    """

    root: str

    @staticmethod
    def default_train_clean_100_from_training_dir(training_dir: str) -> "LibriSpeechConfig":
        return LibriSpeechConfig(
            root=os.path.normpath(
                os.path.join(
                    training_dir,
                    "../datasets/librispeech_data/LibriSpeech/train-clean-100",
                )
            )
        )

    @staticmethod
    def _librispeech_base(training_dir: str) -> str:
        return os.path.normpath(
            os.path.join(training_dir, "../datasets/librispeech_data/LibriSpeech")
        )

    @staticmethod
    def train_clean_100_and_360_roots(training_dir: str) -> list[str]:
        """LibriSpeech train-clean-100 + train-clean-360 split directories."""
        base = LibriSpeechConfig._librispeech_base(training_dir)
        return [
            os.path.join(base, "train-clean-100"),
            os.path.join(base, "train-clean-360"),
        ]

    @staticmethod
    def dev_clean_root(training_dir: str) -> str:
        """LibriSpeech dev-clean split (validation)."""
        return os.path.join(LibriSpeechConfig._librispeech_base(training_dir), "dev-clean")

