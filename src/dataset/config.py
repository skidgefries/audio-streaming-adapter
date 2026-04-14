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
                    "../../../datasets/librispeech_data/LibriSpeech/train-clean-100",
                )
            )
        )

