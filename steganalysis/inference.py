"""Model loading and inference for the steganalysis demo."""

from __future__ import annotations

import pickle
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .features import extract_features
from .models import SteganalysisDNN

BINARY_CHECKPOINT = "final_bin_best.pth"
MULTI_CHECKPOINT = "final_multi_best.pth"
SCALER_FILE = "final_scaler.pkl"
BINARY_ENCODER_FILE = "final_bin_le.pkl"
MULTI_ENCODER_FILE = "final_multi_le.pkl"

ARTIFACT_FILES = (
    BINARY_CHECKPOINT,
    MULTI_CHECKPOINT,
    SCALER_FILE,
    BINARY_ENCODER_FILE,
    MULTI_ENCODER_FILE,
)


class ArtifactsMissingError(RuntimeError):
    """Raised when the model artifacts are not present on disk."""


@dataclass(frozen=True)
class DetectorPaths:
    """Locations of the five artifacts."""

    directory: Path

    def __post_init__(self) -> None:
        missing = [name for name in ARTIFACT_FILES if not (self.directory / name).is_file()]
        if missing:
            raise ArtifactsMissingError(
                "Missing model artifacts in "
                f"{self.directory}: {', '.join(missing)}. "
                "Run scripts/fetch_models.py to download them."
            )

    def __getitem__(self, name: str) -> Path:
        return self.directory / name


class StegoDetector:
    """Loads the artifacts once and answers prediction requests."""

    def __init__(
        self,
        scaler: Any,
        binary_encoder: Any,
        multi_encoder: Any,
        binary_model: SteganalysisDNN,
        multi_model: SteganalysisDNN,
    ) -> None:
        self._scaler = scaler
        self._binary_encoder = binary_encoder
        self._multi_encoder = multi_encoder
        self._binary_model = binary_model
        self._multi_model = multi_model
        self._lock = threading.Lock()

    @classmethod
    def load(cls, artifact_dir: Path | str) -> "StegoDetector":
        paths = DetectorPaths(Path(artifact_dir))

        with paths[SCALER_FILE].open("rb") as handle:
            scaler = pickle.load(handle)
        with paths[BINARY_ENCODER_FILE].open("rb") as handle:
            binary_encoder = pickle.load(handle)
        with paths[MULTI_ENCODER_FILE].open("rb") as handle:
            multi_encoder = pickle.load(handle)

        binary_model = _load_checkpoint(paths[BINARY_CHECKPOINT], len(binary_encoder.classes_))
        multi_model = _load_checkpoint(paths[MULTI_CHECKPOINT], len(multi_encoder.classes_))
        return cls(scaler, binary_encoder, multi_encoder, binary_model, multi_model)

    def predict(self, image: Image.Image) -> dict[str, Any]:
        """Return the binary and payload predictions for ``image``."""
        features = extract_features(image)
        batch = self._scaler.transform(features.reshape(1, -1)).astype(np.float32)

        with self._lock, torch.no_grad():
            tensor = torch.from_numpy(batch)
            binary_probs = torch.softmax(self._binary_model(tensor), dim=1).numpy()[0]
            binary_index = int(np.argmax(binary_probs))
            binary_class = str(self._binary_encoder.classes_[binary_index])

            if binary_class == "stego":
                multi_probs = torch.softmax(self._multi_model(tensor), dim=1).numpy()[0]
            else:
                multi_probs = np.zeros(len(self._multi_encoder.classes_))
                multi_probs[int(self._multi_encoder.transform(["clean"])[0])] = 1.0

        payload_type = str(self._multi_encoder.classes_[int(np.argmax(multi_probs))])

        return {
            "binary_prediction": {
                "class": binary_class,
                "confidence": float(binary_probs[binary_index]),
                "probabilities": {
                    str(label): float(prob)
                    for label, prob in zip(self._binary_encoder.classes_, binary_probs)
                },
            },
            "multiclass_prediction": {
                "payload_type": payload_type,
                "probabilities": {
                    str(label): float(prob)
                    for label, prob in zip(self._multi_encoder.classes_, multi_probs)
                },
            },
        }


def _load_checkpoint(path: Path, num_classes: int) -> SteganalysisDNN:
    model = SteganalysisDNN(num_classes)
    state = torch.load(path, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


_detector: StegoDetector | None = None
_detector_lock = threading.Lock()


def get_detector(artifact_dir: Path | str = "models") -> StegoDetector:
    """Return the process-wide detector, loading it on first use."""
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                _detector = StegoDetector.load(artifact_dir)
    return _detector
