"""Tests for artifact loading and the prediction contract."""

from __future__ import annotations

import pytest
from PIL import Image

from steganalysis.inference import ArtifactsMissingError, StegoDetector
from tests.conftest import MODELS_DIR, make_gradient, requires_artifacts


def test_missing_artifacts_raise(tmp_path) -> None:
    with pytest.raises(ArtifactsMissingError):
        StegoDetector.load(tmp_path)


@requires_artifacts
def test_binary_prediction_shape(detector, sample_image: Image.Image) -> None:
    result = detector.predict(sample_image)
    binary = result["binary_prediction"]

    assert binary["class"] in {"clean", "stego"}
    assert 0.0 <= binary["confidence"] <= 1.0
    assert set(binary["probabilities"]) == {"clean", "stego"}
    assert sum(binary["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)


@requires_artifacts
def test_multiclass_prediction_shape(detector, sample_image: Image.Image) -> None:
    result = detector.predict(sample_image)
    multi = result["multiclass_prediction"]

    assert multi["payload_type"] in {"clean", "html", "js", "ps"}
    assert set(multi["probabilities"]) == {"clean", "html", "js", "ps"}
    assert sum(multi["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)


@requires_artifacts
def test_predictions_are_stable(detector, sample_image: Image.Image) -> None:
    first = detector.predict(sample_image)
    second = detector.predict(sample_image)
    assert first == second


@requires_artifacts
def test_rgb_conversion_does_not_change_rgb_input(detector, sample_image: Image.Image) -> None:
    assert detector.predict(sample_image) == detector.predict(sample_image.convert("RGB"))


@requires_artifacts
def test_grayscale_input_is_accepted(detector) -> None:
    result = detector.predict(make_gradient().convert("L"))
    assert result["binary_prediction"]["class"] in {"clean", "stego"}
