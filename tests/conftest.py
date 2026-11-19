"""Shared fixtures for the test suite."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from steganalysis.inference import ARTIFACT_FILES, StegoDetector

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"

requires_artifacts = pytest.mark.skipif(
    not all((MODELS_DIR / name).is_file() for name in ARTIFACT_FILES),
    reason="model artifacts not installed; run scripts/fetch_models.py",
)


def make_gradient(width: int = 64, height: int = 64) -> Image.Image:
    """Return a deterministic RGB image with a smooth gradient."""
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (x * 255 // max(width - 1, 1), y * 255 // max(height - 1, 1), 128)
    return image


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="session")
def detector() -> StegoDetector:
    return StegoDetector.load(MODELS_DIR)


@pytest.fixture
def sample_image() -> Image.Image:
    return make_gradient()


@pytest.fixture
def sample_png() -> bytes:
    return png_bytes(make_gradient())
