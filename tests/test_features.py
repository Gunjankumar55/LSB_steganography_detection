"""Contract tests for the feature vector."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from steganalysis.features import (
    FEATURE_COUNT,
    IMAGE_SIZE,
    extract_features,
    extract_lsb_bits,
)
from tests.conftest import make_gradient, png_bytes


def _reference_byte_features(lsb_feat: np.ndarray) -> np.ndarray:
    """The scalar implementation the vectorised version must match."""
    chunk_size = 4
    num_chunks = len(lsb_feat) // chunk_size
    trimmed = lsb_feat[: num_chunks * chunk_size].astype(np.int32)
    byte_values = []
    for i in range(0, len(trimmed), chunk_size):
        value = (
            (trimmed[i] << 6)
            | (trimmed[i + 1] << 4)
            | (trimmed[i + 2] << 2)
            | trimmed[i + 3]
        )
        byte_values.append(value)
    byte_values = np.array(byte_values)
    return np.array(
        [
            np.sum((byte_values >= 10) & (byte_values <= 15)) / len(byte_values),
            np.sum((byte_values >= 0) & (byte_values <= 5)) / len(byte_values),
            len(np.unique(byte_values)) / len(byte_values),
        ],
        dtype=np.float32,
    )


def test_feature_count_matches_dimension() -> None:
    assert IMAGE_SIZE == 256
    assert FEATURE_COUNT == 196_614


def test_feature_vector_shape_and_dtype(sample_image: Image.Image) -> None:
    features = extract_features(sample_image)
    assert features.shape == (FEATURE_COUNT,)
    assert features.dtype == np.float32


def test_feature_vector_is_deterministic(sample_image: Image.Image) -> None:
    first = extract_features(sample_image)
    second = extract_features(sample_image)
    np.testing.assert_array_equal(first, second)


def test_lsb_bits_stay_within_mask(sample_image: Image.Image) -> None:
    array = np.array(sample_image)
    two_bits = extract_lsb_bits(array, k=2)
    one_bit = extract_lsb_bits(array, k=1)
    assert two_bits.max() <= 3
    assert one_bit.max() <= 1
    assert len(two_bits) == array.size


def test_byte_features_match_scalar_reference() -> None:
    from steganalysis.features import _byte_distribution_features

    rng = np.random.default_rng(1234)
    lsb_feat = rng.integers(0, 4, size=256 * 256 * 3, dtype=np.uint8)
    np.testing.assert_allclose(
        _byte_distribution_features(lsb_feat),
        _reference_byte_features(lsb_feat),
        rtol=0,
        atol=1e-6,
    )


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P"])
def test_modes_are_normalised(mode: str) -> None:
    image = make_gradient().convert(mode)
    features = extract_features(image)
    assert features.shape == (FEATURE_COUNT,)
    assert np.isfinite(features).all()


def test_accepts_png_round_trip() -> None:
    import io

    image = Image.open(io.BytesIO(png_bytes(make_gradient())))
    assert extract_features(image).shape == (FEATURE_COUNT,)
