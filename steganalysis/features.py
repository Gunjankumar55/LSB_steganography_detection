"""Feature extraction for LSB steganalysis.

The classifier consumes a fixed 196,614-dimensional vector built from a
256x256 RGB image:

* 196,608 values - the two least significant bits of every pixel channel,
  flattened in row-major order.
* 3 values - block entropy, transition rate, and a chi-square statistic
  describing how uniformly the LSB values are distributed.
* 3 values - character-distribution statistics derived from repacking
  four LSB values into one byte.

The feature order and definitions are part of the model contract; changing
them invalidates the published weights.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.stats import entropy

IMAGE_SIZE = 256
LSB_BITS = 2
FEATURE_COUNT = IMAGE_SIZE * IMAGE_SIZE * 3 + 6

_CHUNK_SIZE = 512
_TRANSITION_SAMPLE = 5000
_BYTE_GROUP = 4


def extract_lsb_bits(img_array: np.ndarray, k: int = LSB_BITS) -> np.ndarray:
    """Return the low ``k`` bits of every channel as a flat array."""
    mask = (1 << k) - 1
    return (img_array & mask).flatten()


def _randomness_features(lsb_feat: np.ndarray) -> np.ndarray:
    """Summarise how random the LSB plane looks."""
    blocks = lsb_feat.reshape(-1, _CHUNK_SIZE)
    entropies = []
    for block in blocks[::10]:
        hist = np.bincount(block.astype(np.int32), minlength=4)
        hist = hist / max(hist.sum(), 1)
        entropies.append(entropy(hist + 1e-10))

    mean_entropy = np.mean(entropies)

    sample = lsb_feat[:_TRANSITION_SAMPLE].astype(np.int32)
    transitions = np.sum(np.abs(np.diff(sample)))
    transition_rate = transitions / max(len(sample) - 1, 1)

    hist_full = np.bincount(lsb_feat.astype(np.int32), minlength=4)
    expected = len(lsb_feat) / 4
    chi2_stat = np.sum((hist_full - expected) ** 2 / expected) if expected > 0 else 0

    return np.array([mean_entropy, transition_rate, chi2_stat], dtype=np.float32)


def _byte_distribution_features(lsb_feat: np.ndarray) -> np.ndarray:
    """Repack LSB values into bytes and describe their distribution."""
    num_bytes = len(lsb_feat) // _BYTE_GROUP
    grouped = lsb_feat[: num_bytes * _BYTE_GROUP].astype(np.int32).reshape(-1, _BYTE_GROUP)
    byte_values = (
        (grouped[:, 0] << 6)
        | (grouped[:, 1] << 4)
        | (grouped[:, 2] << 2)
        | grouped[:, 3]
    )

    url_char_count = np.sum((byte_values >= 10) & (byte_values <= 15))
    html_char_count = np.sum((byte_values >= 0) & (byte_values <= 5))
    unique_ratio = len(np.unique(byte_values)) / max(len(byte_values), 1)

    return np.array(
        [
            url_char_count / max(len(byte_values), 1),
            html_char_count / max(len(byte_values), 1),
            unique_ratio,
        ],
        dtype=np.float32,
    )


def extract_features(image: Image.Image, k: int = LSB_BITS) -> np.ndarray:
    """Build the feature vector for ``image``.

    The image is converted to RGB before resizing so that palette and
    greyscale inputs are handled consistently.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

    lsb_feat = extract_lsb_bits(np.array(image), k=k)
    return np.concatenate(
        [
            lsb_feat.astype(np.float32),
            _randomness_features(lsb_feat),
            _byte_distribution_features(lsb_feat),
        ]
    )
