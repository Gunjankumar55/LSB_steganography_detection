"""Download the model artifacts from the project's GitHub release.

The weights are 581 MiB in total, so they are published as release assets
instead of being committed (or stored in Git LFS). Run this once after
cloning:

    python scripts/fetch_models.py

Files are verified against the SHA-256 digests below and skipped when an
identical copy is already present.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

RELEASE_TAG = "v1.0.0"
BASE_URL = (
    "https://github.com/Gunjankumar55/LSB_steganography_detection"
    f"/releases/download/{RELEASE_TAG}"
)

ARTIFACTS: dict[str, str] = {
    "final_bin_best.pth": "937140babfa952b004688c3bb98665434afd2b901658e5e8ff4de0438c107d90",
    "final_bin_le.pkl": "90b3bd41467e805a270b7d2b1ca04eb1ea9d31670d16bf5c4022f81f5c3c6d9f",
    "final_multi_best.pth": "2d10330083813ade2f03ff8365b24abdc732f5728f1470372390863f748073ee",
    "final_multi_le.pkl": "b16da8d351473e04c4cf7ebcac502f4f7410e7832ec44db0f2e2f1c8d6ddf93b",
    "final_scaler.pkl": "0a115a9cf586486270825e41369e3d3e6f2367f6f49c16f1d583d6ac626015a2",
}

_CHUNK = 1 << 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def download(name: str, expected: str, target_dir: Path, force: bool) -> None:
    destination = target_dir / name

    if destination.is_file() and not force:
        if sha256(destination) == expected:
            print(f"  ok       {name}")
            return
        print(f"  mismatch {name} - re-downloading")

    url = f"{BASE_URL}/{name}"
    print(f"  get      {name}")

    with tempfile.NamedTemporaryFile(
        dir=target_dir, prefix=f".{name}.", suffix=".part", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                while chunk := response.read(_CHUNK):
                    handle.write(chunk)
        except (urllib.error.URLError, urllib.error.HTTPError) as error:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise SystemExit(f"Failed to download {name} from {url}: {error}") from error

    actual = sha256(temporary)
    if actual != expected:
        temporary.unlink(missing_ok=True)
        raise SystemExit(
            f"Checksum mismatch for {name}:\n  expected {expected}\n  actual   {actual}"
        )

    temporary.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "models",
        help="destination directory (default: <repo>/models)",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download files that already exist"
    )
    args = parser.parse_args()

    args.dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(ARTIFACTS)} artifacts from release {RELEASE_TAG}")

    for name, expected in ARTIFACTS.items():
        download(name, expected, args.dir, args.force)

    print(f"Done. Artifacts are in {args.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
