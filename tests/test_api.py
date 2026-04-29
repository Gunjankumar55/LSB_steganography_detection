"""HTTP contract tests for the Flask app."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from app import MAX_CONTENT_LENGTH, create_app
from tests.conftest import make_gradient, png_bytes, requires_artifacts


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_index_renders(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert b"Multi-Payload LSB Steganalysis Demo" in response.data


def test_missing_file_field_is_rejected(client) -> None:
    response = client.post("/predict", data={})
    assert response.status_code == 400
    assert "file" in response.get_json()["error"]


def test_non_image_is_rejected(client) -> None:
    response = client.post(
        "/predict",
        data={"file": (io.BytesIO(b"not an image"), "payload.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "The file is not a PNG image."


def test_jpeg_is_rejected(client) -> None:
    buffer = io.BytesIO()
    make_gradient().save(buffer, format="JPEG")
    response = client.post(
        "/predict",
        data={"file": (io.BytesIO(buffer.getvalue()), "photo.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "Only PNG images are supported" in response.get_json()["error"]


def test_tiny_image_is_rejected(client) -> None:
    response = client.post(
        "/predict",
        data={"file": (io.BytesIO(png_bytes(Image.new("RGB", (4, 4)))), "tiny.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "too small" in response.get_json()["error"]


def test_oversized_upload_is_rejected(client) -> None:
    payload = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\0" * MAX_CONTENT_LENGTH)
    response = client.post(
        "/predict",
        data={"file": (payload, "big.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 413
    assert "5 MB" in response.get_json()["error"]


@requires_artifacts
def test_png_prediction_returns_the_documented_schema(client) -> None:
    response = client.post(
        "/predict",
        data={"file": (io.BytesIO(png_bytes(make_gradient())), "sample.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200

    body = response.get_json()
    assert set(body) == {"binary_prediction", "multiclass_prediction"}
    assert set(body["binary_prediction"]) == {"class", "confidence", "probabilities"}
    assert set(body["multiclass_prediction"]) == {"payload_type", "probabilities"}
