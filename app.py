"""Flask entry point for the multi-payload LSB steganalysis demo."""

from __future__ import annotations

import io
import logging
import os

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException

from steganalysis.inference import ArtifactsMissingError, get_detector

MAX_CONTENT_LENGTH = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MIN_IMAGE_EDGE = 8

LOGGER = logging.getLogger(__name__)


class UploadError(Exception):
    """An upload the endpoint refuses to analyse."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


def read_upload() -> Image.Image:
    """Validate the multipart upload and return a decoded PNG."""
    if request.content_length and request.content_length > MAX_CONTENT_LENGTH:
        raise UploadError("File exceeds the 5 MB limit.", 413)

    upload = request.files.get("file")
    if upload is None:
        raise UploadError('Send the image as a multipart field named "file".')

    payload = upload.read()
    if not payload:
        raise UploadError("The uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(payload))
    except UnidentifiedImageError:
        raise UploadError("The file is not a PNG image.") from None
    except OSError:
        raise UploadError("The image could not be decoded.") from None

    if image.format != "PNG":
        raise UploadError(f"Only PNG images are supported (received {image.format}).")

    width, height = image.size
    if width < MIN_IMAGE_EDGE or height < MIN_IMAGE_EDGE:
        raise UploadError("The image is too small to analyse.")
    if width * height > MAX_IMAGE_PIXELS:
        raise UploadError("The image is too large; the limit is 40 megapixels.")

    try:
        image.load()
    except OSError:
        raise UploadError("The image could not be decoded.") from None

    return image


def create_app() -> Flask:
    """Build the Flask application."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/predict")
    def predict():
        image = read_upload()
        return jsonify(get_detector().predict(image))

    @app.errorhandler(UploadError)
    def handle_upload_error(error: UploadError):
        return jsonify({"error": error.message}), error.status

    @app.errorhandler(413)
    def request_entity_too_large(_error):
        return jsonify({"error": "File exceeds the 5 MB limit."}), 413

    @app.errorhandler(ArtifactsMissingError)
    def handle_missing_artifacts(error: ArtifactsMissingError):
        LOGGER.error("Model artifacts unavailable: %s", error)
        return (
            jsonify({"error": "The model artifacts are not installed on the server."}),
            503,
        )

    @app.errorhandler(Exception)
    def handle_unexpected(error: Exception):
        if isinstance(error, HTTPException):
            return error
        LOGGER.exception("Prediction failed")
        return jsonify({"error": "The image could not be analysed."}), 500

    return app


app = create_app()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
