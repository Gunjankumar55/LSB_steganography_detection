"""Flask entry point for the multi-payload LSB steganalysis demo."""

from __future__ import annotations

import io
import traceback

from flask import Flask, jsonify, render_template, request
from PIL import Image

from steganalysis.inference import get_detector

MAX_CONTENT_LENGTH = 5 * 1024 * 1024


def create_app() -> Flask:
    """Build the Flask application."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/predict")
    def predict():
        try:
            upload = request.files["file"]
            image = Image.open(io.BytesIO(upload.read()))
            return jsonify(get_detector().predict(image))
        except Exception as error:  # noqa: BLE001 - surfaced to the demo UI
            traceback.print_exc()
            return jsonify({"error": str(error)}), 500

    @app.errorhandler(413)
    def request_entity_too_large(_error):
        return jsonify({"error": "File size exceeds 5MB"}), 413

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
