
from pathlib import Path
import threading

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge
from ai_edge_litert.interpreter import Interpreter

app = Flask(__name__)

# Maximum image upload size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "pneumonia_model.tflite"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Load lightweight model once
interpreter = Interpreter(
    model_path=str(MODEL_PATH),
    num_threads=1
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prevent simultaneous access to the interpreter
prediction_lock = threading.Lock()


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def preprocess_image(image):
    image = ImageOps.exif_transpose(image)
    image = image.convert("L")
    image = image.resize(
        (160, 160),
        Image.Resampling.LANCZOS
    )

    array = np.asarray(image, dtype=np.float32)

    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)

    # The model normalizes pixels internally
    return array


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "application": "PneumoDetect-AI",
        "status": "ok"
    })


@app.errorhandler(RequestEntityTooLarge)
def file_too_large(error):
    return jsonify({
        "error": "Image size must be less than 10 MB."
    }), 413


@app.route("/predict", methods=["POST"])
def predict():

    if "xray" not in request.files:
        return jsonify({
            "error": "Please select an X-ray image."
        }), 400

    file = request.files["xray"]

    if not file.filename:
        return jsonify({
            "error": "No file selected."
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": "Please upload a PNG, JPG, or JPEG image."
        }), 400

    try:
        with Image.open(file.stream) as image:
            image.load()
            input_array = preprocess_image(image)

        with prediction_lock:
            interpreter.set_tensor(
                input_details[0]["index"],
                input_array
            )

            interpreter.invoke()

            raw_prediction = interpreter.get_tensor(
                output_details[0]["index"]
            )

        pneumonia_score = float(
            np.asarray(raw_prediction).reshape(-1)[0]
        )

        pneumonia_score = float(
            np.clip(pneumonia_score, 0.0, 1.0)
        )

        if pneumonia_score >= 0.5:
            label = "PNEUMONIA"
            confidence = pneumonia_score
        else:
            label = "NORMAL"
            confidence = 1.0 - pneumonia_score

        return jsonify({
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "pneumonia_score": round(
                pneumonia_score * 100, 2
            ),
            "normal_score": round(
                (1.0 - pneumonia_score) * 100, 2
            )
        })

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError
    ):
        return jsonify({
            "error": "The uploaded file is not a valid readable image."
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")

        return jsonify({
            "error": "Prediction failed. Please check server logs."
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
