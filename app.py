"""
app.py  -  LIME Explainer Flask backend
Run:  python app.py
Then open: http://localhost:5000
"""

from __future__ import annotations

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from cnn_models     import CNN_MODEL_INFO
from models         import get_text_model_info, get_tabular_model_info
from lime_explainer import explain_text, explain_tabular, explain_image
from openai_wrapper import validate_openai_key

app = Flask(__name__)
CORS(app)


# --- page ---

@app.route("/")
def index():
    return render_template("index.html")


# --- model metadata ---

@app.route("/api/models/text", methods=["GET"])
def api_text_models():
    return jsonify({"models": get_text_model_info()})


@app.route("/api/models/tabular", methods=["GET"])
def api_tabular_models():
    return jsonify({"models": get_tabular_model_info()})


@app.route("/api/models/cnn", methods=["GET"])
def api_cnn_models():
    return jsonify({"models": CNN_MODEL_INFO})


# --- validation ---

@app.route("/api/validate-key", methods=["POST"])
def api_validate_key():
    data    = request.get_json(force=True)
    api_key = data.get("api_key", "").strip()
    if not api_key:
        return jsonify({"valid": False, "message": "No API key provided."})
    ok, msg = validate_openai_key(api_key)
    return jsonify({"valid": ok, "message": msg})


# --- explain: text ---

@app.route("/api/explain/text", methods=["POST"])
def api_explain_text():
    data = request.get_json(force=True)

    text        = data.get("text", "").strip()
    model_type  = data.get("model_type", "sentiment")
    openai_key  = data.get("openai_key", "").strip()
    task        = data.get("task", "sentiment analysis").strip()
    class_names = data.get("class_names", None)
    n_samples   = int(data.get("n_samples", 150))

    if not text:
        return jsonify({"error": "Please provide some text to explain."}), 400

    if isinstance(class_names, str):
        class_names = [c.strip() for c in class_names.split(",") if c.strip()]
    if not class_names:
        class_names = None

    result = explain_text(
        text        = text,
        model_type  = model_type,
        openai_key  = openai_key,
        task        = task,
        class_names = class_names,
        n_samples   = n_samples,
    )
    status = 200 if result.get("success") else 500
    return jsonify(result), status


# --- explain: tabular ---

@app.route("/api/explain/tabular", methods=["POST"])
def api_explain_tabular():
    data = request.get_json(force=True)

    model_type = data.get("model_type", "iris")
    features   = data.get("features", {})
    n_samples  = int(data.get("n_samples", 500))

    try:
        features = {k: float(v) for k, v in features.items()}
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid feature values: {exc}"}), 400

    if not features:
        return jsonify({"error": "Please provide feature values."}), 400

    result = explain_tabular(
        features   = features,
        model_type = model_type,
        n_samples  = n_samples,
    )
    status = 200 if result.get("success") else 500
    return jsonify(result), status


# --- explain: image ---

@app.route("/api/explain/image", methods=["POST"])
def api_explain_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file. Send multipart/form-data with key 'image'."}), 400

    file       = request.files["image"]
    model_type = request.form.get("model_type", "resnet50")
    n_samples  = int(request.form.get("n_samples", 200))
    num_feat   = int(request.form.get("num_features", 6))

    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    image_bytes = file.read()
    result = explain_image(
        image_bytes  = image_bytes,
        model_type   = model_type,
        n_samples    = n_samples,
        num_features = num_feat,
    )
    status = 200 if result.get("success") else 500
    return jsonify(result), status


# --- entry point ---

if __name__ == "__main__":
    print("=" * 60)
    print("  LIME Explainer  -  http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, port=5000, threaded=True, use_reloader=False)
