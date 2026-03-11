"""
Titanic Survival Analysis — Flask Application
=============================================
Serves the web UI and exposes a JSON prediction API.
All ML logic is kept separate from routing concerns.
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR    = PROJECT_ROOT / "models"
VIS_DIR      = PROJECT_ROOT / "visualizations"


# ─── Load ML Artifacts Once at Startup ───────────────────────────────────────
def load_artifacts():
    """Load model, encoders, scaler, and feature list from disk."""
    try:
        model         = joblib.load(MODEL_DIR / "best_model.joblib")
        encoders      = joblib.load(MODEL_DIR / "encoders.joblib")
        scaler        = joblib.load(MODEL_DIR / "scaler.joblib")
        feature_names = joblib.load(MODEL_DIR / "feature_names.joblib")
        print("✓ ML artifacts loaded successfully.")
        return model, encoders, scaler, feature_names
    except FileNotFoundError as e:
        print(f"⚠ Warning: {e}")
        print("  Run src/pipeline.py first to train and save the model.")
        return None, None, None, None


MODEL, ENCODERS, SCALER, FEATURE_NAMES = load_artifacts()


# ─── Feature Engineering Helpers ─────────────────────────────────────────────
def infer_title(sex: str, age: float) -> str:
    """Infer passenger title from sex and age."""
    if age < 18:
        return "Master" if sex == "male" else "Miss"
    return "Mr" if sex == "male" else "Mrs"


def build_input_row(pclass, sex, age, fare, embarked, sibsp, parch) -> dict:
    """
    Apply the same feature engineering as pipeline.py:
    FamilySize, IsAlone, Title, label-encode categoricals.
    Returns a raw dict ready to be scaled and predicted on.
    """
    family_size = int(sibsp) + int(parch) + 1
    is_alone    = int(family_size == 1)
    title       = infer_title(sex, age)

    sex_enc      = int(ENCODERS["sex"].transform([sex])[0])
    embarked_enc = int(ENCODERS["embarked"].transform([embarked])[0])

    # Handle unseen titles gracefully
    title_classes = list(ENCODERS["title"].classes_)
    if title not in title_classes:
        title = title_classes[0]
    title_enc = int(ENCODERS["title"].transform([title])[0])

    return {
        "Pclass":     int(pclass),
        "Sex":        sex_enc,
        "Age":        float(age),
        "Fare":       float(fare),
        "Embarked":   embarked_enc,
        "FamilySize": family_size,
        "IsAlone":    is_alone,
        "Title":      title_enc,
    }


def predict_survival(input_data: dict):
    """
    Scale input and run inference.
    Returns (prediction: int, probability: float).
    """
    df     = pd.DataFrame([input_data], columns=FEATURE_NAMES)
    scaled = pd.DataFrame(SCALER.transform(df), columns=FEATURE_NAMES)
    pred   = int(MODEL.predict(scaled)[0])
    prob   = float(MODEL.predict_proba(scaled)[0][1])
    return pred, prob


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the main dashboard page."""
    # Check which visualizations are available to pass to template
    vis_files = {
        "correlation_heatmap": (VIS_DIR / "correlation_heatmap.png").exists(),
        "survival_gender":     (VIS_DIR / "survival_gender.png").exists(),
        "survival_pclass":     (VIS_DIR / "survival_pclass.png").exists(),
        "feature_importance":  (VIS_DIR / "feature_importance.png").exists(),
        "roc_curves":          (VIS_DIR / "roc_curves.png").exists(),
        "confusion_matrices":  (VIS_DIR / "confusion_matrices.png").exists(),
    }
    model_ready = MODEL is not None
    return render_template("index.html", vis=vis_files, model_ready=model_ready)


@app.route("/predict", methods=["POST"])
def predict():
    """
    JSON API endpoint for survival prediction.

    Expects JSON body:
        { pclass, sex, age, fare, embarked, sibsp, parch }

    Returns JSON:
        { survived, probability, title, family_size, is_alone, error? }
    """
    if MODEL is None:
        return jsonify({"error": "Model not loaded. Run the pipeline first."}), 503

    try:
        data = request.get_json(force=True)

        # Validate required fields
        required = ["pclass", "sex", "age", "fare", "embarked", "sibsp", "parch"]
        missing  = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Validate ranges
        age  = float(data["age"])
        fare = float(data["fare"])
        if not (0 <= age <= 120):
            return jsonify({"error": "Age must be between 0 and 120"}), 400
        if fare < 0:
            return jsonify({"error": "Fare cannot be negative"}), 400

        input_data  = build_input_row(
            pclass   = data["pclass"],
            sex      = data["sex"],
            age      = age,
            fare     = fare,
            embarked = data["embarked"],
            sibsp    = data["sibsp"],
            parch    = data["parch"],
        )

        pred, prob = predict_survival(input_data)

        return jsonify({
            "survived":    pred,
            "probability": round(prob, 4),
            "title":       infer_title(data["sex"], age),
            "family_size": input_data["FamilySize"],
            "is_alone":    input_data["IsAlone"],
        })

    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/visualizations/<filename>")
def serve_visualization(filename):
    """Serve visualization images from the visualizations folder."""
    return send_from_directory(VIS_DIR, filename)


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # debug=False in production; set via env var
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    port  = int(os.environ.get("PORT", 5000))
    app.run(debug=debug, port=port)