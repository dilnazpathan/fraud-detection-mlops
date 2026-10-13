from pathlib import Path
import json

import joblib


# Locate the project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"

# Production artifact paths
MODEL_PATH = MODELS_DIR / "production_model.joblib"
SCALER_PATH = MODELS_DIR / "production_scaler.joblib"
THRESHOLD_PATH = MODELS_DIR / "production_threshold.json"


def load_production_artifacts():
    """Load the production model, scaler, and fraud threshold."""

    required_files = [
        MODEL_PATH,
        SCALER_PATH,
        THRESHOLD_PATH,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Production artifact not found: {file_path}"
            )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        threshold_config = json.load(file)

    threshold = float(threshold_config["threshold"])

    if not 0 < threshold < 1:
        raise ValueError("Fraud threshold must be between 0 and 1.")

    return model, scaler, threshold


# Load the artifacts for use by the API
MODEL, SCALER, FRAUD_THRESHOLD = load_production_artifacts()