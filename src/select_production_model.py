from pathlib import Path
import shutil

import joblib
import json


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Existing trained artifacts
TUNED_MODEL = MODELS_DIR / "random_forest_tuned.joblib"
THRESHOLD_FILE = MODELS_DIR / "fraud_threshold.json"
SCALER_FILE = MODELS_DIR / "standard_scaler.joblib"

# Production artifacts
PRODUCTION_MODEL = MODELS_DIR / "production_model.joblib"
PRODUCTION_THRESHOLD = MODELS_DIR / "production_threshold.json"
PRODUCTION_SCALER = MODELS_DIR / "production_scaler.joblib"


def main():
    print("SELECTING PRODUCTION MODEL")
    print()

    # Check that all required artifacts exist
    required_files = [
        TUNED_MODEL,
        THRESHOLD_FILE,
        SCALER_FILE,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    # Load and verify the trained model
    model = joblib.load(TUNED_MODEL)

    print("Selected model:", type(model).__name__)
    print("Model file:", TUNED_MODEL.name)

    # Read the optimized threshold
    with open(THRESHOLD_FILE, "r", encoding="utf-8") as file:
        threshold_info = json.load(file)

    print("Fraud threshold:", threshold_info["threshold"])

    # Copy the selected artifacts into production filenames
    shutil.copy2(TUNED_MODEL, PRODUCTION_MODEL)
    shutil.copy2(THRESHOLD_FILE, PRODUCTION_THRESHOLD)
    shutil.copy2(SCALER_FILE, PRODUCTION_SCALER)

    print("\nPRODUCTION ARTIFACTS SAVED")
    print("Model:", PRODUCTION_MODEL)
    print("Threshold:", PRODUCTION_THRESHOLD)
    print("Scaler:", PRODUCTION_SCALER)

    print("\nProduction model feature count:",
          len(model.feature_names_in_))

    print("\nPRODUCTION MODEL SELECTION COMPLETED")


if __name__ == "__main__":
    main()