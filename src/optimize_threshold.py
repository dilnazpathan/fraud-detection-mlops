from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
)


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "random_forest_tuned.joblib"
VALIDATION_PATH = DATA_DIR / "validation_preprocessed.parquet"
THRESHOLD_PATH = MODELS_DIR / "fraud_threshold.json"

# Required minimum fraud recall
TARGET_RECALL = 0.99


def main():
    print("Loading tuned Random Forest model...")

    model = joblib.load(MODEL_PATH)

    print("Loading validation dataset...")

    validation_df = pd.read_parquet(VALIDATION_PATH)

    X_validation = validation_df.drop(columns=["isFraud"])
    y_validation = validation_df["isFraud"]

    # Ensure the feature order matches training.
    X_validation = X_validation[model.feature_names_in_]

    print("Validation transactions:", len(X_validation))
    print("Actual fraud transactions:", int(y_validation.sum()))

    # Get fraud probabilities.
    print("\nCalculating fraud probabilities...")

    y_probability = model.predict_proba(X_validation)[:, 1]

    # Evaluate the default threshold of 0.5.
    default_predictions = (y_probability >= 0.5).astype(int)

    print("\nRESULTS AT DEFAULT THRESHOLD (0.5)")
    print(
        "Precision:",
        round(precision_score(y_validation, default_predictions), 4),
    )
    print(
        "Recall:",
        round(recall_score(y_validation, default_predictions), 4),
    )
    print(
        "F1-score:",
        round(f1_score(y_validation, default_predictions), 4),
    )

    # Calculate precision and recall across possible thresholds.
    precision, recall, thresholds = precision_recall_curve(
        y_validation,
        y_probability,
    )

    # Keep thresholds that achieve the required fraud recall.
    eligible_indices = [
        i for i in range(len(thresholds))
        if recall[i] >= TARGET_RECALL
    ]

    if not eligible_indices:
        raise RuntimeError(
            f"No threshold achieves recall >= {TARGET_RECALL:.0%}."
        )

    # Choose the eligible threshold with the highest precision.
    best_index = max(
        eligible_indices,
        key=lambda i: precision[i],
    )

    best_threshold = float(thresholds[best_index])

    # Calculate predictions using the selected threshold.
    optimized_predictions = (
        y_probability >= best_threshold
    ).astype(int)

    optimized_precision = precision_score(
        y_validation,
        optimized_predictions,
        zero_division=0,
    )
    optimized_recall = recall_score(
        y_validation,
        optimized_predictions,
        zero_division=0,
    )
    optimized_f1 = f1_score(
        y_validation,
        optimized_predictions,
        zero_division=0,
    )
    optimized_accuracy = accuracy_score(
        y_validation,
        optimized_predictions,
    )

    print("\nOPTIMIZED THRESHOLD RESULTS")
    print("Target fraud recall:", f"{TARGET_RECALL:.2%}")
    print("Selected threshold:", round(best_threshold, 6))
    print("Accuracy:", round(optimized_accuracy, 4))
    print("Fraud precision:", round(optimized_precision, 4))
    print("Fraud recall:", round(optimized_recall, 4))
    print("Fraud F1-score:", round(optimized_f1, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_validation, optimized_predictions))

    # Save the selected threshold for later deployment.
    threshold_info = {
        "threshold": best_threshold,
        "target_recall": TARGET_RECALL,
        "validation_precision": float(optimized_precision),
        "validation_recall": float(optimized_recall),
        "validation_f1_score": float(optimized_f1),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(THRESHOLD_PATH, "w", encoding="utf-8") as file:
        json.dump(threshold_info, file, indent=4)

    print("\nThreshold configuration saved to:")
    print(THRESHOLD_PATH)

    print("\nTHRESHOLD OPTIMIZATION COMPLETED")


if __name__ == "__main__":
    main()