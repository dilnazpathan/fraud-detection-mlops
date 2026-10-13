from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_PATH = DATA_DIR / "train_resampled.parquet"
VALIDATION_PATH = DATA_DIR / "validation_preprocessed.parquet"

MODEL_PATH = MODELS_DIR / "logistic_regression_baseline.joblib"


def main():
    # Check that the input datasets exist
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_PATH}")

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation data not found: {VALIDATION_PATH}"
        )

    print("Loading training and validation datasets...")

    train_df = pd.read_parquet(TRAIN_PATH)
    validation_df = pd.read_parquet(VALIDATION_PATH)

    # Separate features and target
    X_train = train_df.drop(columns=["isFraud"])
    y_train = train_df["isFraud"]

    X_validation = validation_df.drop(columns=["isFraud"])
    y_validation = validation_df["isFraud"]

    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_validation))
    print("Number of features:", X_train.shape[1])

    # Create the baseline model
    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train, y_train)

    print("Model training completed.")

    # Predict on the untouched validation dataset
    print("\nEvaluating on validation data...")

    y_pred = model.predict(X_validation)
    y_probability = model.predict_proba(X_validation)[:, 1]

    # Evaluate the model
    print("\nBASELINE MODEL RESULTS")
    print("Accuracy:", round(accuracy_score(y_validation, y_pred), 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_validation,
            y_pred,
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_validation, y_pred))

    print(
        "ROC-AUC:",
        round(roc_auc_score(y_validation, y_probability), 4),
    )

    print(
        "Average Precision:",
        round(average_precision_score(y_validation, y_probability), 4),
    )

    # Save the baseline model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nBaseline model saved to:")
    print(MODEL_PATH)


if __name__ == "__main__":
    main()