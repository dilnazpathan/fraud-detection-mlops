from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_PATH = DATA_DIR / "train_resampled.parquet"
VALIDATION_PATH = DATA_DIR / "validation_preprocessed.parquet"

BASELINE_PATH = MODELS_DIR / "logistic_regression_baseline.joblib"
RESULTS_PATH = REPORTS_DIR / "model_comparison.csv"


def evaluate_model(name, model, X_validation, y_validation):
    """Evaluate a model on the same validation dataset."""

    print(f"\nEvaluating {name}...")

    y_pred = model.predict(X_validation)
    y_probability = model.predict_proba(X_validation)[:, 1]

    accuracy = accuracy_score(y_validation, y_pred)
    precision = precision_score(
        y_validation, y_pred, zero_division=0
    )
    recall = recall_score(
        y_validation, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_validation, y_pred, zero_division=0
    )
    roc_auc = roc_auc_score(y_validation, y_probability)
    average_precision = average_precision_score(
        y_validation, y_probability
    )

    cm = confusion_matrix(y_validation, y_pred)

    print("Accuracy:", round(accuracy, 4))
    print("Fraud Precision:", round(precision, 4))
    print("Fraud Recall:", round(recall, 4))
    print("Fraud F1-score:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc, 4))
    print("Average Precision:", round(average_precision, 4))
    print("Confusion Matrix:")
    print(cm)

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Fraud Precision": precision,
        "Fraud Recall": recall,
        "Fraud F1-score": f1,
        "ROC-AUC": roc_auc,
        "Average Precision": average_precision,
    }


def main():
    print("Loading training and validation datasets...")

    train_df = pd.read_parquet(TRAIN_PATH)
    validation_df = pd.read_parquet(VALIDATION_PATH)

    X_train = train_df.drop(columns=["isFraud"])
    y_train = train_df["isFraud"]

    X_validation = validation_df.drop(columns=["isFraud"])
    y_validation = validation_df["isFraud"]

    # Ensure validation uses the same feature order as training.
    X_validation = X_validation[X_train.columns]

    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_validation))
    print("Features:", X_train.shape[1])

    results = []

    # Model 1: Load the existing Logistic Regression baseline.
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Baseline model not found: {BASELINE_PATH}"
        )

    baseline_model = joblib.load(BASELINE_PATH)

    results.append(
        evaluate_model(
            "Logistic Regression",
            baseline_model,
            X_validation,
            y_validation,
        )
    )

    # Model 2: Random Forest
    print("\nTraining Random Forest...")

    random_forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    random_forest.fit(X_train, y_train)

    joblib.dump(
        random_forest,
        MODELS_DIR / "random_forest.joblib",
    )

    results.append(
        evaluate_model(
            "Random Forest",
            random_forest,
            X_validation,
            y_validation,
        )
    )

    # Model 3: XGBoost
    print("\nTraining XGBoost...")

    xgboost_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    xgboost_model.fit(X_train, y_train)

    joblib.dump(
        xgboost_model,
        MODELS_DIR / "xgboost.joblib",
    )

    results.append(
        evaluate_model(
            "XGBoost",
            xgboost_model,
            X_validation,
            y_validation,
        )
    )

    # Save the comparison table.
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="Average Precision",
        ascending=False,
    )

    results_df.to_csv(RESULTS_PATH, index=False)

    print("\nMODEL COMPARISON SUMMARY")
    print(results_df.round(4).to_string(index=False))

    print("\nComparison saved to:")
    print(RESULTS_PATH)

    print("\nMODEL COMPARISON COMPLETED")


if __name__ == "__main__":
    main()