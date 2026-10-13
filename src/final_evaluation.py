from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODELS_DIR / "random_forest_tuned.joblib"
TEST_PATH = DATA_DIR / "test_preprocessed.parquet"
THRESHOLD_PATH = MODELS_DIR / "fraud_threshold.json"

RESULTS_PATH = REPORTS_DIR / "final_evaluation.csv"


def main():
    # Check required files
    for path in [MODEL_PATH, TEST_PATH, THRESHOLD_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print("Loading tuned Random Forest model...")
    model = joblib.load(MODEL_PATH)

    print("Loading optimized threshold...")
    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        threshold_info = json.load(file)

    threshold = float(threshold_info["threshold"])
    print("Selected threshold:", threshold)

    print("\nLoading untouched test dataset...")
    test_df = pd.read_parquet(TEST_PATH)

    X_test = test_df.drop(columns=["isFraud"])
    y_test = test_df["isFraud"]

    # Ensure feature order matches model training
    X_test = X_test[model.feature_names_in_]

    print("Test transactions:", len(X_test))
    print("Actual fraud transactions:", int(y_test.sum()))
    print("Actual legitimate transactions:", int((y_test == 0).sum()))

    # Get predicted fraud probabilities
    print("\nGenerating test predictions...")
    y_probability = model.predict_proba(X_test)[:, 1]

    # Apply threshold selected using validation data
    y_pred = (y_probability >= threshold).astype(int)

    # Calculate final metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_probability)
    average_precision = average_precision_score(y_test, y_probability)

    print("\nFINAL TEST RESULTS")
    print("Accuracy:", round(accuracy, 4))
    print("Fraud Precision:", round(precision, 4))
    print("Fraud Recall:", round(recall, 4))
    print("Fraud F1-score:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc, 4))
    print("Average Precision:", round(average_precision, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the test evaluation summary
    results = pd.DataFrame([{
        "Model": "Tuned Random Forest",
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Fraud Precision": precision,
        "Fraud Recall": recall,
        "Fraud F1-score": f1,
        "ROC-AUC": roc_auc,
        "Average Precision": average_precision,
        "Test Transactions": len(X_test),
        "Actual Fraud Transactions": int(y_test.sum()),
    }])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)

    print("\nFinal evaluation saved to:")
    print(RESULTS_PATH)

    print("\nFINAL MODEL EVALUATION COMPLETED")


if __name__ == "__main__":
    main()