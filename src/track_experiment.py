from pathlib import Path
import json

import joblib
import pandas as pd
import mlflow


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODELS_DIR / "random_forest_tuned.joblib"
THRESHOLD_PATH = MODELS_DIR / "fraud_threshold.json"

FINAL_RESULTS_PATH = REPORTS_DIR / "final_evaluation.csv"
COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
TUNING_RESULTS_PATH = REPORTS_DIR / "random_forest_tuning_results.csv"

# SQLite tracking database
TRACKING_DB = PROJECT_ROOT / "mlflow.db"


def main():
    # Check required files
    required_files = [
        MODEL_PATH,
        THRESHOLD_PATH,
        FINAL_RESULTS_PATH,
        COMPARISON_PATH,
        TUNING_RESULTS_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # Load the model, threshold, and test results
    model = joblib.load(MODEL_PATH)

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        threshold_info = json.load(file)

    final_results = pd.read_csv(FINAL_RESULTS_PATH).iloc[0]

    # Configure MLflow to use SQLite instead of the file-based
    # tracking backend.
    tracking_uri = f"sqlite:///{TRACKING_DB.resolve().as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.set_experiment("Fraud-Detection-MLOps")

    print("Starting MLflow experiment tracking...")

    with mlflow.start_run(
        run_name="tuned_random_forest_final_evaluation"
    ):
        # Record model parameters
        mlflow.log_params({
            "model_type": type(model).__name__,
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "min_samples_split": model.min_samples_split,
            "min_samples_leaf": model.min_samples_leaf,
            "max_features": str(model.max_features),
            "threshold": threshold_info["threshold"],
            "target_recall": threshold_info["target_recall"],
            "training_resampling": "random_undersampling_10_to_1",
        })

        # Record final test metrics
        mlflow.log_metrics({
            "test_accuracy": float(final_results["Accuracy"]),
            "test_fraud_precision": float(final_results["Fraud Precision"]),
            "test_fraud_recall": float(final_results["Fraud Recall"]),
            "test_fraud_f1_score": float(final_results["Fraud F1-score"]),
            "test_roc_auc": float(final_results["ROC-AUC"]),
            "test_average_precision": float(
                final_results["Average Precision"]
            ),
        })

        # Record model and threshold files
        mlflow.log_artifact(
            str(MODEL_PATH),
            artifact_path="model",
        )
        mlflow.log_artifact(
            str(THRESHOLD_PATH),
            artifact_path="model",
        )

        # Record experiment reports
        mlflow.log_artifact(
            str(FINAL_RESULTS_PATH),
            artifact_path="reports",
        )
        mlflow.log_artifact(
            str(COMPARISON_PATH),
            artifact_path="reports",
        )
        mlflow.log_artifact(
            str(TUNING_RESULTS_PATH),
            artifact_path="reports",
        )

        run_id = mlflow.active_run().info.run_id

    print("\nMLFLOW TRACKING COMPLETED")
    print("Experiment:", experiment.name)
    print("Experiment ID:", experiment.experiment_id)
    print("Run ID:", run_id)
    print("MLflow database:", TRACKING_DB)


if __name__ == "__main__":
    main()