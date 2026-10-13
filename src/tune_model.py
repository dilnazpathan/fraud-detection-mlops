from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
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

MODEL_PATH = MODELS_DIR / "random_forest_tuned.joblib"
RESULTS_PATH = REPORTS_DIR / "random_forest_tuning_results.csv"


def main():
    print("Loading training and validation datasets...")

    train_df = pd.read_parquet(TRAIN_PATH)
    validation_df = pd.read_parquet(VALIDATION_PATH)

    # Separate features and target
    X_train = train_df.drop(columns=["isFraud"])
    y_train = train_df["isFraud"]

    X_validation = validation_df.drop(columns=["isFraud"])
    y_validation = validation_df["isFraud"]

    # Ensure validation columns use the training column order
    X_validation = X_validation[X_train.columns]

    print("Training rows:", len(X_train))
    print("Validation rows:", len(X_validation))

    # Parameters to test
    parameter_grid = {
        "n_estimators": [100, 150],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt"],
    }

    # Create the base model
    random_forest = RandomForestClassifier(
        random_state=42,
        n_jobs=1,
    )

    # Randomized hyperparameter search
    search = RandomizedSearchCV(
        estimator=random_forest,
        param_distributions=parameter_grid,
        n_iter=6,
        scoring="average_precision",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=2,
        refit=True,
    )

    print("\nStarting hyperparameter tuning...")
    print("This may take several minutes.")

    search.fit(X_train, y_train)

    # Get the best model
    best_model = search.best_estimator_

    print("\nBEST PARAMETERS")
    print(search.best_params_)

    print("\nBest cross-validation Average Precision:")
    print(round(search.best_score_, 4))

    # Evaluate the tuned model on validation data
    print("\nEvaluating tuned model on validation data...")

    y_pred = best_model.predict(X_validation)
    y_probability = best_model.predict_proba(X_validation)[:, 1]

    print("\nTUNED RANDOM FOREST RESULTS")
    print("Accuracy:", round(accuracy_score(y_validation, y_pred), 4))
    print(
        "Fraud Precision:",
        round(precision_score(y_validation, y_pred, zero_division=0), 4),
    )
    print(
        "Fraud Recall:",
        round(recall_score(y_validation, y_pred, zero_division=0), 4),
    )
    print(
        "Fraud F1-score:",
        round(f1_score(y_validation, y_pred, zero_division=0), 4),
    )
    print("ROC-AUC:", round(roc_auc_score(y_validation, y_probability), 4))
    print(
        "Average Precision:",
        round(average_precision_score(y_validation, y_probability), 4),
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_validation, y_pred))

    # Save the tuned model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    # Save the search results
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(search.cv_results_)
    results_df.to_csv(RESULTS_PATH, index=False)

    print("\nTuned model saved to:")
    print(MODEL_PATH)

    print("\nTuning results saved to:")
    print(RESULTS_PATH)

    print("\nHYPERPARAMETER TUNING COMPLETED")


if __name__ == "__main__":
    main()