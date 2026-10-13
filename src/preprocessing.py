from pathlib import Path

import joblib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

CHUNK_SIZE = 500_000

# Input datasets
TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEST_PATH = DATA_DIR / "test.csv"

# Numerical features
NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
    "origin_balance_change",
    "destination_balance_change",
    "errorBalanceOrig",
    "errorBalanceDest",
    "hour",
    "day",
]

# Transaction categories for one-hot encoding
TRANSACTION_TYPES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]


def fit_scaler():
    """Fit the scaler using training data only."""

    scaler = StandardScaler()

    print("Fitting scaler on training data...")

    for chunk in pd.read_csv(
        TRAIN_PATH,
        usecols=NUMERIC_FEATURES,
        chunksize=CHUNK_SIZE,
    ):
        scaler.partial_fit(chunk[NUMERIC_FEATURES])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    scaler_path = MODELS_DIR / "standard_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    print("Scaler saved:", scaler_path)

    return scaler


def preprocess_file(input_path, output_path, scaler):
    """Scale numerical features and encode transaction type."""

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    # Remove an old output before rebuilding it.
    output_path.unlink(missing_ok=True)

    writer = None
    total_rows = 0

    try:
        for chunk in pd.read_csv(
            input_path,
            chunksize=CHUNK_SIZE,
        ):
            # Scale numerical features using the training scaler.
            scaled_values = scaler.transform(
                chunk[NUMERIC_FEATURES]
            )

            processed = pd.DataFrame(
                scaled_values,
                columns=NUMERIC_FEATURES,
            )

            # One-hot encode transaction type.
            for transaction_type in TRANSACTION_TYPES:
                column_name = f"type_{transaction_type}"

                processed[column_name] = (
                    chunk["type"].to_numpy() == transaction_type
                ).astype("int8")

            # Keep the fraud label separate from the input features.
            processed["isFraud"] = (
                chunk["isFraud"].astype("int8").to_numpy()
            )

            # Write each chunk to a compressed Parquet file.
            table = pa.Table.from_pandas(
                processed,
                preserve_index=False,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    str(output_path),
                    table.schema,
                    compression="snappy",
                )

            writer.write_table(table)
            total_rows += len(chunk)

    finally:
        if writer is not None:
            writer.close()

    print(f"Processed {total_rows:,} rows")
    print("Saved:", output_path)

    return total_rows


def main():
    print("STARTING PREPROCESSING")

    # Fit only on training data to prevent data leakage.
    scaler = fit_scaler()

    datasets = {
        "train": TRAIN_PATH,
        "validation": VALIDATION_PATH,
        "test": TEST_PATH,
    }

    for name, input_path in datasets.items():
        output_path = DATA_DIR / f"{name}_preprocessed.parquet"

        print(f"\nPreprocessing {name} dataset...")

        preprocess_file(
            input_path,
            output_path,
            scaler,
        )

    print("\nPREPROCESSING COMPLETED")


if __name__ == "__main__":
    main()