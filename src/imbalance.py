from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from imblearn.under_sampling import RandomUnderSampler


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_PATH = DATA_DIR / "train_preprocessed.parquet"
OUTPUT_PATH = DATA_DIR / "train_resampled.parquet"

CHUNK_SIZE = 250_000
RANDOM_STATE = 42
TARGET_RATIO = 0.1  # Fraud / legitimate = 1 / 10


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {INPUT_PATH}")

    print("STEP 1: COUNTING TRAINING CLASSES")

    parquet_file = pq.ParquetFile(INPUT_PATH)

    legitimate_count = 0
    fraud_count = 0

    # First pass: count the two classes without loading all features.
    for batch in parquet_file.iter_batches(
        batch_size=CHUNK_SIZE,
        columns=["isFraud"],
    ):
        labels = batch.column(0).to_numpy()
        legitimate_count += int((labels == 0).sum())
        fraud_count += int((labels == 1).sum())

    print("Legitimate transactions:", legitimate_count)
    print("Fraud transactions:", fraud_count)

    if fraud_count == 0 or legitimate_count == 0:
        raise ValueError("Both transaction classes must be present.")

    # Keep a safety margin of legitimate candidate rows.
    target_legitimate = min(
        legitimate_count,
        fraud_count * 10,
    )

    sample_fraction = min(
        1.0,
        (target_legitimate * 1.25) / legitimate_count,
    )

    print("\nSTEP 2: SELECTING RANDOM CANDIDATES")

    legitimate_samples = []
    fraud_samples = []

    # Second pass: retain every fraud row and sample legitimate rows.
    for batch_number, batch in enumerate(
        parquet_file.iter_batches(batch_size=CHUNK_SIZE)
    ):
        chunk = batch.to_pandas()

        fraud_rows = chunk.loc[chunk["isFraud"] == 1]
        legitimate_rows = chunk.loc[chunk["isFraud"] == 0]

        if not fraud_rows.empty:
            fraud_samples.append(fraud_rows)

        if not legitimate_rows.empty:
            sampled_rows = legitimate_rows.sample(
                frac=sample_fraction,
                random_state=RANDOM_STATE + batch_number,
            )
            legitimate_samples.append(sampled_rows)

    legitimate_candidates = pd.concat(
        legitimate_samples,
        ignore_index=True,
    )

    all_fraud = pd.concat(
        fraud_samples,
        ignore_index=True,
    )

    print("Legitimate candidates:", len(legitimate_candidates))
    print("Fraud rows retained:", len(all_fraud))

    if len(legitimate_candidates) < target_legitimate:
        raise RuntimeError(
            "Not enough legitimate candidates for the requested ratio."
        )

    # Combine the candidate data and apply RandomUnderSampler.
    candidate_data = pd.concat(
        [legitimate_candidates, all_fraud],
        ignore_index=True,
    )

    X = candidate_data.drop(columns=["isFraud"])
    y = candidate_data["isFraud"]

    print("\nSTEP 3: APPLYING RANDOM UNDERSAMPLING")

    sampler = RandomUnderSampler(
        sampling_strategy=TARGET_RATIO,
        random_state=RANDOM_STATE,
    )

    X_resampled, y_resampled = sampler.fit_resample(X, y)

    resampled_data = X_resampled.copy()
    resampled_data["isFraud"] = y_resampled

    # Shuffle the rows before saving.
    resampled_data = resampled_data.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    resampled_data.to_parquet(
        OUTPUT_PATH,
        index=False,
        compression="snappy",
    )

    print("\nIMBALANCE HANDLING COMPLETED")
    print("Total training rows:", len(resampled_data))
    print(
        "Legitimate transactions:",
        int((resampled_data["isFraud"] == 0).sum()),
    )
    print(
        "Fraud transactions:",
        int((resampled_data["isFraud"] == 1).sum()),
    )
    print("Output file:", OUTPUT_PATH)


if __name__ == "__main__":
    main()