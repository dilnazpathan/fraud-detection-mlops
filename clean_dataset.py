import pandas as pd
from pathlib import Path

# Input and output file paths
input_path = Path("data/raw/PS_20174392719_1491204439457_log.csv")
output_path = Path("data/processed/transactions_cleaned.csv")

# Valid transaction types
valid_types = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]

numeric_columns = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

total_rows = 0
cleaned_rows = 0
first_chunk = True

# Create the processed folder if needed
output_path.parent.mkdir(parents=True, exist_ok=True)

# Read the large CSV in chunks
for chunk in pd.read_csv(input_path, chunksize=500_000):
    total_rows += len(chunk)

    # Standardize transaction types
    chunk["type"] = chunk["type"].astype("string").str.strip().str.upper()

    # Convert numeric columns
    for column in numeric_columns:
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")

    # Validate numeric values
    valid = chunk[numeric_columns].notna().all(axis=1)
    valid &= chunk["step"] > 0
    valid &= chunk["amount"] >= 0

    # Check balances are non-negative
    balance_columns = [
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]

    for column in balance_columns:
        valid &= chunk[column] >= 0

    # Check fraud labels and transaction types
    valid &= chunk["isFraud"].isin([0, 1])
    valid &= chunk["isFlaggedFraud"].isin([0, 1])
    valid &= chunk["type"].isin(valid_types)

    # Keep valid transactions
    clean_chunk = chunk.loc[valid].copy()
    cleaned_rows += len(clean_chunk)

    # Write each chunk to the processed CSV
    clean_chunk.to_csv(
        output_path,
        index=False,
        mode="w" if first_chunk else "a",
        header=first_chunk,
    )

    first_chunk = False

print("DATA CLEANING COMPLETED")
print("Original transactions:", total_rows)
print("Cleaned transactions:", cleaned_rows)
print("Rows removed:", total_rows - cleaned_rows)
print("Saved file:", output_path)