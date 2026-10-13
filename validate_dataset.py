import pandas as pd

path = "data/raw/PS_20174392719_1491204439457_log.csv"

total_rows = 0
missing_counts = {}
fraud_counts = {0: 0, 1: 0}
unexpected_targets = {}

for chunk in pd.read_csv(path, chunksize=500_000):
    total_rows += len(chunk)

    for column, count in chunk.isna().sum().items():
        missing_counts[column] = missing_counts.get(column, 0) + int(count)

    for label, count in chunk["isFraud"].value_counts(dropna=False).items():
        if pd.isna(label):
            continue

        if label in fraud_counts:
            fraud_counts[int(label)] += int(count)
        else:
            unexpected_targets[str(label)] = (
                unexpected_targets.get(str(label), 0) + int(count)
            )

print("Total transactions:", total_rows)

print("\nMissing values:")
for column, count in missing_counts.items():
    print(f"{column}: {count}")

print("\nTransaction classes:")
print("Legitimate (0):", fraud_counts[0])
print("Fraud (1):", fraud_counts[1])

if total_rows:
    print(f"Fraud percentage: {fraud_counts[1] / total_rows * 100:.4f}%")

print("\nUnexpected target values:", unexpected_targets or "None")