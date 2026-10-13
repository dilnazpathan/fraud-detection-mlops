import math

import pandas as pd


# Numerical features used during model training
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

# Transaction types used during training
TRANSACTION_TYPES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]

# Final model feature order
MODEL_FEATURES = NUMERIC_FEATURES + [
    f"type_{transaction_type}"
    for transaction_type in TRANSACTION_TYPES
]


def preprocess_transaction(transaction: dict, scaler) -> pd.DataFrame:
    """Validate and preprocess one transaction for prediction."""

    required_fields = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]

    # Check required fields
    for field in required_fields:
        if field not in transaction:
            raise ValueError(f"Missing required field: {field}")

    # Validate transaction type
    transaction_type = str(transaction["type"]).strip().upper()

    if transaction_type not in TRANSACTION_TYPES:
        raise ValueError(
            f"Invalid transaction type: {transaction_type}"
        )

    # Convert numerical inputs and reject invalid numbers
    numeric_input_fields = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFlaggedFraud",
    ]

    values = {}

    for field in numeric_input_fields:
        raw_value = transaction.get(
            field,
            0 if field == "isFlaggedFraud" else None,
        )

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            raise ValueError(f"{field} must be a valid number.")

        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite.")

        values[field] = value

    # Validate step
    if values["step"] < 1 or not values["step"].is_integer():
        raise ValueError("step must be a positive integer.")

    # Validate amount and account balances
    non_negative_fields = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]

    for field in non_negative_fields:
        if values[field] < 0:
            raise ValueError(f"{field} cannot be negative.")

    # Validate the flag
    if values["isFlaggedFraud"] not in (0, 1):
        raise ValueError("isFlaggedFraud must be 0 or 1.")

    step = int(values["step"])

    # Create the features used during model training
    features = {
        "step": step,
        "amount": values["amount"],
        "oldbalanceOrg": values["oldbalanceOrg"],
        "newbalanceOrig": values["newbalanceOrig"],
        "oldbalanceDest": values["oldbalanceDest"],
        "newbalanceDest": values["newbalanceDest"],
        "isFlaggedFraud": int(values["isFlaggedFraud"]),

        "origin_balance_change": (
            values["oldbalanceOrg"] - values["newbalanceOrig"]
        ),

        "destination_balance_change": (
            values["newbalanceDest"] - values["oldbalanceDest"]
        ),

        "errorBalanceOrig": (
            values["oldbalanceOrg"]
            - values["amount"]
            - values["newbalanceOrig"]
        ),

        "errorBalanceDest": (
            values["oldbalanceDest"]
            + values["amount"]
            - values["newbalanceDest"]
        ),

        "hour": (step - 1) % 24,
        "day": ((step - 1) // 24) + 1,
    }

    # Scale the numerical features
    numeric_df = pd.DataFrame(
        [features],
        columns=NUMERIC_FEATURES,
    )

    scaled_values = scaler.transform(numeric_df)

    processed_df = pd.DataFrame(
        scaled_values,
        columns=NUMERIC_FEATURES,
    )

    # Encode transaction type
    for transaction_type_value in TRANSACTION_TYPES:
        column_name = f"type_{transaction_type_value}"

        processed_df[column_name] = int(
            transaction_type == transaction_type_value
        )

    # Return features in the exact model-training order
    return processed_df[MODEL_FEATURES]