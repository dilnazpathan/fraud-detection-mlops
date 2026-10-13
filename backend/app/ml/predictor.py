from .model_loader import MODEL, SCALER, FRAUD_THRESHOLD
from .preprocessor import preprocess_transaction


def predict_transaction(transaction: dict) -> dict:
    """Predict whether a transaction is fraudulent."""

    # Prepare the transaction in the model's expected format
    processed_data = preprocess_transaction(
        transaction,
        SCALER,
    )

    # Calculate the probability of fraud
    fraud_probability = float(
        MODEL.predict_proba(processed_data)[0, 1]
    )

    # Apply the optimized production threshold
    is_fraud = fraud_probability >= FRAUD_THRESHOLD

    if is_fraud:
        prediction = "FRAUD"
    else:
        prediction = "LEGITIMATE"

    # Return the prediction details
    return {
        "prediction": prediction,
        "fraud_probability": round(fraud_probability, 6),
        "fraud_probability_percent": round(
            fraud_probability * 100, 2
        ),
        "threshold": round(FRAUD_THRESHOLD, 6),
    }