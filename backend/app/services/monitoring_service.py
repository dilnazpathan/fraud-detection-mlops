from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.database.models import PredictionRecord


def get_monitoring_summary(db: Session) -> dict:
    """Summarize the predictions saved in the database."""

    total_predictions = (
        db.query(func.count(PredictionRecord.id)).scalar() or 0
    )

    predicted_fraud = (
        db.query(func.count(PredictionRecord.id))
        .filter(PredictionRecord.prediction == "FRAUD")
        .scalar()
        or 0
    )

    predicted_legitimate = (
        db.query(func.count(PredictionRecord.id))
        .filter(PredictionRecord.prediction == "LEGITIMATE")
        .scalar()
        or 0
    )

    average_probability = (
        db.query(func.avg(PredictionRecord.fraud_probability)).scalar()
    )

    average_amount = (
        db.query(func.avg(PredictionRecord.amount)).scalar()
    )

    predicted_fraud_rate = (
        predicted_fraud / total_predictions * 100
        if total_predictions > 0
        else 0.0
    )

    return {
        "total_predictions": total_predictions,
        "predicted_fraud": predicted_fraud,
        "predicted_legitimate": predicted_legitimate,
        "predicted_fraud_rate_percent": round(
            predicted_fraud_rate, 2
        ),
        "average_fraud_probability": round(
            float(average_probability or 0.0), 4
        ),
        "average_transaction_amount": round(
            float(average_amount or 0.0), 2
        ),
    }