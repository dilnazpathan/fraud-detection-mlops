from sqlalchemy.orm import Session

from backend.app.database.models import PredictionRecord


def save_prediction(
    db: Session,
    transaction: dict,
    result: dict,
) -> PredictionRecord:
    """Save a transaction and its prediction to the database."""

    record = PredictionRecord(
        step=transaction["step"],
        transaction_type=transaction["type"],
        amount=transaction["amount"],
        oldbalanceOrg=transaction["oldbalanceOrg"],
        newbalanceOrig=transaction["newbalanceOrig"],
        oldbalanceDest=transaction["oldbalanceDest"],
        newbalanceDest=transaction["newbalanceDest"],
        isFlaggedFraud=transaction.get("isFlaggedFraud", 0),
        prediction=result["prediction"],
        fraud_probability=result["fraud_probability"],
        threshold=result["threshold"],
    )

    try:
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    except Exception:
        db.rollback()
        raise


def get_recent_predictions(
    db: Session,
    limit: int = 100,
) -> list[PredictionRecord]:
    """Return the most recent prediction records."""

    return (
        db.query(PredictionRecord)
        .order_by(PredictionRecord.created_at.desc())
        .limit(limit)
        .all()
    )