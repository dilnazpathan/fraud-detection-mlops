from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.app.database.database import get_db
from backend.app.database.crud import get_recent_predictions


router = APIRouter(
    prefix="/transactions",
    tags=["Transactions"],
)


@router.get("/")
def get_transactions(
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Return recently saved transaction predictions."""

    records = get_recent_predictions(db, limit=limit)

    return {
        "total_returned": len(records),
        "transactions": [
            {
                "record_id": record.id,
                "step": record.step,
                "type": record.transaction_type,
                "amount": record.amount,
                "oldbalanceOrg": record.oldbalanceOrg,
                "newbalanceOrig": record.newbalanceOrig,
                "oldbalanceDest": record.oldbalanceDest,
                "newbalanceDest": record.newbalanceDest,
                "isFlaggedFraud": record.isFlaggedFraud,
                "prediction": record.prediction,
                "fraud_probability": record.fraud_probability,
                "threshold": record.threshold,
                "created_at": record.created_at.isoformat(),
            }
            for record in records
        ],
    }