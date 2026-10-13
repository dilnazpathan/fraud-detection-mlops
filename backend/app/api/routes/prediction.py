from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.database.database import get_db
from backend.app.database.crud import save_prediction
from backend.app.ml.predictor import predict_transaction


router = APIRouter(
    prefix="/prediction",
    tags=["Prediction"],
)


class TransactionRequest(BaseModel):
    step: int = Field(gt=0)

    type: Literal[
        "PAYMENT",
        "TRANSFER",
        "CASH_OUT",
        "DEBIT",
        "CASH_IN",
    ]

    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)
    newbalanceDest: float = Field(ge=0)
    isFlaggedFraud: int = Field(default=0, ge=0, le=1)


@router.post("/predict")
def predict(
    request: TransactionRequest,
    db: Session = Depends(get_db),
):
    """Predict fraud and save the prediction record."""

    transaction = request.model_dump()

    try:
        # Make the prediction
        result = predict_transaction(transaction)

        # Save the transaction and prediction
        record = save_prediction(
            db=db,
            transaction=transaction,
            result=result,
        )

        # Return the prediction and saved record details
        return {
            **result,
            "record_id": record.id,
            "created_at": record.created_at.isoformat(),
        }

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed or could not be saved.",
        ) from error