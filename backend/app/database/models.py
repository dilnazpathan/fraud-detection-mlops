from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String

from backend.app.database.database import Base


class PredictionRecord(Base):
    """Database table for saved fraud predictions."""

    __tablename__ = "prediction_records"

    id = Column(Integer, primary_key=True, index=True)

    # Original transaction details
    step = Column(Integer, nullable=False)
    transaction_type = Column(String(20), nullable=False)
    amount = Column(Float, nullable=False)

    oldbalanceOrg = Column(Float, nullable=False)
    newbalanceOrig = Column(Float, nullable=False)
    oldbalanceDest = Column(Float, nullable=False)
    newbalanceDest = Column(Float, nullable=False)

    isFlaggedFraud = Column(Integer, nullable=False, default=0)

    # Model prediction results
    prediction = Column(String(20), nullable=False)
    fraud_probability = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)

    # Time the prediction was recorded
    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )