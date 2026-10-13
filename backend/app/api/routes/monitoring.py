from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.database.database import get_db
from backend.app.services.monitoring_service import get_monitoring_summary


router = APIRouter(
    prefix="/monitoring",
    tags=["Monitoring"],
)


@router.get("/summary")
def monitoring_summary(db: Session = Depends(get_db)):
    """Return summary metrics for saved predictions."""

    return get_monitoring_summary(db)