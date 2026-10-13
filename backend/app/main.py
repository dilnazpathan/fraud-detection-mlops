from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.api.routes.prediction import router as prediction_router
from backend.app.api.routes.transactions import router as transactions_router
from backend.app.api.routes.monitoring import router as monitoring_router

from backend.app.database.database import Base, engine
from backend.app.database.models import PredictionRecord


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create database tables if they do not exist.
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="Fraud Detection API",
    description="Fraud prediction, transaction history, and monitoring API",
    version="1.0.0",
    lifespan=lifespan,
)

# Register API endpoints.
app.include_router(prediction_router)
app.include_router(transactions_router)
app.include_router(monitoring_router)


@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running",
        "status": "healthy",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "loaded",
    }