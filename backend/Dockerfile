
FROM python:3.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY backend/requirements.txt /app/backend/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend/ /app/backend/

COPY models/production_model.joblib /app/models/production_model.joblib
COPY models/production_scaler.joblib /app/models/production_scaler.joblib
COPY models/production_threshold.json /app/models/production_threshold.json

RUN mkdir -p /data

ENV DATABASE_PATH=/data/predictions.db

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
