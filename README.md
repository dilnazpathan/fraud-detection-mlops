

\# End-to-End Fraud Detection System with ML Model Deployment and Monitoring



\## Project Overview



This project is an end-to-end machine-learning application that predicts potentially fraudulent financial transactions.



It uses a trained Random Forest classification model, a FastAPI backend, a React dashboard, SQLite prediction storage, monitoring metrics, and Docker Compose deployment.



\## Features



\- Financial transaction fraud prediction

\- Fraud probability calculation and configurable decision threshold

\- FastAPI prediction endpoint

\- React-based transaction dashboard

\- SQLite database for saving prediction records

\- Transaction history retrieval

\- Monitoring summaries for saved predictions

\- Dockerized frontend and backend

\- Persistent database storage using a Docker volume

\- MLflow experiment tracking during model development



\## Technology Stack



\- Python

\- Pandas and NumPy

\- Scikit-learn

\- Random Forest classification

\- FastAPI

\- React

\- Vite

\- Nginx

\- SQLite

\- SQLAlchemy

\- Docker and Docker Compose

\- MLflow



\## Application Architecture



1\. A user enters transaction details in the React dashboard.

2\. The frontend sends the transaction to the FastAPI backend.

3\. The backend preprocesses the transaction using the expected model features.

4\. The production Random Forest model calculates the fraud probability.

5\. The configured decision threshold determines the predicted class.

6\. The transaction and prediction result are saved in SQLite.

7\. The dashboard displays the prediction, transaction history, and monitoring metrics.



\## Project Structure



```text

fraud-detection-mlops/

├── backend/

│   ├── app/

│   │   ├── api/

│   │   │   └── routes/

│   │   ├── database/

│   │   ├── ml/

│   │   ├── schemas/

│   │   └── services/

│   ├── Dockerfile

│   └── requirements.txt

├── frontend/

│   ├── src/

│   ├── Dockerfile

│   ├── nginx.conf

│   └── package.json

├── models/

│   ├── production\_model.joblib

│   ├── production\_scaler.joblib

│   └── production\_threshold.json

├── src/

│   └── select\_production\_model.py

├── .dockerignore

└── docker-compose.yml

```



\## Run the Application with Docker



\### Prerequisites



\- Docker Desktop installed and running

\- Docker Compose available

\- Production model artifacts present in the `models/` directory



\### Start the Application



Open a terminal in the project root and run:



```powershell

docker compose up -d --build

```



\### Access the Application



\- \*\*React dashboard:\*\* http://localhost:5173

\- \*\*FastAPI documentation:\*\* http://localhost:8000/docs

\- \*\*API health check:\*\* http://localhost:8000/health

\- \*\*Transaction history:\*\* http://localhost:8000/transactions/

\- \*\*Monitoring summary:\*\* http://localhost:8000/monitoring/summary



\### Check Container Status



```powershell

docker compose ps

```



\### View Logs



```powershell

docker compose logs --tail=100

```



\### Stop the Application



```powershell

docker compose down

```



The named Docker volume stores prediction records separately from the containers. Stopping or recreating the containers does not normally remove this data.



\*\*Do not use `docker compose down -v` unless you intend to delete the associated Docker volumes and their stored data.\*\*



\## API Endpoints



| Method | Endpoint | Purpose |

|---|---|---|

| GET | `/` | API welcome response |

| GET | `/health` | Health check |

| POST | `/prediction/predict` | Predict and save a transaction |

| GET | `/transactions/` | Retrieve saved predictions |

| GET | `/monitoring/summary` | Retrieve prediction summary metrics |



\## Monitoring



The monitoring endpoint reports:



\- Total saved predictions

\- Number classified as fraud

\- Number classified as legitimate

\- Predicted fraud rate

\- Average fraud probability

\- Average transaction amount



These are operational prediction metrics. The current implementation does not establish whether transactions are truly fraudulent or independently measure model drift.



\## Model Artifacts



The production model directory contains:



\- `production\_model.joblib` — trained Random Forest model

\- `production\_scaler.joblib` — fitted numerical scaler

\- `production\_threshold.json` — configured decision threshold



The model expects 18 processed features in a specific order. Input preprocessing must remain consistent with training.



\## Limitations



\- Model predictions are estimates, not proof of fraud.

\- Results depend on training data and preprocessing assumptions.

\- The monitoring metrics describe saved predictions and do not replace verified fraud labels or a full model-drift monitoring system.

\- The application is a demonstration project and should not be used as an unaudited production banking decision system.



\## Future Improvements



\- Add model and data drift detection

\- Track verified fraud outcomes and model performance over time

\- Add automated tests and continuous integration

\- Deploy to a cloud hosting platform

\- Add authentication and access controls

\- Improve monitoring and alerting



\## Author



Developed as an end-to-end machine-learning deployment and MLOps portfolio project.
