import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Use the environment variable in Docker, or the project folder locally.
DATABASE_PATH = Path(
    os.getenv(
        "DATABASE_PATH",
        str(PROJECT_ROOT / "predictions.db"),
    )
)

DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


def get_db():
    """Provide a database session to each API request."""
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()