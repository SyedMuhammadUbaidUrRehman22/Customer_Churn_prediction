"""Shared database helpers."""

from sqlalchemy import create_engine

from src.config import DATABASE_URI, MAX_CONNECTIONS, TIMEOUT_SECONDS


def get_engine():
    if not DATABASE_URI:
        raise RuntimeError(
            "DATABASE_URI is required, for example "
            "mysql+mysqlconnector://user:password@localhost:3306/churn_db"
        )
    return create_engine(
        DATABASE_URI,
        pool_size=MAX_CONNECTIONS,
        pool_pre_ping=True,
        connect_args={"connection_timeout": TIMEOUT_SECONDS},
    )
