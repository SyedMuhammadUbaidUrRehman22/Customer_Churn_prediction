"""Shared database helpers."""

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url

from src.config import DATABASE_URI, MAX_CONNECTIONS, TIMEOUT_SECONDS


def get_engine():
    if not DATABASE_URI:
        raise RuntimeError(
            "DATABASE_URI is required, for example "
            "mysql+mysqlconnector://user:password@localhost:3306/churn_db"
        )
    timeout_key = "connect_timeout" if make_url(DATABASE_URI).drivername == "mysql+pymysql" else "connection_timeout"
    return create_engine(
        DATABASE_URI,
        pool_size=MAX_CONNECTIONS,
        pool_pre_ping=True,
        connect_args={timeout_key: TIMEOUT_SECONDS},
    )
