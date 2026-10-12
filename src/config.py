"""Project settings, read from environment variables and the .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")
DATA_PATH = PROJECT_ROOT / "data" / "ai4i2020.csv"
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"


def require_database_url() -> str:
    """Return the database URL, or raise a clear error if it is not set."""
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env and fill it in."
        )
    return DATABASE_URL
