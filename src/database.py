"""PostgreSQL connection helpers."""

from pathlib import Path

import psycopg2

from src.config import require_database_url


def get_connection():
    """Open and return a connection to the PostgreSQL database."""
    try:
        return psycopg2.connect(require_database_url())
    except psycopg2.OperationalError as exc:
        raise RuntimeError(f"Could not connect to the database: {exc}") from exc


def run_sql_file(conn, path: Path) -> None:
    """Run every statement in a .sql file and commit the result."""
    try:
        sql = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Could not read SQL file {path}: {exc}") from exc

    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    except psycopg2.Error as exc:
        conn.rollback()
        raise RuntimeError(f"SQL file {path} failed: {exc}") from exc
