"""Read the AI4I CSV, reshape it to match the database tables, and load it."""

from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

COLUMN_MAP = {
    "UDI": "machine_id",
    "Product ID": "product_id",
    "Type": "type_code",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
}

CAUSE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

READING_COLUMNS = [
    "machine_id",
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]


def load_raw(path: Path) -> pd.DataFrame:
    """Read the CSV and check that every column we need is present."""
    try:
        raw = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise RuntimeError(f"Could not read {path}: {exc}") from exc

    required = set(COLUMN_MAP) | set(CAUSE_COLUMNS)
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(f"CSV is missing columns: {sorted(missing)}")
    return raw


def build_tables(raw: pd.DataFrame) -> dict:
    """Split the flat CSV into one DataFrame per database table.

    Cause flags are only kept for rows where Machine failure is 1, so rows
    with a cause flag but no failure are not loaded into the cause tables.
    """
    df = raw.rename(columns=COLUMN_MAP)

    machines = df[["machine_id", "product_id", "type_code"]]
    sensor_readings = df[READING_COLUMNS]

    failed = df[df["machine_failure"] == 1]
    failure_events = failed[["machine_id"]].reset_index(drop=True)

    flags = failed[["machine_id"] + CAUSE_COLUMNS].melt(
        id_vars="machine_id", var_name="cause_code", value_name="flag"
    )
    event_causes = flags.loc[flags["flag"] == 1, ["machine_id", "cause_code"]]

    return {
        "machines": machines,
        "sensor_readings": sensor_readings,
        "failure_events": failure_events,
        "event_causes": event_causes.reset_index(drop=True),
    }


def _rows(frame: pd.DataFrame) -> list:
    """Convert a DataFrame to a list of plain Python tuples for psycopg2."""
    return [tuple(row) for row in frame.astype(object).values.tolist()]


def load_tables(conn, tables: dict) -> dict:
    """Insert the prepared tables into PostgreSQL and return row counts."""
    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO machines (machine_id, product_id, type_code) VALUES %s",
                _rows(tables["machines"]),
            )
            execute_values(
                cur,
                "INSERT INTO sensor_readings ("
                + ", ".join(READING_COLUMNS)
                + ") VALUES %s",
                _rows(tables["sensor_readings"]),
            )
            execute_values(
                cur,
                "INSERT INTO failure_events (machine_id) VALUES %s",
                _rows(tables["failure_events"]),
            )

            cur.execute("SELECT machine_id, failure_id FROM failure_events")
            failure_by_machine = dict(cur.fetchall())

            causes = tables["event_causes"].copy()
            causes["failure_id"] = causes["machine_id"].map(failure_by_machine)
            execute_values(
                cur,
                "INSERT INTO event_causes (failure_id, cause_code) VALUES %s",
                _rows(causes[["failure_id", "cause_code"]]),
            )
        conn.commit()
    except psycopg2.Error as exc:
        conn.rollback()
        raise RuntimeError(f"Loading data failed: {exc}") from exc

    return {name: len(frame) for name, frame in tables.items()}
