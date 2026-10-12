"""Command line entry point for the robot failure project."""

import argparse
import sys
from pathlib import Path

from src.config import DATA_PATH, SCHEMA_PATH
from src.database import get_connection, run_sql_file
from src.ingestion import build_tables, load_raw, load_tables
from src.analysis import analyze_dataset
from src.modeling import train_models
from src.clustering import cluster_machines
from src.retrieval import answer_question, build_index
from src.regression import train_regression
from src.regression_advanced import train_advanced_regression

def setup() -> None:
    """Create the PostgreSQL schema, then load and reshape the CSV data."""
    raw = load_raw(DATA_PATH)
    tables = build_tables(raw)

    conn = get_connection()
    try:
        run_sql_file(conn, SCHEMA_PATH)
        counts = load_tables(conn, tables)
    finally:
        conn.close()

    for name, count in counts.items():
        print(f"{name}: {count} rows loaded")


def analyze() -> None:
    """Run exploratory data analysis and feature engineering."""
    output_dir = Path("reports")
    analyze_dataset(DATA_PATH, output_dir)

def train() -> None:
    """Train and compare the machine-failure models."""
    output_dir = Path("models")
    train_models(DATA_PATH, output_dir)

def cluster():
    output_dir = Path("reports")
    cluster_machines(DATA_PATH, output_dir)

def build_parser() -> argparse.ArgumentParser:
    """Define the available commands."""
    parser = argparse.ArgumentParser(
        description="Robot failure prediction system"
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    sub.add_parser(
        "setup",
        help="create the schema and load the data",
    )

    sub.add_parser(
        "analyze",
        help="run EDA and feature engineering",
    )
    sub.add_parser(
    "train",
    help="train and compare machine-failure models",
)
    sub.add_parser(
    "regression",
    help="train and compare regression models",
)
    sub.add_parser(
        "regression-advanced",
        help="compare polynomial and regularized regression models",
    )

    sub.add_parser(
    "cluster",
    help="run K-Means clustering and write cluster labels to PostgreSQL",
)
    sub.add_parser(
        "index",
        help="build the searchable maintenance text index",
    )

    ask_parser = sub.add_parser(
        "ask",
        help="ask a question about machine maintenance data",
    )
    ask_parser.add_argument(
        "question",
        help="natural-language question to answer",
    )
    return parser


def main() -> None:
    """Read the command typed after main.py and run it."""
    args = build_parser().parse_args()

    try:
        if args.command == "setup":
            setup()

        elif args.command == "analyze":
            analyze()

        elif args.command == "train":
             train()  
               
        elif args.command == "regression":
             train_regression(DATA_PATH, Path("models"))

        elif args.command == "regression-advanced":
            train_advanced_regression(DATA_PATH, Path("models"))
            
        elif args.command == "cluster":
             cluster()    
        elif args.command == "index":
            build_index()

        elif args.command == "ask":
            answer_question(args.question)

    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()