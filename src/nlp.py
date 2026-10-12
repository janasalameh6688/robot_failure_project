
"""Compare sensor-based and text-based machine-failure prediction."""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.modeling import prepare_data


DATA_PATH = Path("data/ai4i2020.csv")
OUTPUT_PATH = Path("reports/nlp_model_comparison.csv")
TARGET = "Machine failure"


def create_machine_text(df: pd.DataFrame) -> pd.Series:
    """Convert sensor measurements into text descriptions."""
    return (
        "Machine type "
        + df["Type"].astype(str)
        + " air temperature "
        + df["Air temperature [K]"].round(1).astype(str)
        + " process temperature "
        + df["Process temperature [K]"].round(1).astype(str)
        + " rotational speed "
        + df["Rotational speed [rpm]"].astype(str)
        + " torque "
        + df["Torque [Nm]"].round(1).astype(str)
        + " tool wear "
        + df["Tool wear [min]"].astype(str)
    )


def evaluate(name, model, X_test, y_test) -> dict:
    """Evaluate a fitted classifier."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "model": name,
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "pr_auc": average_precision_score(
            y_test, probabilities
        ),
        "roc_auc": roc_auc_score(
            y_test, probabilities
        ),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # Split once so both experiments use identical records.
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.20,
        random_state=42,
        stratify=df[TARGET],
    )

    y_train = df.loc[train_idx, TARGET]
    y_test = df.loc[test_idx, TARGET]

    # Experiment 1: sensor features with existing preprocessing.
    X_sensor, _, sensor_preprocessor = prepare_data(df)

    sensor_model = Pipeline(
        steps=[
            ("preprocessor", sensor_preprocessor),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    sensor_model.fit(
        X_sensor.loc[train_idx],
        y_train,
    )

    # Experiment 2: text features.
    # TfidfVectorizer is inside the pipeline, so its vocabulary
    # and IDF values are learned from training text only.
    descriptions = create_machine_text(df)

    text_model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    token_pattern=r"(?u)\b\w+\b",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    max_features=5000,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    text_model.fit(
        descriptions.loc[train_idx],
        y_train,
    )

    results = [
        evaluate(
            "Sensor features - Logistic Regression",
            sensor_model,
            X_sensor.loc[test_idx],
            y_test,
        ),
        evaluate(
            "TF-IDF text - Logistic Regression",
            text_model,
            descriptions.loc[test_idx],
            y_test,
        ),
    ]

    results_df = pd.DataFrame(results).sort_values(
        "pr_auc",
        ascending=False,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\nNLP and sensor model comparison")
    print(results_df.to_string(index=False, float_format="%.4f"))
    print(f"\nResults saved to: {OUTPUT_PATH}")
    print(
        "\nNote: machine descriptions are derived from the same "
        "sensor measurements, not additional independent data."
    )


if __name__ == "__main__":
    main()
