"""Compare original sensor features against engineered features."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


DATA_PATH = Path("data/ai4i2020.csv")
OUTPUT_PATH = Path("reports/feature_experiment.csv")
TARGET = "Machine failure"

# These columns are excluded to avoid identifiers and target leakage.
EXCLUDED_COLUMNS = [
    TARGET,
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

ENGINEERED_FEATURES = [
    "Temperature difference [K]",
    "Mechanical power proxy",
    "Tool wear torque interaction",
]


def add_engineered_features(df):
    """Add the three derived sensor features."""
    data = df.copy()

    data["Temperature difference [K]"] = (
        data["Process temperature [K]"]
        - data["Air temperature [K]"]
    )

    data["Mechanical power proxy"] = (
        data["Torque [Nm]"]
        * data["Rotational speed [rpm]"]
    )

    data["Tool wear torque interaction"] = (
        data["Tool wear [min]"]
        * data["Torque [Nm]"]
    )

    return data


def build_pipeline(X, seed=42):
    """Build preprocessing and a Random Forest classifier."""
    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    transformers = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore"
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate(name, model, X_test, y_test):
    """Calculate metrics using predicted probabilities."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "experiment": name,
        "PR-AUC": average_precision_score(
            y_test, probabilities
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y_test, probabilities
        ),
    }


def main():
    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    data = add_engineered_features(df)
    y = data[TARGET]

    # Both experiments use exactly the same split.
    indices = data.index
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    results = []

    for name, include_engineered in [
        ("Original features", False),
        ("Original + engineered features", True),
    ]:
        columns_to_exclude = list(EXCLUDED_COLUMNS)

        if not include_engineered:
            columns_to_exclude.extend(ENGINEERED_FEATURES)

        X = data.drop(
            columns=columns_to_exclude,
            errors="ignore",
        )

        X_train = X.loc[train_indices]
        X_test = X.loc[test_indices]
        y_train = y.loc[train_indices]
        y_test = y.loc[test_indices]

        model = build_pipeline(X_train)
        model.fit(X_train, y_train)

        results.append(
            evaluate(name, model, X_test, y_test)
        )

    results_df = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\nFEATURE ENGINEERING EXPERIMENT")
    print("=" * 55)
    print(results_df.round(4).to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()