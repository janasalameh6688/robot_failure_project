"""Test the contribution of each engineered feature."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
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


DATA_PATH = Path("data/ai4i2020.csv")
OUTPUT_PATH = Path("reports/feature_ablation.csv")
TARGET = "Machine failure"

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
    """Create the same features used in the previous experiment."""
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


def build_model(X):
    """Use the same preprocessing and Random Forest settings."""
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
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main():
    df = pd.read_csv(DATA_PATH)
    data = add_engineered_features(df)
    y = data[TARGET]

    # Make one fixed, stratified split for every experiment.
    train_idx, test_idx = train_test_split(
        data.index,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    experiments = [
        ("A - Original features", []),
        ("B - Temperature difference", [
            "Temperature difference [K]",
        ]),
        ("C - Mechanical power proxy", [
            "Mechanical power proxy",
        ]),
        ("D - Wear-torque interaction", [
            "Tool wear torque interaction",
        ]),
        ("E - All engineered features", ENGINEERED_FEATURES),
    ]

    results = []

    for name, included_features in experiments:
        excluded = list(EXCLUDED_COLUMNS)

        excluded.extend(
            feature
            for feature in ENGINEERED_FEATURES
            if feature not in included_features
        )

        X = data.drop(columns=excluded, errors="ignore")

        X_train = X.loc[train_idx]
        X_test = X.loc[test_idx]
        y_train = y.loc[train_idx]
        y_test = y.loc[test_idx]

        model = build_model(X_train)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        results.append(
            {
                "experiment": name,
                "features_added": (
                    ", ".join(included_features)
                    if included_features
                    else "None"
                ),
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
        )

        print(f"Finished: {name}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "PR-AUC", ascending=False
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\nFEATURE ABLATION RESULTS")
    print("=" * 75)
    print(results_df.round(4).to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()