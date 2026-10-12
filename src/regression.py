
"""Regression models for predicting machine rotational speed."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "Rotational speed [rpm]"

EXCLUDED_COLUMNS = [
    "UDI",
    "Product ID",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def train_regression(data_path: Path, output_dir: Path) -> None:
    """Train regression models and save their evaluation results."""
    df = pd.read_csv(data_path)

    if TARGET not in df.columns:
        raise RuntimeError(f"Required target column not found: {TARGET}")

    # Engineer features without using the target as an input.
    if (
        "Process temperature [K]" in df.columns
        and "Air temperature [K]" in df.columns
    ):
        df["Temperature difference [K]"] = (
            df["Process temperature [K]"] - df["Air temperature [K]"]
        )

    if "Torque [Nm]" in df.columns and "Tool wear [min]" in df.columns:
        df["Tool wear torque interaction"] = (
            df["Torque [Nm]"] * df["Tool wear [min]"]
        )

    y = df[TARGET]
    X = df.drop(columns=[TARGET, *EXCLUDED_COLUMNS], errors="ignore")

    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if not numeric_columns and not categorical_columns:
        raise RuntimeError("No usable predictor columns were found.")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []

    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        metrics = {
            "model": name,
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
            "R2": r2_score(y_test, predictions),
        }
        results.append(metrics)

        print(
            f"{name}: "
            f"MAE={metrics['MAE']:.2f} rpm, "
            f"RMSE={metrics['RMSE']:.2f} rpm, "
            f"R2={metrics['R2']:.4f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "regression_comparison.csv"
    pd.DataFrame(results).sort_values("RMSE").to_csv(
        report_path, index=False
    )

    print(f"\nRegression comparison saved to: {report_path}")
