
"""Compare polynomial regression and regularized regression models."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


TARGET = "Rotational speed [rpm]"

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def train_advanced_regression(
    data_path: Path,
    output_dir: Path,
) -> None:
    """Compare linear, polynomial, Ridge, and Lasso regression."""
    df = pd.read_csv(data_path)

    required_columns = [TARGET, *FEATURES]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]
    if missing_columns:
        raise RuntimeError(
            f"Required dataset columns are missing: {missing_columns}"
        )

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    # Polynomial expansion is limited to four selected sensor features.
    models = {
        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Polynomial Regression (degree 2)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Ridge Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Polynomial Ridge (degree 2)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Polynomial Lasso (degree 2)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=0.1, max_iter=10000)),
        ]),
    }

    results = []

    print("Polynomial regression and regularization comparison\n")

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        results.append({
            "model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

        print(
            f"{name}: MAE={mae:.2f} rpm, "
            f"RMSE={rmse:.2f} rpm, R2={r2:.4f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "advanced_regression_comparison.csv"

    results_df = pd.DataFrame(results).sort_values("RMSE")
    results_df.to_csv(report_path, index=False)

    print(f"\nResults saved to: {report_path}")
    print(f"Best model by test RMSE: {results_df.iloc[0]['model']}")
