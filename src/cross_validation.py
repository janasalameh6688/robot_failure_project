
"""Stratified cross-validation for the robot failure classifier."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.modeling import prepare_data


def run_cross_validation():
    data_path = Path("data/ai4i2020.csv")
    output_path = Path("reports/cross_validation_results.csv")

    df = pd.read_csv(data_path)
    X, y, preprocessor = prepare_data(df)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=150,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    print("Running 5-fold stratified cross-validation...")
    print("This may take a few minutes.")

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False,
    )

    metrics = []

    for metric in scoring:
        scores = results[f"test_{metric}"]

        metrics.append(
            {
                "metric": metric,
                "mean": scores.mean(),
                "std": scores.std(),
            }
        )

        print(
            f"{metric}: {scores.mean():.4f} "
            f"+/- {scores.std():.4f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(metrics).to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_cross_validation()
