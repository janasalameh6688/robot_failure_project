
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.modeling import create_features, prepare_data


def main():
    # Load the original dataset.
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "ai4i2020.csv"
    report_dir = project_root / "reports"
    report_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_path)

    # Reuse the project's existing feature preparation.
    X, y = prepare_data(df)

    # Split the data while preserving the failure rate.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Reuse the existing preprocessing pipeline.
    from src.modeling import build_preprocessor

    preprocessor = build_preprocessor(X_train)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("Training Random Forest...")
    model.fit(X_train, y_train)

    print("Calculating permutation importance...")
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    output_path = report_dir / "permutation_importance.csv"
    importance_df.to_csv(output_path, index=False)

    print("\nTop 15 features:")
    print(importance_df.head(15).to_string(index=False))
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
