"""Compare Random Forest class-weight strategies."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.modeling import prepare_data


DATA_PATH = Path("data/ai4i2020.csv")
REPORT_PATH = Path("reports/random_forest_imbalance_comparison.csv")

WEIGHT_STRATEGIES = {
    "No class weighting": None,
    "Balanced": "balanced",
    "Balanced subsample": "balanced_subsample",
}


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X, y, _ = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    results = []

    print("Random Forest: class-imbalance experiment")
    print(f"Training records: {len(X_train)}")
    print(f"Test records: {len(X_test)}")
    print(f"Test failures: {int(y_test.sum())}")
    print()

    for name, class_weight in WEIGHT_STRATEGIES.items():
        # Build a fresh preprocessor for each independent experiment.
        _, _, preprocessor = prepare_data(df)

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=300,
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1,
            )),
        ])

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        metrics = {
            "model": name,
            "class_weight": str(class_weight),
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
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }

        results.append(metrics)

        print(name)
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-score:  {metrics['f1']:.4f}")
        print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  False alarms: {fp}")
        print(f"  Missed failures: {fn}")
        print()

    results_df = pd.DataFrame(results).sort_values(
        ["pr_auc", "f1"],
        ascending=False,
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(REPORT_PATH, index=False)

    print("Results ranked by PR-AUC, then F1-score:")
    print(
        results_df[
            ["model", "precision", "recall", "f1", "pr_auc", "roc_auc"]
        ].round(4).to_string(index=False)
    )
    print(f"\nReport saved to: {REPORT_PATH}")
    print("Existing champion model was not modified.")


if __name__ == "__main__":
    main()
