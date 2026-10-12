
"""Tune Random Forest decision threshold without using test data for selection."""

from pathlib import Path

import numpy as np
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
VALIDATION_REPORT = Path("reports/random_forest_threshold_validation.csv")
TEST_REPORT = Path("reports/random_forest_threshold_test.csv")


def evaluate_threshold(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "threshold": round(float(threshold), 2),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    X, y, preprocessor = prepare_data(df)

    # Reserve the test set; do not use it to choose a threshold.
    X_remaining, X_test, y_remaining, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Split the remaining 80% into 60% training and 20% validation.
    X_train, X_val, y_train, y_val = train_test_split(
        X_remaining,
        y_remaining,
        test_size=0.25,
        random_state=42,
        stratify=y_remaining,
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            class_weight=None,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    model.fit(X_train, y_train)

    val_probabilities = model.predict_proba(X_val)[:, 1]
    test_probabilities = model.predict_proba(X_test)[:, 1]

    print("Random Forest threshold-tuning experiment")
    print(f"Training records:    {len(X_train)}")
    print(f"Validation records:  {len(X_val)}")
    print(f"Test records:        {len(X_test)}")
    print(f"Validation failures: {int(y_val.sum())}")
    print(f"Test failures:       {int(y_test.sum())}")
    print()

    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)

    validation_results = pd.DataFrame([
        evaluate_threshold(y_val, val_probabilities, threshold)
        for threshold in thresholds
    ])

    validation_results.to_csv(VALIDATION_REPORT, index=False)

    print("Validation results (ranked by F1):")
    print(
        validation_results.sort_values(
            ["f1", "recall"], ascending=False
        )[
            ["threshold", "precision", "recall", "f1",
             "false_positives", "false_negatives"]
        ].round(4).to_string(index=False)
    )

    # Choose the threshold using validation data only.
    best_row = validation_results.sort_values(
        ["f1", "recall"], ascending=False
    ).iloc[0]
    best_threshold = float(best_row["threshold"])

    # Compare default and selected thresholds on the untouched test set.
    test_results = pd.DataFrame([
        evaluate_threshold(y_test, test_probabilities, 0.50),
        evaluate_threshold(
            y_test, test_probabilities, best_threshold
        ),
    ])

    test_results.insert(
        1,
        "selection",
        ["Default threshold", "Selected using validation F1"],
    )

    test_results["pr_auc"] = average_precision_score(
        y_test, test_probabilities
    )
    test_results["roc_auc"] = roc_auc_score(
        y_test, test_probabilities
    )

    test_results.to_csv(TEST_REPORT, index=False)

    print()
    print(f"Threshold selected on validation data: {best_threshold:.2f}")
    print("Final test-set comparison:")
    print(
        test_results[
            ["threshold", "selection", "precision", "recall", "f1",
             "false_positives", "false_negatives", "pr_auc", "roc_auc"]
        ].round(4).to_string(index=False)
    )

    print(f"\nValidation report saved to: {VALIDATION_REPORT}")
    print(f"Test report saved to: {TEST_REPORT}")
    print("Existing champion model was not modified.")


if __name__ == "__main__":
    main()
