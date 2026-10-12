"""Machine-failure model training and comparison."""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


TARGET = "Machine failure"

LEAKAGE_COLUMNS = [
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

ID_COLUMNS = [
    "UDI",
    "Product ID",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create predictive features from machine sensor data."""
    data = df.copy()

    if (
        "Air temperature [K]" in data.columns
        and "Process temperature [K]" in data.columns
    ):
        data["Temperature difference [K]"] = (
            data["Process temperature [K]"]
            - data["Air temperature [K]"]
        )

    if (
        "Rotational speed [rpm]" in data.columns
        and "Torque [Nm]" in data.columns
    ):
        data["Mechanical power proxy"] = (
            data["Rotational speed [rpm]"]
            * data["Torque [Nm]"]
        )

    if (
        "Tool wear [min]" in data.columns
        and "Torque [Nm]" in data.columns
    ):
        data["Tool wear torque interaction"] = (
            data["Tool wear [min]"]
            * data["Torque [Nm]"]
        )

    return data


def prepare_data(df: pd.DataFrame):
    """Prepare features and target while avoiding leakage."""
    data = create_features(df)

    columns_to_remove = [
        TARGET,
        *LEAKAGE_COLUMNS,
        *ID_COLUMNS,
    ]

    X = data.drop(
        columns=columns_to_remove,
        errors="ignore",
    )
    y = data[TARGET]

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ]
    )

    return X, y, preprocessor


def evaluate_model(name, model, X_test, y_test):
    """Evaluate one trained model."""
    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)

        # Convert decision scores to a 0-1 range for reporting.
        minimum = scores.min()
        maximum = scores.max()

        if maximum > minimum:
            probabilities = (
                (scores - minimum)
                / (maximum - minimum)
            )
        else:
            probabilities = scores

    return {
        "model": name,
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_test,
            probabilities,
        ),
    }


def train_models(data_path: Path, output_dir: Path) -> None:
    """Train, compare, and save the champion classifier."""
    df = pd.read_csv(data_path)

    X, y, preprocessor = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "Dummy baseline": DummyClassifier(
            strategy="most_frequent"
        ),
        "Perceptron": Perceptron(
            class_weight="balanced",
            random_state=42,
        ),
        "Logistic regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
        "Linear SVM": SVC(
            kernel="linear",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
        "RBF SVM": SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
        "Decision tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=8,
            random_state=42,
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.5,
            random_state=42,
        ),
    }

    results = []
    trained_models = {}

    print("Training models...")
    print()

    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(
            name,
            pipeline,
            X_test,
            y_test,
        )

        results.append(metrics)
        trained_models[name] = pipeline

        print(
            f"{name}: "
            f"F1={metrics['f1']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"PR-AUC={metrics['pr_auc']:.4f}, "
            f"ROC-AUC={metrics['roc_auc']:.4f}"
        )

    results_df = pd.DataFrame(results)

    # PR-AUC is especially useful here because failures
    # represent only 3.39% of the observations.
    results_df = results_df.sort_values(
        by=["pr_auc", "f1", "recall"],
        ascending=False,
    )

    champion_name = results_df.iloc[0]["model"]
    champion = trained_models[champion_name]

    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "model_comparison.csv"
    results_df.to_csv(
        results_path,
        index=False,
    )

    champion_path = output_dir / "champion_model.joblib"
    joblib.dump(
        champion,
        champion_path,
    )

    report_path = output_dir / "model_report.txt"

    with report_path.open("w", encoding="utf-8") as report:
        report.write("ROBOT FAILURE PROJECT - MODEL REPORT\n")
        report.write("=" * 55 + "\n\n")

        report.write("FEATURE POLICY\n")
        report.write("-" * 30 + "\n")
        report.write(
            "Excluded identifier columns: "
            + ", ".join(ID_COLUMNS)
            + "\n"
        )
        report.write(
            "Excluded failure-mode columns to avoid leakage: "
            + ", ".join(LEAKAGE_COLUMNS)
            + "\n\n"
        )

        report.write("MODEL COMPARISON\n")
        report.write("-" * 30 + "\n")

        for _, row in results_df.iterrows():
            report.write(
                f"{row['model']}: "
                f"precision={row['precision']:.4f}, "
                f"recall={row['recall']:.4f}, "
                f"F1={row['f1']:.4f}, "
                f"ROC-AUC={row['roc_auc']:.4f}, "
                f"PR-AUC={row['pr_auc']:.4f}\n"
            )

        report.write("\n")
        report.write(
            f"CHAMPION MODEL: {champion_name}\n"
        )
        report.write(
            "Champion selected primarily using PR-AUC, "
            "with F1 and recall as secondary criteria.\n"
        )

        report.write("\n")
        report.write("CHAMPION CLASSIFICATION REPORT\n")
        report.write("-" * 30 + "\n")

        predictions = champion.predict(X_test)

        report.write(
            classification_report(
                y_test,
                predictions,
                zero_division=0,
            )
        )
        report.write("\n")
        report.write("CONFUSION MATRIX\n")
        report.write("-" * 30 + "\n")
        cm = confusion_matrix(y_test, predictions)
        report.write(f"{cm[0][0]} TN | {cm[0][1]} FP\n")
        report.write(f"{cm[1][0]} FN | {cm[1][1]} TP\n")

    print()
    print(f"Champion model: {champion_name}")
    print(f"Comparison saved to: {results_path}")
    print(f"Champion saved to: {champion_path}")
    print(f"Report saved to: {report_path}")