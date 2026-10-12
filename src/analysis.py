"""Exploratory analysis and feature engineering for the AI4I dataset."""

from pathlib import Path

import pandas as pd


def analyze_dataset(data_path: Path, output_dir: Path) -> None:
    """Run EDA and feature engineering and save a text report."""
    df = pd.read_csv(data_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    report = []

    report.append("ROBOT FAILURE PROJECT - DATA ANALYSIS REPORT")
    report.append("=" * 55)
    report.append("")

    # Basic dataset information
    report.append("1. DATASET OVERVIEW")
    report.append("-" * 30)
    report.append(f"Rows: {len(df)}")
    report.append(f"Columns: {len(df.columns)}")
    report.append("")

    report.append("Columns:")
    for column in df.columns:
        report.append(f"  - {column}")
    report.append("")

    # Missing values
    report.append("2. MISSING VALUES")
    report.append("-" * 30)

    missing = df.isnull().sum()
    missing_found = False

    for column, count in missing.items():
        if count > 0:
            report.append(f"{column}: {count}")
            missing_found = True

    if not missing_found:
        report.append("No missing values found.")

    report.append("")

    # Target distribution
    target = "Machine failure"

    report.append("3. TARGET DISTRIBUTION")
    report.append("-" * 30)

    if target in df.columns:
        counts = df[target].value_counts().sort_index()

        for value, count in counts.items():
            percentage = count / len(df) * 100
            report.append(
                f"{target} = {value}: {count} rows "
                f"({percentage:.2f}%)"
            )

    report.append("")

    # Numeric statistics
    report.append("4. NUMERIC FEATURE STATISTICS")
    report.append("-" * 30)

    numeric_columns = df.select_dtypes(include="number").columns

    stats = df[numeric_columns].describe().T

    for column, row in stats.iterrows():
        report.append(
            f"{column}: "
            f"mean={row['mean']:.3f}, "
            f"std={row['std']:.3f}, "
            f"min={row['min']:.3f}, "
            f"max={row['max']:.3f}"
        )

    report.append("")

    # Feature engineering
    report.append("5. FEATURE ENGINEERING")
    report.append("-" * 30)

    engineered = df.copy()

    if (
        "Air temperature [K]" in engineered.columns
        and "Process temperature [K]" in engineered.columns
    ):
        engineered["Temperature difference [K]"] = (
            engineered["Process temperature [K]"]
            - engineered["Air temperature [K]"]
        )
        report.append(
            "Created: Temperature difference [K]"
        )

    if (
        "Rotational speed [rpm]" in engineered.columns
        and "Torque [Nm]" in engineered.columns
    ):
        engineered["Mechanical power proxy"] = (
            engineered["Rotational speed [rpm]"]
            * engineered["Torque [Nm]"]
        )
        report.append(
            "Created: Mechanical power proxy"
        )

    if (
        "Tool wear [min]" in engineered.columns
        and "Torque [Nm]" in engineered.columns
    ):
        engineered["Tool wear torque interaction"] = (
            engineered["Tool wear [min]"]
            * engineered["Torque [Nm]"]
        )
        report.append(
            "Created: Tool wear torque interaction"
        )

    report.append("")

    # Correlation with target
    report.append("6. CORRELATION WITH MACHINE FAILURE")
    report.append("-" * 30)

    if target in engineered.columns:
        correlations = (
            engineered.select_dtypes(include="number")
            .corr()[target]
            .drop(target)
            .sort_values(key=abs, ascending=False)
        )

        for column, correlation in correlations.items():
            report.append(
                f"{column}: {correlation:.4f}"
            )

    report.append("")

    # Leakage warning
    report.append("7. MODELING NOTE")
    report.append("-" * 30)
    report.append(
        "The failure-mode indicator columns "
        "(TWF, HDF, PWF, OSF, RNF) are closely related "
        "to Machine failure and should be treated carefully "
        "to avoid target leakage."
    )
    report.append(
        "The main predictive features should focus on "
        "machine characteristics and sensor measurements "
        "available before a failure occurs."
    )
    report.append("")

    report_path = output_dir / "analysis_report.txt"
    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print(f"Analysis complete.")
    print(f"Report saved to: {report_path}")