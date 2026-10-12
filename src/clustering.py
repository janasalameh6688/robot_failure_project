"""K-Means clustering for machine operating profiles."""

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PATH
from src.database import get_connection


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


def prepare_clustering_data(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the dataset and prepare scaled features for clustering."""

    df = pd.read_csv(csv_path)

    # Feature engineering used elsewhere in the project.
    df["Temperature difference [K]"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    )

    df["Mechanical power proxy"] = (
        df["Rotational speed [rpm]"] * df["Torque [Nm]"]
    )

    df["Tool wear torque interaction"] = (
        df["Tool wear [min]"] * df["Torque [Nm]"]
    )

    # Clustering should describe machine operating behavior.
    # Do not use the failure target or failure-mode indicators.
    excluded = set(ID_COLUMNS + LEAKAGE_COLUMNS + [TARGET])

    features = df.drop(columns=excluded)

    # Keep numerical features only.
    features = features.select_dtypes(include="number")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    scaled_df = pd.DataFrame(
        scaled,
        columns=features.columns,
        index=df.index,
    )

    return df, scaled_df


def find_best_k(
    scaled_features: pd.DataFrame,
    min_k: int = 2,
    max_k: int = 6,
) -> tuple[int, pd.DataFrame]:
    """Evaluate several K values using silhouette score."""

    results = []

    for k in range(min_k, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        labels = model.fit_predict(scaled_features)

        score = silhouette_score(
            scaled_features,
            labels,
        )

        results.append(
            {
                "k": k,
                "silhouette_score": score,
            }
        )

    scores = pd.DataFrame(results)

    best_k = int(
        scores.loc[
            scores["silhouette_score"].idxmax(),
            "k",
        ]
    )

    return best_k, scores


def save_clusters_to_database(
    df: pd.DataFrame,
    labels,
) -> None:
    """Write the machine cluster labels into PostgreSQL."""

    conn = get_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS machine_clusters (
                    udi INTEGER PRIMARY KEY,
                    cluster_label INTEGER NOT NULL
                )
                """
            )

            for udi, label in zip(df["UDI"], labels):
                cur.execute(
                    """
                    INSERT INTO machine_clusters
                        (udi, cluster_label)
                    VALUES (%s, %s)
                    ON CONFLICT (udi)
                    DO UPDATE SET
                        cluster_label = EXCLUDED.cluster_label
                    """,
                    (int(udi), int(label)),
                )

        conn.commit()

    finally:
        conn.close()


def cluster_machines(
    csv_path: Path = DATA_PATH,
    output_dir: Path = Path("reports"),
) -> None:
    """Run K-Means, report the best clustering and save labels."""

    output_dir.mkdir(parents=True, exist_ok=True)

    df, scaled_features = prepare_clustering_data(csv_path)

    best_k, scores = find_best_k(scaled_features)

    final_model = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=10,
    )

    labels = final_model.fit_predict(scaled_features)

    final_score = silhouette_score(
        scaled_features,
        labels,
    )

    df["cluster_label"] = labels

    # Cluster sizes.
    cluster_sizes = (
        df["cluster_label"]
        .value_counts()
        .sort_index()
    )

    # Describe the operating profile of each cluster.
    numeric_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Temperature difference [K]",
        "Mechanical power proxy",
        "Tool wear torque interaction",
    ]

    cluster_profiles = (
        df.groupby("cluster_label")[numeric_columns]
        .mean()
        .round(2)
    )

    report_path = output_dir / "cluster_report.txt"

    with report_path.open("w", encoding="utf-8") as report:
        report.write("K-MEANS CLUSTERING REPORT\n")
        report.write("=" * 60 + "\n\n")

        report.write("Silhouette scores by K\n")
        report.write("-" * 60 + "\n")
        report.write(scores.to_string(index=False))
        report.write("\n\n")

        report.write(f"Selected K: {best_k}\n")
        report.write(
            f"Final silhouette score: {final_score:.4f}\n\n"
        )

        report.write("Cluster sizes\n")
        report.write("-" * 60 + "\n")
        report.write(cluster_sizes.to_string())
        report.write("\n\n")

        report.write("Average operating profile by cluster\n")
        report.write("-" * 60 + "\n")
        report.write(cluster_profiles.to_string())
        report.write("\n\n")

        report.write("Interpretation\n")
        report.write("-" * 60 + "\n")

        
        for cluster_id in sorted(df["cluster_label"].unique()):
            profile = cluster_profiles.loc[cluster_id]

            if cluster_id == 0:
                description = (
                    "Higher tool wear and the highest average "
                    "torque among the four clusters."
                )
            elif cluster_id == 1:
                description = (
                    "Highest average rotational speed, lowest "
                    "torque, and lowest mechanical power proxy."
                )
            elif cluster_id == 2:
                description = (
                    "Lowest average tool wear and lowest "
                    "tool-wear–torque interaction."
                )
            elif cluster_id == 3:
                description = (
                    "Highest average air and process temperatures."
                )
            else:
                description = "Distinct operating profile."

            report.write(
                f"\nCluster {cluster_id}: {description}\n"
            )


    save_clusters_to_database(df, labels)

    print(f"Best K: {best_k}")
    print(f"Silhouette score: {final_score:.4f}")
    print("Cluster sizes:")
    print(cluster_sizes)
    print(f"\nReport saved to: {report_path}")
    print("Cluster labels written to PostgreSQL.")