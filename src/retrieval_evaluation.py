
"""Evaluate maintenance retrieval across several numeric queries."""

from pathlib import Path

import pandas as pd

from src.retrieval import retrieve

DATA_PATH = Path("data/ai4i2020.csv")
REPORT_PATH = Path("reports/retrieval_evaluation.csv")
K_VALUES = (1, 3, 5, 10)


def main() -> None:
    """Evaluate retrieval using explicit numeric relevance rules."""
    df = pd.read_csv(DATA_PATH)

    torque_75 = df["Torque [Nm]"].quantile(0.75)
    wear_75 = df["Tool wear [min]"].quantile(0.75)
    speed_25 = df["Rotational speed [rpm]"].quantile(0.25)

    tests = [
        {
            "question": "Find machines with high torque and tool wear",
            "rule": "Torque and tool wear >= their 75th percentiles",
            "mask": (
                (df["Torque [Nm]"] >= torque_75)
                & (df["Tool wear [min]"] >= wear_75)
            ),
        },
        {
            "question": "Find machines with high torque",
            "rule": "Torque >= its 75th percentile",
            "mask": df["Torque [Nm]"] >= torque_75,
        },
        {
            "question": "Find machines with low rotational speed",
            "rule": "Rotational speed <= its 25th percentile",
            "mask": df["Rotational speed [rpm]"] <= speed_25,
        },
    ]

    rows = []

    for test in tests:
        relevant_ids = set(
            df.loc[test["mask"], "UDI"].astype(int)
        )

        results = retrieve(test["question"], top_k=max(K_VALUES))
        retrieved_ids = [
            int(item["machine_id"]) for item in results
        ]

        for k in K_VALUES:
            top_ids = retrieved_ids[:k]
            hits = len(set(top_ids) & relevant_ids)
            recall = hits / len(relevant_ids) if relevant_ids else 0.0
            precision = hits / len(top_ids) if top_ids else 0.0

            rows.append({
                "question": test["question"],
                "relevance_rule": test["rule"],
                "relevant_records_total": len(relevant_ids),
                "k": k,
                "retrieved_records_available": len(top_ids),
                "relevant_hits": hits,
                "recall_at_k": round(recall, 4),
                "precision_at_k": round(precision, 4),
            })

    report = pd.DataFrame(rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)

    print("Multi-query retrieval evaluation complete")
    print(f"Report saved to: {REPORT_PATH}")
    print(
        report[
            [
                "question",
                "k",
                "relevant_hits",
                "recall_at_k",
                "precision_at_k",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
