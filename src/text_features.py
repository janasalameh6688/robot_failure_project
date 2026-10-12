from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.maintenance_text import create_maintenance_description


DATA_PATH = Path("data/ai4i2020.csv")


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    descriptions = df.apply(
        create_maintenance_description, axis=1
    ).tolist()

    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        stop_words="english",
    )

    matrix = vectorizer.fit_transform(descriptions)
    terms = vectorizer.get_feature_names_out()

    print("Dataset shape:", df.shape)
    print("TF-IDF matrix shape:", matrix.shape)

    print("\nMaintenance-related vocabulary:")
    keywords = [
        "machine", "temperature", "air", "process",
        "rotational", "speed", "torque", "tool", "wear",
    ]

    for keyword in keywords:
        print(f"{keyword}: {keyword in terms}")

    print("\nFirst 30 vocabulary terms:")
    print(terms[:30].tolist())

    print("\nSample description:")
    print(descriptions[0])

    print("\nFirst description's non-zero features:")
    row = matrix[0].toarray().flatten()
    indices = row.argsort()[::-1]

    for i in indices[:10]:
        if row[i] > 0:
            print(f"{terms[i]}: {row[i]:.4f}")


if __name__ == "__main__":
    main()