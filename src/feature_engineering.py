import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features from existing machine sensor readings.

    Returns a copy so the original DataFrame is not modified.
    """
    df = df.copy()

    df["Temperature difference [K]"] = (
        df["Process temperature [K]"]
        - df["Air temperature [K]"]
    )

    df["Mechanical power proxy"] = (
        df["Torque [Nm]"]
        * df["Rotational speed [rpm]"]
    )

    df["Tool wear torque interaction"] = (
        df["Tool wear [min]"]
        * df["Torque [Nm]"]
    )

    return df


if __name__ == "__main__":
    data = pd.read_csv("data/ai4i2020.csv")
    data = add_engineered_features(data)

    new_features = [
        "Temperature difference [K]",
        "Mechanical power proxy",
        "Tool wear torque interaction",
    ]

    print("Dataset shape:", data.shape)
    print("\nEngineered feature statistics:")
    print(data[new_features].describe().round(2).to_string())
    print("\nSample engineered features:")
    print(data[new_features + ["Machine failure"]].head().to_string(index=False))