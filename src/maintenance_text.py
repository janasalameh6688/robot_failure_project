import pandas as pd


DATA_PATH = "data/ai4i2020.csv"


def create_maintenance_description(row):
    """Generate a readable description from one sensor record."""
    return (
        f"Machine type {row['Type']} operates at "
        f"{row['Air temperature [K]']} K air temperature, "
        f"{row['Process temperature [K]']} K process temperature, "
        f"{row['Rotational speed [rpm]']} rpm rotational speed, "
        f"{row['Torque [Nm]']} Nm torque, and "
        f"{row['Tool wear [min]']} minutes of tool wear."
    )


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)

    df["maintenance_description"] = df.apply(
        create_maintenance_description, axis=1
    )

    print("Dataset shape:", df.shape)
    print("\nExample maintenance descriptions:\n")

    for description in df["maintenance_description"].head(5):
        print("-", description)

    print("\nOriginal failure labels:")
    print(df["Machine failure"].value_counts().sort_index())
