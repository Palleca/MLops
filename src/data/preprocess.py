import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame, target_column: str = "trip_duration"):
    df = df.copy()

    # remplacer certaines valeurs importantes
    df["passenger_count"] = df["passenger_count"].fillna(1)
    # Supprimer les lignes avec valeurs manquantes
    df = df.dropna()

    # Conversion date
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    # Features temporelles
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_day"] = df["pickup_datetime"].dt.day
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_weekday"] = df["pickup_datetime"].dt.weekday

    # Variable catégorielle binaire
    df["store_and_fwd_flag"] = df["store_and_fwd_flag"].map({"N": 0, "Y": 1})

    # Colonnes utilisées
    feature_columns = [
        "vendor_id",
        "passenger_count",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "store_and_fwd_flag",
        "pickup_hour",
        "pickup_day",
        "pickup_month",
        "pickup_weekday",
    ]

    X = df[feature_columns]
    y = df[target_column]

    return X, y