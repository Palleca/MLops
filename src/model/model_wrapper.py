import pandas as pd


class TaxiDurationModel:
    def __init__(self, model):
        self.model = model

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

        df["pickup_hour"] = df["pickup_datetime"].dt.hour
        df["pickup_day"] = df["pickup_datetime"].dt.day
        df["pickup_month"] = df["pickup_datetime"].dt.month
        df["pickup_weekday"] = df["pickup_datetime"].dt.weekday

        df["store_and_fwd_flag"] = df["store_and_fwd_flag"].map({"N": 0, "Y": 1})

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

        return df[feature_columns]

    def postprocess(self, prediction):
        return prediction

    def predict(self, input_df: pd.DataFrame):
        X = self.preprocess(input_df)
        raw_output = self.model.predict(X)
        output = self.postprocess(raw_output)
        return output