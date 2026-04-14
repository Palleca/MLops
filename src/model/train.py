import os
import joblib
import yaml
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from src.data.preprocess import load_data, preprocess_data


def train():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = config["paths"]["data_path"]
    model_path = config["paths"]["model_path"]
    target_column = config["ml"]["target_column"]
    test_size = config["ml"]["test_size"]
    random_state = config["ml"]["random_state"]

    df = load_data(data_path)
    X, y = preprocess_data(df, target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"RMSE: {rmse:.2f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved in: {model_path}")


if __name__ == "__main__":
    train()