import joblib
import yaml

from src.data.preprocess import load_data, preprocess_data


def predict_examples():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = config["paths"]["data_path"]
    model_path = config["paths"]["model_path"]
    target_column = config["ml"]["target_column"]

    df = load_data(data_path)
    X, _ = preprocess_data(df, target_column)

    model = joblib.load(model_path)

    sample = X.head(5)
    predictions = model.predict(sample)

    print("Predictions:")
    for i, pred in enumerate(predictions, start=1):
        print(f"Trip {i}: {pred:.2f} seconds")


if __name__ == "__main__":
    predict_examples()