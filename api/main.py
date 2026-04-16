import sqlite3
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from src.model.model_wrapper import TaxiDurationModel


app = FastAPI(title="NYC Taxi API")


# Petit registre local de modèles
MODEL_REGISTRY = {
    "v1": {
        "path": "models/model.pkl",
        "created_at": "2026-04-15T00:00:00",
        "version": "v1"
    }
}


def get_model_info(model_version: str | None = None):
    if model_version is None:
        model_version = "v1"
    return MODEL_REGISTRY[model_version]


def load_wrapped_model(model_version: str | None = None):
    model_info = get_model_info(model_version)
    base_model = joblib.load(model_info["path"])
    wrapped_model = TaxiDurationModel(base_model)
    return wrapped_model, model_info


def haversine_meters(lat1, lon1, lat2, lon2):
    R = 6371000  # mètres

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


class TripInput(BaseModel):
    vendor_id: int
    pickup_datetime: str
    passenger_count: int = Field(gt=0)
    pickup_longitude: float = Field(ge=-180, le=180)
    pickup_latitude: float = Field(ge=-90, le=90)
    dropoff_longitude: float = Field(ge=-180, le=180)
    dropoff_latitude: float = Field(ge=-90, le=90)
    store_and_fwd_flag: str
    model_version: str | None = None

    @field_validator("store_and_fwd_flag")
    @classmethod
    def validate_flag(cls, v):
        if v not in {"Y", "N"}:
            raise ValueError("store_and_fwd_flag doit être 'Y' ou 'N'")
        return v

    @field_validator("dropoff_latitude")
    @classmethod
    def validate_distance(cls, v, info):
        data = info.data

        required_fields = [
            "pickup_latitude",
            "pickup_longitude",
            "dropoff_longitude",
        ]

        if all(field in data for field in required_fields):
            distance = haversine_meters(
                data["pickup_latitude"],
                data["pickup_longitude"],
                v,
                data["dropoff_longitude"],
            )
            if distance <= 50:
                raise ValueError("La distance entre pickup et dropoff doit être > 50 mètres")

        return v


class BatchTripInput(BaseModel):
    trips: list[TripInput]
    model_version: str | None = None


def init_db():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            inference_time TEXT,
            model_version TEXT,
            model_path TEXT,
            model_created_at TEXT,
            vendor_id INTEGER,
            pickup_datetime TEXT,
            passenger_count INTEGER,
            pickup_longitude REAL,
            pickup_latitude REAL,
            dropoff_longitude REAL,
            dropoff_latitude REAL,
            store_and_fwd_flag TEXT,
            predicted_duration REAL
        )
    """)

    conn.commit()
    conn.close()


def save_prediction(trip: TripInput, predicted_duration: float, model_info: dict):
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            inference_time,
            model_version,
            model_path,
            model_created_at,
            vendor_id,
            pickup_datetime,
            passenger_count,
            pickup_longitude,
            pickup_latitude,
            dropoff_longitude,
            dropoff_latitude,
            store_and_fwd_flag,
            predicted_duration
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        model_info["version"],
        model_info["path"],
        model_info["created_at"],
        trip.vendor_id,
        trip.pickup_datetime,
        trip.passenger_count,
        trip.pickup_longitude,
        trip.pickup_latitude,
        trip.dropoff_longitude,
        trip.dropoff_latitude,
        trip.store_and_fwd_flag,
        predicted_duration
    ))

    conn.commit()
    conn.close()


@app.on_event("startup")
def startup_event():
    init_db()


@app.get("/")
def root():
    return {"message": "API OK"}


@app.post("/predict")
def predict(trip: TripInput):
    wrapped_model, model_info = load_wrapped_model(trip.model_version)

    input_df = pd.DataFrame([trip.model_dump(exclude={"model_version"})])
    prediction = wrapped_model.predict(input_df)[0]

    save_prediction(trip, float(prediction), model_info)

    return {
        "predicted_trip_duration": float(prediction),
        "unit": "seconds",
        "model_version": model_info["version"]
    }


@app.post("/predict_batch")
def predict_batch(batch: BatchTripInput):
    model_version = batch.model_version

    if model_version is None and len(batch.trips) > 0:
        model_version = batch.trips[0].model_version

    wrapped_model, model_info = load_wrapped_model(model_version)

    rows = []
    for trip in batch.trips:
        row = trip.model_dump(exclude={"model_version"})
        rows.append(row)

    input_df = pd.DataFrame(rows)
    predictions = wrapped_model.predict(input_df)

    for trip, pred in zip(batch.trips, predictions):
        save_prediction(trip, float(pred), model_info)

    return {
        "predictions": [float(pred) for pred in predictions],
        "unit": "seconds",
        "model_version": model_info["version"]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)