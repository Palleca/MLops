import sqlite3
from datetime import datetime

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from src.model.model_wrapper import TaxiDurationModel


# Initialisation API
app = FastAPI(title="NYC Taxi API")


# Charger le modèle
base_model = joblib.load("models/model.pkl")
model = TaxiDurationModel(base_model)


# Schéma d'entrée (IMPORTANT)
class TripInput(BaseModel):
    vendor_id: int
    pickup_datetime: str
    passenger_count: int
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: str


# Base de données SQLite
def init_db():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_time TEXT,
            predicted_duration REAL
        )
    """)

    conn.commit()
    conn.close()


def save_prediction(prediction):
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (prediction_time, predicted_duration)
        VALUES (?, ?)
    """, (
        datetime.now().isoformat(),
        prediction
    ))

    conn.commit()
    conn.close()


# Initialisation DB au démarrage
@app.on_event("startup")
def startup_event():
    init_db()


# Route test
@app.get("/")
def root():
    return {"message": "API OK"}


# Endpoint principal
@app.post("/predict")
def predict(trip: TripInput):

    # Convertir en DataFrame
    input_df = pd.DataFrame([trip.model_dump()])

    # Prédiction
    prediction = model.predict(input_df)[0]

    # Sauvegarde
    save_prediction(float(prediction))

    # Réponse API
    return {
        "predicted_trip_duration": float(prediction),
        "unit": "seconds"
    }


# Lancer serveur
if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)