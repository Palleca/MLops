import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import common


# Chemins depuis config.yml
DATA_PATH = common.CONFIG["paths"]["data_processed"]
MLRUNS_PATH = common.CONFIG["paths"]["mlruns"]

# Infos MLflow
MODEL_NAME = common.CONFIG["mlflow"]["model_name"]
EXPERIMENT_NAME = common.CONFIG["mlflow"]["experiment_name"]

# Conversion du chemin Windows en URI compatible MLflow
TRACKING_URI = Path(MLRUNS_PATH).resolve().as_uri()


def load_data():
    """
    Charger les données prétraitées depuis le fichier .pkl
    """
    with open(DATA_PATH, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    return X_train, X_test, y_train, y_test


def train():
    """
    Entraîner un modèle RandomForest et enregistrer l'expérience dans MLflow
    """
    X_train, X_test, y_train, y_test = load_data()

    # Configuration MLflow
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    #  Démarrage d'un run MLflow
    with mlflow.start_run(run_name="RandomForest"):

        #  Création du modèle
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        # Entraînement
        model.fit(X_train, y_train)

        # Prédictions
        preds = model.predict(X_test)

        # Évaluation
        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)

        # Enregistrement des paramètres et métriques
        mlflow.log_param("model", "RandomForest")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Sauvegarde du modèle dans MLflow (registry)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")


if __name__ == "__main__":
    train()