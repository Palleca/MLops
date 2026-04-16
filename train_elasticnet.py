import pandas as pd
import numpy as np
import pickle
from pathlib import Path

import mlflow
from sklearn.linear_model import ElasticNet

import common as common


# Chemins depuis le fichier config
DATA_PROC_PATH = common.CONFIG["paths"]["data_processed"]
DIR_MLRUNS = common.CONFIG["paths"]["mlruns"]

# Paramètres ML
RANDOM_STATE = common.CONFIG["ml"]["random_state"]

# Paramètres MLflow
EXPERIMENT_NAME = common.CONFIG["mlflow"]["experiment_name"]
MODEL_NAME = common.CONFIG["mlflow"]["model_name"]
ARTIFACT_PATH = common.CONFIG["mlflow"]["artifact_path"]

# Conversion du chemin local en URI compatible MLflow
TRACKING_URI = Path(DIR_MLRUNS).resolve().as_uri()


def load_data():
    """
    Charger les données prétraitées depuis le fichier .pkl
    """
    with open(DATA_PROC_PATH, "rb") as file:
        X_train, X_test, y_train, y_test = pickle.load(file)

    return X_train, X_test, y_train, y_test


def train_and_log_model(model, X_train, X_test, y_train, y_test):
    """
    Entraîner le modèle, le logger dans MLflow, puis évaluer ses performances
    """
    # Entraînement du modèle
    model.fit(X_train, y_train)

    # Inférer automatiquement la signature du modèle
    # à partir des données d'entrée et de sortie
    signature = mlflow.models.infer_signature(X_train, y_train)

    # Enregistrer le modèle dans MLflow
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=ARTIFACT_PATH,
        signature=signature
    )

    # Évaluer le modèle sur le jeu de test
    results = mlflow.evaluate(
        model_info.model_uri,
        data=pd.concat([X_test, y_test], axis=1),
        targets=y_test.name,
        model_type="regressor",
        evaluators=["default"]
    )

    return results


if __name__ == "__main__":

    # Utiliser MLflow en local
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)

    # Charger les données prétraitées
    X_train, X_test, y_train, y_test = load_data()

    # Définir l'expérience MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Grille d'hyperparamètres à tester
    params_alpha = [0.01, 0.1, 1, 10]
    params_l1_ratio = np.arange(0.0, 1.1, 0.5)

    num_iterations = len(params_alpha) * len(params_l1_ratio)

    run_name = "elasticnet"
    k = 0
    best_score = float("inf")
    best_run_id = None

    # Run parent : contient plusieurs runs enfants
    with mlflow.start_run(run_name=run_name, description=run_name) as parent_run:
        for alpha in params_alpha:
            for l1_ratio in params_l1_ratio:
                k += 1
                print(f"\n***** ITERATION {k} sur {num_iterations} *****")

                child_run_name = f"{run_name}_{k:02}"
                model = ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    random_state=RANDOM_STATE
                )

                # Run enfant : un test par combinaison d'hyperparamètres
                with mlflow.start_run(run_name=child_run_name, nested=True) as child_run:
                    results = train_and_log_model(
                        model,
                        X_train,
                        X_test,
                        y_train,
                        y_test
                    )

                    # Logger les hyperparamètres testés
                    mlflow.log_param("alpha", alpha)
                    mlflow.log_param("l1_ratio", l1_ratio)

                    # Garder en mémoire le meilleur modèle selon le RMSE
                    if results.metrics["root_mean_squared_error"] < best_score:
                        best_score = results.metrics["root_mean_squared_error"]
                        best_run_id = child_run.info.run_id

                    print(f"rmse: {results.metrics['root_mean_squared_error']}")
                    print(f"r2: {results.metrics['r2_score']}")

    print("#" * 20)

    # Enregistrer le meilleur modèle dans le Model Registry
    model_uri = f"runs:/{best_run_id}/{ARTIFACT_PATH}"
    mv = mlflow.register_model(model_uri, MODEL_NAME)

    print("Modèle enregistré dans le Model Registry :")
    print(f"Nom : {mv.name}")
    print(f"Version : {mv.version}")
    print(f"Source : {mv.source}")