# Prédiction de la durée des trajets NYC – Projet MLOps

##  Description

Ce projet vise à prédire la durée d’un trajet de taxi à New York à l’aide du machine learning.

L’objectif est de transformer un modèle initialement développé dans un notebook en un pipeline ML structuré et prêt pour la production.

---

## Objectif

Prédire la variable **trip_duration** (durée du trajet en secondes) à partir des caractéristiques du trajet :

* heure de départ
* nombre de passagers
* coordonnées de départ et d’arrivée
* variables dérivées (features)

---

##  Structure du projet

```id="h3np4l"
MLops/
├── data/
├── models/
├── src/
│   ├── data/
│   │   └── preprocess.py
│   └── model/
│       ├── train.py
│       └── predict.py
├── config.yaml
├── requirements.txt
├── README.md
└── .gitignore
```

---

##  Installation

### 1. Créer un environnement virtuel

```bash id="xcs9k3"
python -m venv venv
venv\Scripts\activate
```

### 2. Installer les dépendances

```bash id="9lpxps"
pip install -r requirements.txt
```

---

##  Utilisation

### 🔹 Entraîner le modèle

```bash id="2w0ae6"
python -m src.model.train
```

 Cette commande permet de :

* prétraiter les données
* entraîner le modèle
* sauvegarder le modèle dans `models/model.pkl`

---

### 🔹 Lancer des prédictions

```bash id="98ug6a"
python -m src.model.predict
```

 Cette commande permet de :

* charger le modèle entraîné
* effectuer des prédictions sur des exemples

---

## Feature Engineering

Les variables utilisées sont :

* passenger_count
* coordonnées (pickup et dropoff)
* store_and_fwd_flag (encodée en binaire)
* variables temporelles :

  * heure (pickup_hour)
  * jour (pickup_day)
  * mois (pickup_month)
  * jour de la semaine (pickup_weekday)

---

## Modèle

* Modèle utilisé : **Random Forest Regressor**
* Métrique d’évaluation : **RMSE**

---

## Résultat

Le modèle entraîné est sauvegardé dans :

```id="0v7db3"
models/model.pkl
```

---

## Concepts abordés

* Prétraitement des données
* Feature engineering
* Entraînement de modèle
* Sérialisation du modèle
* Inférence
* Structuration d’un projet ML (bases du MLOps)

---

## Auteur

Bruno – Projet MLOps
