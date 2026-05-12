"""
modelling.py
============
File training untuk MLflow Project (Kriteria 3).
Dijalankan via: mlflow run ./MLProject --env-manager=local

Perbedaan dari Kriteria 2:
- Tracking ke DagsHub (bukan localhost) untuk CI/CD
- Path dataset relatif terhadap lokasi file ini
- Kompatibel dengan mlflow run command
"""

import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)

TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"https://dagshub.com/iqbalrza/Workflow-CI_MuhamadIqbalReza.mlflow"
)

mlflow.set_tracking_uri(TRACKING_URI)

os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ.get("DAGSHUB_USERNAME", "iqbalrza")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ.get("DAGSHUB_TOKEN", "")

experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "Telco Churn CI Pipeline")
mlflow.set_experiment(experiment_name)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "telco_churn_preprocessing")

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

print(f"Data loaded - Train: {X_train.shape}, Test: {X_test.shape}")

with mlflow.start_run(run_name="CI_RandomForest"):

    # Parameters
    n_estimators = int(os.environ.get("N_ESTIMATORS", 200))
    max_depth    = os.environ.get("MAX_DEPTH", "10")
    max_depth    = None if max_depth == "None" else int(max_depth)
    random_state = 42

    mlflow.log_param("n_estimators",  n_estimators)
    mlflow.log_param("max_depth",     max_depth)
    mlflow.log_param("random_state",  random_state)

    # Training
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_pred_prob)

    mlflow.log_metric("accuracy",   accuracy)
    mlflow.log_metric("precision",  precision)
    mlflow.log_metric("recall",     recall)
    mlflow.log_metric("f1_score",   f1)
    mlflow.log_metric("roc_auc",    roc_auc)

    # Log model
    mlflow.sklearn.log_model(
        sk_model=best_model if 'best_model' in dir() else model,
        artifact_path="model"
    )

    run_id = mlflow.active_run().info.run_id
    print(f"\nRun ID: {run_id}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nTraining selesai!")