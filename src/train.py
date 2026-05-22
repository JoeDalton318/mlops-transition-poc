import os
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class TrainingConfig:
    """Training configuration / Configuration d'entraînement."""

    experiment_name: str = "California_Housing_Prediction"
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 100
    max_depth: int | None = 10


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the California housing dataset.

    Charge le dataset California Housing.
    """
    dataset = fetch_california_housing(as_frame=True)
    features = dataset.data
    target = dataset.target
    return features, target


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute evaluation metrics for the regression model.

    Calcule les métriques d'évaluation pour le modèle de régression.
    """
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "r2_score": r2}


def main() -> None:
    """Main entry point for training and logging the model with MLflow.

    Point d'entrée principal pour l'entraînement et l'enregistrement du modèle avec MLflow.
    """
    config = TrainingConfig()
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    features, target = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=config.random_state,
    )

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = evaluate_model(y_test.to_numpy(), predictions)

        mlflow.log_param("n_estimators", config.n_estimators)
        mlflow.log_param("max_depth", config.max_depth)
        mlflow.log_param("random_state", config.random_state)

        mlflow.log_metric("rmse", metrics["rmse"])
        mlflow.log_metric("r2_score", metrics["r2_score"])

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=None,
        )

        print("Run ID:", run.info.run_id)
        print("Metrics:", metrics)


if __name__ == "__main__":
    main()
