"""
Model training pipeline with MLflow integration.

This module encapsulates the training workflow in a reusable run_pipeline() function
that can be triggered externally (e.g., by drift detection or continuous training systems).
It logs all metrics, parameters, and model artifacts to MLflow for reproducibility
and versioning.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any


# Configuration
MODELS_DIR = Path(__file__).parent.parent / 'models'
DATA_FILE = Path(__file__).parent.parent / 'immobilier_france.csv'
MODEL_NAME = 'housing_model'
RANDOM_STATE = 42


def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and prepare the housing dataset.

    Returns:
        Tuple of (features DataFrame, target Series).
    
    Raises:
        FileNotFoundError: If the data file does not exist.
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_FILE}. Run generate_data.py first.")

    df = pd.read_csv(DATA_FILE)

    # Define feature columns (exclude target)
    feature_columns = [
        'Surface_m2',
        'Nb_Pieces',
        'Annee_Construction',
        'Distance_Centre_km',
        'DPE_Energy_Class',
        'Has_Balcony',
        'Has_Parking',
    ]

    X = df[feature_columns]
    y = df['Prix_k_EUR']

    return X, y


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """
    Train a linear regression model on housing features.

    Args:
        X_train: Training features.
        y_train: Training target (price in k EUR).

    Returns:
        Trained LinearRegression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate model performance on test set.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test target.

    Returns:
        Dictionary of evaluation metrics (MSE, RMSE, MAE, R2).
    """
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
    }


def run_pipeline(trigger_source: str = 'manual') -> Dict[str, Any]:
    """
    Execute the complete training pipeline with MLflow logging.

    This function orchestrates data loading, model training, evaluation,
    and artifact logging. It can be called externally by continuous training
    or drift detection systems.

    Args:
        trigger_source: Source of the training trigger
                       ('manual', 'drift_detection', 'scheduled').
                       Used for logging purposes.

    Returns:
        Dictionary containing:
            - model_path: Path to saved model
            - metrics: Evaluation metrics
            - run_id: MLflow run ID
            - status: 'success' or 'failed'
            - message: Descriptive message
    """
    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # MLflow experiment setup
        experiment_name = 'housing_price_prediction'
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(description=f'Training triggered by {trigger_source}'):
            print("Loading data...")
            X, y = load_and_prepare_data()

            # Log dataset info
            mlflow.log_param('dataset_size', len(X))
            mlflow.log_param('trigger_source', trigger_source)
            mlflow.log_param('random_state', RANDOM_STATE)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=RANDOM_STATE
            )

            print("Training model...")
            model = train_model(X_train, y_train)

            print("Evaluating model...")
            metrics = evaluate_model(model, X_test, y_test)

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Save model locally
            model_path = MODELS_DIR / f'{MODEL_NAME}.pkl'
            joblib.dump(model, model_path)
            print(f"Model saved to {model_path}")

            # Log model as artifact
            mlflow.sklearn.log_model(model, 'model', registered_model_name=MODEL_NAME)
            mlflow.log_artifact(str(model_path), artifact_path='models')

            # Get MLflow run ID
            run_id = mlflow.active_run().info.run_id

            result = {
                'model_path': str(model_path),
                'metrics': metrics,
                'run_id': run_id,
                'status': 'success',
                'message': f'Model trained successfully. R2: {metrics["r2"]:.4f}',
            }

            print(f"Training complete. {result['message']}")
            return result

    except Exception as e:
        error_message = f'Training failed: {str(e)}'
        print(f"ERROR: {error_message}")
        return {
            'model_path': None,
            'metrics': None,
            'run_id': None,
            'status': 'failed',
            'message': error_message,
        }


def main() -> None:
    """
    Entry point for manual model training.
    """
    result = run_pipeline(trigger_source='manual')
    if result['status'] == 'success':
        print(f"\nTraining metrics:\n{result['metrics']}")
    else:
        print(f"\nTraining failed: {result['message']}")


if __name__ == '__main__':
    main()