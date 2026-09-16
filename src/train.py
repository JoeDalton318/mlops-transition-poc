"""
Model training pipeline with MLflow integration and champion/challenger promotion.
Pipeline d'entraînement du modèle avec intégration MLflow et promotion champion/challenger.

This module encapsulates the training workflow in a reusable run_pipeline() function
that can be triggered externally (e.g., by drift detection or continuous training systems).
It logs all metrics, parameters, and model artifacts to MLflow for reproducibility
and versioning.

Ce module encapsule le flux d'entraînement dans une fonction run_pipeline() réutilisable
qui peut être déclenchée en externe (par exemple, par la détection de dérive ou les systèmes d'entraînement continu).
Il journalise toutes les métriques, paramètres et artefacts de modèle dans MLflow pour la reproductibilité
et le versionnage.

Promotion logic (added Sept. 2026): each newly trained model is a *challenger*.
It is only promoted to the MLflow Model Registry alias ``champion`` — and only
then copied to the local ``models/housing_model.pkl`` file served by the API —
if it improves on the current champion's R2 by at least PROMOTION_MIN_GAIN, or
if no champion exists yet. This replaces the previous behavior where every run
silently overwrote the serving model with no comparison.

Logique de promotion (ajoutée sept. 2026) : chaque modèle nouvellement entraîné
est un *challenger*. Il n'est promu vers l'alias ``champion`` du Model Registry
MLflow — et copié vers le fichier local ``models/housing_model.pkl`` servi par
l'API qu'à cette condition — que s'il améliore le R2 du champion actuel d'au
moins PROMOTION_MIN_GAIN, ou si aucun champion n'existe encore. Ceci remplace
le comportement précédent où chaque exécution écrasait silencieusement le
modèle servi sans comparaison.
"""

import logging
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from dotenv import load_dotenv

# Charger les variables d'environnement (Load environment variables)
load_dotenv()

logger = logging.getLogger(__name__)


# Configuration (chargée depuis .env ou par défaut)
models_dir_env = os.getenv('MODELS_DIR', 'models')
data_file_env = os.getenv('DATA_FILE_PATH', 'data/immobilier_france.csv')

MODELS_DIR = Path(__file__).parent.parent / models_dir_env
DATA_FILE = Path(__file__).parent.parent / data_file_env
MODEL_NAME = 'housing_model'
RANDOM_STATE = 42
CHAMPION_ALIAS = 'champion'
PROMOTION_METRIC = 'r2'  # higher is better / plus élevé est meilleur
PROMOTION_MIN_GAIN = float(os.getenv('PROMOTION_MIN_GAIN', '0.0'))


def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and prepare the housing dataset.
    Charge et prépare l'ensemble de données immobilières.

    Returns:
        Tuple of (features DataFrame, target Series).
        Tuple de (DataFrame des caractéristiques, Series de la cible).
    
    Raises:
        FileNotFoundError: If the data file does not exist. (Si le fichier de données n'existe pas.)
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_FILE}. Run generate_data.py first.")

    df = pd.read_csv(DATA_FILE)

    # Define feature columns (exclude target)
    # Définir les colonnes de caractéristiques (exclure la cible)
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
    Entraîne un modèle de régression linéaire sur les caractéristiques immobilières.

    Args:
        X_train: Training features. (Caractéristiques d'entraînement.)
        y_train: Training target (price in k EUR). (Cible d'entraînement : prix en k EUR.)

    Returns:
        Trained LinearRegression model. (Modèle de régression linéaire entraîné.)
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate model performance on test set.
    Évalue la performance du modèle sur l'ensemble de test.

    Args:
        model: Trained model. (Modèle entraîné.)
        X_test: Test features. (Caractéristiques de test.)
        y_test: Test target. (Cible de test.)

    Returns:
        Dictionary of evaluation metrics (MSE, RMSE, MAE, R2).
        Dictionnaire des métriques d'évaluation (MSE, RMSE, MAE, R2).
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


def get_current_champion_metric(client: MlflowClient, metric_name: str = PROMOTION_METRIC) -> Optional[float]:
    """
    Fetch the evaluation metric of the model version currently aliased 'champion'.
    Récupère la métrique d'évaluation de la version de modèle actuellement aliasée 'champion'.

    Args:
        client: MLflow tracking/registry client. (Client de suivi/registre MLflow.)
        metric_name: Name of the metric to read from the champion's run. (Nom de la métrique à lire.)

    Returns:
        The champion's metric value, or None if no champion alias exists yet
        (first run, or metric missing on the champion's run).
        La valeur de la métrique du champion, ou None si aucun alias champion
        n'existe encore (premier run, ou métrique absente du run du champion).
    """
    try:
        champion_version = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException:
        logger.info('No existing champion alias found for %s (first promotion).', MODEL_NAME)
        return None

    try:
        champion_run = client.get_run(champion_version.run_id)
        return champion_run.data.metrics.get(metric_name)
    except MlflowException as e:
        logger.warning('Could not read champion run metrics: %s', str(e))
        return None


def find_model_version_for_run(client: MlflowClient, run_id: str) -> Optional[str]:
    """
    Locate the registered model version created for a given MLflow run.
    Localise la version de modèle enregistrée créée pour un run MLflow donné.

    Used instead of relying on a specific mlflow.sklearn.log_model() return
    attribute, to stay robust across minor MLflow client versions.
    Utilisé plutôt que de dépendre d'un attribut spécifique du retour de
    mlflow.sklearn.log_model(), pour rester robuste face aux versions mineures
    du client MLflow.

    Args:
        client: MLflow tracking/registry client. (Client de suivi/registre MLflow.)
        run_id: MLflow run ID that logged the model. (ID du run ayant journalisé le modèle.)

    Returns:
        The matching registered model version number as a string, or None if
        not found. (Le numéro de version correspondant, ou None si introuvable.)
    """
    try:
        versions = client.search_model_versions(f"run_id='{run_id}'")
        matching = [v for v in versions if v.name == MODEL_NAME]
        if not matching:
            return None
        # Most recent registration for this run / enregistrement le plus récent pour ce run
        matching.sort(key=lambda v: int(v.version), reverse=True)
        return matching[0].version
    except MlflowException as e:
        logger.warning('Could not resolve registered model version for run %s: %s', run_id, str(e))
        return None


def promote_challenger_if_better(
    client: MlflowClient,
    run_id: str,
    challenger_metrics: Dict[str, float],
    model_path: Path,
    model: LinearRegression,
) -> Dict[str, Any]:
    """
    Compare the challenger (newly trained model) against the current champion
    and promote it — MLflow alias + local serving file — only if it wins.
    Compare le challenger (modèle nouvellement entraîné) au champion actuel et
    ne le promeut (alias MLflow + fichier local de service) que s'il l'emporte.

    Promotion gate: promote if there is no existing champion, or if
    challenger_metrics[PROMOTION_METRIC] > champion_metric + PROMOTION_MIN_GAIN.
    Porte de promotion : promotion si aucun champion n'existe, ou si la
    métrique du challenger dépasse celle du champion d'au moins PROMOTION_MIN_GAIN.

    Args:
        client: MLflow tracking/registry client. (Client de suivi/registre MLflow.)
        run_id: Run ID of the challenger. (ID du run du challenger.)
        challenger_metrics: Evaluation metrics of the challenger. (Métriques du challenger.)
        model_path: Local path the serving file should be written to if promoted.
                    (Chemin local du fichier de service à écrire en cas de promotion.)
        model: The trained challenger model object, for the local fallback save.
               (L'objet modèle challenger entraîné, pour la sauvegarde locale.)

    Returns:
        Dictionary with 'promoted' (bool), 'champion_metric_before' (float or None),
        'challenger_metric' (float), and 'registered_version' (str or None).
        Dictionnaire avec 'promoted' (bool), 'champion_metric_before' (float ou None),
        'challenger_metric' (float) et 'registered_version' (str ou None).
    """
    champion_metric_before = get_current_champion_metric(client)
    challenger_metric = challenger_metrics[PROMOTION_METRIC]

    should_promote = bool(
        champion_metric_before is None
        or challenger_metric > (champion_metric_before + PROMOTION_MIN_GAIN)
    )

    registered_version = None

    if should_promote:
        registered_version = find_model_version_for_run(client, run_id)
        if registered_version is not None:
            try:
                client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, registered_version)
                logger.info(
                    "Promoted version %s to alias '%s' (R2 %.4f vs previous champion %s).",
                    registered_version,
                    CHAMPION_ALIAS,
                    challenger_metric,
                    f'{champion_metric_before:.4f}' if champion_metric_before is not None else 'none',
                )
            except MlflowException as e:
                # Alias API unavailable/misconfigured on this MLflow backend: degrade
                # gracefully rather than failing the whole training run.
                # API d'alias indisponible/mal configurée sur ce backend MLflow :
                # dégradation gracieuse plutôt que faire échouer tout l'entraînement.
                logger.warning('Could not set registry alias (%s); serving file still updated locally.', str(e))
        else:
            logger.warning(
                'Promotion decided but no registered model version was found for run %s; '
                'serving file still updated locally.', run_id
            )

        # Fichier local servi par l'API : uniquement mis à jour en cas de promotion
        # (Local file served by the API: only updated on promotion)
        joblib.dump(model, model_path)
    else:
        logger.info(
            'Challenger not promoted (R2 %.4f does not beat champion %.4f + min_gain %.4f). '
            'Serving file unchanged.',
            challenger_metric, champion_metric_before, PROMOTION_MIN_GAIN
        )

    return {
        'promoted': should_promote,
        'champion_metric_before': champion_metric_before,
        'challenger_metric': challenger_metric,
        'registered_version': registered_version,
    }


def run_pipeline(trigger_source: str = 'manual') -> Dict[str, Any]:
    """
    Execute the complete training pipeline with MLflow logging.
    Exécute le pipeline d'entraînement complet avec la journalisation MLflow.

    This function orchestrates data loading, model training, evaluation,
    and artifact logging. It can be called externally by continuous training
    or drift detection systems.
    Cette fonction orchestre le chargement des données, l'entraînement du modèle, l'évaluation,
    et la journalisation des artefacts. Elle peut être appelée en externe par les systèmes
    d'entraînement continu ou de détection de dérive.

    Args:
        trigger_source: Source of the training trigger ('manual', 'drift_detection', 'scheduled').
                        Used for logging purposes.
                        Source du déclencheur d'entraînement. Utilisé à des fins de journalisation.

    Returns:
        Dictionary containing: (Dictionnaire contenant :)
            - model_path: Path to saved model, only meaningful if promoted (Chemin vers le modèle sauvegardé)
            - metrics: Evaluation metrics of the challenger (Métriques d'évaluation du challenger)
            - run_id: MLflow run ID (ID d'exécution MLflow)
            - status: 'success' or 'failed' (Statut : 'success' ou 'failed')
            - message: Descriptive message (Message descriptif)
            - promoted: whether this run's model became the new champion (bool)
                        (si le modèle de ce run est devenu le nouveau champion)
            - champion_metric_before: previous champion's R2, or None (R2 du champion précédent, ou None)
    """
    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # MLflow experiment setup
        experiment_name = 'housing_price_prediction'
        mlflow.set_experiment(experiment_name)
        client = MlflowClient()

        with mlflow.start_run(description=f'Training triggered by {trigger_source}'):
            logger.info("Loading data...")
            X, y = load_and_prepare_data()

            # Log dataset info
            mlflow.log_param('dataset_size', len(X))
            mlflow.log_param('trigger_source', trigger_source)
            mlflow.log_param('random_state', RANDOM_STATE)
            mlflow.log_param('promotion_min_gain', PROMOTION_MIN_GAIN)
            mlflow.log_param('promotion_metric', PROMOTION_METRIC)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=RANDOM_STATE
            )

            logger.info("Training model (challenger)...")
            model = train_model(X_train, y_train)

            logger.info("Evaluating model...")
            metrics = evaluate_model(model, X_test, y_test)

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Journaliser le modèle comme artefact et l'enregistrer dans le Model Registry
            # (Log model as artifact and register it in the Model Registry)
            mlflow.sklearn.log_model(model, 'model', registered_model_name=MODEL_NAME)

            # Récupérer l'ID d'exécution MLflow (Get MLflow run ID)
            run_id = mlflow.active_run().info.run_id

            # Porte de promotion champion/challenger avant de servir le modèle
            # (Champion/challenger promotion gate before serving the model)
            model_path = MODELS_DIR / f'{MODEL_NAME}.pkl'
            promotion = promote_challenger_if_better(client, run_id, metrics, model_path, model)

            mlflow.log_param('promoted', promotion['promoted'])
            if promotion['champion_metric_before'] is not None:
                mlflow.log_metric('champion_metric_before', promotion['champion_metric_before'])

            if promotion['promoted']:
                mlflow.log_artifact(str(model_path), artifact_path='models')
                logger.info("Model saved to %s (promoted to '%s')", model_path, CHAMPION_ALIAS)
            else:
                logger.info(
                    "Challenger R2=%.4f did not beat champion R2=%.4f; serving file unchanged.",
                    metrics['r2'], promotion['champion_metric_before'],
                )

            result = {
                'model_path': str(model_path) if promotion['promoted'] else None,
                'metrics': metrics,
                'run_id': run_id,
                'status': 'success',
                'message': (
                    f"Model trained successfully. R2: {metrics['r2']:.4f} "
                    f"({'promoted to champion' if promotion['promoted'] else 'kept as challenger, not promoted'})."
                ),
                'promoted': promotion['promoted'],
                'champion_metric_before': promotion['champion_metric_before'],
                'registered_version': promotion['registered_version'],
            }

            logger.info("Training complete. %s", result['message'])
            return result

    except Exception as e:
        error_message = f'Training failed: {str(e)}'
        logger.error(error_message)
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
    Point d'entrée pour l'entraînement manuel du modèle.
    """
    result = run_pipeline(trigger_source='manual')
    if result['status'] == 'success':
        print(f"\nTraining metrics:\n{result['metrics']}")
    else:
        print(f"\nTraining failed: {result['message']}")


if __name__ == '__main__':
    main()