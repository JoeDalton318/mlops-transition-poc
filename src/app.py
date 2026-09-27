"""
FastAPI application for housing price prediction.
Application FastAPI pour la prédiction des prix immobiliers.

Provides a REST API endpoint that accepts housing feature inputs,
loads the trained model, and returns price predictions.
Fournit un point de terminaison API REST qui accepte les entrées de caractéristiques immobilières,
charge le modèle entraîné et renvoie les prédictions de prix.

Includes comprehensive error handling and request validation via Pydantic.
Inclut une gestion complète des erreurs et une validation des requêtes via Pydantic.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import joblib
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv
import logging

# Charger les variables d'environnement (Load environment variables)
load_dotenv()

# Configuration de la journalisation (Configure logging)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation de l'application FastAPI (Initialize FastAPI app)
app = FastAPI(
    title='Housing Price Prediction API',
    description='MLOps PoC: Predict French real estate prices / Prédiction des prix immobiliers',
    version='1.0.0',
)

# Configuration du modèle (Model configuration)
# Utilise la variable d'environnement MODELS_DIR ou 'models' par défaut
models_dir_env = os.getenv('MODELS_DIR', 'models')
MODEL_PATH = Path(__file__).parent.parent / models_dir_env / 'housing_model.pkl'

# Ordre des variables tel qu'utilisé à l'entraînement (src/train.py::load_and_prepare_data).
# Le vecteur d'inférence est construit comme un DataFrame portant ces noms, et non comme une
# liste positionnelle : sans les noms, scikit-learn accepte silencieusement un ordre de colonnes
# différent de celui de l'entraînement et renvoie des prédictions fausses sans aucune erreur.
# Feature order used at training time. The inference vector is a named DataFrame rather than a
# positional list: without names, scikit-learn silently accepts a different column order and
# returns wrong predictions with no error.
FEATURE_COLUMNS: List[str] = [
    'Surface_m2',
    'Nb_Pieces',
    'Annee_Construction',
    'Distance_Centre_km',
    'DPE_Energy_Class',
    'Has_Balcony',
    'Has_Parking',
]

# Domaine effectivement couvert par le jeu d'entraînement (data/immobilier_france.csv, 500
# lignes) : bornes issues des seuils de generate_data.py et relevées sur le fichier réel.
# Les bornes du schéma Pydantic ci-dessous restent volontairement plus larges — elles vérifient
# la plausibilité physique d'une demande, pas la validité du modèle. Une demande située hors de
# ce domaine reçoit donc une prédiction, mais celle-ci est explicitement signalée comme une
# extrapolation.
# Domain actually covered by the training set. The Pydantic bounds below stay deliberately
# wider: they check physical plausibility, not model validity. Requests outside this domain are
# answered, but the answer is explicitly flagged as an extrapolation.
TRAINING_DOMAIN: Dict[str, Tuple[float, float]] = {
    'Surface_m2': (30.0, 300.0),
    'Nb_Pieces': (1, 5),
    'Annee_Construction': (1950, 2024),
    'Distance_Centre_km': (0.5, 50.0),
    'DPE_Energy_Class': (1, 7),
    'Has_Balcony': (0, 1),
    'Has_Parking': (0, 1),
}


class HousingFeatures(BaseModel):
    """
    Request schema for housing price prediction.
    Schéma de requête pour la prédiction des prix immobiliers.

    All numeric fields represent real property characteristics from the dataset.
    Tous les champs numériques représentent de vraies caractéristiques immobilières du jeu de données.
    DPE_Energy_Class is a numerical encoding (1=worst, 7=best).
    DPE_Energy_Class est un encodage numérique (1=pire, 7=meilleur).
    """
    Surface_m2: float = Field(..., gt=0, le=300, description='Living area in square meters / Surface habitable en mètres carrés')
    Nb_Pieces: int = Field(..., ge=1, le=10, description='Number of rooms / Nombre de pièces')
    Annee_Construction: int = Field(..., ge=1900, le=2024, description='Construction year / Année de construction')
    Distance_Centre_km: float = Field(..., ge=0, le=100, description='Distance to city center in km / Distance du centre-ville en km')
    DPE_Energy_Class: int = Field(..., ge=1, le=7, description='Energy efficiency rating (1-7) / Évaluation de l\'efficacité énergétique (1-7)')
    Has_Balcony: int = Field(..., ge=0, le=1, description='Binary: 1 if has balcony, 0 otherwise / Binaire : 1 si avec balcon, 0 sinon')
    Has_Parking: int = Field(..., ge=0, le=1, description='Binary: 1 if has parking, 0 otherwise / Binaire : 1 si avec parking, 0 sinon')

    @field_validator('Nb_Pieces', 'DPE_Energy_Class', 'Has_Balcony', 'Has_Parking', mode='after')
    @classmethod
    def validate_integers(cls, v: int) -> int:
        """
        Ensure integer fields are actually integers.
        Assure que les champs entiers sont réellement des entiers.
        """
        return int(v)

    class Config:
        """
        Pydantic config for example documentation.
        Configuration Pydantic pour la documentation des exemples.
        """
        json_schema_extra = {
            'example': {
                'Surface_m2': 120.5,
                'Nb_Pieces': 3,
                'Annee_Construction': 1995,
                'Distance_Centre_km': 5.2,
                'DPE_Energy_Class': 5,
                'Has_Balcony': 1,
                'Has_Parking': 1,
            }
        }


class PredictionResponse(BaseModel):
    """
    Response schema for price predictions.
    Schéma de réponse pour les prédictions de prix.

    The two domain fields state whether the request lies inside the range of values the model
    was actually trained on. A prediction outside that range is an extrapolation: it is returned,
    but it must not be read with the same confidence.
    Les deux champs de domaine indiquent si la demande se situe dans l'intervalle de valeurs sur
    lequel le modèle a réellement été entraîné. Une prédiction hors de cet intervalle est une
    extrapolation : elle est renvoyée, mais ne doit pas être lue avec la même confiance.
    """
    predicted_price_k_eur: float = Field(..., description='Predicted price in thousands of EUR / Prix prédit en milliers d\'EUR')
    input_features: HousingFeatures
    model_version: str = '1.0.0'
    out_of_training_domain: bool = Field(default=False, description='True if at least one feature lies outside the training domain / Vrai si au moins une variable sort du domaine d\'entraînement')
    domain_warnings: List[str] = Field(default_factory=list, description='One message per feature outside the training domain / Un message par variable hors du domaine d\'entraînement')


def check_training_domain(feature_values: Dict[str, float]) -> List[str]:
    """
    List the features whose value lies outside the training domain.
    Liste les variables dont la valeur sort du domaine d'entraînement.

    Args:
        feature_values: Mapping feature name -> submitted value. (Association nom de variable ->
                        valeur soumise.)

    Returns:
        One human-readable message per out-of-domain feature, in the declaration order of
        TRAINING_DOMAIN; an empty list when the whole request lies inside the domain.
        (Un message lisible par variable hors domaine, dans l'ordre de déclaration de
        TRAINING_DOMAIN ; liste vide si toute la demande est dans le domaine.)
    """
    messages: List[str] = []
    for name, (low, high) in TRAINING_DOMAIN.items():
        value = feature_values[name]
        if value < low or value > high:
            messages.append(
                f'{name}={value} outside training domain [{low}, {high}]'
            )
    return messages


def load_model() -> LinearRegression:
    """
    Load the trained model from disk.
    Charge le modèle entraîné depuis le disque.

    Design note: the API intentionally loads a local file rather than resolving
    the MLflow Model Registry 'champion' alias directly at request/startup time.
    src/train.py's promotion gate (see promote_challenger_if_better()) only ever
    (re)writes this file when a newly trained challenger actually beats the
    current champion on R2 — so this local file is, by construction, always a
    copy of whatever model currently holds the 'champion' alias. This "shadow
    copy" pattern decouples the serving path from the MLflow tracking server's
    availability: the API keeps serving predictions even if MLflow is down,
    which would not be the case if it resolved the alias over the network on
    every reload.
    Note de conception : l'API charge volontairement un fichier local plutôt que
    de résoudre l'alias 'champion' du Model Registry MLflow directement au
    démarrage/à chaque requête. La porte de promotion de src/train.py (voir
    promote_challenger_if_better()) ne (ré)écrit ce fichier que lorsqu'un
    challenger nouvellement entraîné bat effectivement le champion actuel sur le
    R2 — ce fichier local est donc, par construction, toujours une copie du
    modèle qui détient l'alias 'champion'. Ce motif de "copie miroir" découple
    le chemin de service de la disponibilité du serveur de tracking MLflow :
    l'API continue de servir des prédictions même si MLflow est indisponible,
    ce qui ne serait pas le cas si elle résolvait l'alias sur le réseau à
    chaque rechargement.

    Returns:
        Loaded sklearn model object. (L'objet du modèle sklearn chargé.)

    Raises:
        FileNotFoundError: If model file does not exist. (Si le fichier du modèle n'existe pas.)
    """
    if not MODEL_PATH.exists():
        logger.error(f'Model not found at {MODEL_PATH}')
        raise FileNotFoundError(
            f'Model not found at {MODEL_PATH}. '
            'Please train the model first using src/train.py'
        )

    model = joblib.load(MODEL_PATH)
    logger.info(f'Model loaded successfully from {MODEL_PATH}')
    return model


# Charger le modèle au démarrage (Load model at startup)
try:
    model = load_model()
    logger.info('Application started with model loaded')
except FileNotFoundError as e:
    logger.warning(f'Model not available at startup: {e}')
    model = None


@app.get('/health', tags=['System'])
async def health_check() -> dict:
    """
    Health check endpoint for monitoring.
    Point de terminaison de vérification de l'état (santé) pour la surveillance.

    Returns:
        Status dictionary indicating application health.
        Dictionnaire d'état indiquant la santé de l'application.
    """
    model_status = 'loaded' if model is not None else 'not_loaded'
    return {
        'status': 'healthy',
        'model_status': model_status,
    }


@app.post('/predict', response_model=PredictionResponse, tags=['Prediction'])
async def predict_price(features: HousingFeatures) -> PredictionResponse:
    """
    Predict housing price based on property features.
    Prédit le prix du logement en fonction des caractéristiques de la propriété.

    This endpoint loads the trained model and generates a price prediction
    for the given housing characteristics. All inputs are validated via Pydantic.
    Ce point de terminaison charge le modèle entraîné et génère une prédiction de prix
    pour les caractéristiques immobilières données. Toutes les entrées sont validées via Pydantic.

    Args:
        features: HousingFeatures request body. (Corps de la requête des caractéristiques.)

    Returns:
        PredictionResponse containing predicted price and input features.
        Réponse de prédiction contenant le prix prédit et les caractéristiques d'entrée.

    Raises:
        HTTPException: If model is not loaded or prediction fails. (Si le modèle n'est pas chargé ou que la prédiction échoue.)
    """
    if model is None:
        logger.error('Prediction attempted but model is not loaded')
        raise HTTPException(
            status_code=503,
            detail='Model is not available. Please train the model first.',
        )

    try:
        # Construire le vecteur d'inférence avec les noms de colonnes de l'entraînement :
        # scikit-learn peut alors détecter lui-même une incohérence de schéma, au lieu de
        # dépendre silencieusement de l'ordre des valeurs.
        # Build the inference vector with the training column names, so that scikit-learn can
        # detect a schema mismatch itself instead of silently depending on value order.
        feature_values = features.model_dump()
        feature_frame = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)

        # Générer la prédiction (Generate prediction)
        prediction = model.predict(feature_frame)[0]

        # Signaler une éventuelle extrapolation hors du domaine d'entraînement
        # (Flag a possible extrapolation outside the training domain)
        domain_warnings = check_training_domain(feature_values)
        if domain_warnings:
            logger.warning('Prediction outside training domain: %s', '; '.join(domain_warnings))

        logger.info(f'Prediction generated: {prediction:.2f} k EUR for input {features}')

        return PredictionResponse(
            predicted_price_k_eur=round(prediction, 2),
            input_features=features,
            model_version='1.0.0',
            out_of_training_domain=bool(domain_warnings),
            domain_warnings=domain_warnings,
        )

    except Exception as e:
        # Sécurité : la trace/le message d'exception interne (chemins de fichiers,
        # détails de bibliothèque, etc.) est journalisé côté serveur pour le
        # diagnostic, mais jamais renvoyé tel quel au client -- un detail HTTP trop
        # bavard est une fuite d'information exploitable en production.
        # Security: the internal exception message/traceback (file paths, library
        # internals, etc.) is logged server-side for diagnostics, but never echoed
        # back to the client -- an overly verbose HTTP detail is an exploitable
        # information leak in production.
        logger.exception('Prediction error')
        raise HTTPException(
            status_code=500,
            detail='Prediction failed due to an internal error. Please contact the API operator if this persists.',
        )


@app.get('/', tags=['Info'])
async def root() -> dict:
    """
    Root endpoint with API documentation reference.
    Point de terminaison racine avec la référence de la documentation API.
    """
    return {
        'message': 'Housing Price Prediction API',
        'docs': '/docs',
        'health': '/health',
    }