"""
FastAPI application for housing price prediction.

Provides a REST API endpoint that accepts housing feature inputs,
loads the trained model, and returns price predictions.
Includes comprehensive error handling and request validation via Pydantic.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import joblib
from pathlib import Path
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title='Housing Price Prediction API',
    description='MLOps PoC: Predict French real estate prices',
    version='1.0.0',
)

# Model configuration
MODEL_PATH = Path(__file__).parent.parent / 'models' / 'housing_model.pkl'


class HousingFeatures(BaseModel):
    """
    Request schema for housing price prediction.

    All numeric fields represent real property characteristics from the dataset.
    DPE_Energy_Class is a numerical encoding (1=worst, 7=best).
    """
    Surface_m2: float = Field(..., gt=0, le=300, description='Living area in square meters')
    Nb_Pieces: int = Field(..., ge=1, le=10, description='Number of rooms')
    Annee_Construction: int = Field(..., ge=1900, le=2024, description='Construction year')
    Distance_Centre_km: float = Field(..., ge=0, le=100, description='Distance to city center in km')
    DPE_Energy_Class: int = Field(..., ge=1, le=7, description='Energy efficiency rating (1-7)')
    Has_Balcony: int = Field(..., ge=0, le=1, description='Binary: 1 if has balcony, 0 otherwise')
    Has_Parking: int = Field(..., ge=0, le=1, description='Binary: 1 if has parking, 0 otherwise')

    @field_validator('Nb_Pieces', 'DPE_Energy_Class', 'Has_Balcony', 'Has_Parking', mode='after')
    @classmethod
    def validate_integers(cls, v: int) -> int:
        """Ensure integer fields are actually integers."""
        return int(v)

    class Config:
        """Pydantic config for example documentation."""
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
    """Response schema for price predictions."""
    predicted_price_k_eur: float = Field(..., description='Predicted price in thousands of EUR')
    input_features: HousingFeatures
    model_version: str = '1.0.0'


def load_model():
    """
    Load the trained model from disk.

    Returns:
        Loaded sklearn model object.
    
    Raises:
        FileNotFoundError: If model file does not exist.
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


# Load model at startup
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

    Returns:
        Status dictionary indicating application health.
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

    This endpoint loads the trained model and generates a price prediction
    for the given housing characteristics. All inputs are validated via Pydantic.

    Args:
        features: HousingFeatures request body.

    Returns:
        PredictionResponse containing predicted price and input features.

    Raises:
        HTTPException: If model is not loaded or prediction fails.
    """
    if model is None:
        logger.error('Prediction attempted but model is not loaded')
        raise HTTPException(
            status_code=503,
            detail='Model is not available. Please train the model first.',
        )

    try:
        # Prepare features as array matching model training order
        feature_array = [
            [
                features.Surface_m2,
                features.Nb_Pieces,
                features.Annee_Construction,
                features.Distance_Centre_km,
                features.DPE_Energy_Class,
                features.Has_Balcony,
                features.Has_Parking,
            ]
        ]

        # Generate prediction
        prediction = model.predict(feature_array)[0]

        logger.info(f'Prediction generated: {prediction:.2f} k EUR for input {features}')

        return PredictionResponse(
            predicted_price_k_eur=round(prediction, 2),
            input_features=features,
            model_version='1.0.0',
        )

    except Exception as e:
        logger.error(f'Prediction error: {str(e)}')
        raise HTTPException(
            status_code=500,
            detail=f'Prediction failed: {str(e)}',
        )


@app.get('/', tags=['Info'])
async def root() -> dict:
    """Root endpoint with API documentation reference."""
    return {
        'message': 'Housing Price Prediction API',
        'docs': '/docs',
        'health': '/health',
    }