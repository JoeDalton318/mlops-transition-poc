"""
Unit tests for the FastAPI prediction endpoint.

This module provides basic smoke tests to verify API functionality
before deployment. Tests cover health checks and request validation.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from app import app


@pytest.fixture
def client():
    """
    Provide a TestClient instance for API testing.

    Returns:
        TestClient configured for the FastAPI application.
    """
    return TestClient(app)


def test_health_check(client):
    """
    Test the health check endpoint.

    Verifies that the API is running and responsive.
    This is a basic smoke test to catch obvious deployment failures.
    """
    response = client.get('/health')
    assert response.status_code == 200
    data = response.json()
    assert 'status' in data
    assert data['status'] == 'healthy'


def test_root_endpoint(client):
    """
    Test the root endpoint.

    Verifies that the API provides documentation and info endpoints.
    """
    response = client.get('/')
    assert response.status_code == 200
    data = response.json()
    assert 'docs' in data
    assert data['docs'] == '/docs'


def test_prediction_endpoint_with_valid_input(client):
    """
    Test the prediction endpoint with valid housing features.

    Sends a complete, valid request to verify the prediction pipeline
    works end-to-end. Returns 200 on success or 503 if model not loaded.
    """
    valid_features = {
        'Surface_m2': 120.5,
        'Nb_Pieces': 3,
        'Annee_Construction': 1995,
        'Distance_Centre_km': 5.2,
        'DPE_Energy_Class': 5,
        'Has_Balcony': 1,
        'Has_Parking': 1,
    }

    response = client.post('/predict', json=valid_features)

    # Accept both 200 (model loaded) and 503 (model not available in CI)
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        assert 'predicted_price_k_eur' in data
        assert isinstance(data['predicted_price_k_eur'], (int, float))
        assert data['predicted_price_k_eur'] > 0
        assert 'input_features' in data


def test_prediction_endpoint_with_invalid_input(client):
    """
    Test the prediction endpoint with invalid input.

    Verifies that the API properly rejects malformed requests
    with appropriate HTTP status codes.
    """
    invalid_features = {
        'Surface_m2': -50.0,  # Invalid: negative area
        'Nb_Pieces': 3,
        'Annee_Construction': 1995,
        'Distance_Centre_km': 5.2,
        'DPE_Energy_Class': 5,
        'Has_Balcony': 1,
        'Has_Parking': 1,
    }

    response = client.post('/predict', json=invalid_features)
    assert response.status_code == 422  # Validation error


def test_prediction_endpoint_missing_fields(client):
    """
    Test the prediction endpoint with incomplete request.

    Verifies that the API rejects requests missing required fields.
    """
    incomplete_features = {
        'Surface_m2': 120.5,
        'Nb_Pieces': 3,
        # Missing other required fields
    }

    response = client.post('/predict', json=incomplete_features)
    assert response.status_code == 422  # Validation error


if __name__ == '__main__':
    pytest.main([__file__, '-v'])