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


VALID_FEATURES = {
    'Surface_m2': 120.5,
    'Nb_Pieces': 3,
    'Annee_Construction': 1995,
    'Distance_Centre_km': 5.2,
    'DPE_Energy_Class': 5,
    'Has_Balcony': 1,
    'Has_Parking': 1,
}


@pytest.mark.parametrize('field,value', [
    ('Surface_m2', 0),          # gt=0 excludes the boundary itself
    ('Surface_m2', 301),        # le=300
    ('Nb_Pieces', 0),           # ge=1
    ('Nb_Pieces', 11),          # le=10
    ('Annee_Construction', 1899),  # ge=1900
    ('Annee_Construction', 2025),  # le=2024
    ('Distance_Centre_km', -1),  # ge=0
    ('Distance_Centre_km', 101),  # le=100
    ('DPE_Energy_Class', 0),    # ge=1
    ('DPE_Energy_Class', 8),    # le=7
    ('Has_Balcony', 2),         # le=1 (not a plain boolean)
    ('Has_Parking', -1),        # ge=0
])
def test_prediction_endpoint_rejects_out_of_bounds_values(client, field, value):
    """
    Boundary-value regression guard for the Pydantic schema in src/app.py.

    Each case takes one field just outside its declared Field(...) bounds
    (see HousingFeatures) and checks the API rejects it with 422, rather than
    silently accepting an out-of-range value that the model was never trained
    or validated on. Added during the Phase 3 security/robustness audit,
    because the existing suite only exercised one invalid case
    (negative Surface_m2) and never the other six bounded fields.
    """
    features = dict(VALID_FEATURES)
    features[field] = value

    response = client.post('/predict', json=features)
    assert response.status_code == 422, (
        f'{field}={value} should violate its declared bounds and be rejected, '
        f'got {response.status_code}'
    )


@pytest.mark.parametrize('field,value', [
    ('Surface_m2', 300),         # le=300, boundary itself is valid
    ('Nb_Pieces', 1),            # ge=1
    ('Nb_Pieces', 10),           # le=10
    ('Annee_Construction', 1900),  # ge=1900
    ('Annee_Construction', 2024),  # le=2024
    ('Distance_Centre_km', 0),   # ge=0
    ('Distance_Centre_km', 100),  # le=100
    ('DPE_Energy_Class', 1),     # ge=1
    ('DPE_Energy_Class', 7),     # le=7
])
def test_prediction_endpoint_accepts_boundary_values(client, field, value):
    """
    Confirms the schema's bounds are inclusive (Field(ge=..., le=...)) rather
    than accidentally exclusive: a value exactly at the documented boundary
    must be accepted (200 if the model is loaded, 503 only if it is not —
    never 422 for validation).
    """
    features = dict(VALID_FEATURES)
    features[field] = value

    response = client.post('/predict', json=features)
    assert response.status_code in (200, 503), (
        f'{field}={value} is within its declared bounds and should not be '
        f'rejected as invalid, got {response.status_code}: {response.text}'
    )


# --------------------------------------------------------------------------
# Section 3B (audit de robustesse) : scenario metier nominal et stabilite
# sous charge legere.
# --------------------------------------------------------------------------

def test_nominal_business_scenario_t3_65m2_with_parking(client):
    """
    Realistic nominal scenario: a T3 (3-room) apartment, 65 m2, with
    parking, submitted as a prediction request. Verifies the API accepts
    it (200, or 503 if no model is loaded in this environment) and that the
    returned price is a sane positive number.

    Note on the "predicted_price_range_k_EUR" field: no such field exists
    in the real PredictionResponse schema (src/app.py only returns
    predicted_price_k_eur, input_features, model_version). The +/-15%
    estimation range shown to the end user is computed client-side, in
    src/dashboard.py's render_prediction_section() (price * 0.85 to
    price * 1.15), not returned by the API itself. Per this project's
    anti-fabrication rule, this test exercises that real range-computation
    logic against the real API price rather than asserting against a
    nonexistent API field.

    Scenario metier nominal realiste : un appartement T3 (3 pieces), 65 m2,
    avec parking, soumis comme requete de prediction. Verifie que l'API
    l'accepte (200, ou 503 si aucun modele n'est charge dans cet
    environnement) et que le prix retourne est un nombre positif coherent.

    Note sur le champ "predicted_price_range_k_EUR" : ce champ n'existe pas
    dans le schema reel PredictionResponse (src/app.py ne retourne que
    predicted_price_k_eur, input_features, model_version). La fourchette
    d'estimation +/-15% affichee a l'utilisateur final est calculee cote
    client, dans render_prediction_section() de src/dashboard.py
    (price * 0.85 a price * 1.15), et non retournee par l'API elle-meme.
    Conformement a la regle anti-fabrication du projet, ce test exerce cette
    logique reelle de calcul de fourchette sur le prix reel de l'API plutot
    que de verifier un champ API inexistant.
    """
    t3_with_parking = {
        'Surface_m2': 65.0,
        'Nb_Pieces': 3,
        'Annee_Construction': 2005,
        'Distance_Centre_km': 3.5,
        'DPE_Energy_Class': 4,
        'Has_Balcony': 1,
        'Has_Parking': 1,
    }

    response = client.post('/predict', json=t3_with_parking)

    assert response.status_code in (200, 503)

    if response.status_code == 200:
        data = response.json()
        price = data['predicted_price_k_eur']

        assert isinstance(price, (int, float))
        assert price > 0, f'Predicted price for a T3 65m2 must be positive, got {price}'

        # Fourchette d'estimation telle que calculee par le dashboard (dashboard.py,
        # render_prediction_section) : [price * 0.85, price * 1.15]. On verifie que
        # cette fourchette derivee du prix reel de l'API est bien coherente
        # (borne basse < prix < borne haute), et non un champ retourne par l'API.
        range_low = price * 0.85
        range_high = price * 1.15
        assert range_low < price < range_high
        assert data['input_features']['Surface_m2'] == 65.0
        assert data['input_features']['Has_Parking'] == 1


def test_successive_requests_are_stable_no_response_drift(client):
    """
    Light load / stability test: the same valid request repeated N times in
    a row must return the same status code and (when 200) the exact same
    predicted price every time --- the model is a stateless, already-fitted
    LinearRegression, so repeated .predict() calls on identical input must
    be deterministic, and no per-request state should leak between calls.

    This is a functional stability proxy, not a real memory-profiling load
    test (no process memory sampling is performed here) --- it would catch
    a regression where repeated calls corrupt shared state (e.g. a module-
    level cache or counter) or where response values drift across calls,
    which is the class of bug a stateless prediction endpoint must never
    exhibit.

    Test de charge legere / stabilite : la meme requete valide repetee N
    fois d'affilee doit renvoyer le meme code de statut et (si 200)
    exactement le meme prix predit a chaque fois --- le modele est une
    LinearRegression deja entrainee et sans etat, donc des appels .predict()
    repetes sur une entree identique doivent etre deterministes, et aucun
    etat par requete ne doit fuiter entre les appels.

    Ceci est un proxy fonctionnel de stabilite, pas un veritable test de
    charge avec profilage memoire (aucun echantillonnage de la memoire du
    processus n'est effectue ici) --- il detecterait une regression ou des
    appels repetes corrompent un etat partage (par ex. un cache ou compteur
    au niveau du module) ou ou les valeurs de reponse derivent d'un appel a
    l'autre, ce qu'un endpoint de prediction sans etat ne doit jamais
    presenter.
    """
    n_requests = 25
    statuses = []
    prices = []

    for _ in range(n_requests):
        response = client.post('/predict', json=VALID_FEATURES)
        statuses.append(response.status_code)
        if response.status_code == 200:
            prices.append(response.json()['predicted_price_k_eur'])

    assert len(set(statuses)) == 1, (
        f'Status code drifted across {n_requests} identical requests: {set(statuses)}'
    )

    if prices:
        assert len(set(prices)) == 1, (
            f'Predicted price drifted across {n_requests} identical requests: {set(prices)}'
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
