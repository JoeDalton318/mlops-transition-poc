"""
Robustness / failure-mode tests for the FastAPI serving layer (src/app.py).

Complements tests/test_api.py (schema validation, happy-path smoke tests)
with tests that specifically target what happens when the served model
artifact is missing, corrupted, or absent at prediction time --- the
scenarios most likely to occur in production once drift-triggered
retraining and manual redeploys start touching models/housing_model.pkl.

Tests de robustesse / gestion des pannes pour la couche de service FastAPI
(src/app.py). Complète tests/test_api.py (validation de schéma, tests de
fumée du chemin nominal) avec des tests ciblant spécifiquement ce qui se
passe lorsque l'artefact de modèle servi est manquant, corrompu, ou absent
au moment de la prédiction --- les scénarios les plus susceptibles de
survenir en production une fois que le réentraînement déclenché par la
dérive et les redéploiements manuels commencent à modifier
models/housing_model.pkl.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import app  # noqa: E402


VALID_FEATURES = {
    'Surface_m2': 120.5,
    'Nb_Pieces': 3,
    'Annee_Construction': 1995,
    'Distance_Centre_km': 5.2,
    'DPE_Energy_Class': 5,
    'Has_Balcony': 1,
    'Has_Parking': 1,
}


# --------------------------------------------------------------------------
# load_model() in isolation: exercised directly against a monkeypatched
# app.MODEL_PATH, independent of whatever model happens to be loaded into
# the module-level `app.model` global at import time.
# --------------------------------------------------------------------------

def test_load_model_raises_filenotfounderror_when_path_missing(monkeypatch, tmp_path):
    """
    load_model() must raise a FileNotFoundError with an actionable message
    (pointing at src/train.py) when the configured model path does not
    exist on disk, rather than letting joblib raise an unrelated error.

    load_model() doit lever une FileNotFoundError avec un message
    exploitable (pointant vers src/train.py) lorsque le chemin de modèle
    configuré n'existe pas sur le disque, plutôt que de laisser joblib
    lever une erreur non pertinente.
    """
    missing_path = tmp_path / 'does_not_exist' / 'housing_model.pkl'
    monkeypatch.setattr(app, 'MODEL_PATH', missing_path)

    with pytest.raises(FileNotFoundError, match='train the model first'):
        app.load_model()


def test_load_model_raises_on_corrupted_model_file(monkeypatch, tmp_path):
    """
    A model file that exists but contains garbage bytes (e.g. a truncated
    write, a disk-full event during joblib.dump(), or filesystem
    corruption) must cause load_model() to raise rather than returning a
    seemingly-valid object that fails unpredictably later on .predict().

    Un fichier de modèle qui existe mais contient des octets invalides (par
    ex. écriture tronquée, disque plein pendant joblib.dump(), corruption du
    système de fichiers) doit faire lever une exception à load_model()
    plutôt que de renvoyer un objet apparemment valide qui échouerait de
    façon imprévisible plus tard lors de .predict().
    """
    corrupted_path = tmp_path / 'corrupted_model.pkl'
    corrupted_path.write_bytes(b'this is not a valid pickle/joblib artifact \x00\xff\x13')
    monkeypatch.setattr(app, 'MODEL_PATH', corrupted_path)

    with pytest.raises(Exception):
        app.load_model()


def test_load_model_succeeds_on_valid_model_file(tmp_path):
    """
    Sanity check / control case for the two tests above: a genuinely valid
    joblib-serialized model must load back successfully, so the failures
    verified above are attributable to the missing/corrupted file and not
    to a broken load_model() implementation.

    Vérification de contrôle pour les deux tests précédents : un modèle
    réellement valide sérialisé avec joblib doit se recharger avec succès,
    afin que les échecs vérifiés ci-dessus soient attribuables au fichier
    manquant/corrompu et non à une implémentation défaillante de
    load_model().
    """
    import joblib
    from sklearn.linear_model import LinearRegression

    model = LinearRegression().fit([[1], [2], [3]], [1, 2, 3])
    model_path = tmp_path / 'valid_model.pkl'
    joblib.dump(model, model_path)

    import app as app_module
    original_path = app_module.MODEL_PATH
    app_module.MODEL_PATH = model_path
    try:
        loaded = app_module.load_model()
    finally:
        app_module.MODEL_PATH = original_path

    assert hasattr(loaded, 'predict')


# --------------------------------------------------------------------------
# /predict endpoint behavior when the module-level `model` global is None
# (mirrors what happens at real startup when models/housing_model.pkl is
# absent --- see the try/except FileNotFoundError around load_model() at
# module import time in src/app.py).
# --------------------------------------------------------------------------

@pytest.fixture
def client():
    """
    Provide a TestClient bound to the real FastAPI app object.
    Fournit un TestClient lié à l'objet application FastAPI réel.
    """
    return TestClient(app.app)


def test_predict_returns_503_when_model_not_loaded(monkeypatch, client):
    """
    With app.model forced to None (simulating a startup where
    models/housing_model.pkl was absent), /predict must respond 503 with an
    explicit "train the model first" message, not a 500 crash or a silent
    wrong prediction.

    Avec app.model forcé à None (simulant un démarrage où
    models/housing_model.pkl était absent), /predict doit répondre 503 avec
    un message explicite "entraîner le modèle d'abord", et non un plantage
    500 ou une prédiction erronée silencieuse.
    """
    monkeypatch.setattr(app, 'model', None)

    response = client.post('/predict', json=VALID_FEATURES)

    assert response.status_code == 503
    assert 'not available' in response.json()['detail'].lower()


def test_health_check_reports_not_loaded_when_model_is_none(monkeypatch, client):
    """/health must reflect model_status='not_loaded' when app.model is None."""
    monkeypatch.setattr(app, 'model', None)

    response = client.get('/health')

    assert response.status_code == 200
    assert response.json()['model_status'] == 'not_loaded'


def test_predict_succeeds_when_a_valid_model_is_injected(monkeypatch, client):
    """
    Control case: with a genuinely fitted model injected into app.model,
    /predict must return 200 with a numeric predicted_price_k_eur --- this
    confirms the 503 in the tests above is really caused by model is None,
    not by an unrelated regression in the endpoint.

    Cas de contrôle : avec un modèle réellement entraîné injecté dans
    app.model, /predict doit renvoyer 200 avec un predicted_price_k_eur
    numérique --- ceci confirme que le 503 des tests précédents est bien dû
    à model is None, et non à une régression indépendante du endpoint.
    """
    from sklearn.linear_model import LinearRegression
    import pandas as pd

    X = pd.DataFrame({
        'Surface_m2': [50, 100, 150],
        'Nb_Pieces': [2, 3, 4],
        'Annee_Construction': [1990, 2000, 2010],
        'Distance_Centre_km': [10, 5, 1],
        'DPE_Energy_Class': [3, 4, 5],
        'Has_Balcony': [0, 1, 1],
        'Has_Parking': [0, 1, 1],
    })
    y = [100, 200, 300]
    fake_model = LinearRegression().fit(X, y)
    monkeypatch.setattr(app, 'model', fake_model)

    response = client.post('/predict', json=VALID_FEATURES)

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data['predicted_price_k_eur'], (int, float))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
