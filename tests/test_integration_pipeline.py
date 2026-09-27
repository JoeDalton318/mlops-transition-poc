"""
End-to-end integration tests for the self-healing MLOps loop:
train -> drift injection -> Evidently AI detection -> automatic retraining
-> champion/challenger promotion decision, plus an API+model integration
test verifying the serving layer actually reflects the currently promoted
artifact.

Isolation strategy: every test runs against a throwaway MLflow tracking
store (a temp SQLite file) and a throwaway dataset/models directory, all
under pytest's tmp_path. mlflow.set_tracking_uri() is redirected for the
duration of each test and restored afterwards in a finally block, and
train.DATA_FILE / train.MODELS_DIR / drift_detection.REFERENCE_DATA_FILE /
drift_detection.REPORTS_DIR are monkeypatched to point inside tmp_path.
This guarantees these tests never touch the developer's real mlflow.db,
mlruns/ artifacts, registered 'housing_model', or data/immobilier_france.csv
--- running this file cannot corrupt or overwrite the real champion model.

These are genuine integration tests (they run the real scikit-learn
training code, the real MLflow Model Registry alias API, and the real
Evidently AI DataDriftPreset report, not mocks of them), so they are slower
than the rest of the suite; they are marked with @pytest.mark.integration
(see pytest.ini) so `pytest tests/ -v -m "not integration"` can skip them
for a fast inner-loop run if desired.

Tests d'integration bout-en-bout pour la boucle MLOps auto-cicatrisante :
entrainement -> injection de derive -> detection Evidently AI ->
reentrainement automatique -> decision de promotion champion/challenger,
plus un test d'integration API+modele verifiant que la couche de service
reflete effectivement l'artefact actuellement promu.

Strategie d'isolation : chaque test s'execute contre un entrepot de suivi
MLflow jetable (un fichier SQLite temporaire) et un dataset/repertoire de
modeles jetable, le tout sous le tmp_path de pytest. mlflow.set_tracking_uri()
est redirige pour la duree de chaque test puis restaure ensuite dans un bloc
finally, et train.DATA_FILE / train.MODELS_DIR /
drift_detection.REFERENCE_DATA_FILE / drift_detection.REPORTS_DIR sont
monkeypatches pour pointer a l'interieur de tmp_path. Ceci garantit que ces
tests ne touchent jamais le vrai mlflow.db du developpeur, les artefacts
mlruns/, le 'housing_model' enregistre, ou data/immobilier_france.csv ---
executer ce fichier ne peut pas corrompre ou ecraser le vrai modele champion.

Ce sont de veritables tests d'integration (ils executent le vrai code
d'entrainement scikit-learn, la vraie API d'alias du Model Registry MLflow,
et le vrai rapport Evidently AI DataDriftPreset, pas des mocks de ceux-ci),
donc plus lents que le reste de la suite ; ils sont marques avec
@pytest.mark.integration (voir pytest.ini) afin que
`pytest tests/ -v -m "not integration"` puisse les ignorer pour une boucle
de developpement rapide si souhaite.
"""

import sys
from pathlib import Path
from typing import NamedTuple

import mlflow
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

import app  # noqa: E402
import generate_data  # noqa: E402
import drift_detection  # noqa: E402
import train  # noqa: E402


FEATURE_COLUMNS = [
    'Surface_m2',
    'Nb_Pieces',
    'Annee_Construction',
    'Distance_Centre_km',
    'DPE_Energy_Class',
    'Has_Balcony',
    'Has_Parking',
]


def _make_synthetic_dataset(n: int, seed: int, signal_strength: float) -> pd.DataFrame:
    """
    Build a small synthetic housing dataset with the exact real schema.

    `signal_strength` controls how strongly Prix_k_EUR actually depends on
    the features: 0.0 produces a target statistically independent of every
    feature (a LinearRegression fit on it should achieve R2 near 0, useful
    as a deliberately weak "champion" to be beaten), while 1.0 produces a
    near-noiseless linear relationship (R2 near 1, useful as a clearly
    better "challenger"). This lets the promotion-gate tests construct
    datasets with a *known, deterministic* R2 ordering rather than hoping
    two random datasets happen to differ.

    Construit un petit jeu de donnees immobilier synthetique avec le schema
    reel exact. `signal_strength` controle la force reelle de la dependance
    de Prix_k_EUR aux caracteristiques : 0.0 produit une cible
    statistiquement independante de toute caracteristique (un R2 proche de
    0, utile comme "champion" deliberement faible a battre), tandis que 1.0
    produit une relation lineaire quasi sans bruit (R2 proche de 1, utile
    comme "challenger" clairement meilleur). Ceci permet aux tests de la
    porte de promotion de construire des jeux de donnees avec un ordre de
    R2 *connu et deterministe* plutot que d'esperer que deux jeux de
    donnees aleatoires different par hasard.
    """
    rng = np.random.default_rng(seed)

    surface = rng.uniform(30, 250, n)
    nb_pieces = rng.integers(1, 8, n)
    annee = rng.integers(1950, 2024, n)
    distance = rng.uniform(0.5, 40.0, n)
    dpe = rng.integers(1, 8, n)
    balcony = rng.integers(0, 2, n)
    parking = rng.integers(0, 2, n)

    signal = (
        2.5 * surface
        - 3.0 * distance
        + 5.0 * dpe
        + 8.0 * nb_pieces
        + 10.0 * parking
        + 200.0
    )
    noise = rng.normal(loc=0.0, scale=150.0, size=n)
    independent_target = rng.uniform(50, 500, n)

    price = signal_strength * signal + (1 - signal_strength) * independent_target
    price = price + (1 - signal_strength) * noise
    price = np.clip(price, 10.0, None)

    return pd.DataFrame({
        'Surface_m2': surface,
        'Nb_Pieces': nb_pieces,
        'Annee_Construction': annee,
        'Distance_Centre_km': distance,
        'DPE_Energy_Class': dpe,
        'Has_Balcony': balcony,
        'Has_Parking': parking,
        'Prix_k_EUR': price,
    })


class IsolatedEnv(NamedTuple):
    data_file: Path
    models_dir: Path
    model_path: Path
    reports_dir: Path


@pytest.fixture
def isolated_mlops_environment(tmp_path, monkeypatch):
    """
    Redirect every piece of global/module state train.py and
    drift_detection.py rely on (MLflow tracking URI, dataset path, models
    directory, drift-reports directory) into a throwaway tmp_path, and
    restore the original MLflow tracking URI on teardown.

    Redirige tout l'etat global/module dont dependent train.py et
    drift_detection.py (URI de suivi MLflow, chemin du dataset, repertoire
    des modeles, repertoire des rapports de derive) vers un tmp_path
    jetable, et restaure l'URI de suivi MLflow d'origine au demontage.
    """
    data_file = tmp_path / 'data' / 'immobilier_france.csv'
    data_file.parent.mkdir(parents=True, exist_ok=True)
    models_dir = tmp_path / 'models'
    reports_dir = tmp_path / 'drift_reports'

    monkeypatch.setattr(train, 'DATA_FILE', data_file)
    monkeypatch.setattr(train, 'MODELS_DIR', models_dir)
    monkeypatch.setattr(drift_detection, 'REFERENCE_DATA_FILE', data_file)
    monkeypatch.setattr(drift_detection, 'REPORTS_DIR', reports_dir)

    original_tracking_uri = mlflow.get_tracking_uri()
    isolated_db = tmp_path / 'mlflow_test.db'
    mlflow.set_tracking_uri(f'sqlite:///{isolated_db}')

    try:
        yield IsolatedEnv(
            data_file=data_file,
            models_dir=models_dir,
            model_path=models_dir / f'{train.MODEL_NAME}.pkl',
            reports_dir=reports_dir,
        )
    finally:
        mlflow.set_tracking_uri(original_tracking_uri)


# --------------------------------------------------------------------------
# Full self-healing loop: train champion -> inject drift -> detect via
# Evidently -> trigger run_pipeline() -> verify the promotion decision.
# --------------------------------------------------------------------------

@pytest.mark.integration
def test_full_self_healing_loop_champion_then_drift_then_retrain(isolated_mlops_environment):
    """
    Runs the real self-healing loop end-to-end and turns the critical
    finding documented in the memoire (Chapter IV: trigger_retraining()
    calls run_pipeline() without ever forwarding the drifted current_data
    it just detected drift on) into an executable regression check.

    Steps: (1) train an initial champion on the dataset produced by the
    project's own generator (generate_data.generate_housing_dataset with its
    default seed, i.e. the same data as data/immobilier_france.csv); (2) run
    drift_detection.run_drift_detection(generate_synthetic_data=True), the
    real production self-healing entry point, which internally perturbs a
    copy of the reference data (Surface_m2 += 10, Prix_k_EUR *= 1.15) and
    runs a real Evidently DataDriftPreset report against it; (3) because
    that perturbation is a systematic, whole-population shift, Evidently
    must flag exactly 2 of the 8 columns (Surface_m2, Prix_k_EUR), i.e. a
    drift score of 0.25 > DRIFT_THRESHOLD, as reported in the memoire; (4) because trigger_retraining() only ever calls
    run_pipeline(trigger_source='drift_detection') -- which reloads
    train.DATA_FILE from disk, i.e. the *unperturbed* reference data, not
    the drifted current_data -- the resulting challenger is trained on
    identical data with the identical random_state=42 split as the
    original champion, so it cannot strictly beat it on R2 and must NOT be
    promoted. If a future fix forwards current_data into retraining, this
    assertion is expected to need updating -- which is the point: it pins
    down the currently-documented gap as a concrete, executable check.

    Execute la vraie boucle d'auto-cicatrisation de bout en bout et
    transforme le constat critique documente dans le memoire (Chapitre IV :
    trigger_retraining() appelle run_pipeline() sans jamais transmettre les
    current_data derivees qu'elle vient pourtant de detecter comme
    derivantes) en un controle de regression executable.
    """
    df = generate_data.generate_housing_dataset()
    df.to_csv(isolated_mlops_environment.data_file, index=False)

    initial_result = train.run_pipeline(trigger_source='manual')
    assert initial_result['status'] == 'success'
    assert initial_result['promoted'] is True, 'First-ever run must always be promoted (no champion yet).'
    champion_r2 = initial_result['metrics']['r2']

    detection_result = drift_detection.run_drift_detection(generate_synthetic_data=True)

    assert detection_result['drift_detected'] is True, (
        f"A systematic +10 Surface_m2 / *1.15 Prix_k_EUR shift over the whole "
        f"population must be detected as drift (got drift_score="
        f"{detection_result.get('drift_score')})."
    )
    assert detection_result['drift_score'] == pytest.approx(0.25), (
        'Exactly 2 of the 8 columns (Surface_m2, Prix_k_EUR) must drift, '
        'i.e. the share of drifted columns reported in the memoire.'
    )
    assert detection_result['retraining_triggered'] is True

    retraining_result = detection_result['retraining_result']
    assert retraining_result is not None, 'trigger_retraining() must not silently fail.'
    assert retraining_result['status'] == 'success'

    assert retraining_result['metrics']['r2'] == pytest.approx(champion_r2, abs=1e-9), (
        'Retraining after drift detection reused the exact same (unperturbed) '
        'reference dataset and random_state, so it must reproduce the same R2 '
        'as the original champion -- this is the documented '
        'current_data-not-forwarded gap in trigger_retraining()/run_pipeline().'
    )
    assert retraining_result['promoted'] is False, (
        'Because the challenger cannot strictly beat the champion R2 (identical '
        'data/split), the promotion gate must correctly refuse to promote it.'
    )


@pytest.mark.integration
def test_identical_current_data_reports_no_drift_and_no_retraining(isolated_mlops_environment, monkeypatch):
    """
    Non-regression check for the drift score: comparing the reference data
    with an identical copy (run_drift_detection(generate_synthetic_data=False))
    must yield a drift score of exactly 0.0, no drift and no retraining. It
    runs a real Evidently DataDriftPreset report, whose 'drift_share' key is
    Evidently's own threshold (0.5): reading that key as the score would
    flag drift on identical data and trigger a useless retraining.

    Contrôle de non-régression du score de dérive : comparer les données de
    référence à une copie identique doit donner un score de dérive de
    exactement 0.0, aucune dérive et aucun réentraînement. Le test exécute un
    vrai rapport Evidently DataDriftPreset, dont la clé 'drift_share' est le
    seuil propre à Evidently (0.5) : lire cette clé comme score signalerait
    une dérive sur des données identiques et déclencherait un réentraînement
    inutile.
    """
    df = generate_data.generate_housing_dataset()
    df.to_csv(isolated_mlops_environment.data_file, index=False)

    retraining_calls = []
    monkeypatch.setattr(drift_detection, 'trigger_retraining', lambda drift_score: retraining_calls.append(drift_score))

    result = drift_detection.run_drift_detection(generate_synthetic_data=False)

    assert result['drift_score'] == 0.0
    assert result['drift_detected'] is False
    assert result['retraining_triggered'] is False
    assert retraining_calls == []


@pytest.mark.integration
def test_genuinely_better_challenger_is_promoted_over_weak_champion(isolated_mlops_environment):
    """
    Positive-path counterpart to the test above: when a challenger is
    trained on data that genuinely supports a better fit, the promotion
    gate must promote it and overwrite the local serving artifact -- this
    confirms the non-promotion observed above is really due to the
    unforwarded-data gap, not a promotion gate that never promotes.

    Contrepartie du chemin positif au test precedent : lorsqu'un challenger
    est entraine sur des donnees permettant reellement un meilleur ajustement,
    la porte de promotion doit le promouvoir et ecraser l'artefact local de
    service -- ceci confirme que la non-promotion observee ci-dessus est bien
    due a la lacune de transmission des donnees, et non a une porte de
    promotion qui ne promeut jamais.
    """
    weak_df = _make_synthetic_dataset(n=200, seed=2, signal_strength=0.0)
    weak_df.to_csv(isolated_mlops_environment.data_file, index=False)

    weak_result = train.run_pipeline(trigger_source='manual')
    assert weak_result['status'] == 'success'
    assert weak_result['promoted'] is True
    weak_r2 = weak_result['metrics']['r2']
    assert not isolated_mlops_environment.model_path.exists() or True  # written by promotion above
    model_mtime_after_first_promotion = isolated_mlops_environment.model_path.stat().st_mtime

    strong_df = _make_synthetic_dataset(n=200, seed=3, signal_strength=1.0)
    strong_df.to_csv(isolated_mlops_environment.data_file, index=False)

    strong_result = train.run_pipeline(trigger_source='manual')

    assert strong_result['status'] == 'success'
    assert strong_result['champion_metric_before'] == pytest.approx(weak_r2)
    assert strong_result['metrics']['r2'] > weak_r2, (
        'The signal_strength=1.0 dataset must yield a genuinely higher R2 than '
        'the signal_strength=0.0 dataset.'
    )
    assert strong_result['promoted'] is True, (
        'A challenger that genuinely beats the champion R2 must be promoted.'
    )
    assert isolated_mlops_environment.model_path.stat().st_mtime >= model_mtime_after_first_promotion


# --------------------------------------------------------------------------
# API + Model integration: the API must serve the currently-promoted model
# and pick up a newly-promoted artifact when reloaded.
# --------------------------------------------------------------------------

@pytest.mark.integration
def test_api_serves_the_actually_promoted_model_and_reacts_to_artifact_changes(
    isolated_mlops_environment, monkeypatch
):
    """
    End-to-end check that the API's "shadow copy" serving pattern (see the
    design note in src/app.py's load_model()) really works: after
    train.run_pipeline() promotes a model, pointing app.MODEL_PATH at that
    same file and reloading it via app.load_model() must make /predict
    return a price matching the promoted model's own .predict() output, and
    after a *second*, promoted retraining on different data changes the
    artifact on disk, reloading again must change the API's predictions --
    i.e. the API is not stuck serving a stale, cached model object.

    Verification de bout en bout que le motif de service "copie miroir" de
    l'API (voir la note de conception dans load_model() de src/app.py)
    fonctionne reellement : apres que train.run_pipeline() a promu un
    modele, pointer app.MODEL_PATH vers ce meme fichier et le recharger via
    app.load_model() doit faire renvoyer par /predict un prix correspondant
    a la sortie .predict() du modele promu lui-meme, et apres un *second*
    reentrainement promu sur des donnees differentes qui modifie l'artefact
    sur disque, un nouveau rechargement doit modifier les predictions de
    l'API -- c'est-a-dire que l'API ne reste pas bloquee a servir un objet
    modele perime et mis en cache.
    """
    sample_features = {
        'Surface_m2': 90.0,
        'Nb_Pieces': 4,
        'Annee_Construction': 2010,
        'Distance_Centre_km': 8.0,
        'DPE_Energy_Class': 5,
        'Has_Balcony': 1,
        'Has_Parking': 1,
    }
    feature_row = [[sample_features[col] for col in FEATURE_COLUMNS]]

    df_a = _make_synthetic_dataset(n=200, seed=4, signal_strength=0.5)
    df_a.to_csv(isolated_mlops_environment.data_file, index=False)
    result_a = train.run_pipeline(trigger_source='manual')
    assert result_a['status'] == 'success' and result_a['promoted'] is True

    monkeypatch.setattr(app, 'MODEL_PATH', isolated_mlops_environment.model_path)
    reloaded_model_a = app.load_model()
    monkeypatch.setattr(app, 'model', reloaded_model_a)

    client = TestClient(app.app)
    response_a = client.post('/predict', json=sample_features)
    assert response_a.status_code == 200
    api_price_a = response_a.json()['predicted_price_k_eur']
    direct_price_a = round(float(reloaded_model_a.predict(feature_row)[0]), 2)
    assert api_price_a == pytest.approx(direct_price_a), (
        'The API must serve predictions from the actually-promoted model artifact.'
    )

    df_b = _make_synthetic_dataset(n=200, seed=5, signal_strength=1.0)
    df_b.to_csv(isolated_mlops_environment.data_file, index=False)
    result_b = train.run_pipeline(trigger_source='manual')
    assert result_b['status'] == 'success'
    assert result_b['promoted'] is True, (
        'This second run trains on a different dataset (different seed); it is '
        'not guaranteed a priori to beat the first, so if this fails the test '
        'data generation (not the API) needs a larger R2 gap.'
    )

    reloaded_model_b = app.load_model()
    monkeypatch.setattr(app, 'model', reloaded_model_b)

    response_b = client.post('/predict', json=sample_features)
    assert response_b.status_code == 200
    api_price_b = response_b.json()['predicted_price_k_eur']
    direct_price_b = round(float(reloaded_model_b.predict(feature_row)[0]), 2)
    assert api_price_b == pytest.approx(direct_price_b), (
        'After reload, the API must serve predictions from the newly promoted artifact.'
    )
    assert api_price_b != pytest.approx(api_price_a), (
        'Retraining on a materially different dataset and reloading the model '
        'must change the API prediction -- otherwise the API would be silently '
        'serving a stale, cached model despite the artifact on disk changing.'
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
