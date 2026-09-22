"""
Unit tests for the training pipeline (src/train.py).

Covers data loading/schema, model training/evaluation, and the
champion/challenger promotion gate in isolation (via a lightweight fake
MLflow client, dependency-injected — no live MLflow tracking server needed).

Tests unitaires pour le pipeline d'entraînement (src/train.py).

Couvre le chargement/schéma des données, l'entraînement/évaluation du modèle,
et la porte de promotion champion/challenger de façon isolée (via un client
MLflow factice injecté en dépendance — aucun serveur de tracking MLflow réel
n'est nécessaire).
"""

import contextlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

# Add src directory to path for imports (mirrors tests/test_api.py)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import train  # noqa: E402


EXPECTED_FEATURE_COLUMNS = [
    'Surface_m2',
    'Nb_Pieces',
    'Annee_Construction',
    'Distance_Centre_km',
    'DPE_Energy_Class',
    'Has_Balcony',
    'Has_Parking',
]


def test_load_and_prepare_data_schema():
    """
    The loaded dataset must expose exactly the 7 expected feature columns
    (matching the Pydantic schema in src/app.py) and a numeric target with
    no missing values, since a mismatch here would silently break both
    training and the API's prediction contract.
    """
    X, y = train.load_and_prepare_data()

    assert list(X.columns) == EXPECTED_FEATURE_COLUMNS
    assert len(X) == len(y)
    assert len(X) > 0
    assert X.isnull().sum().sum() == 0
    assert y.isnull().sum() == 0
    assert y.name == 'Prix_k_EUR'
    assert pd.api.types.is_numeric_dtype(y)


def test_train_model_returns_fitted_linear_regression():
    """A trained model must be a fitted LinearRegression exposing coef_/intercept_."""
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame({
        'a': rng.normal(size=50),
        'b': rng.normal(size=50),
    })
    y_train = 2 * X_train['a'] - 3 * X_train['b'] + 1

    model = train.train_model(X_train, y_train)

    assert isinstance(model, LinearRegression)
    assert hasattr(model, 'coef_')
    assert len(model.coef_) == 2


def test_evaluate_model_returns_expected_metric_keys():
    """evaluate_model() must return exactly {mse, rmse, mae, r2}, with rmse == sqrt(mse)."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame({'x': rng.normal(size=100)})
    y = 5 * X['x'] + 2 + rng.normal(scale=0.01, size=100)

    model = LinearRegression().fit(X, y)
    metrics = train.evaluate_model(model, X, y)

    assert set(metrics.keys()) == {'mse', 'rmse', 'mae', 'r2'}
    assert metrics['rmse'] == pytest.approx(np.sqrt(metrics['mse']), rel=1e-6)
    assert metrics['r2'] <= 1.0 + 1e-9


def test_evaluate_model_near_perfect_fit_on_noiseless_linear_data():
    """
    With a noiseless, perfectly linear relationship, a LinearRegression must
    achieve R2 very close to 1 on a held-out split — this is a sanity check
    on evaluate_model()'s correctness, independent of the real dataset.
    """
    rng = np.random.default_rng(2)
    n = 200
    X = pd.DataFrame({
        'x1': rng.normal(size=n),
        'x2': rng.normal(size=n),
    })
    y = 10 + 3 * X['x1'] - 2 * X['x2']

    X_train, y_train = X.iloc[:150], y.iloc[:150]
    X_test, y_test = X.iloc[150:], y.iloc[150:]

    model = train.train_model(X_train, y_train)
    metrics = train.evaluate_model(model, X_test, y_test)

    assert metrics['r2'] > 0.999


def test_real_dataset_baseline_r2_above_threshold():
    """
    Regression guard: the real dataset (data/immobilier_france.csv) must
    yield R2 > 0.85 with the project's train/test split configuration, per
    the project's data-quality requirement. This is the exact check that
    would have caught the original price-formula saturation bug (77.8% of
    targets clipped at the same value), which produced R2 ~= 0.42.
    """
    from sklearn.model_selection import train_test_split

    X, y = train.load_and_prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=train.RANDOM_STATE
    )
    model = train.train_model(X_train, y_train)
    metrics = train.evaluate_model(model, X_test, y_test)

    assert metrics['r2'] > 0.85, (
        f"R2={metrics['r2']:.4f} is below the 0.85 data-quality threshold; "
        "check data/immobilier_france.csv for saturation/clipping artifacts."
    )


# --------------------------------------------------------------------------
# Champion/challenger promotion gate — tested via a fake MLflow client
# (dependency-injected into promote_challenger_if_better), so this does not
# require a live MLflow tracking server.
# --------------------------------------------------------------------------

class MlflowExceptionLike(Exception):
    """Stand-in for mlflow.exceptions.MlflowException in these unit tests."""


@dataclass
class _FakeModelVersion:
    name: str
    version: str
    run_id: str


@dataclass
class _FakeRun:
    metrics: Dict[str, float] = field(default_factory=dict)


class FakeMlflowClient:
    """
    Minimal stand-in for mlflow.tracking.MlflowClient exposing only the
    methods promote_challenger_if_better() actually calls.
    """

    def __init__(self, champion_version: Optional[str] = None, champion_metrics: Optional[Dict[str, float]] = None):
        self._aliases: Dict[str, str] = {}
        self._runs: Dict[str, _FakeRun] = {}
        self._versions: List[_FakeModelVersion] = []
        if champion_version is not None:
            self._aliases['champion'] = champion_version
            self._runs[f'run-for-{champion_version}'] = _FakeRun(metrics=champion_metrics or {})
            self._versions.append(_FakeModelVersion(name=train.MODEL_NAME, version=champion_version, run_id=f'run-for-{champion_version}'))

    def register_challenger_run(self, run_id: str, version: str):
        self._versions.append(_FakeModelVersion(name=train.MODEL_NAME, version=version, run_id=run_id))

    def get_model_version_by_alias(self, name, alias):
        if alias not in self._aliases:
            raise MlflowExceptionLike('alias not found')
        version = self._aliases[alias]
        return next(v for v in self._versions if v.name == name and v.version == version)

    def get_run(self, run_id):
        return type('R', (), {'data': type('D', (), {'metrics': self._runs.get(run_id, _FakeRun()).metrics})()})()

    def search_model_versions(self, filter_str):
        run_id = filter_str.split("'")[1]
        return [v for v in self._versions if v.run_id == run_id]

    def set_registered_model_alias(self, name, alias, version):
        self._aliases[alias] = version


@pytest.fixture(autouse=True)
def _patch_mlflow_exception(monkeypatch):
    """Make train.py's except MlflowException clauses catch our fake exception too."""
    monkeypatch.setattr(train, 'MlflowException', MlflowExceptionLike)


def test_promotion_first_run_always_promotes(tmp_path):
    """With no existing champion, the first trained model must always be promoted."""
    client = FakeMlflowClient(champion_version=None)
    client.register_challenger_run('run-1', '1')
    model = LinearRegression().fit([[1], [2]], [1, 2])
    model_path = tmp_path / 'model.pkl'

    result = train.promote_challenger_if_better(
        client, 'run-1', {'r2': 0.5}, model_path, model
    )

    assert result['promoted'] is True
    assert result['champion_metric_before'] is None
    assert model_path.exists()


def test_promotion_better_challenger_is_promoted(tmp_path):
    """A challenger strictly beating the champion's R2 must be promoted."""
    client = FakeMlflowClient(champion_version='1', champion_metrics={'r2': 0.80})
    client.register_challenger_run('run-2', '2')
    model = LinearRegression().fit([[1], [2]], [1, 2])
    model_path = tmp_path / 'model.pkl'

    result = train.promote_challenger_if_better(
        client, 'run-2', {'r2': 0.90}, model_path, model
    )

    assert result['promoted'] is True
    assert result['champion_metric_before'] == pytest.approx(0.80)
    assert model_path.exists()


def test_promotion_worse_or_equal_challenger_is_not_promoted(tmp_path):
    """
    A challenger that does not strictly beat the champion (equal or worse R2)
    must not be promoted, and the serving file must not be (re)written.
    """
    client = FakeMlflowClient(champion_version='1', champion_metrics={'r2': 0.90})
    client.register_challenger_run('run-2', '2')
    model = LinearRegression().fit([[1], [2]], [1, 2])
    model_path = tmp_path / 'model.pkl'

    result_equal = train.promote_challenger_if_better(
        client, 'run-2', {'r2': 0.90}, model_path, model
    )
    result_worse = train.promote_challenger_if_better(
        client, 'run-2', {'r2': 0.10}, model_path, model
    )

    assert result_equal['promoted'] is False
    assert result_worse['promoted'] is False
    assert not model_path.exists()


def test_run_pipeline_logs_governance_tags(monkeypatch, tmp_path):
    """
    run_pipeline() must record the four governance/audit tags on every run
    (model_type, dataset_name, pipeline_stage, environment), with values
    derived from the trigger source and from the promotion decision: a first
    manual run is promoted ('initial_training', 'production'), while a
    drift-triggered run that does not beat the champion stays in 'staging'
    ('continuous_training'). MLflow calls are replaced by in-memory fakes, so
    no tracking server or database is needed.

    run_pipeline() doit enregistrer les quatre étiquettes de gouvernance et
    d'audit à chaque exécution (model_type, dataset_name, pipeline_stage,
    environment), avec des valeurs dérivées de la source du déclenchement et
    de la décision de promotion : un premier entraînement manuel est promu
    ('initial_training', 'production'), tandis qu'un réentraînement déclenché
    par la dérive qui ne bat pas le champion reste en 'staging'
    ('continuous_training'). Les appels MLflow sont remplacés par des doubles
    en mémoire : aucun serveur de suivi ni base de données n'est nécessaire.
    """
    rng = np.random.default_rng(3)
    n = 40
    data = pd.DataFrame({
        'Surface_m2': rng.uniform(30, 200, n),
        'Nb_Pieces': rng.integers(1, 6, n),
        'Annee_Construction': rng.integers(1950, 2024, n),
        'Distance_Centre_km': rng.uniform(0.5, 30, n),
        'DPE_Energy_Class': rng.integers(1, 8, n),
        'Has_Balcony': rng.integers(0, 2, n),
        'Has_Parking': rng.integers(0, 2, n),
    })
    data['Prix_k_EUR'] = 3 * data['Surface_m2'] - 2 * data['Distance_Centre_km'] + 50
    data_file = tmp_path / 'immobilier_france.csv'
    data.to_csv(data_file, index=False)
    monkeypatch.setattr(train, 'DATA_FILE', data_file)
    monkeypatch.setattr(train, 'MODELS_DIR', tmp_path / 'models')

    tags: Dict[str, str] = {}
    active_run_id = {'value': 'run-1'}
    monkeypatch.setattr(train.mlflow, 'set_experiment', lambda name: None)
    monkeypatch.setattr(train.mlflow, 'start_run', lambda **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(train.mlflow, 'log_param', lambda key, value: None)
    monkeypatch.setattr(train.mlflow, 'log_metric', lambda key, value: None)
    monkeypatch.setattr(train.mlflow, 'log_artifact', lambda *args, **kwargs: None)
    monkeypatch.setattr(train.mlflow, 'set_tag', lambda key, value: tags.__setitem__(key, value))
    monkeypatch.setattr(train.mlflow.sklearn, 'log_model', lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train.mlflow, 'active_run',
        lambda: SimpleNamespace(info=SimpleNamespace(run_id=active_run_id['value'])),
    )

    # 1) Premier entraînement manuel, aucun champion : promu (First manual run, no champion: promoted)
    first_client = FakeMlflowClient(champion_version=None)
    first_client.register_challenger_run('run-1', '1')
    monkeypatch.setattr(train, 'MlflowClient', lambda: first_client)

    first = train.run_pipeline(trigger_source='manual')

    assert first['status'] == 'success', first['message']
    assert first['promoted'] is True
    assert tags == {
        'model_type': 'LinearRegression',
        'dataset_name': 'immobilier_france.csv',
        'pipeline_stage': 'initial_training',
        'environment': 'production',
    }

    # 2) Réentraînement par dérive face à un champion imbattable : non promu
    #    (Drift-triggered retraining against an unbeatable champion: not promoted)
    tags.clear()
    active_run_id['value'] = 'run-2'
    drift_client = FakeMlflowClient(champion_version='1', champion_metrics={'r2': 1.0})
    drift_client.register_challenger_run('run-2', '2')
    monkeypatch.setattr(train, 'MlflowClient', lambda: drift_client)

    retrained = train.run_pipeline(trigger_source='drift_detection')

    assert retrained['status'] == 'success', retrained['message']
    assert retrained['promoted'] is False
    assert tags == {
        'model_type': 'LinearRegression',
        'dataset_name': 'immobilier_france.csv',
        'pipeline_stage': 'continuous_training',
        'environment': 'staging',
    }


# --------------------------------------------------------------------------
# Section 3A (audit de robustesse) : cohérence économique des coefficients,
# monotonicité, reproductibilité stricte, et gestion des jeux de données
# vides/manquants.
# --------------------------------------------------------------------------

def test_model_coefficient_signs_match_real_estate_logic():
    """
    Economic sanity check on the real dataset: a linear regression trained
    on data/immobilier_france.csv must learn a strictly positive coefficient
    for Surface_m2 (larger living area => higher price) and a strictly
    negative coefficient for Distance_Centre_km (farther from the city
    center => lower price). A model violating either sign would be
    economically nonsensical even if its R2 looked acceptable.

    Vérification de cohérence économique sur le jeu de données réel : une
    régression linéaire entraînée sur data/immobilier_france.csv doit
    apprendre un coefficient strictement positif pour Surface_m2 (plus de
    surface habitable => prix plus élevé) et strictement négatif pour
    Distance_Centre_km (plus loin du centre-ville => prix plus bas). Un
    modèle violant l'un de ces signes serait économiquement incohérent même
    avec un R2 apparemment correct.
    """
    X, y = train.load_and_prepare_data()
    model = train.train_model(X, y)

    surface_idx = EXPECTED_FEATURE_COLUMNS.index('Surface_m2')
    distance_idx = EXPECTED_FEATURE_COLUMNS.index('Distance_Centre_km')

    assert model.coef_[surface_idx] > 0, (
        f"Surface_m2 coefficient should be > 0, got {model.coef_[surface_idx]:.4f}"
    )
    assert model.coef_[distance_idx] < 0, (
        f"Distance_Centre_km coefficient should be < 0, got {model.coef_[distance_idx]:.4f}"
    )


def test_monotonicity_increasing_surface_increases_predicted_price():
    """
    Sensitivity/monotonicity test: holding every other feature fixed,
    increasing Surface_m2 must strictly increase the predicted price. This
    is the behavioral counterpart to the coefficient-sign test above --- it
    checks the effect at prediction time rather than inspecting coef_
    directly, so it would also catch a bug in how features are ordered
    when building the prediction input array (see src/app.py's
    feature_array construction).

    Test de sensibilité/monotonicité : à caractéristiques égales par
    ailleurs, augmenter Surface_m2 doit strictement augmenter le prix
    prédit. C'est le pendant comportemental du test précédent sur le signe
    du coefficient --- il vérifie l'effet au moment de la prédiction plutôt
    que d'inspecter coef_ directement, ce qui détecterait aussi un bug dans
    l'ordre des caractéristiques lors de la construction du tableau
    d'entrée de prédiction (voir la construction de feature_array dans
    src/app.py).
    """
    X, y = train.load_and_prepare_data()
    model = train.train_model(X, y)

    base_row = {
        'Surface_m2': 60.0,
        'Nb_Pieces': 3,
        'Annee_Construction': 2000,
        'Distance_Centre_km': 5.0,
        'DPE_Energy_Class': 4,
        'Has_Balcony': 1,
        'Has_Parking': 1,
    }
    larger_row = dict(base_row, Surface_m2=120.0)

    X_base = pd.DataFrame([base_row])[EXPECTED_FEATURE_COLUMNS]
    X_larger = pd.DataFrame([larger_row])[EXPECTED_FEATURE_COLUMNS]

    price_base = model.predict(X_base)[0]
    price_larger = model.predict(X_larger)[0]

    assert price_larger > price_base, (
        f"Doubling Surface_m2 (60 -> 120 m2) should increase predicted price, "
        f"got {price_base:.2f} -> {price_larger:.2f} k EUR"
    )


def test_train_model_strict_reproducibility_with_fixed_random_state():
    """
    Reproducibility guard: two independent training runs using the exact
    same train_test_split(random_state=train.RANDOM_STATE) on the same
    dataset must produce numerically identical model weights. Linear
    regression's normal-equation solver is itself deterministic (no
    internal randomness), so the only source of nondeterminism would be a
    different train/test split --- which RANDOM_STATE=42 is meant to
    prevent. This is a regression guard for MLOps reproducibility
    (traceability requirement discussed in the memoire, Chapter III).

    Garde-fou de reproductibilité : deux entraînements indépendants
    utilisant exactement le même train_test_split(random_state=
    train.RANDOM_STATE) sur le même jeu de données doivent produire des
    poids de modèle numériquement identiques. Le solveur de la régression
    linéaire (équations normales) est lui-même déterministe (pas
    d'aléatoire interne) ; la seule source de non-déterminisme serait un
    découpage train/test différent --- ce que RANDOM_STATE=42 est censé
    empêcher. Ceci est un garde-fou de reproductibilité MLOps (exigence de
    traçabilité discutée dans le mémoire, Chapitre III).
    """
    from sklearn.model_selection import train_test_split

    X, y = train.load_and_prepare_data()

    X_train_1, _, y_train_1, _ = train_test_split(
        X, y, test_size=0.2, random_state=train.RANDOM_STATE
    )
    X_train_2, _, y_train_2, _ = train_test_split(
        X, y, test_size=0.2, random_state=train.RANDOM_STATE
    )

    model_1 = train.train_model(X_train_1, y_train_1)
    model_2 = train.train_model(X_train_2, y_train_2)

    np.testing.assert_array_equal(model_1.coef_, model_2.coef_)
    assert model_1.intercept_ == pytest.approx(model_2.intercept_, abs=1e-12)


def test_load_and_prepare_data_raises_filenotfounderror_when_missing(monkeypatch, tmp_path):
    """
    train.py must fail loudly (FileNotFoundError) with an actionable message
    when the dataset file does not exist, rather than letting pandas raise
    an opaque error further down the stack.

    train.py doit échouer explicitement (FileNotFoundError) avec un message
    exploitable lorsque le fichier de données n'existe pas, plutôt que de
    laisser pandas lever une erreur opaque plus loin dans la pile d'appels.
    """
    missing_path = tmp_path / 'does_not_exist.csv'
    monkeypatch.setattr(train, 'DATA_FILE', missing_path)

    with pytest.raises(FileNotFoundError, match='Run generate_data.py first'):
        train.load_and_prepare_data()


def test_load_and_prepare_data_raises_on_completely_empty_file(monkeypatch, tmp_path):
    """
    A zero-byte dataset file (e.g. a failed/interrupted generate_data.py run,
    or a corrupted checkout) has no header for pandas to parse and must
    surface as pandas.errors.EmptyDataError rather than being silently
    swallowed or crashing later during training with a confusing KeyError.

    Un fichier de données de taille nulle (par ex. exécution de
    generate_data.py interrompue/échouée, ou checkout corrompu) n'a pas
    d'en-tête que pandas puisse analyser, et doit se manifester par
    pandas.errors.EmptyDataError plutôt que d'être silencieusement avalé ou
    de planter plus tard pendant l'entraînement avec un KeyError confus.
    """
    empty_file = tmp_path / 'empty.csv'
    empty_file.write_text('')
    monkeypatch.setattr(train, 'DATA_FILE', empty_file)

    with pytest.raises(pd.errors.EmptyDataError):
        train.load_and_prepare_data()


def test_load_and_prepare_data_header_only_returns_empty_frame(monkeypatch, tmp_path):
    """
    A header-only CSV (valid columns, zero data rows) is not an I/O error,
    so load_and_prepare_data() must return an empty-but-well-formed
    (X, y) pair rather than raising --- the failure belongs downstream, in
    train_model() itself (see the next test).

    Un CSV ne contenant que l'en-tête (colonnes valides, zéro ligne de
    données) n'est pas une erreur d'E/S : load_and_prepare_data() doit donc
    renvoyer un couple (X, y) vide mais bien formé plutôt que de lever une
    exception --- l'échec doit se produire en aval, dans train_model()
    elle-même (voir le test suivant).
    """
    header_only = tmp_path / 'header_only.csv'
    header_only.write_text(
        'Surface_m2,Nb_Pieces,Annee_Construction,Distance_Centre_km,'
        'DPE_Energy_Class,Has_Balcony,Has_Parking,Prix_k_EUR\n'
    )
    monkeypatch.setattr(train, 'DATA_FILE', header_only)

    X, y = train.load_and_prepare_data()

    assert list(X.columns) == EXPECTED_FEATURE_COLUMNS
    assert len(X) == 0
    assert len(y) == 0


def test_train_model_raises_on_empty_training_data():
    """
    Calling train_model() with zero training samples must raise (sklearn's
    own ValueError for a 0-sample array), not silently return a degenerate
    fitted model --- an empty dataset must halt the pipeline, not produce a
    model that predicts an arbitrary constant.

    Appeler train_model() avec zéro échantillon d'entraînement doit lever
    une exception (ValueError natif de scikit-learn pour un tableau à 0
    échantillon), et non renvoyer silencieusement un modèle dégénéré ---
    un jeu de données vide doit arrêter le pipeline, pas produire un modèle
    qui prédit une constante arbitraire.
    """
    X_empty = pd.DataFrame({col: [] for col in EXPECTED_FEATURE_COLUMNS})
    y_empty = pd.Series([], dtype=float, name='Prix_k_EUR')

    with pytest.raises(ValueError):
        train.train_model(X_empty, y_empty)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
