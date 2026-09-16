"""
Data quality / schema tests for data/immobilier_france.csv.

These tests exist specifically because the original synthetic dataset
generator had a silent bug: 77.8% of Prix_k_EUR values were saturated at the
upper clipping bound (500 k EUR), producing a near-degenerate regression
target (real R2 ~= 0.42 on the buggy dataset). No test in the original suite
would have caught this. These tests would.

Tests de qualité/schéma des données pour data/immobilier_france.csv.

Ces tests existent spécifiquement parce que le générateur de données
synthétiques d'origine contenait un bug silencieux : 77,8 % des valeurs de
Prix_k_EUR étaient saturées au plafond de clipping (500 k EUR), produisant
une cible de régression quasi dégénérée (R2 réel ~= 0,42 sur le jeu de
données défectueux). Aucun test de la suite d'origine ne l'aurait détecté.
Ceux-ci le font.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

DATA_FILE = Path(__file__).parent.parent / 'data' / 'immobilier_france.csv'

EXPECTED_COLUMNS = [
    'Surface_m2',
    'Nb_Pieces',
    'Annee_Construction',
    'Distance_Centre_km',
    'DPE_Energy_Class',
    'Has_Balcony',
    'Has_Parking',
    'Prix_k_EUR',
]


@pytest.fixture(scope='module')
def df():
    """Load the real dataset once per module. (Charge le jeu de données réel une fois par module.)"""
    assert DATA_FILE.exists(), f'{DATA_FILE} not found — run generate_data.py first.'
    return pd.read_csv(DATA_FILE)


def test_columns_match_expected_schema(df):
    """
    The CSV columns must exactly match the API/training schema, in order.
    Les colonnes du CSV doivent correspondre exactement au schéma API/entraînement, dans l'ordre.
    """
    assert list(df.columns) == EXPECTED_COLUMNS


def test_no_missing_values(df):
    """No column may contain missing values. (Aucune colonne ne doit contenir de valeur manquante.)"""
    assert df.isnull().sum().sum() == 0


def test_no_duplicate_rows(df):
    """
    The dataset must not contain exact duplicate rows.
    Le jeu de données ne doit pas contenir de lignes strictement dupliquées.
    """
    assert df.duplicated().sum() == 0


def test_row_count_is_reasonable(df):
    """
    The dataset must be large enough for a meaningful train/test split.
    Le jeu de données doit être assez volumineux pour un découpage train/test pertinent.
    """
    assert len(df) >= 100


@pytest.mark.parametrize('column,lo,hi', [
    ('Surface_m2', 30, 300),
    ('Nb_Pieces', 1, 10),
    ('Annee_Construction', 1900, 2024),
    ('Distance_Centre_km', 0, 100),
    ('DPE_Energy_Class', 1, 7),
])
def test_feature_within_pydantic_bounds(df, column, lo, hi):
    """
    Bounds must stay within the ranges enforced by the Pydantic schema in
    src/app.py (HousingFeatures) — otherwise the API would reject valid
    training rows if they were ever replayed as prediction requests.
    """
    assert df[column].min() >= lo
    assert df[column].max() <= hi


@pytest.mark.parametrize('column', ['Has_Balcony', 'Has_Parking'])
def test_binary_columns_are_strictly_0_or_1(df, column):
    """
    Has_Balcony/Has_Parking must only ever take the values 0 or 1.
    Has_Balcony/Has_Parking ne doivent prendre que les valeurs 0 ou 1.
    """
    assert set(df[column].unique()) <= {0, 1}


def test_target_is_positive(df):
    """
    A real-estate price can never be negative or zero.
    Un prix immobilier ne peut jamais être négatif ou nul.
    """
    assert (df['Prix_k_EUR'] > 0).all()


def test_target_not_saturated_at_bounds(df):
    """
    Regression guard for the original clipping-saturation bug: no more than
    2% of rows may sit at the dataset's own min or max Prix_k_EUR value. On
    the buggy generator, 389/500 rows (77.8%) were exactly at 500.0.
    """
    value_counts = df['Prix_k_EUR'].value_counts()
    most_common_count = value_counts.iloc[0] if len(value_counts) else 0
    saturation_ratio = most_common_count / len(df)

    assert saturation_ratio <= 0.02, (
        f"{most_common_count}/{len(df)} rows ({saturation_ratio:.1%}) share the same "
        "Prix_k_EUR value — likely a clipping/saturation artifact in generate_data.py."
    )


def test_target_has_realistic_spread(df):
    """A degenerate (near-constant) target would make the regression problem meaningless."""
    coefficient_of_variation = df['Prix_k_EUR'].std() / df['Prix_k_EUR'].mean()
    assert coefficient_of_variation > 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
