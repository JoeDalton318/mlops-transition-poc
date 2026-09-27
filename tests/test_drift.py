"""
Unit tests for the drift detection pipeline (src/drift_detection.py).

Focuses on the pure-Python logic (drift status extraction, reference dataset
loading, retraining trigger wiring) that does not require running an actual
Evidently Report computation, so these tests stay fast and do not depend on
network access or a live MLflow tracking server. The full
Report(metrics=[DataDriftPreset()]).run(...) computation itself is exercised
by actually running `python src/drift_detection.py` (see the memoire's
Chapter IV for the resulting real drift score) and by the integration tests
(tests/test_integration_pipeline.py), not unit-tested here. The mocked report
dicts below reproduce the exact structure of a real Evidently 0.4.18 report.

Tests unitaires pour le pipeline de détection de dérive (src/drift_detection.py).

Se concentre sur la logique Python pure (extraction du statut de dérive,
chargement des données de référence, déclenchement du réentraînement) qui ne
nécessite pas de calcul Evidently réel, afin que ces tests restent rapides et
sans dépendance réseau ou à un serveur MLflow actif. Le calcul
Report(metrics=[DataDriftPreset()]).run(...) lui-même est exercé en exécutant
réellement `python src/drift_detection.py` (voir le Chapitre IV du mémoire
pour le score de dérive réel obtenu) et par les tests d'intégration
(tests/test_integration_pipeline.py), pas testé unitairement ici. Les
dictionnaires de rapport simulés ci-dessous reproduisent la structure exacte
d'un vrai rapport Evidently 0.4.18.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import drift_detection  # noqa: E402


def test_create_reference_dataset_matches_real_data_file():
    """The reference dataset must load without error and match the real CSV schema."""
    df = drift_detection.create_reference_dataset()

    assert len(df) > 0
    assert 'Prix_k_EUR' in df.columns
    assert df.isnull().sum().sum() == 0


def _evidently_report_dict(number_of_drifted_columns: int, number_of_columns: int = 8) -> dict:
    """
    Build a report dict with the exact structure produced by Evidently 0.4.18
    (Report(metrics=[DataDriftPreset()]).as_dict()): a DatasetDriftMetric
    result, whose 'drift_share' key is Evidently's own dataset-level threshold
    (0.5 by default), followed by a DataDriftTable result.

    Construit un dictionnaire de rapport ayant la structure exacte produite par
    Evidently 0.4.18 : un résultat DatasetDriftMetric, dont la clé 'drift_share'
    est le seuil propre à Evidently au niveau du jeu de données (0.5 par défaut),
    suivi d'un résultat DataDriftTable.
    """
    share = number_of_drifted_columns / number_of_columns
    common = {
        'number_of_columns': number_of_columns,
        'number_of_drifted_columns': number_of_drifted_columns,
        'share_of_drifted_columns': share,
        'dataset_drift': share >= 0.5,
    }
    return {
        'metrics': [
            {'metric': 'DatasetDriftMetric', 'result': {'drift_share': 0.5, **common}},
            {'metric': 'DataDriftTable', 'result': dict(common)},
        ]
    }


def test_check_drift_status_below_threshold_returns_false():
    """
    1 drifted column out of 8 (0.125) is below DRIFT_THRESHOLD and must not flag
    drift, even though the report also carries Evidently's 'drift_share' = 0.5.
    This is the regression check for reading the threshold instead of the share.
    """
    detected, score = drift_detection.check_drift_status(_evidently_report_dict(1))

    assert detected is False
    assert score == pytest.approx(0.125)


def test_check_drift_status_above_threshold_returns_true():
    """2 drifted columns out of 8 (0.25, the PoC's real result) must flag drift."""
    detected, score = drift_detection.check_drift_status(_evidently_report_dict(2))

    assert detected is True
    assert score == pytest.approx(0.25)


def test_check_drift_status_exactly_at_threshold_does_not_trigger():
    """
    The comparison is strict (drift_score > DRIFT_THRESHOLD), so a report
    landing exactly on the threshold must not be flagged as drifted.
    """
    detected, score = drift_detection.check_drift_status(_evidently_report_dict(1, number_of_columns=5))

    assert score == pytest.approx(drift_detection.DRIFT_THRESHOLD)
    assert detected is False


def test_check_drift_status_falls_back_to_drift_share_when_share_is_absent():
    """
    If 'share_of_drifted_columns' is absent from every metric, the legacy
    'drift_share' key is used as a fallback.
    """
    report_dict = {'metrics': [{'result': {'drift_share': 0.45}}]}

    detected, score = drift_detection.check_drift_status(report_dict)

    assert detected is True
    assert score == pytest.approx(0.45)


def test_check_drift_status_missing_drift_keys_defaults_safely():
    """A report with neither drift key must not crash and must default to no drift."""
    report_dict = {'metrics': [{'result': {'some_other_key': 1.0}}]}

    detected, score = drift_detection.check_drift_status(report_dict)

    assert detected is False
    assert score == 0.0


def test_check_drift_status_malformed_report_returns_safe_default():
    """A malformed report dict must be handled gracefully (no exception), not crash the pipeline."""
    detected, score = drift_detection.check_drift_status({'unexpected': 'shape'})

    assert detected is False
    assert score == 0.0


def test_trigger_retraining_calls_run_pipeline_with_drift_source():
    """
    trigger_retraining() must invoke train.run_pipeline with
    trigger_source='drift_detection', so retraining runs are distinguishable
    in MLflow from manual runs.
    """
    fake_result = {'status': 'success', 'metrics': {'r2': 0.9}, 'message': 'ok'}
    with patch.object(drift_detection, 'run_pipeline', return_value=fake_result) as mock_run:
        result = drift_detection.trigger_retraining(drift_score=0.35)

    mock_run.assert_called_once_with(trigger_source='drift_detection')
    assert result == fake_result


def test_trigger_retraining_handles_pipeline_exception():
    """If run_pipeline raises, trigger_retraining() must catch it and return None, not crash."""
    with patch.object(drift_detection, 'run_pipeline', side_effect=RuntimeError('boom')):
        result = drift_detection.trigger_retraining(drift_score=0.5)

    assert result is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
