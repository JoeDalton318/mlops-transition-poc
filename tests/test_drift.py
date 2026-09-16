"""
Unit tests for the drift detection pipeline (src/drift_detection.py).

Focuses on the pure-Python logic (drift status extraction, reference dataset
loading, retraining trigger wiring) that does not require running an actual
Evidently Report computation, so these tests stay fast and do not depend on
network access or a live MLflow tracking server. The full
Report(metrics=[DataDriftPreset()]).run(...) computation itself is exercised
by actually running `python src/drift_detection.py` (see the memoire's
Chapter IV for the resulting real drift score), not unit-tested here.

Tests unitaires pour le pipeline de détection de dérive (src/drift_detection.py).

Se concentre sur la logique Python pure (extraction du statut de dérive,
chargement des données de référence, déclenchement du réentraînement) qui ne
nécessite pas de calcul Evidently réel, afin que ces tests restent rapides et
sans dépendance réseau ou à un serveur MLflow actif. Le calcul
Report(metrics=[DataDriftPreset()]).run(...) lui-même est exercé en exécutant
réellement `python src/drift_detection.py` (voir le Chapitre IV du mémoire
pour le score de dérive réel obtenu), pas testé unitairement ici.
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


def test_check_drift_status_below_threshold_returns_false():
    """A report with drift_share below DRIFT_THRESHOLD must not flag drift."""
    report_dict = {
        'metrics': [
            {'result': {'drift_share': 0.05}},
        ]
    }

    detected, score = drift_detection.check_drift_status(report_dict)

    assert detected is False
    assert score == pytest.approx(0.05)


def test_check_drift_status_above_threshold_returns_true():
    """A report with drift_share above DRIFT_THRESHOLD (0.2) must flag drift."""
    report_dict = {
        'metrics': [
            {'result': {'drift_share': 0.45}},
        ]
    }

    detected, score = drift_detection.check_drift_status(report_dict)

    assert detected is True
    assert score == pytest.approx(0.45)


def test_check_drift_status_exactly_at_threshold_does_not_trigger():
    """
    The comparison is strict (drift_score > DRIFT_THRESHOLD), so a report
    landing exactly on the threshold must not be flagged as drifted.
    """
    report_dict = {'metrics': [{'result': {'drift_share': drift_detection.DRIFT_THRESHOLD}}]}

    detected, score = drift_detection.check_drift_status(report_dict)

    assert detected is False
    assert score == pytest.approx(drift_detection.DRIFT_THRESHOLD)


def test_check_drift_status_missing_drift_share_defaults_safely():
    """A report with no 'drift_share' key must not crash and must default to no drift."""
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
