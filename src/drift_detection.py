"""
Data drift detection with continuous training triggering.

This module monitors incoming data against a reference dataset using
Evidently AI. If drift is detected, it automatically triggers model retraining
(self-healing MLOps pipeline) to maintain prediction accuracy.

The script generates drift reports and logs retraining events for monitoring.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Tuple, Optional
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import sys

# Import training pipeline
from train import run_pipeline, load_and_prepare_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Configuration
REFERENCE_DATA_FILE = Path(__file__).parent.parent / 'immobilier_france.csv'
REPORTS_DIR = Path(__file__).parent.parent / 'drift_reports'
DRIFT_THRESHOLD = 0.2  # Trigger retraining if drift score > 20%


def create_reference_dataset() -> pd.DataFrame:
    """
    Load the reference dataset for drift comparison.

    The reference dataset represents the distribution the model was trained on.
    Future data is compared against this baseline to detect drift.

    Returns:
        Reference DataFrame.

    Raises:
        FileNotFoundError: If reference data file does not exist.
    """
    if not REFERENCE_DATA_FILE.exists():
        raise FileNotFoundError(f'Reference dataset not found at {REFERENCE_DATA_FILE}')

    df = pd.read_csv(REFERENCE_DATA_FILE)
    logger.info(f'Reference dataset loaded: {len(df)} records')
    return df


def generate_drift_report(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict:
    """
    Generate Evidently AI drift report comparing current data to reference.

    Args:
        reference_data: Baseline dataset the model was trained on.
        current_data: New incoming data to check for drift.

    Returns:
        Dictionary containing report metadata, HTML path, and drift metrics.
    """
    logger.info('Generating drift detection report...')

    # Create report with data drift preset
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)

    # Ensure reports directory exists
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = REPORTS_DIR / f'drift_report_{timestamp}.html'
    report.save_html(str(report_path))

    logger.info(f'Drift report saved to {report_path}')

    # Extract drift metrics from report
    report_dict = report.as_dict()

    return {
        'timestamp': timestamp,
        'report_path': str(report_path),
        'report_dict': report_dict,
    }


def check_drift_status(report_dict: Dict) -> Tuple[bool, float]:
    """
    Programmatically extract drift detection status from Evidently report.

    This function checks if data drift was detected and returns a drift score.
    The drift is considered significant if the overall drift share exceeds
    the configured threshold.

    Args:
        report_dict: Report dictionary from Evidently AI.

    Returns:
        Tuple of (drift_detected: bool, drift_score: float).
    """
    try:
        # Navigate Evidently report structure
        metrics = report_dict.get('metrics', [])

        # Look for data drift metric results
        drift_detected = False
        drift_score = 0.0

        for metric in metrics:
            if 'result' in metric and isinstance(metric['result'], dict):
                result = metric['result']
                # Check for drift share metric
                if 'drift_share' in result:
                    drift_score = result['drift_share']
                    drift_detected = drift_score > DRIFT_THRESHOLD

        logger.info(f'Drift score: {drift_score:.4f} (threshold: {DRIFT_THRESHOLD})')

        if drift_detected:
            logger.warning(f'DRIFT DETECTED: Score {drift_score:.4f} exceeds threshold')
        else:
            logger.info('No significant drift detected')

        return drift_detected, drift_score

    except Exception as e:
        logger.error(f'Error checking drift status: {str(e)}')
        return False, 0.0


def trigger_retraining(drift_score: float) -> Optional[Dict]:
    """
    Trigger model retraining when drift is detected.

    This function implements the self-healing mechanism. When data drift
    exceeds the threshold, it automatically calls the training pipeline
    to retrain the model on fresh data, adapting to distribution changes.

    Args:
        drift_score: Detected drift score for logging.

    Returns:
        Result dictionary from training pipeline, or None if retraining failed.
    """
    logger.warning(f'Initiating self-healing retraining (drift_score: {drift_score:.4f})')

    try:
        # Call training pipeline
        result = run_pipeline(trigger_source='drift_detection')

        if result['status'] == 'success':
            logger.info(
                f'Self-healing retraining completed successfully. '
                f'New R2 score: {result["metrics"]["r2"]:.4f}'
            )
        else:
            logger.error(f'Self-healing retraining failed: {result["message"]}')

        return result

    except Exception as e:
        logger.error(f'Error during retraining: {str(e)}')
        return None


def run_drift_detection(generate_synthetic_data: bool = True) -> Dict:
    """
    Execute the complete drift detection and self-healing pipeline.

    This function orchestrates the workflow:
    1. Load reference (training) data
    2. Optionally generate synthetic current data for demo
    3. Generate Evidently drift report
    4. Check if drift is significant
    5. Trigger retraining if drift detected

    Args:
        generate_synthetic_data: If True, generates synthetic data to simulate drift.
                               If False, uses immobilier_france.csv as current data.
                               (Used for demo; production would fetch live data)

    Returns:
        Dictionary with results of drift detection and any retraining.
    """
    logger.info('Starting drift detection pipeline...')

    try:
        # Load reference data
        reference_data = create_reference_dataset()

        # Prepare current data
        if generate_synthetic_data:
            logger.info('Generating synthetic current data for demonstration...')
            # Add slight perturbations to reference data to simulate drift
            current_data = reference_data.copy()
            current_data['Surface_m2'] += 10  # Systematic shift
            current_data['Prix_k_EUR'] *= 1.15  # Market price increase
        else:
            current_data = reference_data.copy()

        # Generate drift report
        report_info = generate_drift_report(reference_data, current_data)

        # Check drift status
        drift_detected, drift_score = check_drift_status(report_info['report_dict'])

        result = {
            'timestamp': report_info['timestamp'],
            'report_path': report_info['report_path'],
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'retraining_triggered': False,
            'retraining_result': None,
        }

        # Trigger retraining if drift detected
        if drift_detected:
            retraining_result = trigger_retraining(drift_score)
            result['retraining_triggered'] = True
            result['retraining_result'] = retraining_result

        logger.info(f'Drift detection pipeline completed. Result: {result}')
        return result

    except Exception as e:
        logger.error(f'Drift detection pipeline failed: {str(e)}')
        return {
            'status': 'failed',
            'message': str(e),
        }


def main() -> None:
    """Entry point for drift detection script."""
    logger.info('Drift Detection and Self-Healing Pipeline Started')
    result = run_drift_detection(generate_synthetic_data=True)

    if result['drift_detected']:
        print(f"\nDrift Detection Results:")
        print(f"  Drift Score: {result['drift_score']:.4f}")
        print(f"  Report: {result['report_path']}")
        print(f"  Retraining Triggered: {result['retraining_triggered']}")
        if result['retraining_result']:
            print(f"  Retraining Status: {result['retraining_result']['status']}")
    else:
        print(f"No significant drift detected (score: {result['drift_score']:.4f})")


if __name__ == '__main__':
    main()