"""
Streamlit interactive dashboard for MLOps monitoring and control.
Tableau de bord Streamlit interactif pour la surveillance et le contrôle MLOps.

This application provides a unified interface for:
- Making price predictions via the FastAPI backend
- Visualizing the latest data drift detection report
- Manually triggering the continuous training (CT) pipeline
- Monitoring system health

Cette application fournit une interface unifiée pour :
- Faire des prédictions de prix via le backend FastAPI
- Visualiser le dernier rapport de détection de dérive de données
- Déclencher manuellement le pipeline d'entraînement continu (CT)
- Surveiller la santé du système
"""

import streamlit as st
import requests
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import pandas as pd
import logging

# Configure logging
# Configuration de la journalisation
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
# Paramètres de configuration
API_BASE_URL = 'http://localhost:8000'
DRIFT_REPORTS_DIR = Path(__file__).parent.parent / 'drift_reports'

# Language translations
# Traductions linguistiques
TRANSLATIONS = {
    'en': {
        'title': 'Housing MLOps Dashboard',
        'subtitle': 'Production-ready monitoring and control center for the housing price prediction model with continuous training capabilities.',
        'api_status': 'API Status: Online',
        'api_offline': 'API Status: Offline',
        'model_loaded': 'Model Status: Loaded',
        'model_not_loaded': 'Model Status: Not Loaded',
        'model_unknown': 'Model Status: Unknown',
        'dashboard_updated': 'Dashboard Updated',
        'price_prediction': 'Price Prediction',
        'property_characteristics': 'Property Characteristics',
        'living_area': 'Living Area (m²)',
        'num_rooms': 'Number of Rooms',
        'construction_year': 'Construction Year',
        'distance_city': 'Distance to City Center (km)',
        'energy_amenities': 'Energy & Amenities',
        'energy_efficiency': 'Energy Efficiency (DPE Class)',
        'energy_best': '1 = Worst efficiency, 7 = Best efficiency',
        'has_balcony': 'Has Balcony?',
        'yes': 'Yes',
        'no': 'No',
        'has_parking': 'Has Parking?',
        'get_prediction': 'Get Price Prediction',
        'predicted_price': 'Predicted Price',
        'price_range': 'Range',
        'view_features': 'View Input Features',
        'drift_monitoring': 'Data Drift Monitoring',
        'latest_report': 'Latest Drift Report',
        'report_generated': 'Report generated',
        'actions': 'Actions',
        'run_ct': 'Run Continuous Training',
        'run_ct_help': 'Manually trigger drift detection and retraining if needed',
        'no_reports': 'No drift reports available yet. Run the pipeline to generate one.',
        'system_info': 'System Information',
        'model_pipeline': 'Model Pipeline',
        'mlops_level': 'MLOps Level 1/2',
        'ct_enabled': 'Continuous Training Enabled',
        'self_healing': 'Self-Healing Architecture',
        'components': 'Components',
        'api_backend': 'FastAPI Backend: /predict',
        'drift_detection': 'Evidently AI: Drift Detection',
        'experiment_tracking': 'MLflow: Experiment Tracking',
        'data_sources': 'Data Sources',
        'synthetic_dataset': 'Synthetic Housing Dataset',
        'features_count': '7 Property Features',
        'request_timeout': 'Request timeout: API did not respond within 5 seconds',
        'connection_error': 'Connection error: Cannot reach API at http://localhost:8000',
        'ensure_api': 'Ensure the FastAPI server is running: uvicorn src.app:app --reload',
        'unexpected_error': 'Unexpected error',
        'pipeline_running': 'Running drift detection and continuous training...',
        'pipeline_success': 'Continuous training pipeline completed successfully',
        'pipeline_failed': 'Pipeline failed',
        'pipeline_timeout': 'Pipeline timeout: Execution took longer than 120 seconds',
        'api_error': 'API Error',
    },
    'fr': {
        'title': 'Tableau de Bord MLOps Immobilier',
        'subtitle': 'Centre de surveillance et de contrôle prêt pour la production pour le modèle de prédiction des prix immobiliers avec capacités d\'entraînement continu.',
        'api_status': 'État de l\'API : En ligne',
        'api_offline': 'État de l\'API : Hors ligne',
        'model_loaded': 'État du modèle : Chargé',
        'model_not_loaded': 'État du modèle : Non chargé',
        'model_unknown': 'État du modèle : Inconnu',
        'dashboard_updated': 'Tableau de bord mis à jour',
        'price_prediction': 'Prédiction de Prix',
        'property_characteristics': 'Caractéristiques de la Propriété',
        'living_area': 'Surface Habitable (m²)',
        'num_rooms': 'Nombre de Pièces',
        'construction_year': 'Année de Construction',
        'distance_city': 'Distance au Centre-Ville (km)',
        'energy_amenities': 'Énergie et Équipements',
        'energy_efficiency': 'Efficacité Énergétique (Classe DPE)',
        'energy_best': '1 = Pire efficacité, 7 = Meilleure efficacité',
        'has_balcony': 'Avec Balcon ?',
        'yes': 'Oui',
        'no': 'Non',
        'has_parking': 'Avec Parking ?',
        'get_prediction': 'Obtenir la Prédiction de Prix',
        'predicted_price': 'Prix Prédit',
        'price_range': 'Plage',
        'view_features': 'Afficher les Caractéristiques d\'Entrée',
        'drift_monitoring': 'Surveillance de la Dérive de Données',
        'latest_report': 'Dernier Rapport de Dérive',
        'report_generated': 'Rapport généré',
        'actions': 'Actions',
        'run_ct': 'Exécuter l\'Entraînement Continu',
        'run_ct_help': 'Déclencher manuellement la détection de dérive et le réentraînement si nécessaire',
        'no_reports': 'Aucun rapport de dérive disponible encore. Exécutez le pipeline pour en générer un.',
        'system_info': 'Informations Système',
        'model_pipeline': 'Pipeline de Modèle',
        'mlops_level': 'Niveau MLOps 1/2',
        'ct_enabled': 'Entraînement Continu Activé',
        'self_healing': 'Architecture Auto-Cicatrisante',
        'components': 'Composants',
        'api_backend': 'Backend FastAPI : /predict',
        'drift_detection': 'Evidently AI : Détection de Dérive',
        'experiment_tracking': 'MLflow : Suivi des Expériences',
        'data_sources': 'Sources de Données',
        'synthetic_dataset': 'Ensemble de Données Immobilier Synthétique',
        'features_count': '7 Caractéristiques de Propriété',
        'request_timeout': 'Délai d\'attente dépassé : l\'API n\'a pas répondu en 5 secondes',
        'connection_error': 'Erreur de connexion : Impossible d\'accéder à l\'API à http://localhost:8000',
        'ensure_api': 'Assurez-vous que le serveur FastAPI est en cours d\'exécution : uvicorn src.app:app --reload',
        'unexpected_error': 'Erreur inattendue',
        'pipeline_running': 'Exécution de la détection de dérive et de l\'entraînement continu...',
        'pipeline_success': 'Pipeline d\'entraînement continu complété avec succès',
        'pipeline_failed': 'Échec du pipeline',
        'pipeline_timeout': 'Délai d\'attente du pipeline : L\'exécution a dépassé 120 secondes',
        'api_error': 'Erreur API',
    }
}

# Page configuration
# Configuration de la page
st.set_page_config(
    page_title='Housing MLOps Dashboard',
    page_icon='🏠',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Language selector in sidebar
# Sélecteur de langue dans la barre latérale
if 'language' not in st.session_state:
    st.session_state.language = 'fr'

with st.sidebar:
    language = st.radio(
        'Language / Langue',
        options=['en', 'fr'],
        format_func=lambda x: 'English' if x == 'en' else 'Français',
        horizontal=True
    )
    st.session_state.language = language

# Get translation function
# Fonction de traduction
def t(key):
    """Get translated text"""
    return TRANSLATIONS[st.session_state.language].get(key, key)

# Custom CSS for professional styling
# CSS personnalisé pour un style professionnel
st.markdown("""
    <style>
        .metric-container {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .success-box {
            background-color: #d4edda;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #28a745;
        }
        .error-box {
            background-color: #f8d7da;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #dc3545;
        }
        .info-box {
            background-color: #d1ecf1;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #17a2b8;
        }
    </style>
""", unsafe_allow_html=True)


def check_api_health() -> Tuple[bool, Optional[str]]:
    """
    Check if the FastAPI backend is operational.
    Vérifie si le backend FastAPI est opérationnel.

    Returns:
        Tuple of (is_healthy: bool, model_status: Optional[str])
    """
    try:
        response = requests.get(f'{API_BASE_URL}/health', timeout=2)
        if response.status_code == 200:
            data = response.json()
            return True, data.get('model_status', 'unknown')
    except requests.exceptions.ConnectionError:
        logger.warning('API health check failed: Connection refused')
    except Exception as e:
        logger.warning(f'API health check failed: {str(e)}')

    return False, None


def get_latest_drift_report() -> Optional[Path]:
    """
    Retrieve the most recent drift detection report file.
    Récupère le fichier de rapport de détection de dérive le plus récent.

    Returns:
        Path to the latest drift report HTML file, or None if none exist.
    """
    if not DRIFT_REPORTS_DIR.exists():
        return None

    report_files = sorted(
        DRIFT_REPORTS_DIR.glob('drift_report_*.html'),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    return report_files[0] if report_files else None


def send_prediction_request(features: Dict) -> Optional[Dict]:
    """
    Send housing features to the FastAPI prediction endpoint.
    Envoie les caractéristiques immobilières au point de terminaison de prédiction FastAPI.

    Args:
        features: Dictionary of housing features matching the Pydantic schema.

    Returns:
        Prediction response dictionary, or None if request fails.
    """
    try:
        response = requests.post(
            f'{API_BASE_URL}/predict',
            json=features,
            timeout=5,
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f'{t("api_error")} {response.status_code}: {response.text}')
            return None

    except requests.exceptions.Timeout:
        st.error(t('request_timeout'))
        return None
    except requests.exceptions.ConnectionError:
        st.error(t('connection_error'))
        st.info(t('ensure_api'))
        return None
    except Exception as e:
        st.error(f'{t("unexpected_error")}: {str(e)}')
        return None


def trigger_continuous_training() -> bool:
    """
    Manually trigger the drift detection and continuous training pipeline.
    Déclenchez manuellement le pipeline de détection de dérive et d'entraînement continu.

    Returns:
        True if subprocess execution was successful, False otherwise.
    """
    try:
        with st.spinner(t('pipeline_running')):
            script_path = Path(__file__).parent / 'drift_detection.py'

            result = subprocess.run(
                ['python', str(script_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode == 0:
                st.success(t('pipeline_success'))
                logger.info('CT pipeline executed successfully')
                return True
            else:
                st.error(f'{t("pipeline_failed")}: {result.stderr}')
                logger.error(f'CT pipeline failed: {result.stderr}')
                return False

    except subprocess.TimeoutExpired:
        st.error(t('pipeline_timeout'))
        logger.error('CT pipeline timeout')
        return False
    except Exception as e:
        st.error(f'{t("unexpected_error")}: {str(e)}')
        logger.error(f'CT pipeline error: {str(e)}')
        return False


def render_header() -> None:
    """
    Render the dashboard header with system status.
    Affiche l'en-tête du tableau de bord avec l'état du système.
    """
    st.title(t('title'))
    st.markdown(t('subtitle'))

    # System status
    # État du système
    col1, col2, col3 = st.columns(3)

    api_healthy, model_status = check_api_health()

    with col1:
        if api_healthy:
            st.success(t('api_status'))
        else:
            st.error(t('api_offline'))

    with col2:
        if model_status == 'loaded':
            st.success(t('model_loaded'))
        elif model_status == 'not_loaded':
            st.warning(t('model_not_loaded'))
        else:
            st.error(t('model_unknown'))

    with col3:
        st.info(f'{t("dashboard_updated")}: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    st.divider()


def render_prediction_section() -> None:
    """
    Render the prediction input form and results section.
    Affiche le formulaire d'entrée de prédiction et la section des résultats.
    """
    st.header(t('price_prediction'))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(t('property_characteristics'))
        surface_m2 = st.slider(
            t('living_area'),
            min_value=30.0,
            max_value=300.0,
            value=120.0,
            step=5.0,
        )
        nb_pieces = st.number_input(
            t('num_rooms'),
            min_value=1,
            max_value=10,
            value=3,
            step=1,
        )
        annee_construction = st.slider(
            t('construction_year'),
            min_value=1950,
            max_value=2024,
            value=1995,
            step=1,
        )
        distance_centre_km = st.slider(
            t('distance_city'),
            min_value=0.5,
            max_value=50.0,
            value=5.0,
            step=0.5,
        )

    with col2:
        st.subheader(t('energy_amenities'))
        dpe_energy_class = st.select_slider(
            t('energy_efficiency'),
            options=list(range(1, 8)),
            value=5,
            help=t('energy_best'),
        )
        has_balcony = st.selectbox(
            t('has_balcony'),
            options=[0, 1],
            format_func=lambda x: t('yes') if x == 1 else t('no'),
        )
        has_parking = st.selectbox(
            t('has_parking'),
            options=[0, 1],
            format_func=lambda x: t('yes') if x == 1 else t('no'),
        )

    # Prepare features dictionary
    # Préparez le dictionnaire de caractéristiques
    features = {
        'Surface_m2': surface_m2,
        'Nb_Pieces': nb_pieces,
        'Annee_Construction': annee_construction,
        'Distance_Centre_km': distance_centre_km,
        'DPE_Energy_Class': dpe_energy_class,
        'Has_Balcony': has_balcony,
        'Has_Parking': has_parking,
    }

    # Prediction button
    # Bouton de prédiction
    if st.button(t('get_prediction'), use_container_width=True, type='primary'):
        prediction_result = send_prediction_request(features)

        if prediction_result:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)

            price = prediction_result.get('predicted_price_k_eur', 0)
            st.metric(
                label=t('predicted_price'),
                value=f'{price:,.0f} k EUR',
                delta=f'{t("price_range")}: {price * 0.85:.0f} - {price * 1.15:.0f} k EUR',
            )

            st.markdown('</div>', unsafe_allow_html=True)

            # Display input features for reference
            # Afficher les caractéristiques d'entrée pour référence
            with st.expander(t('view_features')):
                features_df = pd.DataFrame([features]).T
                features_df.columns = ['Value']
                st.dataframe(features_df)

    st.divider()


def render_drift_monitoring_section() -> None:
    """
    Render the data drift monitoring section.
    Affiche la section de surveillance de la dérive de données.
    """
    st.header(t('drift_monitoring'))

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader(t('latest_report'))

        latest_report = get_latest_drift_report()

        if latest_report:
            st.info(f'{t("report_generated")}: {latest_report.name}')

            try:
                with open(latest_report, 'r', encoding='utf-8') as f:
                    report_html = f.read()

                st.components.v1.html(report_html, height=600, scrolling=True)

            except Exception as e:
                st.error(f'{t("unexpected_error")}: {str(e)}')
                st.markdown(f'[Open report in browser]({latest_report})')
        else:
            st.warning(t('no_reports'))

    with col2:
        st.subheader(t('actions'))
        if st.button(
            t('run_ct'),
            use_container_width=True,
            type='secondary',
            help=t('run_ct_help'),
        ):
            success = trigger_continuous_training()
            if success:
                st.rerun()

    st.divider()


def render_system_info() -> None:
    """
    Render system information section.
    Affiche la section d'informations système.
    """
    st.header(t('system_info'))

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.markdown(
            '<div class="info-box">'
            f'<strong>{t("model_pipeline")}</strong><br>'
            f'{t("mlops_level")}<br>'
            f'{t("ct_enabled")}<br>'
            f'{t("self_healing")}'
            '</div>',
            unsafe_allow_html=True
        )

    with info_col2:
        st.markdown(
            '<div class="info-box">'
            f'<strong>{t("components")}</strong><br>'
            f'{t("api_backend")}<br>'
            f'{t("drift_detection")}<br>'
            f'{t("experiment_tracking")}'
            '</div>',
            unsafe_allow_html=True
        )

    with info_col3:
        st.markdown(
            '<div class="info-box">'
            f'<strong>{t("data_sources")}</strong><br>'
            f'immobilier_france.csv<br>'
            f'{t("synthetic_dataset")}<br>'
            f'{t("features_count")}'
            '</div>',
            unsafe_allow_html=True
        )


def main() -> None:
    """
    Main entry point for the Streamlit dashboard application.
    Point d'entrée principal de l'application de tableau de bord Streamlit.
    """
    render_header()
    render_prediction_section()
    render_drift_monitoring_section()
    render_system_info()


if __name__ == '__main__':
    main()

# Custom CSS for professional styling
st.markdown("""
    <style>
        .metric-container {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .success-box {
            background-color: #d4edda;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #28a745;
        }
        .error-box {
            background-color: #f8d7da;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #dc3545;
        }
        .info-box {
            background-color: #d1ecf1;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #17a2b8;
        }
    </style>
""", unsafe_allow_html=True)


def check_api_health() -> Tuple[bool, Optional[str]]:
    """
    Check if the FastAPI backend is operational.

    Attempts a connection to the /health endpoint to verify
    the API is running and responsive. This is called on page load
    to provide immediate feedback on system status.

    Returns:
        Tuple of (is_healthy: bool, model_status: Optional[str])
        Model status is 'loaded', 'not_loaded', or None if API unavailable.
    """
    try:
        response = requests.get(f'{API_BASE_URL}/health', timeout=2)
        if response.status_code == 200:
            data = response.json()
            return True, data.get('model_status', 'unknown')
    except requests.exceptions.ConnectionError:
        logger.warning('API health check failed: Connection refused')
    except Exception as e:
        logger.warning(f'API health check failed: {str(e)}')

    return False, None


def get_latest_drift_report() -> Optional[Path]:
    """
    Retrieve the most recent drift detection report file.

    Scans the drift_reports directory and returns the path to the
    most recently generated HTML report. Used to display the latest
    monitoring results in the dashboard.

    Returns:
        Path to the latest drift report HTML file, or None if none exist.
    """
    if not DRIFT_REPORTS_DIR.exists():
        return None

    report_files = sorted(
        DRIFT_REPORTS_DIR.glob('drift_report_*.html'),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    return report_files[0] if report_files else None


def send_prediction_request(features: Dict) -> Optional[Dict]:
    """
    Send housing features to the FastAPI prediction endpoint.

    Constructs a JSON request with the provided features and sends
    it to the /predict endpoint. Handles network errors gracefully
    with appropriate user-facing error messages.

    Args:
        features: Dictionary of housing features matching the Pydantic schema.

    Returns:
        Prediction response dictionary, or None if request fails.
    """
    try:
        response = requests.post(
            f'{API_BASE_URL}/predict',
            json=features,
            timeout=5,
        )

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f'API Error {response.status_code}: {response.text}')
            return None

    except requests.exceptions.Timeout:
        st.error('Request timeout: API did not respond within 5 seconds')
        return None
    except requests.exceptions.ConnectionError:
        st.error('Connection error: Cannot reach API at http://localhost:8000')
        st.info('Ensure the FastAPI server is running: uvicorn src.app:app --reload')
        return None
    except Exception as e:
        st.error(f'Unexpected error: {str(e)}')
        return None


def trigger_continuous_training() -> bool:
    """
    Manually trigger the drift detection and continuous training pipeline.

    Executes the src/drift_detection.py script as a subprocess, which will
    generate a drift report and automatically retrain the model if drift
    is detected. This button allows operators to force a CT cycle without
    waiting for scheduled execution.

    Returns:
        True if subprocess execution was successful, False otherwise.
    """
    try:
        with st.spinner('Running drift detection and continuous training...'):
            script_path = Path(__file__).parent / 'drift_detection.py'

            result = subprocess.run(
                ['python', str(script_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode == 0:
                st.success('Continuous training pipeline completed successfully')
                logger.info('CT pipeline executed successfully')
                return True
            else:
                st.error(f'Pipeline failed: {result.stderr}')
                logger.error(f'CT pipeline failed: {result.stderr}')
                return False

    except subprocess.TimeoutExpired:
        st.error('Pipeline timeout: Execution took longer than 120 seconds')
        logger.error('CT pipeline timeout')
        return False
    except Exception as e:
        st.error(f'Failed to execute pipeline: {str(e)}')
        logger.error(f'CT pipeline error: {str(e)}')
        return False


def render_header() -> None:
    """
    Render the dashboard header with system status.

    Displays the title, description, and real-time API health status
    in a professional format.
    """
    st.title('Housing MLOps Dashboard')
    st.markdown(
        'Production-ready monitoring and control center for the housing price '
        'prediction model with continuous training capabilities.'
    )

    # System status
    col1, col2, col3 = st.columns(3)

    api_healthy, model_status = check_api_health()

    with col1:
        if api_healthy:
            st.success('✓ API Status: Online')
        else:
            st.error('✗ API Status: Offline')

    with col2:
        if model_status == 'loaded':
            st.success('✓ Model Status: Loaded')
        elif model_status == 'not_loaded':
            st.warning('⚠ Model Status: Not Loaded')
        else:
            st.error('✗ Model Status: Unknown')

    with col3:
        st.info(f'Dashboard Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    st.divider()


def render_prediction_section() -> None:
    """
    Render the prediction input form and results section.

    Provides a user-friendly interface for entering housing features,
    sending prediction requests, and displaying results with formatted
    price prediction and confidence information.
    """
    st.header('Price Prediction')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Property Characteristics')
        surface_m2 = st.slider(
            'Living Area (m²)',
            min_value=30.0,
            max_value=300.0,
            value=120.0,
            step=5.0,
        )
        nb_pieces = st.number_input(
            'Number of Rooms',
            min_value=1,
            max_value=10,
            value=3,
            step=1,
        )
        annee_construction = st.slider(
            'Construction Year',
            min_value=1950,
            max_value=2024,
            value=1995,
            step=1,
        )
        distance_centre_km = st.slider(
            'Distance to City Center (km)',
            min_value=0.5,
            max_value=50.0,
            value=5.0,
            step=0.5,
        )

    with col2:
        st.subheader('Energy & Amenities')
        dpe_energy_class = st.select_slider(
            'Energy Efficiency (DPE Class)',
            options=list(range(1, 8)),
            value=5,
            help='1 = Worst efficiency, 7 = Best efficiency',
        )
        has_balcony = st.selectbox(
            'Has Balcony?',
            options=[0, 1],
            format_func=lambda x: 'Yes' if x == 1 else 'No',
        )
        has_parking = st.selectbox(
            'Has Parking?',
            options=[0, 1],
            format_func=lambda x: 'Yes' if x == 1 else 'No',
        )

    # Prepare features dictionary
    features = {
        'Surface_m2': surface_m2,
        'Nb_Pieces': nb_pieces,
        'Annee_Construction': annee_construction,
        'Distance_Centre_km': distance_centre_km,
        'DPE_Energy_Class': dpe_energy_class,
        'Has_Balcony': has_balcony,
        'Has_Parking': has_parking,
    }

    # Prediction button
    if st.button('Get Price Prediction', use_container_width=True, type='primary'):
        prediction_result = send_prediction_request(features)

        if prediction_result:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)

            price = prediction_result.get('predicted_price_k_eur', 0)
            st.metric(
                label='Predicted Price',
                value=f'{price:,.0f} k EUR',
                delta=f'Range: {price * 0.85:.0f} - {price * 1.15:.0f} k EUR',
            )

            st.markdown('</div>', unsafe_allow_html=True)

            # Display input features for reference
            with st.expander('View Input Features'):
                features_df = pd.DataFrame([features]).T
                features_df.columns = ['Value']
                st.dataframe(features_df)

    st.divider()


def render_drift_monitoring_section() -> None:
    """
    Render the data drift monitoring section.

    Displays the latest drift detection report (if available) and provides
    information about the model's stability. This section helps operators
    understand when model retraining might be necessary.
    """
    st.header('Data Drift Monitoring')

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader('Latest Drift Report')

        latest_report = get_latest_drift_report()

        if latest_report:
            st.info(f'Report generated: {latest_report.name}')

            try:
                with open(latest_report, 'r', encoding='utf-8') as f:
                    report_html = f.read()

                st.components.v1.html(report_html, height=600, scrolling=True)

            except Exception as e:
                st.error(f'Failed to load report: {str(e)}')
                st.markdown(f'[Open report in browser]({latest_report})')
        else:
            st.warning('No drift reports available yet. Run the pipeline to generate one.')

    with col2:
        st.subheader('Actions')
        if st.button(
            'Run Continuous Training',
            use_container_width=True,
            type='secondary',
            help='Manually trigger drift detection and retraining if needed',
        ):
            success = trigger_continuous_training()
            if success:
                st.rerun()

    st.divider()


def render_system_info() -> None:
    """
    Render system information section.

    Displays technical details about the MLOps pipeline, model versions,
    and configuration for operational transparency.
    """
    st.header('System Information')

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.markdown(
            '<div class="info-box">'
            '<strong>Model Pipeline</strong><br>'
            'MLOps Level 1/2<br>'
            'Continuous Training Enabled<br>'
            'Self-Healing Architecture'
            '</div>',
            unsafe_allow_html=True
        )

    with info_col2:
        st.markdown(
            '<div class="info-box">'
            '<strong>Components</strong><br>'
            'FastAPI Backend: /predict<br>'
            'Evidently AI: Drift Detection<br>'
            'MLflow: Experiment Tracking'
            '</div>',
            unsafe_allow_html=True
        )

    with info_col3:
        st.markdown(
            '<div class="info-box">'
            '<strong>Data Sources</strong><br>'
            'immobilier_france.csv<br>'
            'Synthetic Housing Dataset<br>'
            '7 Property Features'
            '</div>',
            unsafe_allow_html=True
        )


def main() -> None:
    """
    Main entry point for the Streamlit dashboard application.

    Orchestrates all dashboard sections in logical order:
    1. Header with system status
    2. Prediction interface
    3. Drift monitoring and CT controls
    4. System information
    """
    render_header()
    render_prediction_section()
    render_drift_monitoring_section()
    render_system_info()


if __name__ == '__main__':
    main()