"""
Tableau de bord Streamlit interactif pour la surveillance et le contrôle MLOps.
Interactive Streamlit dashboard for MLOps monitoring and control.
"""

import streamlit as st
import requests
import subprocess
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import pandas as pd
import logging
from dotenv import load_dotenv

# Charger les variables d'environnement (Load environment variables)
load_dotenv()

# Configuration de la journalisation
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paramètres de configuration (support Docker/Env)
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')
reports_dir_env = os.getenv('DRIFT_REPORTS_DIR', 'drift_reports')
data_file_env = os.getenv('DATA_FILE_PATH', 'data/immobilier_france.csv')

DRIFT_REPORTS_DIR = Path(__file__).parent.parent / reports_dir_env
DATA_FILE = Path(__file__).parent.parent / data_file_env

# Traductions linguistiques / Translations
TRANSLATIONS = {
    'en': {
        'title': 'Housing MLOps Dashboard',
        'subtitle': 'Production-ready monitoring and control center for the housing price prediction model.',
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
        'price_range': 'Estimated Range',
        'view_features': 'View Input Features',
        'out_of_domain': 'Outside the training domain: this prediction is an extrapolation.',
        'drift_monitoring': 'Data Drift Monitoring',
        'latest_report': 'Latest Drift Report',
        'report_generated': 'Report generated',
        'actions': 'Actions',
        'run_ct': 'Run Continuous Training',
        'run_ct_help': 'Manually trigger drift detection and retraining',
        'no_reports': 'No drift reports available yet.',
        'system_info': 'System Information & Data',
        'model_pipeline': 'Model Pipeline',
        'mlops_level': 'MLOps Level 1/2',
        'ct_enabled': 'Continuous Training Enabled',
        'self_healing': 'Self-Healing Architecture',
        'components': 'Components',
        'api_backend': 'FastAPI Backend: /predict',
        'drift_detection': 'Evidently AI: Drift Detection',
        'experiment_tracking': 'MLflow: Experiment Tracking',
        'dataset_insights': 'Dataset Insights (Real Data)',
        'total_properties': 'Total Properties',
        'avg_price': 'Average Price',
        'price_distribution': 'Price Distribution (k€)',
        'request_timeout': 'Request timeout: API did not respond',
        'connection_error': 'Connection error: Cannot reach API',
        'ensure_api': 'Ensure the FastAPI server is running.',
        'unexpected_error': 'Unexpected error',
        'pipeline_running': 'Running drift detection and continuous training...',
        'pipeline_success': 'Continuous training pipeline completed successfully',
        'pipeline_failed': 'Pipeline failed',
        'pipeline_timeout': 'Pipeline timeout: Execution took too long',
        'api_error': 'API Error',
        'data_missing': 'Data file not found. Run generate_data.py to see insights.',
    },
    'fr': {
        'title': 'Tableau de Bord MLOps Immobilier',
        'subtitle': 'Centre de contrôle en production pour le modèle de prédiction des prix immobiliers.',
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
        'price_range': 'Fourchette Estimée',
        'view_features': 'Afficher les Caractéristiques',
        'out_of_domain': 'Hors du domaine d\'entraînement : cette prédiction est une extrapolation.',
        'drift_monitoring': 'Surveillance de la Dérive de Données',
        'latest_report': 'Dernier Rapport de Dérive',
        'report_generated': 'Rapport généré',
        'actions': 'Actions',
        'run_ct': 'Exécuter l\'Entraînement Continu',
        'run_ct_help': 'Déclencher manuellement le réentraînement',
        'no_reports': 'Aucun rapport de dérive disponible.',
        'system_info': 'Informations Système et Données',
        'model_pipeline': 'Pipeline de Modèle',
        'mlops_level': 'Niveau MLOps 1/2',
        'ct_enabled': 'Entraînement Continu Activé',
        'self_healing': 'Architecture Auto-Cicatrisante',
        'components': 'Composants',
        'api_backend': 'Backend FastAPI : /predict',
        'drift_detection': 'Evidently AI : Détection de Dérive',
        'experiment_tracking': 'MLflow : Suivi des Expériences',
        'dataset_insights': 'Aperçu des Données (Données Réelles)',
        'total_properties': 'Propriétés',
        'avg_price': 'Prix Moyen',
        'price_distribution': 'Distribution des Prix (k€)',
        'request_timeout': 'Délai d\'attente dépassé pour l\'API',
        'connection_error': 'Erreur : Impossible d\'accéder à l\'API',
        'ensure_api': 'Assurez-vous que le serveur FastAPI est actif.',
        'unexpected_error': 'Erreur inattendue',
        'pipeline_running': 'Exécution de l\'entraînement continu...',
        'pipeline_success': 'Pipeline d\'entraînement complété avec succès',
        'pipeline_failed': 'Échec du pipeline',
        'pipeline_timeout': 'Délai d\'attente du pipeline dépassé',
        'api_error': 'Erreur API',
        'data_missing': 'Fichier de données introuvable. Exécutez generate_data.py.',
    }
}

# Configuration de la page
st.set_page_config(
    page_title='Housing MLOps Dashboard',
    page_icon='Home',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Sélecteur de langue
if 'language' not in st.session_state:
    st.session_state.language = 'fr'

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2558/2558055.png", width=50) # Generic house icon
    st.markdown("### Configuration")
    language = st.radio(
        'Language / Langue',
        options=['en', 'fr'],
        format_func=lambda x: 'English' if x == 'en' else 'Français',
        horizontal=True
    )
    st.session_state.language = language

def t(key: str) -> str:
    """Get translated text"""
    return TRANSLATIONS[st.session_state.language].get(key, key)

# Design CSS (Premium Dark/Light adaptation)
st.markdown("""
    <style>
        .metric-container {
            background-color: rgba(240, 242, 246, 0.1);
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            border: 1px solid rgba(128, 128, 128, 0.2);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .success-box {
            background-color: rgba(40, 167, 69, 0.15);
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #28a745;
            margin-bottom: 20px;
        }
        .info-box {
            background-color: rgba(23, 162, 184, 0.1);
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #17a2b8;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)


def check_api_health() -> Tuple[bool, Optional[str]]:
    """
    Poll the API's /health endpoint to determine whether it is reachable and
    whether it reports a loaded model.
    Interroge le point de terminaison /health de l'API pour déterminer si
    elle est joignable et si elle signale un modèle chargé.

    Returns:
        Tuple (is_healthy, model_status): is_healthy is True only on an
        HTTP 200 response; model_status is the API's reported status
        ('loaded'/'not_loaded'/'unknown') or None if the API could not be
        reached at all. (Tuple (en_ligne, statut_modele) ; en_ligne vaut
        True uniquement sur une réponse HTTP 200 ; statut_modele est le
        statut renvoyé par l'API, ou None si l'API est injoignable.)
    """
    try:
        response = requests.get(f'{API_BASE_URL}/health', timeout=2)
        if response.status_code == 200:
            return True, response.json().get('model_status', 'unknown')
    except Exception:
        # Volontairement large (API injoignable, timeout, DNS, JSON invalide, ...) :
        # toute panne de sondage de santé doit dégrader gracieusement vers "hors ligne",
        # jamais interrompre le rendu du tableau de bord. Exclut explicitement
        # BaseException (KeyboardInterrupt/SystemExit) contrairement à un `except:` nu.
        # Deliberately broad (unreachable API, timeout, DNS, invalid JSON, ...): any
        # health-poll failure must degrade gracefully to "offline", never crash the
        # dashboard render. Explicitly excludes BaseException (KeyboardInterrupt/
        # SystemExit), unlike a bare `except:`.
        pass
    return False, None


def get_latest_drift_report() -> Optional[Path]:
    """
    Locate the most recently generated drift report, if any.
    Localise le rapport de dérive généré le plus récemment, s'il en existe.

    Returns:
        Path to the newest 'drift_report_*.html' file in DRIFT_REPORTS_DIR
        (by modification time), or None if the directory does not exist or
        contains no report yet. (Chemin du rapport 'drift_report_*.html' le
        plus récent (par date de modification), ou None si le répertoire
        n'existe pas ou ne contient encore aucun rapport.)
    """
    if not DRIFT_REPORTS_DIR.exists():
        return None
    report_files = sorted(DRIFT_REPORTS_DIR.glob('drift_report_*.html'), key=lambda p: p.stat().st_mtime, reverse=True)
    return report_files[0] if report_files else None


def send_prediction_request(features: Dict) -> Optional[Dict]:
    """
    Send a prediction request to the API and surface any failure to the
    user via the Streamlit UI, rather than letting it propagate.
    Envoie une requête de prédiction à l'API et signale tout échec à
    l'utilisateur via l'interface Streamlit, plutôt que de le laisser se
    propager.

    Args:
        features: Housing feature values matching the API's HousingFeatures
                  schema. (Valeurs des caractéristiques du bien, conformes
                  au schéma HousingFeatures de l'API.)

    Returns:
        The API's JSON response as a dict on success (HTTP 200), or None on
        any failure (non-200 response, connection error, or unexpected
        exception) -- the failure itself is reported via st.error/st.info.
        (La réponse JSON de l'API sous forme de dict en cas de succès
        (HTTP 200), ou None en cas d'échec (réponse non-200, erreur de
        connexion, exception imprévue) -- l'échec lui-même est signalé via
        st.error/st.info.)
    """
    try:
        response = requests.post(f'{API_BASE_URL}/predict', json=features, timeout=5)
        if response.status_code == 200:
            return response.json()
        st.error(f'{t("api_error")} {response.status_code}')
    except requests.exceptions.ConnectionError:
        st.error(t('connection_error'))
        st.info(t('ensure_api'))
    except Exception as e:
        st.error(f'{t("unexpected_error")}: {str(e)}')
    return None


def trigger_continuous_training() -> bool:
    """
    Manually trigger the drift-detection / continuous-training pipeline by
    running src/drift_detection.py as a subprocess, and report the outcome
    in the Streamlit UI.
    Déclenche manuellement le pipeline de détection de dérive / entraînement
    continu en exécutant src/drift_detection.py en sous-processus, et
    signale le résultat dans l'interface Streamlit.

    Returns:
        True if the subprocess exited with code 0 (success), False on a
        non-zero exit code, a timeout (120s), or any other exception --
        each failure path also displays an explanatory message via
        st.error. (True si le sous-processus s'est terminé avec le code 0
        (succès), False en cas de code de sortie non nul, de dépassement de
        délai (120 s), ou de toute autre exception -- chaque cas d'échec
        affiche également un message explicatif via st.error.)
    """
    try:
        with st.spinner(t('pipeline_running')):
            script_path = Path(__file__).parent / 'drift_detection.py'
            result = subprocess.run(['python', str(script_path)], capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                st.success(t('pipeline_success'))
                return True
            st.error(f'{t("pipeline_failed")}: {result.stderr}')
    except subprocess.TimeoutExpired:
        st.error(t('pipeline_timeout'))
    except Exception as e:
        st.error(f'{t("unexpected_error")}: {str(e)}')
    return False


def render_header() -> None:
    """
    Render the dashboard's header: title, subtitle, and a live 3-column
    status strip (API reachability, model-loaded status, last-refresh time).
    Affiche l'en-tête du tableau de bord : titre, sous-titre, et une bande
    de statut en direct sur 3 colonnes (joignabilité de l'API, statut de
    chargement du modèle, heure de dernier rafraîchissement).
    """
    st.title(t('title'))
    st.markdown(f"*{t('subtitle')}*")
    
    col1, col2, col3 = st.columns(3)
    api_healthy, model_status = check_api_health()

    with col1:
        if api_healthy:
            st.success(f"{t('api_status')}")
        else:
            st.error(f"{t('api_offline')}")

    with col2:
        if model_status == 'loaded':
            st.success(f"{t('model_loaded')}")
        elif model_status == 'not_loaded':
            st.warning(f"{t('model_not_loaded')}")
        else:
            st.error(f"{t('model_unknown')}")

    with col3:
        st.info(f"{t('dashboard_updated')}: {datetime.now().strftime('%H:%M:%S')}")
    st.divider()


def render_prediction_section() -> None:
    """
    Render the price-prediction form (7 housing features matching the
    API's HousingFeatures schema) and, on submission, display the predicted
    price along with a client-computed +/-15% estimation range.
    Affiche le formulaire de prédiction de prix (7 caractéristiques
    immobilières conformes au schéma HousingFeatures de l'API) et, à la
    soumission, affiche le prix prédit accompagné d'une fourchette
    d'estimation +/-15 % calculée côté client.
    """
    st.header(f"{t('price_prediction')}")

    # Utilisation d'un formulaire pour éviter les rechargements intempestifs
    # Using a form to avoid unnecessary reloads
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(t('property_characteristics'))
            # L'attribut 'key' empêche Streamlit de recréer l'élément (et de causer DuplicateElementId) quand la langue change
            # The 'key' attribute prevents Streamlit from recreating the element (causing DuplicateElementId) when the language changes
            surface_m2 = st.slider(t('living_area'), 30.0, 300.0, 120.0, 5.0, key="slider_surface_m2")
            nb_pieces = st.number_input(t('num_rooms'), 1, 10, 3, 1, key="num_pieces")
            annee_construction = st.slider(t('construction_year'), 1950, 2024, 1995, 1, key="slider_annee")
            distance_centre_km = st.slider(t('distance_city'), 0.5, 50.0, 5.0, 0.5, key="slider_distance")

        with col2:
            st.subheader(t('energy_amenities'))
            dpe_energy_class = st.select_slider(t('energy_efficiency'), options=list(range(1, 8)), value=5, help=t('energy_best'), key="slider_dpe")
            has_balcony = st.selectbox(t('has_balcony'), options=[0, 1], format_func=lambda x: t('yes') if x == 1 else t('no'), key="select_balcony")
            has_parking = st.selectbox(t('has_parking'), options=[0, 1], format_func=lambda x: t('yes') if x == 1 else t('no'), key="select_parking")

        submit_button = st.form_submit_button(label=t('get_prediction'), type='primary', use_container_width=True)

    if submit_button:
        features = {
            'Surface_m2': surface_m2,
            'Nb_Pieces': nb_pieces,
            'Annee_Construction': annee_construction,
            'Distance_Centre_km': distance_centre_km,
            'DPE_Energy_Class': dpe_energy_class,
            'Has_Balcony': has_balcony,
            'Has_Parking': has_parking,
        }
        
        prediction_result = send_prediction_request(features)
        if prediction_result:
            price = prediction_result.get('predicted_price_k_eur', 0)
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.metric(
                label=t('predicted_price'),
                value=f"{price:,.0f} k€",
                delta=f"{t('price_range')}: {price * 0.85:.0f} - {price * 1.15:.0f} k€"
            )
            st.markdown('</div>', unsafe_allow_html=True)

            if prediction_result.get('out_of_training_domain'):
                st.warning(
                    t('out_of_domain') + ' ' + ' | '.join(prediction_result.get('domain_warnings', []))
                )

            with st.expander(t('view_features')):
                st.json(features)
    st.divider()


def render_drift_monitoring_section() -> None:
    """
    Render the drift-monitoring section: embeds the latest Evidently AI
    drift report (if any) and exposes the manual "Run Continuous Training"
    action button (see trigger_continuous_training()).
    Affiche la section de surveillance de la dérive : intègre le dernier
    rapport de dérive Evidently AI (le cas échéant) et expose le bouton
    d'action manuelle « Run Continuous Training »
    (voir trigger_continuous_training()).
    """
    st.header(f"{t('drift_monitoring')}")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(t('latest_report'))
        latest_report = get_latest_drift_report()
        if latest_report:
            st.info(f"{t('report_generated')}: {latest_report.name}")
            try:
                with open(latest_report, 'r', encoding='utf-8') as f:
                    st.components.v1.html(f.read(), height=500, scrolling=True)
            except Exception as e:
                st.error(f"{t('unexpected_error')}: {str(e)}")
        else:
            st.warning(t('no_reports'))

    with col2:
        st.subheader(t('actions'))
        if st.button(t('run_ct'), use_container_width=True, type='secondary', help=t('run_ct_help')):
            if trigger_continuous_training():
                st.rerun()
    st.divider()


def render_system_info() -> None:
    """
    Render dataset insights (row count, average price, price distribution)
    computed from the real dataset when available, plus a static summary of
    the pipeline's components and MLOps capabilities.
    Affiche des indicateurs sur le jeu de données (nombre de lignes, prix
    moyen, distribution des prix) calculés à partir du jeu de données réel
    lorsqu'il est disponible, ainsi qu'un résumé statique des composants du
    pipeline et des capacités MLOps.
    """
    st.header(f"{t('system_info')}")

    # Charger les vraies données pour affichage
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
        
        st.markdown(f"#### {t('dataset_insights')}")
        m1, m2, m3 = st.columns(3)
        m1.metric(t('total_properties'), f"{len(df):,}")
        m2.metric(t('avg_price'), f"{df['Prix_k_EUR'].mean():.0f} k€")
        m3.metric("Max Price", f"{df['Prix_k_EUR'].max():.0f} k€")
        
        st.markdown(f"**{t('price_distribution')}**")
        st.bar_chart(df['Prix_k_EUR'].value_counts(bins=20).sort_index())
    else:
        st.warning(t('data_missing'))

    st.markdown("---")
    
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.markdown(
            f'<div class="info-box">'
            f'<strong>{t("model_pipeline")}</strong><br>'
            f'• {t("mlops_level")}<br>'
            f'• {t("ct_enabled")}<br>'
            f'• {t("self_healing")}'
            f'</div>',
            unsafe_allow_html=True
        )

    with info_col2:
        st.markdown(
            f'<div class="info-box">'
            f'<strong>{t("components")}</strong><br>'
            f'• {t("api_backend")}<br>'
            f'• {t("drift_detection")}<br>'
            f'• {t("experiment_tracking")}'
            f'</div>',
            unsafe_allow_html=True
        )


def main() -> None:
    """
    Entry point for the Streamlit dashboard: renders the four sections in
    order (header/status, prediction form, drift monitoring, system info).
    Point d'entrée du tableau de bord Streamlit : affiche les quatre
    sections dans l'ordre (en-tête/statut, formulaire de prédiction,
    surveillance de la dérive, informations système).
    """
    render_header()
    render_prediction_section()
    render_drift_monitoring_section()
    render_system_info()


if __name__ == '__main__':
    main()