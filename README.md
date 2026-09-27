# mlops-transition-poc

PoC de transition DevOps → MLOps pour mémoire de Master (Big Data & IA) — « Du DevOps au MLOps, un nouveau paradigme ».

Le projet prédit le prix d'un bien immobilier français (en k€) à partir de sept caractéristiques, et démontre une boucle MLOps complète : entraînement suivi (MLflow), promotion champion/challenger, service de prédiction (FastAPI), détection de dérive (Evidently AI) avec réentraînement automatique, tableau de bord de supervision (Streamlit), le tout conteneurisé (Docker Compose).

## Architecture

```text
                     ┌───────────────────────────────────────────────┐
                     │                MLflow Server                  │
                     │   Tracking (runs, métriques) + Model Registry  │
                     │        alias 'champion' <-> versions           │
                     └───────────────▲─────────────────────▲─────────┘
                                     │ log/register          │ set alias
                                     │                        │
   generate_data.py ──► data/*.csv ──► src/train.py (challenger)
                                     │
                                     │ promotion gate (R² > champion + PROMOTION_MIN_GAIN)
                                     ▼
                        models/housing_model.pkl  (copie locale "miroir" du champion)
                                     │
                                     │ chargé au démarrage / après réentraînement
                                     ▼
   src/app.py (FastAPI /predict, /health) ◄──────────────► src/dashboard.py (Streamlit)
                                                                     │
                                                                     │ déclenche (bouton UI)
                                                                     ▼
                        src/drift_detection.py (Evidently AI DataDriftPreset)
                                     │
                                     │ si dérive > DRIFT_THRESHOLD
                                     ▼
                        trigger_retraining() ──► src/train.py (auto-cicatrisation)
```

Trois services conteneurisés (`docker-compose.yml`) : `mlflow` (serveur de tracking, port 5000),
`api` (FastAPI, port 8000), `dashboard` (Streamlit, port 8501), reliés par un réseau Docker
interne dédié (`mlops_net`).

**Limite architecturale documentée et assumée** : `trigger_retraining()` appelle
`run_pipeline()` sans lui transmettre les données courantes (potentiellement dérivées) sur
lesquelles la dérive vient d'être détectée — `run_pipeline()` recharge systématiquement
`data/immobilier_france.csv` depuis le disque. Concrètement, un réentraînement déclenché par
la dérive reproduit donc le même jeu d'entraînement (et, `random_state` étant fixé, le même
modèle) que le champion existant, et ne peut par construction jamais le battre strictement au
sens de la porte de promotion. Ce constat est analysé de façon critique dans le mémoire
(Chapitre IV) comme illustration des limites concrètes de l'automatisation MLOps sans pipeline
d'ingestion de données de production, et fait l'objet d'un test de non-régression exécutable
(`tests/test_integration_pipeline.py::test_full_self_healing_loop_champion_then_drift_then_retrain`).

## Résultats de référence

Obtenus lors d'une exécution de référence locale (Python 3.12, `random_state=42`,
paramètres par défaut de `.env.example`) ; reproductibles à l'identique grâce à la graine fixe
partagée par `generate_data.py` et `src/train.py` (voir `tests/test_train.py::test_train_model_strict_reproducibility_with_fixed_random_state`) :

- **R² du modèle champion** : 0,9780 (jeu de test, split 80/20)
- **Score de dérive** (perturbation de démonstration `Surface_m2 += 10`, `Prix_k_EUR *= 1.15`,
  `src/drift_detection.py::run_drift_detection`) : 0,25 (2 colonnes sur 8 en dérive, part lue dans
  `share_of_drifted_columns`) — au-dessus du seuil `DRIFT_THRESHOLD`
  (0,20), déclenchant le réentraînement automatique.
- **Suite de tests** : 86/86 tests passants (`pytest tests/ -v`), voir la section « Tests ».

## Stack technique

- `scikit-learn` (`LinearRegression`) pour le modèle de régression
- `MLflow` pour le tracking des expériences et le Model Registry (alias `champion`/`challenger`)
- `FastAPI` + `uvicorn` pour l'API de prédiction, validation stricte des entrées via `Pydantic`
- `Evidently AI` pour la détection de dérive des données (`DataDriftPreset`)
- `Streamlit` pour le tableau de bord de supervision et de contrôle
- `Docker` / `docker-compose` pour la conteneurisation (3 services : `mlflow`, `api`, `dashboard`)
- `GitHub Actions` pour l'intégration continue (tests, génération des données, entraînement, build Docker)

## Structure du projet

```text
memoire-mlops-demo/
├── src/
│   ├── train.py            # Pipeline d'entraînement + tracking MLflow + promotion champion/challenger
│   ├── app.py               # API FastAPI de prédiction (/predict, /health)
│   ├── drift_detection.py   # Détection de dérive Evidently AI + déclenchement du réentraînement
│   └── dashboard.py         # Tableau de bord Streamlit (prédiction, dérive, pilotage du réentraînement)
├── tests/                          # Suite pytest (86 tests, voir "Tests" ci-dessous)
│   ├── test_train.py               # Unitaires : entraînement, coefficients, reproductibilité
│   ├── test_app_robustness.py      # Robustesse : modèle manquant/corrompu, API 503
│   ├── test_data_schema.py         # Schéma et qualité du jeu de données
│   ├── test_api.py                 # Fonctionnels : API (bornes Pydantic, scénario métier)
│   ├── test_dashboard.py           # Fonctionnels : utilitaires du tableau de bord Streamlit
│   ├── test_drift.py                # Détection de dérive (logique pure, rapports simulés à la structure Evidently réelle)
│   └── test_integration_pipeline.py # Intégration E2E : boucle self-healing, API + modèle réel
├── data/
│   └── immobilier_france.csv  # Jeu de données synthétique (voir "Jeu de données" ci-dessous)
├── generate_data.py          # Génération du jeu de données synthétique
├── docker-compose.yml        # Services mlflow / api / dashboard
├── Dockerfile
├── requirements.txt
├── pytest.ini                 # Marqueur `integration` (tests d'intégration plus lents, isolables)
├── .env.example               # Modèle de configuration (copier en .env)
└── README.md
```

## Jeu de données

Le jeu de données (`data/immobilier_france.csv`, 500 lignes) est **synthétique**, généré par `generate_data.py` à partir d'une formule linéaire calibrée (7 variables : `Surface_m2`, `Nb_Pieces`, `Annee_Construction`, `Distance_Centre_km`, `DPE_Energy_Class`, `Has_Balcony`, `Has_Parking`) plus un bruit gaussien, avec une graine fixe (`random_state=42`) pour la reproductibilité.

Ce choix a été documenté explicitement : une intégration de données réelles (Demandes de Valeurs Foncières — DVF, data.gouv.fr) a été envisagée, mais nécessiterait un travail d'enrichissement significatif (les DVF ne comportent pas nativement de classe DPE, ni de variables balcon/parking, et un rapprochement fiable avec ces informations dépasse le cadre du présent PoC). Le jeu synthétique reste donc utilisé, avec une formule corrigée pour éviter tout artefact de saturation (voir `generate_data.py` pour le détail et la justification).

## Installation locale

1. Créer un environnement virtuel Python :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Installer les dépendances :

```powershell
pip install -r requirements.txt
```

3. Copier `.env.example` vers `.env` et ajuster si besoin :

```powershell
copy .env.example .env
```

`.env` n'est jamais versionné (voir `.gitignore`) ; cette étape est requise aussi bien pour
l'exécution locale que pour `docker-compose up` (chaque service Docker charge `.env` via
`env_file`).

## Utilisation

### Générer les données et entraîner le modèle

```powershell
python generate_data.py
python src/train.py
```

`train.py` entraîne un `LinearRegression`, journalise les métriques (MSE, RMSE, MAE, R²) et les paramètres dans MLflow, enregistre le modèle dans le Model Registry, puis ne le promeut à l'alias `champion` — et ne met à jour le fichier local servi par l'API (`models/housing_model.pkl`) — que s'il améliore le R² du champion existant (voir la section « Promotion champion/challenger »).

### Démarrer l'API

```powershell
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

#### `POST /predict`

Exemple de requête JSON :

```json
{
  "Surface_m2": 120.5,
  "Nb_Pieces": 3,
  "Annee_Construction": 1995,
  "Distance_Centre_km": 5.2,
  "DPE_Energy_Class": 5,
  "Has_Balcony": 1,
  "Has_Parking": 1
}
```

#### `GET /health`

Renvoie l'état de l'API et si le modèle est chargé.

### Démarrer le tableau de bord Streamlit

```powershell
streamlit run src/dashboard.py
```

Le tableau de bord permet d'obtenir une prédiction via l'API, de visualiser le dernier rapport de dérive Evidently, et de déclencher manuellement le pipeline de détection de dérive / réentraînement.

### Détection de dérive et réentraînement

```powershell
python src/drift_detection.py
```

Compare les données courantes à la référence d'entraînement avec Evidently AI (`DataDriftPreset`) ; si le score de dérive dépasse le seuil configuré (`DRIFT_THRESHOLD`, 0.2 par défaut), déclenche automatiquement `src/train.py`, dont la porte de promotion décide si le modèle réentraîné remplace le champion en service.

### Lancer avec Docker Compose

Nécessite un fichier `.env` à la racine (voir « Installation locale », étape 3) : les trois
services le chargent via `env_file`, et `docker-compose up` échoue s'il est absent.

```powershell
docker-compose up --build
```

Démarre les trois services : `mlflow` (serveur de tracking, port 5000), `api` (FastAPI, port
8000), `dashboard` (Streamlit, port 8501), reliés par le réseau interne `mlops_net` (voir
« Architecture »). Le service `dashboard` reçoit une surcharge explicite de
`MLFLOW_TRACKING_URI` (`http://mlflow:5000`) pour que le bouton « Run Continuous Training »
(qui exécute `src/drift_detection.py` en sous-processus) atteigne bien le serveur MLflow via le
réseau Docker interne, plutôt que la valeur `.env` par défaut (`http://localhost:5000`) qui
pointerait vers le conteneur `dashboard` lui-même.

## Promotion champion/challenger

Chaque exécution de `src/train.py` entraîne un modèle *challenger*. Il n'est promu (alias MLflow `champion` + copie vers `models/housing_model.pkl` servie par l'API) que si son R² dépasse celui du champion actuel d'au moins `PROMOTION_MIN_GAIN` (configurable, `0.0` par défaut = toute amélioration stricte), ou s'il s'agit du tout premier entraînement. Ce mécanisme évite qu'un réentraînement automatique (déclenché par la détection de dérive) ne dégrade silencieusement le modèle en production.

## Variables d'environnement

Voir `.env.example`. Principales variables : `API_HOST`/`API_PORT`, `DASHBOARD_PORT`, `MLFLOW_TRACKING_URI`/`MLFLOW_PORT`/`MLFLOW_BACKEND_STORE_URI`/`MLFLOW_DEFAULT_ARTIFACT_ROOT`, `DATA_FILE_PATH`, `MODELS_DIR`, `DRIFT_REPORTS_DIR`, `PROMOTION_MIN_GAIN`.

## Tests

Prérequis : le jeu de données doit exister avant de lancer la suite (plusieurs tests,
notamment l'intégralité de `test_data_schema.py`, lisent `data/immobilier_france.csv` et
échouent explicitement s'il est absent — voir « Générer les données et entraîner le modèle »
ci-dessus) :

```powershell
python generate_data.py
python src/train.py
pytest tests/ -v
```

**86 tests** au total, répartis en six catégories :

| Catégorie | Fichier(s) | Tests | Contenu |
|---|---|---|---|
| Unitaires (entraînement) | `test_train.py` | 16 | Schéma des données, métriques, signe des coefficients économiques (`Surface_m2`>0, `Distance_Centre_km`<0), monotonicité, reproductibilité stricte (`RANDOM_STATE=42`), dataset vide/manquant, porte de promotion champion/challenger, enregistrement des tags de gouvernance MLflow (`model_type`, `dataset_name`, `pipeline_stage`, `environment`) |
| Robustesse (API/modèle) | `test_app_robustness.py` | 6 | Modèle manquant/corrompu (`load_model`), réponse 503 si modèle absent, cas de contrôle avec modèle valide |
| Schéma de données | `test_data_schema.py` | 14 | Colonnes, bornes Pydantic, valeurs binaires, non-saturation du prix (garde-fou du bug historique) |
| Fonctionnels (API) | `test_api.py` | 28 | `/health`, `/predict`, validation des 7 bornes Pydantic (valeurs limites acceptées/rejetées), scénario métier nominal (T3 65 m² + parking), stabilité sur requêtes successives |
| Fonctionnels (dashboard) | `test_dashboard.py` | 9 | `check_api_health`, `get_latest_drift_report`, `send_prediction_request` (y compris gestion gracieuse de l'API hors-ligne) |
| Dérive | `test_drift.py` | 9 | Extraction de la part de colonnes en dérive (`share_of_drifted_columns`, repli sur `drift_share`), seuillage, déclenchement du réentraînement (logique pure, rapports simulés reproduisant la structure d'un vrai rapport Evidently 0.4.18) |
| **Intégration E2E** | `test_integration_pipeline.py` | 4 | Boucle self-healing complète (entraînement → dérive Evidently réelle de 0,25 → réentraînement → décision de promotion), non-régression du score de dérive (données identiques : score 0, aucun réentraînement), promotion positive avec challenger réellement meilleur, réactivité API/modèle aux changements d'artefact |

Les 4 tests d'intégration exécutent un vrai calcul Evidently AI et un vrai registre MLflow
(redirigé vers une base SQLite temporaire, isolée du `mlflow.db` réel — voir la fixture
`isolated_mlops_environment`) : plus lents que le reste de la suite, ils sont marqués
`@pytest.mark.integration` (déclaré dans `pytest.ini`). Pour une boucle de développement rapide
sans eux :

```powershell
pytest tests/ -v -m "not integration"
```

## Sécurité et bonnes pratiques

- Utilisateur non-root dans le conteneur Docker
- Validation stricte des données entrantes avec Pydantic
- Pas de secrets codés en dur : configuration via `.env` (non versionné) et `python-dotenv`
- Gestion des dépendances épinglées via `requirements.txt`
- Actions GitHub épinglées par SHA complet en CI

## License

Ce projet est sous licence MIT. Voir le fichier `LICENSE`.
