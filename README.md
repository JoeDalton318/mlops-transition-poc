# mlops-transition-poc — preuve de concept MLOps

Preuve de concept réalisée dans le cadre du mémoire de Mastère « Du DevOps au MLOps, un nouveau paradigme » (IPSSI, Mastère Développement, Big Data & Intelligence Artificielle).

Le projet prédit le prix d'un bien immobilier français, exprimé en milliers d'euros, à partir de sept caractéristiques. Il n'a pas pour objet la performance du modèle : il sert à observer, en conditions contrôlées, ce qu'exige une boucle MLOps complète — entraînement suivi, promotion champion/challenger, service de prédiction, détection de dérive, réentraînement conditionnel et supervision.

---

## Table des matières

1. [Contexte et problématique](#1-contexte-et-problématique)
2. [Objectifs](#2-objectifs)
3. [Fonctionnalités](#3-fonctionnalités)
4. [Architecture](#4-architecture)
5. [Structure du dépôt](#5-structure-du-dépôt)
6. [Technologies et dépendances](#6-technologies-et-dépendances)
7. [Prérequis](#7-prérequis)
8. [Installation](#8-installation)
9. [Configuration](#9-configuration)
10. [Lancement](#10-lancement)
11. [Tests](#11-tests)
12. [Utilisation](#12-utilisation)
13. [Pipeline de données et de modèle](#13-pipeline-de-données-et-de-modèle)
14. [Entraînement, évaluation, promotion](#14-entraînement-évaluation-promotion)
15. [Déploiement](#15-déploiement)
16. [Sécurité et gestion des secrets](#16-sécurité-et-gestion-des-secrets)
17. [Limites connues](#17-limites-connues)
18. [Pistes d'amélioration](#18-pistes-damélioration)
19. [Procédure de vérification depuis un clone propre](#19-procédure-de-vérification-depuis-un-clone-propre)
20. [Dépannage](#20-dépannage)
21. [Intégration continue](#21-intégration-continue)
22. [Références](#22-références)

---

## 1. Contexte et problématique

Le DevOps repose sur une hypothèse implicite : le comportement d'une application est spécifié par son code source. Un système d'apprentissage automatique contredit cette hypothèse. Son comportement est appris à partir de données, sa justesse ne s'estime que statistiquement, et il peut se dégrader en production sans qu'aucune ligne de code n'ait changé.

Ce dépôt met cette hypothèse à l'épreuve sur un cas minimal mais complet : que faut-il, au-delà du code, pour reconstruire un modèle en service, décider de sa mise en production, détecter sa dégradation et y réagir ?

## 2. Objectifs

**Fonctionnels**

- Servir une prédiction de prix immobilier à partir de sept caractéristiques, via une API HTTP.
- Superviser le système et déclencher un réentraînement depuis une interface web.
- Détecter un changement de distribution des données d'entrée et archiver un rapport daté.

**Techniques**

- Tracer chaque entraînement : paramètres, métriques, modèle, étiquettes de gouvernance.
- N'installer un nouveau modèle qu'après comparaison chiffrée avec le modèle en service.
- Rendre chaque résultat reproductible à graine fixée, et vérifiable par des tests automatisés.
- Conteneuriser l'ensemble et exécuter une chaîne d'intégration continue à chaque modification.

## 3. Fonctionnalités

| Fonctionnalité | Point d'entrée | Détail |
|---|---|---|
| Génération du jeu de données | `generate_data.py` | 500 lignes synthétiques, graine fixe `random_state=42` |
| Entraînement et suivi | `src/train.py` | `LinearRegression`, journalisation MLflow, enregistrement au Model Registry |
| Porte de promotion | `src/train.py::promote_challenger_if_better` | Le challenger ne devient champion que s'il bat strictement le R² du champion |
| API de prédiction | `src/app.py` | FastAPI : `GET /`, `GET /health`, `POST /predict`, documentation interactive sur `/docs` |
| Contrôle du domaine de validité | `src/app.py::check_training_domain` | Chaque réponse indique si la demande sort du domaine couvert par les données d'entraînement |
| Détection de dérive | `src/drift_detection.py` | Evidently AI `DataDriftPreset`, rapport HTML horodaté, seuil du projet 0,20 |
| Réentraînement conditionnel | `src/drift_detection.py::trigger_retraining` | Appelle le pipeline d'entraînement lorsque la dérive dépasse le seuil |
| Tableau de bord | `src/dashboard.py` | Streamlit : état des services, prédiction, dernier rapport de dérive, bouton de réentraînement |
| Conteneurisation | `docker-compose.yml` | Trois services : `mlflow`, `api`, `dashboard` |
| Intégration continue | `.github/workflows/mlops-pipeline.yml` | Lint, génération des données, entraînement, tests, construction et vérification de l'image |

## 4. Architecture

```text
                     ┌───────────────────────────────────────────────┐
                     │                MLflow Server                  │
                     │   Tracking (runs, métriques) + Model Registry │
                     │        alias 'champion' <-> versions          │
                     └───────────────▲─────────────────────▲─────────┘
                                     │ log/register        │ set alias
                                     │                     │
   generate_data.py ──► data/*.csv ──► src/train.py (challenger)
                                     │
                                     │ porte de promotion (R² > champion + PROMOTION_MIN_GAIN)
                                     ▼
                        models/housing_model.pkl  (copie locale « miroir » du champion)
                                     │
                                     │ chargé au démarrage de l'API
                                     ▼
   src/app.py (FastAPI /predict, /health) ◄──────────────► src/dashboard.py (Streamlit)
                                                                     │
                                                                     │ déclenche (bouton UI)
                                                                     ▼
                        src/drift_detection.py (Evidently AI DataDriftPreset)
                                     │
                                     │ si part de colonnes en dérive > DRIFT_THRESHOLD
                                     ▼
                        trigger_retraining() ──► src/train.py
```

Trois services conteneurisés reliés par le réseau interne `mlops_net` : `mlflow` (port 5000), `api` (port 8000), `dashboard` (port 8501).

Deux choix de conception méritent d'être signalés :

- **L'API sert une copie locale du champion** (`models/housing_model.pkl`) plutôt que de résoudre l'alias MLflow à chaque requête. Le service continue de répondre si MLflow est indisponible, au prix d'une mise à jour qui n'est effective qu'au redémarrage de l'API.
- **Le service `dashboard` reçoit une surcharge explicite de `MLFLOW_TRACKING_URI`** (`http://mlflow:5000`). Sans elle, le sous-processus de réentraînement lancé depuis l'interface viserait `localhost:5000` à l'intérieur de son propre conteneur, où aucun serveur MLflow n'écoute.

## 5. Structure du dépôt

```text
mlops-transition-poc/
├── src/
│   ├── train.py              # Entraînement + tracking MLflow + porte de promotion
│   ├── app.py                # API FastAPI (/, /health, /predict)
│   ├── drift_detection.py    # Détection de dérive Evidently + déclenchement du réentraînement
│   └── dashboard.py          # Tableau de bord Streamlit
├── tests/                    # 89 tests pytest (détail en section 11)
│   ├── test_train.py
│   ├── test_api.py
│   ├── test_app_robustness.py
│   ├── test_data_schema.py
│   ├── test_dashboard.py
│   ├── test_drift.py
│   └── test_integration_pipeline.py
├── .github/workflows/mlops-pipeline.yml
├── generate_data.py          # Génération du jeu de données synthétique
├── docker-compose.yml        # Services mlflow / api / dashboard
├── Dockerfile                # Image unique partagée par les trois services
├── requirements.txt          # Dépendances épinglées
├── pytest.ini                # Marqueur `integration`
├── .env.example              # Modèle de configuration, à copier en .env
├── LICENSE                   # MIT
└── README.md
```

Les répertoires suivants sont **générés à l'exécution et absents du dépôt** (voir `.gitignore`) : `data/`, `models/`, `mlruns/`, `drift_reports/`, ainsi que le fichier `mlflow.db`. C'est une conséquence assumée du choix de ne versionner ni données ni artefacts binaires ; la section 19 en tient compte pas à pas.

## 6. Technologies et dépendances

| Domaine | Outil | Version épinglée |
|---|---|---|
| Modèle | scikit-learn | 1.3.2 |
| Suivi et registre | MLflow | 2.10.0 |
| Dérive | Evidently AI | 0.4.18 |
| API | FastAPI / uvicorn / Pydantic | 0.103.1 / 0.23.2 / 2.6.0 |
| Interface | Streamlit | 1.40.2 |
| Données | pandas / numpy | 2.2.3 / 1.26.4 |
| Tests | pytest / httpx | 8.3.4 / 0.27.0 |
| Base de suivi | SQLAlchemy | 2.0.50 |

`SQLAlchemy` est épinglé volontairement : la version 2.1 a supprimé `FallbackAsyncAdaptedQueuePool`, que MLflow 2.10.0 importe au démarrage de son magasin de suivi. Sans cet épinglage, l'entraînement échoue à l'installation d'une version récente.

La liste complète et faisant foi est `requirements.txt`.

## 7. Prérequis

| Prérequis | Version vérifiée | Comment contrôler |
|---|---|---|
| Python | 3.12 (3.12.10 en local, 3.12.14 en CI) | `python --version` |
| Git | toute version récente | `git --version` |
| Docker Desktop avec Compose v2 | 29.7.2 en local | `docker --version` puis `docker compose version` |

Docker n'est nécessaire que pour la section 15. L'ensemble du pipeline (données, entraînement, API, tableau de bord, dérive, tests) fonctionne sans Docker.

Prévoir environ 2 Go d'espace disque pour l'environnement virtuel et l'image Docker, et un accès réseau pour l'installation des dépendances.

## 8. Installation

```powershell
git clone https://github.com/JoeDalton318/mlops-transition-poc.git
cd mlops-transition-poc
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Sous Linux ou macOS, remplacer les deux dernières lignes d'activation par :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 9. Configuration

Copier le modèle fourni, puis l'ajuster si nécessaire :

```powershell
copy .env.example .env
```

```bash
cp .env.example .env
```

**Le fichier `.env` est obligatoire pour Docker** : les trois services le chargent via `env_file`, et `docker compose` refuse de démarrer s'il est absent. Pour une exécution purement locale, chaque variable possède la même valeur par défaut dans le code : le fichier est alors facultatif, mais recommandé pour éviter toute divergence.

| Variable | Défaut | Utilisée par | Obligatoire |
|---|---|---|---|
| `API_HOST` | `0.0.0.0` | `docker-compose.yml` (service `api`) | non |
| `API_PORT` | `8000` | `docker-compose.yml` (service `api`) | non |
| `API_BASE_URL` | `http://localhost:8000` | `src/dashboard.py` | non |
| `DASHBOARD_PORT` | `8501` | `docker-compose.yml` (service `dashboard`) | non |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | client MLflow (`src/train.py`) | non |
| `MLFLOW_PORT` | `5000` | `docker-compose.yml` (service `mlflow`) | non |
| `MLFLOW_BACKEND_STORE_URI` | `sqlite:///mlflow.db` | serveur MLflow | non |
| `MLFLOW_DEFAULT_ARTIFACT_ROOT` | `./mlruns` | serveur MLflow | non |
| `DATA_FILE_PATH` | `data/immobilier_france.csv` | `train.py`, `drift_detection.py`, `dashboard.py` | non |
| `MODELS_DIR` | `models` | `train.py`, `app.py` | non |
| `DRIFT_REPORTS_DIR` | `drift_reports` | `drift_detection.py`, `dashboard.py` | non |
| `PROMOTION_MIN_GAIN` | `0.0` | `src/train.py` | non |

Deux seuils ne sont **pas** configurables par variable d'environnement et se modifient dans le code :

- `DRIFT_THRESHOLD = 0.2` dans `src/drift_detection.py` : part de colonnes en dérive au-delà de laquelle le réentraînement est déclenché ;
- `RANDOM_STATE = 42` dans `src/train.py` et `generate_data.py` : graine de reproductibilité.

## 10. Lancement

### Sans Docker

Dans l'ordre, depuis la racine du dépôt et l'environnement virtuel activé :

```powershell
python generate_data.py          # crée data/immobilier_france.csv
python src/train.py              # entraîne, journalise, promeut le premier champion
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

Dans un second terminal :

```powershell
streamlit run src/dashboard.py
```

Et, si l'on souhaite l'interface MLflow :

```powershell
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

### Avec Docker

```powershell
docker compose up --build
```

**Prérequis impératif** : `python src/train.py` doit avoir été exécuté au moins une fois avant la construction de l'image, car le `Dockerfile` contient `COPY mlruns /app/mlruns` et ce répertoire n'existe pas dans un dépôt fraîchement cloné. Sans cela, la construction échoue avec un message du type `"/mlruns": not found`.

Interfaces disponibles une fois les trois services démarrés :

- MLflow : <http://localhost:5000>
- API : <http://localhost:8000> — documentation interactive sur <http://localhost:8000/docs>
- Tableau de bord : <http://localhost:8501>

Arrêt propre :

```powershell
docker compose down
```

## 11. Tests

Le jeu de données doit exister avant la suite : plusieurs tests lisent `data/immobilier_france.csv`, absent du dépôt par choix.

```powershell
python generate_data.py
python src/train.py
pytest tests/ -v
```

Pour une boucle de développement rapide, sans les tests d'intégration :

```powershell
pytest tests/ -v -m "not integration"
```

**89 tests**, répartis en sept fichiers :

| Fichier | Tests | Ce qui est vérifié |
|---|---|---|
| `test_api.py` | 31 | `/health`, `/predict`, les 7 bornes Pydantic (valeurs limites acceptées et rejetées), scénario métier, stabilité sur requêtes successives, cohérence de l'ordre des variables entre entraînement et inférence, signalement d'extrapolation |
| `test_train.py` | 16 | Chargement des données, métriques, signe et monotonicité des coefficients, reproductibilité stricte, jeu vide ou absent, porte de promotion, étiquettes de gouvernance |
| `test_data_schema.py` | 14 | Colonnes, bornes, valeurs binaires, non-saturation de la cible (garde-fou d'un bug historique du générateur) |
| `test_dashboard.py` | 9 | `check_api_health`, `get_latest_drift_report`, `send_prediction_request`, dégradation gracieuse quand l'API est hors ligne |
| `test_drift.py` | 9 | Extraction de la part de colonnes en dérive, seuillage strict, repli sur `drift_share`, déclenchement du réentraînement |
| `test_app_robustness.py` | 6 | Modèle manquant ou corrompu, réponse 503, cas de contrôle |
| `test_integration_pipeline.py` | 4 | Boucle complète avec un vrai calcul Evidently et un vrai registre MLflow isolé ; non-régression du score de dérive ; promotion d'un challenger réellement meilleur ; réactivité de l'API au changement d'artefact |

Les quatre tests d'intégration sont marqués `@pytest.mark.integration` (déclaré dans `pytest.ini`) et s'exécutent contre une base SQLite temporaire : ils ne touchent jamais le `mlflow.db` du poste.

## 12. Utilisation

### API

`GET /health` renvoie l'état du service et du modèle :

```json
{"status": "healthy", "model_status": "loaded"}
```

`POST /predict` attend les sept caractéristiques :

```json
{
  "Surface_m2": 120,
  "Nb_Pieces": 3,
  "Annee_Construction": 1995,
  "Distance_Centre_km": 5,
  "DPE_Energy_Class": 5,
  "Has_Balcony": 0,
  "Has_Parking": 0
}
```

et renvoie, avec le champion de référence :

```json
{
  "predicted_price_k_eur": 431.89,
  "input_features": { "...": "..." },
  "model_version": "1.0.0",
  "out_of_training_domain": false,
  "domain_warnings": []
}
```

Une entrée hors bornes est refusée avec un code 422 et un message explicite. Par exemple `Surface_m2 = 500` produit `Input should be less than or equal to 300`.

**Domaine de validité.** Les bornes du schéma vérifient la plausibilité physique d'une demande ; elles sont plus larges que l'intervalle de valeurs réellement couvert par le jeu d'entraînement. Une demande située hors de cet intervalle reçoit donc une prédiction, accompagnée d'un signalement explicite et d'un avertissement dans les journaux du service :

```json
{
  "predicted_price_k_eur": 368.13,
  "out_of_training_domain": true,
  "domain_warnings": [
    "Nb_Pieces=8 outside training domain [1, 5]",
    "Annee_Construction=1920 outside training domain [1950, 2024]",
    "Distance_Centre_km=80.0 outside training domain [0.5, 50.0]"
  ]
}
```

Le domaine de référence est déclaré dans `src/app.py::TRAINING_DOMAIN` : `Surface_m2` de 30 à 300, `Nb_Pieces` de 1 à 5, `Annee_Construction` de 1950 à 2024, `Distance_Centre_km` de 0,5 à 50, `DPE_Energy_Class` de 1 à 7, `Has_Balcony` et `Has_Parking` à 0 ou 1. Ces valeurs correspondent exactement aux bornes du générateur et ont été relevées sur le fichier réel. Le tableau de bord affiche le même avertissement sous le prix prédit.

### Tableau de bord

<http://localhost:8501> — l'interface s'ouvre en anglais ; le sélecteur de langue se trouve dans le panneau de gauche. Elle affiche l'état de l'API et du modèle, un formulaire de prédiction, le dernier rapport de dérive et un bouton « Exécuter l'Entraînement Continu » qui lance `src/drift_detection.py` en sous-processus (délai maximal : 120 secondes).

### Détection de dérive en ligne de commande

```powershell
python src/drift_detection.py
```

Le script compare les données de référence à une copie perturbée de façon contrôlée (`Surface_m2 += 10`, `Prix_k_EUR *= 1.15`), écrit un rapport HTML horodaté dans `drift_reports/` et déclenche le réentraînement si la part de colonnes en dérive dépasse 0,20.

## 13. Pipeline de données et de modèle

1. **Génération** — `generate_data.py` produit 500 lignes à partir d'une formule linéaire sur sept variables, plus un bruit gaussien d'écart-type 25. Graine fixe.
2. **Chargement** — `train.py::load_and_prepare_data` sépare les sept variables explicatives de la cible `Prix_k_EUR`.
3. **Découpage** — 80 / 20, `random_state=42`.
4. **Entraînement** — `LinearRegression` sans mise à l'échelle : le modèle est linéaire, les variables sont déjà dans des ordres de grandeur exploitables.
5. **Évaluation** — MSE, RMSE, MAE, R² sur le jeu de test.
6. **Journalisation** — paramètres, métriques, modèle et quatre étiquettes de gouvernance : `model_type`, `dataset_name`, `pipeline_stage`, `environment`.
7. **Enregistrement** — une nouvelle version est créée dans le Model Registry à chaque exécution.
8. **Promotion** — voir section 14.
9. **Service** — l'API charge `models/housing_model.pkl` au démarrage.
10. **Surveillance** — `drift_detection.py` compare les distributions et archive un rapport.
11. **Réentraînement** — déclenché si la dérive dépasse le seuil, avec la limite documentée en section 17.

Le jeu de données est **synthétique**. Une intégration des Demandes de Valeurs Foncières (data.gouv.fr) a été envisagée puis écartée : les DVF ne comportent ni classe énergétique, ni information sur le balcon ou le parking, et le rapprochement nécessaire dépassait le cadre de la preuve de concept.

## 14. Entraînement, évaluation, promotion

Chaque exécution de `src/train.py` produit un *challenger*. Il ne devient *champion* — alias MLflow et fichier servi par l'API — que s'il dépasse strictement le R² du champion actuel, augmenté de `PROMOTION_MIN_GAIN`, ou s'il s'agit du tout premier entraînement.

Résultats de référence, obtenus avec la graine 42 et reproduits depuis un clone propre le 27 septembre 2026 :

| Métrique | Valeur |
|---|---|
| R² (jeu de test) | 0,9780 |
| RMSE | 22,80 k€ |
| MAE | 18,95 k€ |
| Part de colonnes en dérive (perturbation de démonstration) | 0,25, soit 2 colonnes sur 8 |
| Valeur *p* du test de Kolmogorov-Smirnov sur `Surface_m2` | 0,013431 |

Sur les données perturbées et sans réentraînement, le champion tombe à R² 0,9295 et RMSE 46,91 : la dérive simulée a donc un effet mesurable sur la qualité des prédictions.

## 15. Déploiement

Le dépôt fournit une **chaîne d'intégration continue, pas de livraison continue** : aucune image n'est publiée sur un registre et aucun déploiement n'est automatisé. Le déploiement se fait localement avec `docker compose up --build`.

L'image est construite à partir de `python:3.12-slim`, installe les dépendances épinglées, copie `src/` et `mlruns/`, puis bascule sur un utilisateur non privilégié (`mlopsuser`). Les trois services partagent cette image et surchargent son point d'entrée.

Deux remarques utiles à la lecture :

- `docker-compose.yml` monte le dépôt dans `/app` (`volumes: .:/app`). Le code exécuté est donc celui du poste, pas celui figé dans l'image : pratique en développement, à revoir pour un vrai déploiement.
- Le service `api` démarre `uvicorn --reload`, adapté au développement et non à la production.

## 16. Sécurité et gestion des secrets

**État actuel du dépôt**

- Le projet n'utilise **aucun secret** : ni clé d'API, ni mot de passe, ni jeton. `.env.example` ne contient que des ports, des URL locales et des chemins relatifs.
- `.env` est exclu par `.gitignore` et n'a **jamais été suivi par Git** (vérifié sur l'ensemble de l'historique avec `git log --all --diff-filter=A -- .env`). Aucune valeur sensible n'a donc été exposée.
- `.env.example` est versionné volontairement, via une exception explicite dans `.gitignore` (`!.env.example`), parce que l'installation en dépend.
- Les dépendances sont épinglées à la version exacte, et les actions GitHub sont référencées par empreinte SHA complète.
- Le conteneur s'exécute sous un utilisateur non privilégié.
- Les entrées de l'API sont validées par Pydantic ; les erreurs internes sont journalisées côté serveur mais jamais renvoyées au client.

**Procédure si un secret devait être introduit**

1. Ajouter la variable dans `.env` uniquement.
2. Déclarer son **nom** dans `.env.example`, avec une valeur fictive explicite (`VOTRE_VALEUR`), jamais une valeur réaliste.
3. Ne jamais la recopier dans le README, les annexes, les tests, les journaux ou les captures d'écran.
4. En cas d'exposition accidentelle, révoquer et renouveler la valeur avant toute réécriture d'historique.

**Limites de sécurité assumées pour une preuve de concept**

- L'API n'a ni authentification, ni limitation de débit, ni configuration CORS : elle ne doit pas être exposée au-delà de la machine de démonstration.
- Le modèle est chargé avec `joblib`, qui repose sur `pickle` : charger un fichier `.pkl` d'origine inconnue revient à exécuter son contenu. Ici, le fichier est toujours produit localement par `src/train.py`.
- Aucune signature d'artefact, aucun contrôle d'accès au registre de modèles.

## 17. Limites connues

1. **La boucle d'apprentissage n'est pas fermée.** `trigger_retraining()` appelle `run_pipeline()` sans lui transmettre les données dérivées ; le pipeline recharge le fichier de référence depuis le disque. À graine fixée, le challenger reproduit donc exactement le champion et n'est jamais promu. Ce comportement est volontairement figé par un test de non-régression (`test_full_self_healing_loop_champion_then_drift_then_retrain`).
2. **Les bornes de l'API restent plus larges que le domaine d'entraînement**, par choix : l'API répond à une demande physiquement plausible même si le modèle n'a pas été entraîné sur ce type de bien. L'extrapolation n'est plus silencieuse — elle est signalée dans la réponse et dans les journaux (section 12) — mais elle reste une extrapolation, dont la qualité n'est pas mesurée.
3. **`model_version` est une constante.** L'API renvoie toujours `"1.0.0"`, sans lien avec la version du Model Registry : le maillon prédiction → modèle est manquant.
4. **Aucun journal des prédictions.** Une prédiction ne peut pas être reliée après coup au modèle qui l'a produite.
5. **Pas d'orchestrateur ni de planification.** La détection de dérive est lancée à la main, en ligne de commande ou par le bouton du tableau de bord.
6. **Le déploiement n'est effectif qu'au redémarrage de l'API**, puisque le modèle est chargé une fois au démarrage.
7. **Données synthétiques, dérive simulée, échelle réduite** : 500 lignes, un seul modèle, pas de dimension temporelle.
8. **Un seul découpage 80/20**, sans validation croisée : la comparaison de deux R² ne fournit pas d'intervalle de confiance.
9. **Le lint de la CI ne bloque jamais** : l'étape est déclarée `continue-on-error: true` et se termine par `|| true`.

## 18. Pistes d'amélioration

Par ordre de valeur décroissante :

1. Transmettre les données courantes au réentraînement et évaluer champion et challenger sur un jeu commun.
2. Journaliser chaque prédiction avec la version du modèle et un horodatage.
3. Remplacer la constante `model_version` par la version réelle issue du registre.
4. Mesurer la dégradation du modèle hors domaine, pour passer d'un signalement qualitatif à une incertitude chiffrée.
5. Ajouter un test de significativité à la porte de promotion, ou un déploiement fantôme.
6. Introduire un orchestrateur et une planification de la surveillance.
7. Publier l'image sur un registre et automatiser le déploiement, avec validation humaine.
8. Rendre le lint bloquant et étendre les règles retenues.

## 19. Procédure de vérification depuis un clone propre

Cette procédure a été exécutée le 27 septembre 2026 sur une copie fraîchement clonée du dépôt. Les résultats indiqués sont ceux effectivement observés, sauf mention contraire explicite.

| # | Étape | Commande | Résultat attendu |
|---|---|---|---|
| 1 | Cloner | `git clone https://github.com/JoeDalton318/mlops-transition-poc.git` | Le dépôt est récupéré |
| 2 | Se placer dans le projet | `cd mlops-transition-poc` | — |
| 3 | Vérifier les outils | `python --version` puis `docker compose version` | Python 3.12.x ; Compose v2 |
| 4 | Créer l'environnement | `python -m venv .venv` puis activation | Invite préfixée par `(.venv)` |
| 5 | Installer | `pip install -r requirements.txt` | Installation sans erreur |
| 6 | Configurer | `copy .env.example .env` | Le fichier `.env` existe |
| 7 | Générer les données | `python generate_data.py` | `data/immobilier_france.csv`, 500 lignes |
| 8 | Entraîner | `python src/train.py` | `r2` = 0.9779899998391454 ; `models/housing_model.pkl` créé |
| 9 | Contrôle de style | `flake8 src/ tests/ --select=E9,F63,F7,F82` | Aucune erreur |
| 10 | Tests | `pytest tests/ -v` | 89 tests passants |
| 11 | Lancer l'API | `uvicorn src.app:app --port 8000` | `GET /health` renvoie `model_status: loaded` |
| 12 | Prédire | `POST /predict` avec l'exemple de la section 12 | `predicted_price_k_eur` = 431.89, `out_of_training_domain` = false |
| 13 | Dérive | `python src/drift_detection.py` | `Drift score: 0.2500` ; rapport HTML dans `drift_reports/` |
| 14 | Docker | `docker compose up --build` puis `docker compose down` | Trois services démarrés, puis arrêtés proprement |

**Ce qui a été exécuté et observé** (clone propre, 27/09/2026) : étapes 1, 2, 6, 7, 8, 10 et 13. L'entraînement a reproduit exactement les métriques de référence, et la détection de dérive a donné 0,2500 puis un challenger non promu, conformément à la section 17.

**Ce qui n'a pas été réexécuté dans ce clone** : les étapes 4 et 5, parce que l'environnement virtuel du poste a été réutilisé pour gagner du temps ; l'étape 14, pour ne pas occuper les ports déjà utilisés par l'instance de démonstration. Ces trois étapes sont en revanche exécutées à chaque exécution de la chaîne d'intégration continue, sur une machine vierge (section 21). Niveau de confiance élevé, mais la vérification revient à l'utilisateur sur son propre poste.

**Avant correction du présent dépôt**, la même procédure échouait à l'étape 6 : `.env.example` était exclu par la règle `.env.*` de `.gitignore`, donc absent du clone, et `docker compose` s'arrêtait sur `env file ... .env not found`. L'exception `!.env.example` corrige ce point.

## 20. Dépannage

| Symptôme | Cause | Correction |
|---|---|---|
| `Dataset not found at ... Run generate_data.py first.` | Le jeu de données n'est pas versionné | `python generate_data.py` |
| Tests en échec avec des erreurs de collecte sur `test_data_schema.py` | Même cause | Générer les données avant `pytest` |
| `Model not found at ... Please train the model first` ; l'API répond 503 | Aucun modèle promu localement | `python src/train.py` |
| `env file ... .env not found` au démarrage de Docker | `.env` absent | `copy .env.example .env` |
| `"/mlruns": not found` pendant `docker build` | `mlruns/` n'existe pas encore | Exécuter `python src/train.py` avant la construction |
| `cannot import name 'FallbackAsyncAdaptedQueuePool'` | SQLAlchemy 2.1 incompatible avec MLflow 2.10.0 | Réinstaller les dépendances épinglées (`pip install -r requirements.txt`) |
| `port is already allocated` | Un autre programme occupe 5000, 8000 ou 8501 | Libérer le port ; ne pas modifier les ports dans `.env` sans adapter `docker-compose.yml`, dont les adresses internes `mlflow:5000` et `api:8000` sont fixées |
| Le tableau de bord affiche « État de l'API : Hors ligne » | L'API n'est pas démarrée ou n'est pas joignable | Démarrer l'API, ou `docker compose restart api` |
| Le bouton de réentraînement renvoie « Échec du pipeline » | MLflow injoignable depuis le conteneur | `docker compose restart mlflow dashboard` |
| Le rapport Evidently affiche « Dataset Drift is NOT detected » | Seuil interne d'Evidently à 0,5, différent du seuil du projet (0,20) | Comportement normal : lire la part de colonnes en dérive, 0,25 |

## 21. Intégration continue

`.github/workflows/mlops-pipeline.yml` s'exécute à chaque poussée et à chaque demande de fusion sur `main` :

1. récupération du code et installation de Python 3.12 ;
2. installation des dépendances épinglées ;
3. lint flake8 restreint aux erreurs bloquantes — étape non bloquante ;
4. génération des données puis entraînement, nécessaires aux tests et à la construction de l'image ;
5. exécution de la suite pytest ;
6. construction de l'image Docker et vérification de son démarrage.

Dernière exécution vérifiée : 27 septembre 2026, 89 tests passants sous Python 3.12.14.

## 22. Références

- Documentation MLflow : <https://mlflow.org/docs/latest/>
- Documentation Evidently AI : <https://docs.evidentlyai.com/>
- Documentation FastAPI : <https://fastapi.tiangolo.com/>
- Documentation Streamlit : <https://docs.streamlit.io/>
- Documentation Docker Compose : <https://docs.docker.com/compose/>
- Google Cloud, *MLOps: Continuous Delivery and Automation Pipelines in Machine Learning*
- D. Sculley et al., *Hidden Technical Debt in Machine Learning Systems*, NeurIPS, 2015
- E. Breck et al., *The ML Test Score*, IEEE Big Data, 2017

## Licence

MIT — voir le fichier `LICENSE`.
