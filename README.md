# mlops-transition-poc

PoC de transition DevOps → MLOps pour mémoire de Master.

Ce projet démontre une pipeline sécurisée et reproductible avec :

- `scikit-learn` pour le modèle de régression
- `MLflow` pour le tracking et le versioning du modèle
- `FastAPI` + `uvicorn` pour l'API de prédiction
- `Docker` pour la containerisation
- `Pydantic` pour la validation stricte des entrées

## Structure du projet

```text
memoire-mlops-demo/
├── src/
│   ├── train.py
│   └── app.py
├── requirements.txt
├── Dockerfile
├── README.md
└── LICENSE
```

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

## Entraînement du modèle

Lancer l'entraînement et le tracking MLflow :

```powershell
python src/train.py
```

Les artefacts MLflow sont stockés localement dans `mlruns/` par défaut.

## Démarrage de l'API

Lancer le service FastAPI :

```powershell
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

### Endpoint

- `POST /predict`

Exemple de requête JSON :

```json
{
  "MedInc": 8.3252,
  "HouseAge": 41,
  "AveRooms": 6.9841,
  "AveBedrms": 1.0238,
  "Population": 322,
  "AveOccup": 2.5556,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

## Conteneurisation

Construire l'image Docker :

```powershell
docker build -t mlops-transition-poc .
```

Lancer le conteneur :

```powershell
docker run --rm -p 8000:8000 mlops-transition-poc
```

## Sécurité et bonnes pratiques

- Utilisation d'un utilisateur non-root dans le conteneur
- Validation stricte des données entrantes avec Pydantic
- Pas de secrets codés en dur
- Gestion des dépendances à travers `requirements.txt`

## License

Ce projet est sous licence MIT. Voir le fichier `LICENSE`.
