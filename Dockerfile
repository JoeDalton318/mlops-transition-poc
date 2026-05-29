# 1. Image de base légère et officielle Python
FROM python:3.10-slim

# 2. Bonnes pratiques de sécurité : Création d'un utilisateur non-root
RUN groupadd --system mlopsuser && useradd --system --gid mlopsuser --create-home --home-dir /home/mlopsuser mlopsuser

# 3. Définition du répertoire de travail dans le conteneur
WORKDIR /app

# 4. Installation des dépendances système de base
RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 5. Copie et installation des dépendances Python
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 6. Copie du code source de l'API
COPY src /app/src

# 7. CRUCIAL : Copie du registre des modèles MLflow (pour que l'API trouve le modèle)
COPY mlruns /app/mlruns

# 8. Attribution des droits au nouvel utilisateur
RUN chown -R mlopsuser:mlopsuser /app

# 9. Changement d'utilisateur (on quitte le mode administrateur/root)
USER mlopsuser

# 10. Ouverture du port 8000
EXPOSE 8000

# 11. Commande de démarrage
ENTRYPOINT ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]