# Dockerfile - Version finale, auto-suffisante et recommandée
FROM python:3.12-slim
WORKDIR /app

# Copier le fichier qui liste les dépendances
COPY requirements.txt .

# --- Installation des Dépendances ---
# 1. Installer torch séparément car il utilise une source spécifique
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Installer toutes les autres dépendances depuis le requirements.txt (généré par Poetry)
RUN pip install --no-cache-dir -r requirements.txt

# --- Copie du Code Source et des Scripts ---
COPY src/ /app/src/
COPY api /app/api/
COPY scripts/ /app/scripts/
COPY .env .

# --- DÉFINITION DES VARIABLES D'ENVIRONNEMENT ---
# Indique à Python où trouver le module 'puls_events_rag'
ENV PYTHONPATH="/app/src"
# Définit l'URL d'Ollama spécifiquement pour l'environnement Docker
ENV OLLAMA_BASE_URL="http://host.docker.internal:11434"

# --- CRÉATION DE LA BASE DE DONNÉES (PIPELINE COMPLET) ---
# Étape 1 : Télécharger les données et créer le fichier CSV
RUN python scripts/fetch_data.py

# Étape 2 : Créer la base vectorielle à partir du CSV
RUN python scripts/create_vector_store.py

# --- Configuration du Serveur ---
EXPOSE 5000

# Commande pour lancer l'application avec Gunicorn
CMD ["python", "-m", "gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "300", "--preload", "--access-logfile", "-", "api.main:app"]