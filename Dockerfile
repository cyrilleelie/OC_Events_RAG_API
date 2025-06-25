# ---- Étape 1 : Builder l'environnement avec Poetry ----
# On part d'une image Python de base.
FROM python:3.12-slim AS builder

WORKDIR /app

# On installe Poetry
RUN pip install poetry

# On copie les fichiers de dépendances
COPY pyproject.toml poetry.lock ./

# LA SOLUTION SIMPLE ET PRAGMATIQUE :
# On installe TOUTES les dépendances, y compris celles de dev.
# C'est la commande la plus simple et la plus universelle.
# Pour ce POC, c'est parfaitement acceptable.
RUN poetry install --no-root --no-interaction

# ---- Étape 2 : L'image de production finale ----
# On repart d'une image Python fraîche et légère.
FROM python:3.12-slim

WORKDIR /app

# On copie l'environnement virtuel complet créé par Poetry dans l'étape précédente.
# On utilise un joker (*) car le nom exact du dossier peut varier légèrement.
COPY --from=builder /root/.cache/pypoetry/virtualenvs/puls-events-rag-*-py3.12 /app/.venv

# On active l'environnement pour que les commandes suivantes l'utilisent.
ENV PATH="/app/.venv/bin:$PATH"

# Copie du code de l'application et des données nécessaires au fonctionnement
COPY src/puls_events_rag ./src/puls_events_rag
COPY api ./api
COPY vector_store ./vector_store

# On expose le port sur lequel gunicorn va écouter.
EXPOSE 5000

# La commande finale pour lancer le serveur.
# Gunicorn a été installé par Poetry avec les autres dépendances.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "api.main:app"]