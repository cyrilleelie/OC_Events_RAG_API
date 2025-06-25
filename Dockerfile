# Dockerfile - Version finale et la plus robuste

FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .

# Installation optimisée
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN grep -v "nvidia" requirements.txt > requirements.cpu.txt
RUN pip install --no-cache-dir -r requirements.cpu.txt
RUN pip install --no-cache-dir gunicorn langchain-community

ENV PYTHONPATH="/app/src"

COPY src/ /app/src/
# On copie le package de notre application API
COPY api /app/api/
# On copie la base de données vectorielle
COPY vector_store /app/vector_store/

# ON AJOUTE LA COPIE DU FICHIER .ENV
COPY .env .

EXPOSE 5000

# ON AJOUTE L'OPTION --timeout 120 (120 secondes)
CMD ["python", "-m", "gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "300", "--preload", "--pythonpath", "/app/src", "api.main:app"]