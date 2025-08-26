# src/puls_events_rag/config.py
import os

# --- AJOUT : Sélecteur de fournisseur de LLM ---
# Lit la variable d'environnement LLM_PROVIDER.
# Accepte "local" (pour Ollama) ou "api" (pour l'API Mistral).
# Par défaut, utilise "local" si la variable n'est pas définie.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mistral_local").lower()

# --- Configuration de la Source de Données (Opendatasoft) ---
PORTAL_URL = "https://data.nantesmetropole.fr"
DATASET_ID = "244400404_agenda-evenements-nantes-metropole_v2"
API_ENDPOINT = f"{PORTAL_URL}/api/explore/v2.1/catalog/datasets/{DATASET_ID}/records"

# --- Configuration des Chemins (Paths) ---
DATA_PATH = f"data/events_{DATASET_ID}.csv"
VECTOR_STORE_PATH = "vector_store"

# --- Configuration des Modèles ---
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL_NAME = "mistral-large-latest"

# --- Configuration d'Ollama ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- Configuration de la Chaîne RAG ---
PROMPT_TEMPLATE = """
Tu es un assistant spécialisé dans les événements culturels de la métropole nantaise.
Réponds à la question de l'utilisateur de manière claire et amicale, en te basant EXCLUSIVEMENT sur le contexte suivant.
Si l'information n'est pas dans le contexte, dis poliment que tu ne sais pas.
Ne cherche pas d'informations en dehors.
Cite le nom de l'événement, le lieu et la date si possible.

CONTEXTE :
{context}

QUESTION :
{question}

RÉPONSE :
"""
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RETRIEVER_K = 4