# src/puls_events_rag/config.py

# --- Configuration de la Source de Données (Opendatasoft) ---
PORTAL_URL = "https://data.nantesmetropole.fr"
DATASET_ID = "244400404_agenda-evenements-nantes-nantes-metropole"
API_ENDPOINT = f"{PORTAL_URL}/api/explore/v2.1/catalog/datasets/{DATASET_ID}/records"

# --- Configuration des Chemins (Paths) ---
DATA_PATH = f"data/events_{DATASET_ID}.csv"
VECTOR_STORE_PATH = "vector_store"

# --- Configuration des Modèles ---
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL_NAME = "mistral-large-latest"

# --- Configuration de la Chaîne RAG ---
# Le prompt est maintenant centralisé ici pour une modification facile.
PROMPT_TEMPLATE = """
Tu es un assistant spécialisé dans les événements culturels de la métropole nantaise.
Réponds à la question de l'utilisateur de manière claire et amicale, en te basant EXCLUSIVEMENT sur le contexte suivant.
Si l'information n'est pas dans le contexte, dis poliment que tu ne sais pas. Ne cherche pas d'informations en dehors.
Cite le nom de l'événement et le lieu si possible.

CONTEXTE :
{context}

QUESTION :
{question}

RÉPONSE :
"""

# Paramètres pour le découpage de texte (chunking)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Nombre de documents pertinents à récupérer
RETRIEVER_K = 4