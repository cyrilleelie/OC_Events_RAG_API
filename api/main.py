# api/main.py (Version finale avec sources)

from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
import logging

# On importe notre fonction qui retourne maintenant la chaîne ET le retriever
from puls_events_rag.rag_chain import create_rag_chain

# Configuration du logging
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# --- Chargement de la chaîne RAG et du Retriever ---
# Fait une seule fois au démarrage pour être plus efficace
try:
    rag_chain, retriever = create_rag_chain()
    logging.info("Chaîne RAG et retriever initialisés avec succès.")
except Exception as e:
    logging.error(f"Erreur critique lors de l'initialisation : {e}")
    rag_chain, retriever = None, None

# --- Endpoints de l'API ---

@app.route("/", methods=["GET"])
def health_check():
    """Vérifie que le serveur API fonctionne."""
    return jsonify(status="Puls-Events RAG API with Flask is running")

@app.route("/ask", methods=["POST"])
def ask_question():
    """
    Reçoit une question, l'envoie à la chaîne RAG et retourne la réponse et ses sources.
    """
    if not rag_chain or not retriever:
        return jsonify(error="Le service RAG n'est pas initialisé correctement."), 500

    data = request.get_json()
    if not data or "question" not in data or not isinstance(data["question"], str):
        return jsonify(error="La requête doit être un JSON avec une clé 'question' de type string."), 400

    question = data["question"]
    logging.info(f"Question reçue : {question}")

    try:
        # 1. Obtenir la réponse finale de la chaîne RAG
        answer = rag_chain.invoke(question)
        
        # 2. Obtenir les documents source que le retriever a trouvés
        retrieved_docs = retriever.invoke(question)
        sources = [doc.page_content for doc in retrieved_docs]
        
        logging.info(f"Réponse générée : {answer[:100]}...")
        
        # 3. Retourner un objet JSON complet avec la réponse et les sources
        return jsonify({
            "answer": answer,
            "sources": sources
        })
        
    except Exception as e:
        logging.error(f"Erreur lors de l'invocation de la chaîne RAG: {e}")
        return jsonify(error="Une erreur interne est survenue lors de la génération de la réponse."), 500

# --- Lancement de l'API (pour le développement) ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)