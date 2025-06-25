# api/main.py (Version avec FLASK)

from flask import Flask, request, jsonify
from puls_events_rag.rag_chain import create_rag_chain
from dotenv import load_dotenv
load_dotenv()

# --- Initialisation de l'application Flask ---
app = Flask(__name__)

# --- Chargement de la chaîne RAG ---
# La logique est la même : on charge le modèle une seule fois au démarrage.
try:
    rag_chain = create_rag_chain()
except Exception as e:
    print(f"Erreur critique lors du chargement de la chaîne RAG : {e}")
    rag_chain = None

# --- Endpoints de l'API ---

@app.route("/", methods=["GET"])
def health_check():
    """Vérifie que le serveur API fonctionne."""
    return jsonify(status="Puls-Events RAG API with Flask is running")

@app.route("/ask", methods=["POST"])
def ask_question():
    """
    Reçoit une question, l'envoie à la chaîne RAG et retourne la réponse générée.
    """
    if rag_chain is None:
        return jsonify(error="La chaîne RAG n'a pas pu être initialisée."), 500

    # Avec Flask, la validation des données est manuelle.
    # 1. On récupère le corps de la requête en JSON.
    data = request.get_json()

    # 2. On vérifie que les données sont correctes.
    if not data or "question" not in data or not isinstance(data["question"], str):
        # On retourne une erreur 400 (Bad Request) si la question est manquante ou mal formatée.
        return jsonify(error="La requête doit être un JSON avec une clé 'question' de type string."), 400

    question = data["question"]
    print(f"Question reçue : {question}")

    try:
        response = rag_chain.invoke(question)
        print(f"Réponse générée : {response}")
        return jsonify(answer=response)
    except Exception as e:
        print(f"Erreur lors de l'invocation de la chaîne RAG: {e}")
        return jsonify(error="Une erreur interne est survenue lors de la génération de la réponse."), 500


# --- Lancement de l'API (pour le développement) ---
# Permet de lancer avec `python api/main.py` si besoin.
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)