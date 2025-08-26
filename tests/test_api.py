# tests/test_api.py

import pytest
import requests
import os

# --- Configuration de l'URL de l'API ---
# Lit l'URL depuis une variable d'environnement pour plus de flexibilité,
# avec une valeur par défaut pour le développement local.
API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:5001")

def test_health_check_endpoint():
    """Teste si l'endpoint racine '/' répond correctement."""
    try:
        response = requests.get(f"{API_BASE_URL}/")
        response.raise_for_status() # Lève une exception si le statut est une erreur
    except requests.exceptions.ConnectionError as e:
        pytest.fail(f"La connexion à l'API a échoué. Assurez-vous que le conteneur Docker est bien lancé. Erreur: {e}")

    # Assertions
    assert response.status_code == 200
    assert response.json()["status"] == "Puls-Events RAG API with Flask is running"

def test_ask_endpoint_success():
    """Teste un appel réussi à l'endpoint '/ask' et vérifie la nouvelle structure de réponse."""
    headers = {"Content-Type": "application/json"}
    payload = {"question": "Quelle est l'exposition à la HAB Galerie ?"}
    
    response = requests.post(f"{API_BASE_URL}/ask", headers=headers, json=payload)
    
    # On vérifie la réponse
    assert response.status_code == 200
    response_data = response.json()
    
    # Vérifications de la structure de la réponse
    assert "answer" in response_data
    assert isinstance(response_data["answer"], str)
    assert len(response_data["answer"]) > 0
    
    # On vérifie aussi la présence des sources
    assert "sources" in response_data
    assert isinstance(response_data["sources"], list)

def test_ask_endpoint_missing_question():
    """Teste le cas où la clé 'question' est manquante dans la requête."""
    headers = {"Content-Type": "application/json"}
    payload = {"mauvaise_cle": "une valeur"} # Payload incorrect
    
    response = requests.post(f"{API_BASE_URL}/ask", headers=headers, json=payload)
    
    # L'API doit renvoyer une erreur 400 (Bad Request)
    assert response.status_code == 400
    response_data = response.json()
    assert "error" in response_data
    assert "doit être un JSON avec une clé 'question'" in response_data["error"]

