# tests/test_api.py

import pytest
import requests
import json

# L'URL de notre API locale en cours d'exécution
API_BASE_URL = "http://127.0.0.1:5000"

def test_health_check_endpoint():
    """Teste si l'endpoint racine '/' répond correctement."""
    response = requests.get(API_BASE_URL + "/")
    
    # Assertions : on vérifie que les conditions sont vraies
    assert response.status_code == 200  # La requête a-t-elle réussi ?
    assert response.json()["status"] == "Puls-Events RAG API with Flask is running"

def test_ask_endpoint_success():
    """Teste un appel réussi à l'endpoint '/ask'."""
    headers = {"Content-Type": "application/json"}
    payload = {"question": "Quelle est l'exposition à la HAB Galerie ?"}
    
    # On envoie la requête POST à notre API
    response = requests.post(API_BASE_URL + "/ask", headers=headers, json=payload)
    
    # On vérifie la réponse
    assert response.status_code == 200
    response_data = response.json()
    assert "answer" in response_data  # La clé 'answer' est-elle dans la réponse ?
    assert isinstance(response_data["answer"], str) # La réponse est-elle une chaîne de caractères ?
    assert len(response_data["answer"]) > 0 # La réponse n'est-elle pas vide ?

def test_ask_endpoint_missing_question():
    """Teste le cas où la clé 'question' est manquante dans la requête."""
    headers = {"Content-Type": "application/json"}
    payload = {"mauvaise_cle": "une valeur"} # Payload incorrect
    
    response = requests.post(API_BASE_URL + "/ask", headers=headers, json=payload)
    
    # L'API doit renvoyer une erreur 400 (Bad Request)
    assert response.status_code == 400
    response_data = response.json()
    assert "error" in response_data
    assert "doit être un JSON avec une clé 'question'" in response_data["error"]