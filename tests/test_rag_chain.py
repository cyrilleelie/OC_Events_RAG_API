# tests/test_rag_chain.py

import pytest
from unittest.mock import patch

# --- IMPORTS CORRIGÉS ---
# On importe les classes depuis les mêmes paquets que rag_chain.py
from langchain_ollama import ChatOllama
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

# On importe la fonction à tester
from puls_events_rag.rag_chain import create_rag_chain

# --- Tests Unitaires pour le sélecteur de LLM ---

@patch('puls_events_rag.rag_chain.FAISS.load_local')
def test_selects_ollama_provider(mock_faiss_load, monkeypatch):
    """
    Vérifie que ChatOllama est bien sélectionné quand LLM_PROVIDER="mistral_local".
    """
    monkeypatch.setenv("LLM_PROVIDER", "mistral_local")
    
    rag_chain, _ = create_rag_chain()
    llm = rag_chain.steps[2]
    
    assert isinstance(llm, ChatOllama)

@patch('puls_events_rag.rag_chain.FAISS.load_local')
def test_selects_mistral_api_provider(mock_faiss_load, monkeypatch):
    """
    Vérifie que ChatMistralAI est bien sélectionné quand LLM_PROVIDER="mistral_api".
    """
    monkeypatch.setenv("LLM_PROVIDER", "mistral_api")
    monkeypatch.setenv("MISTRAL_API_KEY", "fake_key")
    
    rag_chain, _ = create_rag_chain()
    llm = rag_chain.steps[2]
    
    assert isinstance(llm, ChatMistralAI)

@patch('puls_events_rag.rag_chain.FAISS.load_local')
def test_selects_google_api_provider(mock_faiss_load, monkeypatch):
    """
    Vérifie que ChatGoogleGenerativeAI est bien sélectionné quand LLM_PROVIDER="google_api".
    """
    monkeypatch.setenv("LLM_PROVIDER", "google_api")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake_key")
    
    rag_chain, _ = create_rag_chain()
    llm = rag_chain.steps[2]
    
    assert isinstance(llm, ChatGoogleGenerativeAI)

@patch('puls_events_rag.rag_chain.FAISS.load_local')
def test_raises_error_on_invalid_provider(mock_faiss_load, monkeypatch):
    """
    Vérifie que le système lève une erreur si le fournisseur de LLM est inconnu.
    """
    monkeypatch.setenv("LLM_PROVIDER", "provider_inconnu")
    
    with pytest.raises(ValueError, match="Fournisseur de LLM non valide"):
        create_rag_chain()
