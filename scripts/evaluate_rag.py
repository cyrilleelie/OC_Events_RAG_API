# scripts/evaluate_rag.py

import os
import json
import pandas as pd
from dotenv import load_dotenv

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

from puls_events_rag.rag_chain import create_rag_chain
from puls_events_rag import config

# --- Chargement des dépendances pour le retriever ---
# (Nécessaire pour récupérer le contexte)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Constantes ---
EVAL_DATASET_PATH = "evaluation_dataset.json"
VECTOR_STORE_PATH = "vector_store"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- Chargement de l'environnement ---
load_dotenv()

def run_evaluation():
    """
    Lance le processus d'évaluation complet de la chaîne RAG.
    """
    # --- 1. Charger le jeu de test annoté ---
    print("Étape 1/5 : Chargement du jeu de test annoté...")
    with open(EVAL_DATASET_PATH, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    questions = [item['question'] for item in eval_data]
    ground_truths = [item['ground_truth'] for item in eval_data]
    print(f"-> {len(questions)} questions chargées.")

    # --- 2. Initialiser la chaîne RAG et le retriever ---
    print("\nÉtape 2/5 : Initialisation de la chaîne RAG et du retriever...")
    rag_chain = create_rag_chain()
    
    # On a besoin du retriever pour récupérer le contexte pour chaque question
    embeddings_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings_model, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={'k': 4})
    print("-> Chaîne et retriever prêts.")

    # --- 3. Générer les réponses et contextes pour chaque question ---
    print("\nÉtape 3/5 : Génération des réponses et contextes (cela peut prendre un moment)...")
    answers = []
    contexts = []
    for i, question in enumerate(questions):
        print(f"  - Traitement de la question {i+1}/{len(questions)} : '{question[:50]}...'")
        # Obtenir la réponse de la chaîne RAG
        answers.append(rag_chain.invoke(question))
        # Obtenir le contexte utilisé
        retrieved_docs = retriever.invoke(question)
        contexts.append([doc.page_content for doc in retrieved_docs])
    print("-> Réponses et contextes générés.")

    # --- 4. Préparer le dataset pour Ragas ---
    print("\nÉtape 4/5 : Préparation du dataset pour Ragas...")
    # Ragas attend un format de dictionnaire spécifique
    dataset_dict = {
        "question": questions,
        "ground_truth": ground_truths,
        "answer": answers,
        "contexts": contexts,
    }
    # Conversion en objet Dataset de la bibliothèque `datasets`
    ragas_dataset = Dataset.from_dict(dataset_dict)
    print("-> Dataset Ragas créé.")

    # --- 5. Lancer l'évaluation Ragas ---
    print("\nÉtape 5/5 : Lancement de l'évaluation Ragas...")
    # Définition des métriques à calculer
    metrics_to_evaluate = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
    
    result = evaluate(ragas_dataset, metrics=metrics_to_evaluate)
    print("-> Évaluation terminée.")

    # Affichage des résultats sous forme de tableau
    print("\n--- RÉSULTATS DE L'ÉVALUATION ---")
    df_results = result.to_pandas()
    print(df_results)
    print("---------------------------------")
    
    return df_results

if __name__ == "__main__":
    run_evaluation()