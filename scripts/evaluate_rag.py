# scripts/evaluate_rag.py (Version avec sauvegarde des rapports)

import os
import json
import asyncio
import traceback
from datetime import datetime
from dotenv import load_dotenv

print("--- DÉBUT DU SCRIPT D'ÉVALUATION ---")
print("Chargement des librairies...")
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from puls_events_rag.rag_chain import create_rag_chain
print("Librairies importées avec succès.")

print("Chargement du fichier .env...")
load_dotenv()
print("-> Fichier .env chargé.")

EVAL_DATASET_PATH = "evaluation_dataset.json"

def run_evaluation():
    """
    Lance l'évaluation complète de la chaîne RAG et sauvegarde les résultats.
    """
    # ... (Toutes les étapes de 1 à 4 restent identiques) ...
    
    print("\nÉtape 1/4 : Chargement du jeu de test annoté...")
    with open(EVAL_DATASET_PATH, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    questions = [item['question'] for item in eval_data]
    ground_truths = [item['ground_truth'] for item in eval_data]
    print(f"-> {len(questions)} questions chargées.")

    print("\nÉtape 2/4 : Initialisation de la chaîne RAG...")
    rag_chain, retriever = create_rag_chain()
    print("-> Chaîne et retriever prêts.")

    print("\nÉtape 3/4 : Génération des réponses et contextes...")
    answers = rag_chain.batch(questions)
    contexts = []
    retrieved_docs_list = retriever.batch(questions)
    for retrieved_docs in retrieved_docs_list:
        contexts.append([doc.page_content for doc in retrieved_docs])
    print("-> Réponses et contextes générés.")

    print("\nÉtape 4/4 : Préparation et lancement de l'évaluation Ragas...")
    dataset_dict = {
        "question": questions,
        "ground_truth": ground_truths,
        "answer": answers,
        "contexts": contexts,
    }
    ragas_dataset = Dataset.from_dict(dataset_dict)
    
    result = evaluate(ragas_dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    print("-> Évaluation terminée.")

    df_results = result.to_pandas()
    print("\n--- RÉSULTATS DE L'ÉVALUATION ---")
    print(df_results)
    print("---------------------------------")
    
    # --- NOUVELLE SECTION : SAUVEGARDE DU RAPPORT ---
    print("\nÉtape 5/5 : Sauvegarde du rapport d'évaluation...")
    
    # Créer un horodatage pour un nom de fichier unique
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename_base = f"report_{timestamp}"
    
    # Créer le dossier 'reports' s'il n'existe pas
    os.makedirs("reports", exist_ok=True)
    
    # Sauvegarder en CSV
    csv_path = f"reports/{report_filename_base}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"-> Rapport CSV sauvegardé dans : {csv_path}")
    
    # Sauvegarder en Markdown
    md_path = f"reports/{report_filename_base}.md"
    df_results.to_markdown(md_path, index=False)
    print(f"-> Rapport Markdown sauvegardé dans : {md_path}")
    
    return df_results

if __name__ == "__main__":
    try:
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        run_evaluation()

    except Exception as e:
        print("\n\n--- ERREUR CRITIQUE ---")
        print(f"Le script a échoué avec l'erreur suivante : {e}")
        print("\n--- TRACEBACK DÉTAILLÉ ---")
        traceback.print_exc()
        print("------------------------")