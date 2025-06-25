# puls_events_rag/rag_chain.py

import os
from dotenv import load_dotenv
from . import config
from langchain.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
# Charger les variables d'environnement (MISTRAL_API_KEY)
load_dotenv()

VECTOR_STORE_PATH = "vector_store"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MISTRAL_MODEL = "mistral-large-latest" # Ou un autre modèle Mistral de votre choix

def create_rag_chain():
    """
    Crée et retourne une chaîne RAG (Retrieval-Augmented Generation) complète.
    """
    # --- 1. Initialiser le Retriever ---
    print("Initialisation du Retriever...")
    vector_store = FAISS.load_local(
        config.VECTOR_STORE_PATH, # <-- MODIFIÉ
        HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME), # <-- MODIFIÉ
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={'k': config.RETRIEVER_K}) # <-- MODIFIÉ
    print("-> Retriever initialisé.")

    # --- 2. Définir le Prompt Template ---
    print("Définition du Prompt Template...")
    # On utilise directement le template du fichier de config
    prompt = PromptTemplate(template=config.PROMPT_TEMPLATE, input_variables=["context", "question"]) # <-- MODIFIÉ
    print("-> Prompt défini.")

    # --- 3. Initialiser le LLM ---
    print("Initialisation du LLM (Mistral)...")
    llm = ChatMistralAI(model_name=config.LLM_MODEL_NAME, temperature=0.3) # <-- MODIFIÉ
    print("-> LLM initialisé.")
    
    # --- 4. Construire la chaîne RAG avec LCEL ---
    print("Construction de la chaîne RAG...")
    
    def format_docs(docs):
        # Concatène le contenu des documents récupérés en une seule chaîne de caractères.
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("-> Chaîne RAG construite avec succès.")
    
    return rag_chain

# --- Bloc de test ---
# Ce bloc ne s'exécute que si on lance ce fichier directement (python puls_events_rag/rag_chain.py)
if __name__ == '__main__':
    print("--- Lancement du test de la chaîne RAG ---")
    
    # Création de la chaîne
    chain = create_rag_chain()
    
    # Exemples de questions pour tester
    question1 = "Y a-t-il des expositions de photographie en ce moment ?"
    question2 = "Je cherche un concert de Jazz"
    question3 = "Que faire avec des enfants ?"
    
    print("\n--- TEST 1 ---")
    print(f"Question : {question1}")
    response1 = chain.invoke(question1)
    print("Réponse du RAG :")
    print(response1)
    
    print("\n--- TEST 2 ---")
    print(f"Question : {question2}")
    response2 = chain.invoke(question2)
    print("Réponse du RAG :")
    print(response2)