# puls_events_rag/rag_chain.py

import os
from dotenv import load_dotenv

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
    print("Initialisation du Retriever depuis la base FAISS...")
    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError(f"Le dossier de la base vectorielle n'a pas été trouvé : {VECTOR_STORE_PATH}")
    
    # Charger le modèle d'embedding (le même que pour la création)
    embeddings_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    
    # Charger la base de données vectorielle FAISS
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings_model, allow_dangerous_deserialization=True)
    
    # Créer le retriever pour chercher les 4 documents les plus pertinents
    retriever = vector_store.as_retriever(search_kwargs={'k': 4})
    print("-> Retriever initialisé.")

    # --- 2. Définir le Prompt Template ---
    print("Définition du Prompt Template...")
    template = """
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
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    print("-> Prompt défini.")

    # --- 3. Initialiser le LLM ---
    print("Initialisation du LLM (Mistral)...")
    llm = ChatMistralAI(model_name=MISTRAL_MODEL, temperature=0.3)
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