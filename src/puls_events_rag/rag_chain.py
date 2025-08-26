# src/puls_events_rag/rag_chain.py

import os
from . import config
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_rag_chain():
    """
    Crée et retourne une chaîne RAG et son retriever,
    en sélectionnant le LLM (local ou API) selon la configuration.
    """
    # --- 1. Initialiser le Retriever ---
    print("Initialisation du Retriever...")
    vector_store = FAISS.load_local(
        config.VECTOR_STORE_PATH,
        HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME),
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={'k': config.RETRIEVER_K})
    print("-> Retriever initialisé.")

    # --- 2. Définir le Prompt Template ---
    print("Définition du Prompt Template...")
    prompt = PromptTemplate(template=config.PROMPT_TEMPLATE, input_variables=["context", "question"])
    print("-> Prompt défini.")

    # --- 3. Initialiser le LLM (logique de sélection) ---
    llm_provider = os.getenv("LLM_PROVIDER", "mistral_local").lower()

    if llm_provider == "mistral_local":
        print("Initialisation du LLM (Ollama local)...")
        llm = ChatOllama(model="mistral", temperature=0.3, base_url=config.OLLAMA_BASE_URL)
        print("-> LLM local initialisé.")
    elif llm_provider == "mistral_api":
        print("Initialisation du LLM (API Mistral)...")
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY non configurée.")
        llm = ChatMistralAI(
            model_name=config.LLM_MODEL_NAME,
            temperature=0.3,
            api_key=api_key
        )
        print("-> LLM API Mistral initialisé.")
    elif llm_provider == "google_api":
        print("Initialisation du LLM (API Google Gemini)...")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY non configurée.")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
        print("-> LLM API Gemini initialisé.")
    else:
        raise ValueError(f"Fournisseur de LLM non valide : '{llm_provider}'.")

    # --- 4. Construire la chaîne RAG ---
    print("Construction de la chaîne RAG...")
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("-> Chaîne RAG construite avec succès.")
    
    return rag_chain, retriever
