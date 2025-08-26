# puls_events_rag/rag_chain.py

from . import config
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama # Pour le modèle local
from langchain_mistralai import ChatMistralAI # Pour l'API Mistral
from langchain_google_genai import ChatGoogleGenerativeAI # Pour l'API Google
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_rag_chain():
    """
    Crée et retourne une chaîne RAG et son retriever,
    en sélectionnant le LLM (local ou API) selon la configuration.
    """
    # --- 1. Initialiser le Retriever (ne change pas) ---
    print("Initialisation du Retriever...")
    vector_store = FAISS.load_local(
        config.VECTOR_STORE_PATH,
        HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME),
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={'k': config.RETRIEVER_K})
    print("-> Retriever initialisé.")

    # --- 2. Définir le Prompt Template (ne change pas) ---
    print("Définition du Prompt Template...")
    prompt = PromptTemplate(template=config.PROMPT_TEMPLATE, input_variables=["context", "question"])
    print("-> Prompt défini.")

    # --- 3. Initialiser le LLM (logique de sélection mise à jour) ---
    if config.LLM_PROVIDER == "mistral_local":
        print("Initialisation du LLM (Ollama local)...")
        llm = ChatOllama(model="mistral", temperature=0.1, base_url=config.OLLAMA_BASE_URL)
        print("-> LLM local initialisé.")
    
    elif config.LLM_PROVIDER == "mistral_api": # On peut renommer ça en "mistral"
        print("Initialisation du LLM (API Mistral)...")
        if not config.MISTRAL_API_KEY:
            raise ValueError("MISTRAL_API_KEY non configurée.")
        llm = ChatMistralAI(model_name=config.LLM_MODEL_NAME, temperature=0.1, api_key=config.MISTRAL_API_KEY)
        print("-> LLM API Mistral initialisé.")

    elif config.LLM_PROVIDER == "google_api": # <-- AJOUT DE L'OPTION GEMINI
        print("Initialisation du LLM (API Google Gemini)...")
        if not config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY non configurée.")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, google_api_key=config.GOOGLE_API_KEY)
        print("-> LLM API Gemini initialisé.")

    else:
        raise ValueError(f"Fournisseur de LLM non valide : '{config.LLM_PROVIDER}'.")

    # --- 4. Construire la chaîne RAG (ne change pas) ---
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