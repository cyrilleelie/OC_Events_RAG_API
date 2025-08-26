# app_streamlit.py (Version finale avec sources)

import streamlit as st
from puls_events_rag.rag_chain import create_rag_chain

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Puls-Events Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Chatbot pour les Événements Culturels")
st.caption("Posez vos questions sur les événements de la métropole nantaise.")

# --- Mise en cache de la chaîne RAG et du Retriever ---
@st.cache_resource
def load_rag_system():
    """Charge la chaîne RAG et le retriever et les met en cache."""
    return create_rag_chain()

# On charge la chaîne et le retriever
rag_chain, retriever = load_rag_system()

# --- Interface de chat ---

# Initialiser l'historique de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher les messages de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Si le message de l'assistant a des sources, on les affiche
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Voir les sources utilisées"):
                for i, source in enumerate(message["sources"]):
                    st.info(f"**Source {i+1}**\n\n---\n\n{source}")

# Zone de saisie pour la question de l'utilisateur
if user_question := st.chat_input("Que cherchez-vous ?"):
    # Ajouter la question à l'historique et l'afficher
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Afficher la réponse du chatbot
    with st.chat_message("assistant"):
        with st.spinner("Je cherche la meilleure réponse..."):
            # 1. Obtenir la réponse
            response_text = rag_chain.invoke(user_question)
            
            # 2. Obtenir les sources
            retrieved_docs = retriever.invoke(user_question)
            sources = [doc.page_content for doc in retrieved_docs]
            
            # 3. Afficher la réponse et les sources
            st.markdown(response_text)
            with st.expander("Voir les sources utilisées"):
                for i, source in enumerate(sources):
                    st.info(f"**Source {i+1}**\n\n---\n\n{source}")
    
    # Ajouter la réponse complète (avec sources) à l'historique
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text,
        "sources": sources
    })