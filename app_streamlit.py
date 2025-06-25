# app_streamlit.py

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

# --- Mise en cache de la chaîne RAG ---
# Streamlit relance le script à chaque interaction de l'utilisateur.
# Pour éviter de recharger les lourds modèles à chaque fois,
# on utilise le décorateur @st.cache_resource.
# La chaîne RAG sera créée une seule fois et réutilisée.
@st.cache_resource
def load_rag_chain():
    """Charge la chaîne RAG et la met en cache."""
    return create_rag_chain()

# On charge la chaîne (cette opération sera lente la première fois, puis instantanée)
rag_chain = load_rag_chain()

# --- Interface de chat ---

# Initialiser l'historique de chat dans l'état de la session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher les messages de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie pour la question de l'utilisateur
if user_question := st.chat_input("Que cherchez-vous ?"):
    # Ajouter la question de l'utilisateur à l'historique et l'afficher
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Afficher la réponse du chatbot
    with st.chat_message("assistant"):
        # Afficher un spinner pendant que le RAG travaille
        with st.spinner("Je cherche la meilleure réponse..."):
            response = rag_chain.invoke(user_question)
            st.markdown(response)
    
    # Ajouter la réponse du chatbot à l'historique
    st.session_state.messages.append({"role": "assistant", "content": response})