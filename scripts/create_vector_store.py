# scripts/create_vector_store.py

import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
DATA_PATH = "data/events_244400404_agenda-evenements-nantes-nantes-metropole.csv"
VECTOR_STORE_PATH = "vector_store" # Le dossier où sera sauvegardé l'index FAISS
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- 1. Chargement des données ---
print("Étape 1/5 : Chargement des données depuis le fichier CSV...")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Le fichier de données n'a pas été trouvé : {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
df.fillna("", inplace=True) # Remplacer les NaN par des chaînes vides
print(f"-> {len(df)} événements chargés.")

# --- 2. Création des Documents LangChain ---
# Chaque ligne du CSV devient un "Document" avec du contenu (page_content) et des métadonnées.
# Les métadonnées sont cruciales pour retrouver l'information originale.
print("\nÉtape 2/5 : Création des documents LangChain...")
documents = []
for _, row in df.iterrows():
    # On concatène les informations textuelles les plus pertinentes pour la recherche
    page_content = (
        f"Titre : {row.get('title', '')}\n"
        f"Description : {row.get('description', '')}\n"
        f"Lieu : {row.get('location_name', '')}\n"
        f"Mots-clés : {row.get('keywords', '')}"
    )
    
    metadata = {
        "source": DATA_PATH,
        "event_uid": row.get('uid', ''),
        "title": row.get('title', ''),
        "url": row.get('url', ''),
        "location": row.get('location_name', ''),
        "date_text": row.get('date_text', '')
    }
    documents.append(Document(page_content=page_content, metadata=metadata))
print(f"-> {len(documents)} documents créés.")

# --- 3. Découpage des documents en chunks ---
print("\nÉtape 3/5 : Découpage des documents en chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Taille de chaque chunk en caractères
    chunk_overlap=200, # Chevauchement entre les chunks pour ne pas perdre de contexte
    length_function=len,
)
split_chunks = text_splitter.split_documents(documents)
print(f"-> Les documents ont été découpés en {len(split_chunks)} chunks.")

# --- 4. Création des embeddings et de l'index FAISS ---
# Cette étape peut prendre du temps la première fois car elle télécharge le modèle.
print("\nÉtape 4/5 : Chargement du modèle d'embedding (peut être long au premier lancement)...")
embeddings_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)

print("\nCréation de l'index vectoriel FAISS...")
# C'est ici que LangChain fait la magie : il prend les chunks, calcule les embeddings
# et les stocke dans une base de données FAISS.
vector_store = FAISS.from_documents(documents=split_chunks, embedding=embeddings_model)
print("-> Index FAISS créé en mémoire.")

# --- 5. Sauvegarde de l'index ---
print("\nÉtape 5/5 : Sauvegarde de l'index FAISS sur le disque...")
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
vector_store.save_local(VECTOR_STORE_PATH)
print(f"\nSUCCÈS ! L'index vectoriel a été créé et sauvegardé dans le dossier '{VECTOR_STORE_PATH}'.")