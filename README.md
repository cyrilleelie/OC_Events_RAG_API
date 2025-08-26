# Projet : POC Chatbot RAG - Puls-Events

Ce projet est un Proof of Concept (POC) d'un chatbot intelligent conçu pour la société fictive Puls-Events. Le chatbot est capable de répondre à des questions en langage naturel sur des événements culturels en s'appuyant sur une architecture RAG (Retrieval-Augmented Generation) et les données de la métropole nantaise.

## 🎯 Objectifs du POC

* **Démontrer la faisabilité technique** d'un système RAG combinant recherche sémantique et LLM.
* **Fournir une API RESTful** robuste et exploitable pour tester la solution.
* **Évaluer quantitativement** la qualité et la pertinence des réponses générées.
* **Proposer un artefact conteneurisé** avec Docker, prêt pour une démo ou un déploiement.

## ✨ Technologies Utilisées

* **Langage :** Python 3.12
* **Gestion de dépendances :** Poetry
* **API & Serveur :** Flask & Gunicorn
* **Interface de Chat :** Streamlit
* **Orchestration RAG :** LangChain
* **Base Vectorielle :** FAISS (CPU)
* **Modèle d'Embedding :** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
* **LLM de Génération :**
    * **Local :** N'importe quel modèle compatible via **Ollama** (ex: Mistral 7B)
    * **API :** **Mistral AI** ou **Google Gemini**
* **Tests & Évaluation :** Pytest & Ragas
* **Conteneurisation :** Docker

## 📂 Structure du Projet

```
puls-events-rag/
├── Dockerfile
├── README.md                 # Ce fichier
├── app_streamlit.py          # Interface Streamlit
├── api/
│   └── main.py               # API Flask
├── evaluation_dataset.json   # Jeu de test pour l'évaluation Ragas
├── pyproject.toml            # Dépendances et configuration Poetry
├── poetry.lock
├── requirements.txt          # Généré pour le build Docker
│
├── data/                     # Données CSV (ignoré par Git)
│
├── reports/                  # Rapports d'évaluation (ignoré par Git)
│
├── scripts/                  # Scripts pour le pipeline de données et l'évaluation
│
├── src/                      # Code source de la librairie principale
│   └── puls_events_rag/
│       ├── config.py         # <-- FICHIER DE CONFIGURATION CENTRAL
│       └── rag_chain.py
│
├── tests/                    # Tests automatisés
│
└── vector_store/             # Index FAISS (ignoré par Git)
```

## 🛠️ Configuration du Chatbot

Ce projet est conçu pour être flexible. La configuration se fait via le fichier `src/puls_events_rag/config.py` et des variables d'environnement dans un fichier `.env`.

### Fichier `config.py`

Vous pouvez modifier `src/puls_events_rag/config.py` pour :
* **Changer la source de données :** Modifiez les variables `PORTAL_URL` et `DATASET_ID`.
* **Adapter le "rôle" de l'assistant :** Modifiez la variable `PROMPT_TEMPLATE`.
* **Tester d'autres modèles d'embedding.**
* **Ajuster les performances du RAG :** Modifiez `CHUNK_SIZE`, `CHUNK_OVERLAP` et `RETRIEVER_K`.

### Fichier `.env` (Sélecteur de LLM et Clés API)

Créez un fichier `.env` à la racine pour choisir le LLM et fournir les clés API nécessaires.

* **`LLM_PROVIDER`** : C'est la variable clé qui détermine quel modèle sera utilisé.
    * `"mistral_local"` : Pour utiliser un modèle via Ollama.
    * `"mistral_api"` : Pour utiliser l'API de Mistral AI.
    * `"google_api"` : Pour utiliser l'API de Google Gemini.
* **Clés API** : Fournissez les clés correspondantes au fournisseur choisi. `OPENAI_API_KEY` est requise pour le script d'évaluation Ragas.

**Exemple pour utiliser Ollama en local :**
```env
LLM_PROVIDER="mistral_local"
OPENAI_API_KEY="sk-..." # Uniquement pour l'évaluation Ragas
```

**Exemple pour utiliser l'API Google Gemini :**
```env
LLM_PROVIDER="google_api"
GOOGLE_API_KEY="AIzaSy..."
OPENAI_API_KEY="sk-..."
```

## 🚀 Installation et Pipeline de Données

**Prérequis :** Git, Python 3.12+, Poetry, et Docker Desktop. Si vous utilisez un modèle local, **Ollama** doit être installé et en cours d'exécution.

1.  **Cloner le dépôt :**
    ```bash
    git clone https://github.com/cyrilleelie/OC_Events_RAG_API
    cd OC_Events_RAG_API
    ```

2.  **Configurer le fichier `.env`** comme expliqué dans la section précédente.

3.  **Installer les dépendances Python :**
    ```bash
    poetry install
    ```

4.  **Exécuter le pipeline de données** (indispensable avant la première utilisation) :
    * `poetry run python scripts/fetch_data.py`
    * `poetry run python scripts/create_vector_store.py`

## 💬 Lancement de l'Interface de Chat (Streamlit)

C'est la méthode la plus simple pour interagir avec le chatbot en local. Le modèle utilisé dépendra de la variable `LLM_PROVIDER` dans votre fichier `.env`.

```bash
poetry run streamlit run app_streamlit.py
```
Votre navigateur s'ouvrira automatiquement sur l'interface du chatbot.

## 🐳 Lancement via Docker (API)

Cette méthode lance l'API Flask conteneurisée.

1.  **Construire l'image Docker :**
    ```bash
    docker build -t puls-events-rag-api .
    ```

2.  **Lancer le conteneur Docker :**
    La commande varie selon le LLM que vous souhaitez utiliser.

    * **Mode 1 : Modèle local (Ollama) - (Par défaut)**
        *Prérequis : Ollama doit tourner sur votre machine.*
        ```bash
        docker run -p 5001:5000 puls-events-rag-api
        ```

    * **Mode 2 : API Google Gemini**
        ```bash
        docker run -p 5001:5000 -e LLM_PROVIDER="google_api" --env-file .env puls-events-rag-api
        ```

    * **Mode 3 : API Mistral**
        ```bash
        docker run -p 5001:5000 -e LLM_PROVIDER="mistral_api" --env-file .env puls-events-rag-api
        ```
        
3.  **Tester l'API :**
    Envoyez une requête `POST` à `http://127.0.0.1:5001/ask` avec un corps JSON :
    ```json
    {
        "question": "Y a-t-il des expositions sur l'île de Nantes ?"
    }
    ```

## ✅ Tests et Évaluation

1.  **Tests fonctionnels de l'API :**
    * Lancez l'API (localement ou via Docker).
    * Dans un autre terminal, lancez : `poetry run pytest`

2.  **Évaluation de la qualité du RAG :**
    * **Principe :** Le script `scripts/evaluate_rag.py` compare les réponses du chatbot (générées avec le `LLM_PROVIDER` de votre `.env`) à une "vérité terrain" (`evaluation_dataset.json`). Il utilise Ragas pour noter la performance.
    * **Prérequis :** L'évaluation Ragas nécessite une clé `OPENAI_API_KEY` dans votre fichier `.env` pour son LLM "juge".
    * **Lancer l'évaluation :**
        ```bash
        poetry run python scripts/evaluate_rag.py
        ```
    * Les résultats sont affichés et sauvegardés dans le dossier `reports/`.
