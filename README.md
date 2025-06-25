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
* **Orchestration RAG :** LangChain
* **Base Vectorielle :** FAISS (CPU)
* **Modèle d'Embedding :** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
* **LLM de Génération :** Mistral AI
* **Tests & Évaluation :** Pytest & Ragas
* **Conteneurisation :** Docker

## 📂 Structure du Projet

```
puls-events-rag/
├── .dockerignore
├── Dockerfile
├── README.md                 # Ce fichier
├── main.py                   # Point d'entrée de l'API Flask
├── evaluation_dataset.json   # Jeu de test pour l'évaluation Ragas
├── pyproject.toml            # Dépendances et configuration Poetry
├── poetry.lock
├── requirements.txt          # Généré pour le build Docker
│
├── data/                     # Données CSV (ignoré par Git)
│
├── scripts/                  # Scripts pour le traitement des données et l'évaluation
│
├── src/                      # Code source de la librairie principale
│   └── puls_events_rag/
│       ├── init.py
│       ├── config.py         # <-- FICHIER DE CONFIGURATION CENTRAL
│       └── rag_chain.py
│
├── tests/                    # Tests automatisés
│
└── vector_store/             # Index FAISS (ignoré par Git)
```

## 🛠️ Configuration du Chatbot

Ce projet est conçu pour être flexible. Tous les paramètres clés sont centralisés dans le fichier suivant pour une adaptation facile : `src/puls_events_rag/config.py`.

Vous pouvez modifier ce fichier pour :
* **Changer la source de données :** Modifiez les variables `PORTAL_URL` et `DATASET_ID` pour cibler un autre jeu de données Opendatasoft.
* **Adapter le "rôle" de l'assistant :** Modifiez la variable `PROMPT_TEMPLATE` pour changer la personnalité, les instructions ou le domaine d'expertise du chatbot.
* **Tester d'autres modèles :** Changez les noms des modèles dans `EMBEDDING_MODEL_NAME` et `LLM_MODEL_NAME`.
* **Ajuster les performances du RAG :** Modifiez les paramètres `CHUNK_SIZE`, `CHUNK_OVERLAP` et `RETRIEVER_K`.

## 🚀 Installation et Usage

**Prérequis :** Git, Python 3.12+, Poetry, et Docker Desktop.

1.  **Cloner le dépôt :**
    ```bash
    git clone https://github.com/cyrilleelie/OC_Events_RAG_API
    cd OC_Events_RAG_API
    ```

2.  **Configurer les secrets :**
    * Créez un fichier `.env` à la racine du projet.
    * Ajoutez les clés API nécessaires. **Attention : la clé requise dépend de la source de données que vous avez configurée dans `config.py`.**
        ```env
        # Clé API pour le LLM de génération
        MISTRAL_API_KEY="sk-xxxxxxxxxx"

        # Clé API pour la source de données (si nécessaire)
        # Par exemple, pour l'API Opendatasoft :
        OPENDATA_API_KEY="xxxxxxxxxx"
        ```

3.  **Installer les dépendances :**
    ```bash
    poetry install
    ```

4.  **Exécuter le pipeline de données :**
    * `poetry run python scripts/fetch_data.py`
    * `poetry run python scripts/create_vector_store.py`

## 🐳 Lancement via Docker (Méthode recommandée)

1.  **Générer `requirements.txt` (si pas à jour) :**
    ```bash
    poetry export -f requirements.txt --output requirements.txt --without-hashes
    ```

2.  **Construire l'image Docker :**
    ```bash
    docker build -t puls-events-rag-api .
    ```

3.  **Lancer le conteneur Docker :**
    *La première exécution peut être longue, le temps de charger les modèles en mémoire.*
    ```bash
    docker run -p 5001:5000 puls-events-rag-api
    ```

4.  **Tester l'API conteneurisée :**
    Envoyez une requête `POST` à `http://127.0.0.1:5001/ask` avec un client API comme Postman.
    **Corps JSON :**
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
    * **Principe :** Le script `scripts/evaluate_rag.py` compare les réponses du chatbot à une "vérité terrain" définie dans le fichier `evaluation_dataset.json`.

    * **Personnaliser le jeu de test :**
      Vous pouvez (et devriez !) adapter le fichier `evaluation_dataset.json` à vos propres cas d'usage.
        * **Ouvrez** le fichier `evaluation_dataset.json`.
        * **Ajoutez ou modifiez** les paires de questions (`question`) et de réponses idéales (`ground_truth`).
        * La `ground_truth` doit être la réponse factuelle et parfaite que vous attendez, basée sur les données que vous avez collectées.

        **Exemple d'entrée dans `evaluation_dataset.json` :**
        ```json
        {
            "question": "Parle-moi de l'exposition à la HAB Galerie.",
            "ground_truth": "L'exposition à la HAB Galerie est celle de l'artiste Gloria Friedmann, intitulée 'Combien de terres faut-il à l’homme ?'. L'entrée est libre et elle explore le thème de la cupidité humaine."
        }
        ```

    * **Lancer l'évaluation :**
      ```bash
      poetry run python scripts/evaluate_rag.py
