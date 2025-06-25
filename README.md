# Projet : POC de Chatbot RAG pour Puls-Events

Ce projet est un Proof of Concept (POC) d'un chatbot intelligent basé sur une architecture RAG. Le chatbot répond aux questions des utilisateurs sur des événements culturels en se basant sur les données de l'API Open Agenda.

## Architecture

- **Modèle de langage (LLM)** : Mistral
- **Recherche sémantique** : FAISS
- **Orchestration** : LangChain
- **API** : FastAPI
- **Gestion de dépendances** : Poetry

## Instructions d'installation et de reproduction

**Prérequis :** [Python](https://www.python.org/) (>=3.8) et [Poetry](https://python-poetry.org/) installés.

1.  **Cloner le dépôt :**
    ```bash
    git clone <URL_DU_DEPOT>
    cd puls-events-rag
    ```

2.  **Installer les dépendances :**
    Cette commande lit le fichier `pyproject.toml`, crée un environnement virtuel et installe toutes les dépendances.
    ```bash
    poetry install
    ```

3.  **Configurer les variables d'environnement :**
    Créez un fichier `.env` à la racine et ajoutez votre clé API Mistral.
    ```bash
    # (Vous pouvez créer un .env.example pour guider les utilisateurs)
    cp .env.example .env
    # Éditez le fichier .env pour y mettre votre clé
    ```

4.  **Activer l'environnement virtuel :**
    Pour exécuter des commandes dans le contexte du projet :
    ```bash
    poetry shell
    ```