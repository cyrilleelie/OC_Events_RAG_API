# scripts/fetch_data.py (VERSION FINALE ET CORRIGÉE)

import os
import re
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv
from puls_events_rag import config

# --- Configuration Opendatasoft ---
load_dotenv()
API_KEY = os.getenv("OPENDATASOFT_API_KEY")

PORTAL_URL = "https://data.nantesmetropole.fr"
DATASET_ID = "244400404_agenda-evenements-nantes-nantes-metropole"
API_ENDPOINT = f"{PORTAL_URL}/api/explore/v2.1/catalog/datasets/{DATASET_ID}/records"

RECORDS_PER_PAGE = 100

os.makedirs("data", exist_ok=True)
OUTPUT_FILE = config.DATA_PATH

def clean_html(raw_html: str) -> str:
    """Supprime les balises HTML et les retours à la ligne excessifs."""
    if not isinstance(raw_html, str):
        return ""
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext.replace('\n', ' ').strip()

def fetch_all_records() -> list:
    """Récupère tous les enregistrements d'un jeu de données Opendatasoft."""
    all_records = []
    offset = 0
    one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    where_clause = f"date > date'{one_year_ago}'"

    print(f"Début de la récupération pour le dataset '{DATASET_ID}'...")
    
    while True:
        print(f"Récupération de {RECORDS_PER_PAGE} enregistrements (offset: {offset})...")
        params = {
            "limit": RECORDS_PER_PAGE, "offset": offset,
            "where": where_clause, "apikey": API_KEY
        }
        try:
            response = requests.get(config.API_ENDPOINT, params=params)
            response.raise_for_status()
            data = response.json()
            records = data.get("results", [])
            if not records:
                break
            all_records.extend(records)
            if len(records) < RECORDS_PER_PAGE:
                break
            offset += RECORDS_PER_PAGE
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de l'appel API : {e}")
            break
    print(f"Récupération terminée. Total d'enregistrements bruts : {len(all_records)}")
    return all_records

def process_and_structure_records(records: list) -> pd.DataFrame:
    """
    Traite la liste d'enregistrements bruts pour la structurer dans un DataFrame,
    en utilisant les BONS noms de champs.
    """
    processed_data = []
    for fields in records: # On peut directement itérer sur les records
        
        # ON CORRIGE LA CONDITION 'IF' AVEC LES BONS NOMS
        if fields.get("nom") and fields.get("description"):
            
            # ON CORRIGE LE MAPPING AVEC LES BONS NOMS
            processed_data.append({
                "uid": fields.get("id_manif"),
                "title": fields.get("nom"),
                "description": clean_html(fields.get("description")),
                "date_text": fields.get("date"), # On renomme pour plus de clarté
                "location_name": fields.get("lieu"),
                "location_address": fields.get("adresse"),
                # On nettoie les virgules en trop dans les mots-clés
                "keywords": fields.get("type", "").replace(",", " ").strip(),
                "url": fields.get("url_site") or fields.get("lien_agenda")
            })
            
    df = pd.DataFrame(processed_data)
    
    # Ces opérations devraient maintenant fonctionner sans erreur
    df.dropna(subset=["uid", "description"], inplace=True)
    df.drop_duplicates(subset=["uid"], inplace=True)
    df = df[df["description"] != ""].reset_index(drop=True)

    print(f"{len(df)} événements propres et structurés, prêts à être sauvegardés.")
    return df

if __name__ == "__main__":
    raw_records = fetch_all_records()
    if raw_records:
        structured_df = process_and_structure_records(raw_records)
        structured_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSUCCÈS ! Les données ont été sauvegardées dans '{OUTPUT_FILE}'")
    else:
        print("\nLa liste d'événements bruts est vide. Le fichier CSV n'a donc pas été créé.")