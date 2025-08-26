# tests/test_data_pipeline.py

import pytest
import pandas as pd
from io import StringIO

# Supposons que vous avez extrait la logique de nettoyage dans une fonction
# dans un fichier `scripts/data_processing.py` pour la rendre testable.
# Si ce n'est pas le cas, vous devrez le faire.
# from scripts.data_processing import clean_dataframe 

# --- Mock de la fonction de nettoyage pour l'exemple ---
# Vous devrez remplacer ceci par votre véritable fonction importée.
def clean_dataframe(df):
    """Nettoie le DataFrame des événements."""
    # Remplacer les NaN uniquement dans les colonnes de type texte
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna("")
    return df
# --- Fin du Mock ---


@pytest.fixture
def sample_dirty_dataframe():
    """Crée un DataFrame pandas "sale" pour les tests."""
    csv_data = """id,nom,description,participants
1,Concert Rock,,150
2,Expo Photo,Une belle exposition,
3,Atelier Peinture,Pour les enfants,25.0
"""
    return pd.read_csv(StringIO(csv_data))

def test_data_cleaning(sample_dirty_dataframe):
    """
    Teste la fonction de nettoyage des données.
    Vérifie que les NaN dans les colonnes texte sont bien remplacés
    et que les colonnes numériques ne sont pas affectées.
    """
    df_dirty = sample_dirty_dataframe
    
    # On applique la fonction de nettoyage
    df_cleaned = clean_dataframe(df_dirty)
    
    # --- Assertions ---
    
    # 1. Vérifier que le NaN dans la colonne 'nom' a été remplacé
    # (Ici, il n'y en a pas, mais c'est un bon test à avoir)
    assert df_cleaned.loc[0, 'nom'] == 'Concert Rock'
    
    # 2. Vérifier que le NaN dans la colonne 'description' a été remplacé
    assert df_cleaned.loc[0, 'description'] == ""
    
    # 3. Vérifier que le NaN dans la colonne numérique 'participants' n'a PAS été remplacé par ""
    # pandas conserve le NaN pour les colonnes numériques, ce qui est correct.
    assert pd.isna(df_cleaned.loc[1, 'participants'])
    
    # 4. Vérifier que les autres valeurs sont intactes
    assert df_cleaned.loc[2, 'nom'] == 'Atelier Peinture'
    assert df_cleaned.loc[2, 'participants'] == 25.0

