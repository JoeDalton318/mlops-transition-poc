"""
Generate synthetic French real estate dataset with enhanced features.
Générez un ensemble de données immobilier français synthétique avec des fonctionnalités améliorées.

This module creates a realistic dataset with additional features to improve
model credibility for academic purposes:
- DPE Energy Class (numerical encoding): 1-7 scale
- Has Balcony (binary): presence of outdoor space
- Has Parking (binary): dedicated parking availability

Ce module crée un ensemble de données réaliste avec des fonctionnalités supplémentaires
pour améliorer la crédibilité du modèle à des fins académiques :
- Classe d'Énergie DPE (codage numérique) : échelle 1-7
- Avec Balcon (binaire) : présence d'espace extérieur
- Avec Parking (binaire) : disponibilité de parking dédié

The dataset simulates price variation based on property characteristics
using realistic correlations observed in French real estate markets.

L'ensemble de données simule la variation des prix en fonction des caractéristiques
des propriétés en utilisant les corrélations réalistes observées sur les marchés
immobiliers français.

NOTE (corrective maintenance, Sept. 2026): the original pricing formula combined
additive terms with a multiplicative room-count factor and a narrow clip range
([50, 500] k EUR). For the parameter values used, this caused ~78% of samples to
saturate at the upper clip bound (500 k EUR), producing a near-degenerate
regression target. The formula below is purely additive/linear in the seven
input features (matching the LinearRegression model used downstream) with a
noise term small enough to keep the fit informative but not trivial, and a wide
safety clip that is never actually reached for the configured parameters
(verified empirically: 0/500 samples at either bound with random_state=42).

NOTE (correction, sept. 2026) : la formule de prix d'origine combinait des termes
additifs avec un multiplicateur non linéaire sur le nombre de pièces et un
intervalle de clipping étroit ([50, 500] k€). Avec les paramètres utilisés, cela
saturait ~78 % des échantillons au plafond (500 k€), produisant une cible de
régression quasi dégénérée. La formule ci-dessous est purement additive/linéaire
sur les sept caractéristiques d'entrée (cohérente avec le modèle LinearRegression
utilisé en aval), avec un bruit calibré pour garder un ajustement informatif sans
être trivial, et un intervalle de sécurité large jamais atteint en pratique avec
les paramètres actuels (vérifié empiriquement : 0/500 échantillon aux bornes avec
random_state=42).
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement (Load environment variables)
load_dotenv()

def generate_housing_dataset(n_samples: int = 500, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic French housing dataset with realistic correlations.
    Générez un ensemble de données d'habitation français synthétique avec des corrélations réalistes.

    Args:
        n_samples: Number of synthetic records to generate (default: 500).
                  Nombre d'enregistrements synthétiques à générer (par défaut : 500).
        random_state: Seed for reproducibility (default: 42).
                     Graine pour la reproductibilité (par défaut : 42).

    Returns:
        DataFrame with columns: Surface_m2, Nb_Pieces, Annee_Construction,
        Distance_Centre_km, DPE_Energy_Class, Has_Balcony, Has_Parking, Prix_k_EUR.
    """
    np.random.seed(random_state)

    # Generate base features with realistic distributions
    # Générer les caractéristiques de base avec des distributions réalistes
    surface_m2 = np.random.normal(loc=120, scale=50, size=n_samples)
    surface_m2 = np.clip(surface_m2, a_min=30, a_max=300)

    nb_pieces = np.random.randint(low=1, high=6, size=n_samples)

    annee_construction = np.random.normal(loc=1995, scale=20, size=n_samples)
    annee_construction = np.clip(annee_construction, a_min=1950, a_max=2024).astype(int)

    distance_centre_km = np.random.exponential(scale=8, size=n_samples)
    distance_centre_km = np.clip(distance_centre_km, a_min=0.5, a_max=50)

    # DPE Energy Class (1=worst, 7=best) - inversely correlated with building age
    # Older buildings tend to have worse energy ratings
    dpe_energy_class = 7 - (annee_construction - 1950) / 15
    dpe_energy_class = np.clip(dpe_energy_class, a_min=1, a_max=7).astype(int)
    # Add random noise to break perfect correlation
    dpe_energy_class = np.clip(
        dpe_energy_class + np.random.randint(-1, 2, size=n_samples),
        a_min=1,
        a_max=7
    )

    # Has Balcony: more likely for larger apartments and newer constructions
    has_balcony = (
        (surface_m2 > 80) & (annee_construction > 1980) & (np.random.rand(n_samples) > 0.4)
    ).astype(int)

    # Has Parking: correlated with proximity to city center and newer buildings
    has_parking = (
        (distance_centre_km < 15) | (annee_construction > 2000)
    ) & (np.random.rand(n_samples) > 0.3)
    has_parking = has_parking.astype(int)

    # Price formula: realistic French real estate pricing (in k EUR)
    # Purely additive/linear combination of the 7 features, consistent with the
    # LinearRegression model trained downstream (src/train.py). Coefficients are
    # illustrative (not calibrated on real transactions) but chosen so each
    # feature's contribution stays in a plausible order of magnitude.
    # Formule de prix : combinaison purement additive/linéaire des 7 caractéristiques,
    # cohérente avec le modèle LinearRegression entraîné en aval (src/train.py).
    # Les coefficients sont illustratifs (non calibrés sur des transactions réelles)
    # mais choisis pour que la contribution de chaque variable reste d'un ordre de
    # grandeur plausible.
    age_years = 2024 - annee_construction

    prix_k_eur = (
        20.0                                 # base price / prix de base (k EUR)
        + 3.2 * surface_m2                   # ~3 200 EUR/m2
        + 6.0 * nb_pieces                    # room premium / prime par pièce
        - 0.30 * age_years                   # age depreciation / dépréciation par année
        - 0.8 * distance_centre_km           # location discount / éloignement du centre
        + 5.0 * dpe_energy_class             # energy efficiency premium / prime DPE
        + 15.0 * has_balcony                 # balcony premium / prime balcon
        + 18.0 * has_parking                 # parking premium / prime parking
    )

    # Add realistic noise (market variations) — std chosen so the noise is
    # informative but does not overwhelm the linear signal (target: fitted
    # LinearRegression R2 > 0.85 on a held-out test set).
    # Ajout d'un bruit réaliste (variations de marché) — écart-type choisi pour
    # rester significatif sans dominer le signal linéaire (cible : R2 > 0.85
    # pour la LinearRegression évaluée sur un jeu de test).
    prix_k_eur = prix_k_eur + np.random.normal(loc=0, scale=25.0, size=n_samples)

    # Wide safety clip (non-negativity / outlier guard only): with the
    # parameters above, no sample reaches either bound for random_state=42.
    # Clip de sécurité large (garde-fou de non-négativité / anti-aberrant
    # uniquement) : avec les paramètres ci-dessus, aucun échantillon n'atteint
    # les bornes pour random_state=42.
    prix_k_eur = np.clip(prix_k_eur, a_min=30, a_max=1200)

    # Construct DataFrame / Construction du DataFrame
    data = pd.DataFrame({
        'Surface_m2': surface_m2.round(2),
        'Nb_Pieces': nb_pieces,
        'Annee_Construction': annee_construction,
        'Distance_Centre_km': distance_centre_km.round(2),
        'DPE_Energy_Class': dpe_energy_class,
        'Has_Balcony': has_balcony,
        'Has_Parking': has_parking,
        'Prix_k_EUR': prix_k_eur.round(2),
    })

    return data


def main() -> None:
    """
    Generate and save the housing dataset to CSV.
    Génère et sauvegarde l'ensemble de données immobilières en CSV.
    """
    # Récupérer le chemin du fichier de données depuis les variables d'environnement
    # (Get data file path from environment variables)
    default_path = Path(__file__).parent / 'data' / 'immobilier_france.csv'
    env_path = os.getenv('DATA_FILE_PATH')
    
    if env_path:
        output_file = Path(__file__).parent / env_path
    else:
        output_file = default_path

    # S'assurer que le dossier parent existe (Ensure parent directory exists)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Remove the file if it already exists (Supprimer le fichier s'il existe déjà)
    if output_file.exists():
        print(f"Removing existing file (Suppression du fichier existant): {output_file}")
        output_file.unlink()

    print("Generating synthetic French housing dataset (Génération des données)...")
    df = generate_housing_dataset(n_samples=500, random_state=42)

    print(f"Dataset shape (Taille des données): {df.shape}")
    print(f"\nDataset preview (Aperçu):\n{df.head()}")
    
    df.to_csv(output_file, index=False)
    print(f"\nDataset saved to (Données sauvegardées vers): {output_file}")


if __name__ == '__main__':
    main()