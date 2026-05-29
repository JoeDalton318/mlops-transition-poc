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
"""

import pandas as pd
import numpy as np
from pathlib import Path


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
    # Base: surface and location are primary drivers
    # Adjustments: building age, energy class, amenities
    prix_base = (surface_m2 * 5.5) + (distance_centre_km * (-1.2))

    # Age penalty: older buildings worth less (depreciation ~0.3% per year)
    age_factor = (2024 - annee_construction) * 0.003
    prix_adjusted = prix_base * (1 - age_factor)

    # Energy class premium: better efficiency adds value
    energy_premium = (dpe_energy_class - 3.5) * 8

    # Amenities: balcony and parking add value
    balcony_value = has_balcony * 15
    parking_value = has_parking * 20

    # Room count multiplier
    room_multiplier = 1 + (nb_pieces * 0.08)

    prix_k_eur = (prix_adjusted + energy_premium + balcony_value + parking_value) * room_multiplier

    # Add realistic noise (market variations)
    prix_k_eur += np.random.normal(loc=0, scale=25, size=n_samples)
    prix_k_eur = np.clip(prix_k_eur, a_min=50, a_max=500)

    # Construct DataFrame
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
    
    Outputs a file 'immobilier_france.csv' in the current directory
    with 500 synthetic French housing records.
    """
    output_file = Path(__file__).parent / 'immobilier_france.csv'

    print("Generating synthetic French housing dataset...")
    df = generate_housing_dataset(n_samples=500, random_state=42)

    print(f"Dataset shape: {df.shape}")
    print(f"\nDataset preview:\n{df.head()}")
    print(f"\nDataset statistics:\n{df.describe()}")

    df.to_csv(output_file, index=False)
    print(f"\nDataset saved to: {output_file}")


if __name__ == '__main__':
    main()