"""
=============================================================
 src/preprocessing.py — Pipeline de Préparation des Données
=============================================================
Fonctions réutilisables pour le nettoyage, l'imputation,
le feature engineering et la transformation du dataset Titanic.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
#  1. CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Charge le dataset CSV et affiche un résumé rapide."""
    df = pd.read_csv(filepath)
    print(f"✅ Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"   Colonnes : {list(df.columns)}")
    return df


def show_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Affiche les valeurs manquantes sous forme de tableau."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    result = pd.DataFrame({
        "Manquantes": missing,
        "Pourcentage (%)": missing_pct
    })
    result = result[result["Manquantes"] > 0].sort_values("Pourcentage (%)", ascending=False)
    print("\n📊 Valeurs manquantes :")
    print(result)
    return result


# ─────────────────────────────────────────────
#  2. EXTRACTION DU TITRE
# ─────────────────────────────────────────────

def extract_title(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait le titre (Mr, Mrs, Miss, Master, Rare) du nom."""
    df = df.copy()
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)

    # Regrouper les titres rares
    rare_titles = [
        "Lady", "Countess", "Capt", "Col", "Don", "Dr",
        "Major", "Rev", "Sir", "Jonkheer", "Dona"
    ]
    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    print(f"✅ Titres extraits : {dict(df['Title'].value_counts())}")
    return df


# ─────────────────────────────────────────────
#  3. IMPUTATION DES VALEURS MANQUANTES
# ─────────────────────────────────────────────

def impute_age(df: pd.DataFrame) -> pd.DataFrame:
    """Impute l'âge par la médiane selon le groupe (Sex, Pclass, Title)."""
    df = df.copy()
    missing_before = df["Age"].isnull().sum()

    # Imputation par médiane groupée (Title, Pclass)
    age_medians = df.groupby(["Title", "Pclass"])["Age"].median()

    for idx in df[df["Age"].isnull()].index:
        title = df.loc[idx, "Title"]
        pclass = df.loc[idx, "Pclass"]
        try:
            df.loc[idx, "Age"] = age_medians.loc[(title, pclass)]
        except KeyError:
            df.loc[idx, "Age"] = df["Age"].median()

    print(f"✅ Age imputé : {missing_before} valeurs manquantes → 0")
    return df


def impute_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Impute le port d'embarquement par le mode (S)."""
    df = df.copy()
    missing = df["Embarked"].isnull().sum()
    df["Embarked"] = df["Embarked"].fillna("S")
    print(f"✅ Embarked imputé : {missing} valeurs → mode 'S'")
    return df


def impute_fare(df: pd.DataFrame) -> pd.DataFrame:
    """Impute le tarif par la médiane de la classe."""
    df = df.copy()
    missing = df["Fare"].isnull().sum()
    if missing > 0:
        df["Fare"] = df.groupby("Pclass")["Fare"].transform(
            lambda x: x.fillna(x.median())
        )
        print(f"✅ Fare imputé : {missing} valeurs → médiane par classe")
    return df


# ─────────────────────────────────────────────
#  4. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée les nouvelles variables dérivées."""
    df = df.copy()

    # Taille de la famille
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # Voyageur seul
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Présence de cabine (indicateur de statut)
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    # Pont (première lettre de la cabine)
    df["Deck"] = df["Cabin"].apply(lambda x: str(x)[0] if pd.notna(x) else "U")

    # Tranche d'âge
    df["AgeBin"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 50, 80],
        labels=["Enfant", "Ado", "JeuneAdulte", "Mature", "Senior"]
    )

    # Tranche de tarif (bins fixes pour compatibilité batch + inférence)
    df["FareBin"] = pd.cut(
        df["Fare"],
        bins=[-1, 7.91, 14.45, 31.0, 600],
        labels=["Low", "Mid", "High", "VeryHigh"]
    )

    print("✅ Features créées : FamilySize, IsAlone, HasCabin, Deck, AgeBin, FareBin")
    return df


# ─────────────────────────────────────────────
#  5. ENCODAGE
# ─────────────────────────────────────────────

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode les variables catégorielles."""
    df = df.copy()

    # Encodage fixe pour Sex (female=0, male=1) — cohérent batch et inférence
    df["Sex"] = df["Sex"].map({"female": 0, "male": 1}).fillna(0).astype(int)

    # One-Hot Encoding pour Embarked, Title, Deck, AgeBin, FareBin
    categorical_cols = ["Embarked", "Title", "Deck", "AgeBin", "FareBin"]
    existing_cols = [c for c in categorical_cols if c in df.columns]
    df = pd.get_dummies(df, columns=existing_cols, drop_first=True, dtype=int)

    print(f"✅ Encodage terminé — {df.shape[1]} colonnes après transformation")
    return df


# ─────────────────────────────────────────────
#  6. NETTOYAGE FINAL & SÉLECTION
# ─────────────────────────────────────────────

def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les colonnes inutiles pour la modélisation."""
    cols_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_drops)
    print(f"✅ Colonnes supprimées : {existing_drops}")
    return df


# ─────────────────────────────────────────────
#  7. PIPELINE COMPLÈTE
# ─────────────────────────────────────────────

def full_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline complète de preprocessing :
    Title → Imputation → Features → Encodage → Nettoyage
    """
    print("\n" + "=" * 60)
    print("🔧 PIPELINE DE PREPROCESSING")
    print("=" * 60)

    df = extract_title(df)
    df = impute_age(df)
    df = impute_embarked(df)
    df = impute_fare(df)
    df = create_features(df)
    df = encode_features(df)
    df = drop_useless_columns(df)

    print("=" * 60)
    print(f"✅ Pipeline terminée — Shape finale : {df.shape}")
    print(f"   Colonnes : {list(df.columns)}")
    print("=" * 60 + "\n")
    return df


def prepare_train_test(df: pd.DataFrame, target: str = "Survived",
                       test_size: float = 0.2, random_state: int = 42):
    """
    Sépare les données en train/test et normalise les features.
    
    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Normalisation
    scaler = StandardScaler()
    feature_names = X_train.columns.tolist()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)

    print(f"✅ Split train/test : {X_train.shape[0]} train, {X_test.shape[0]} test")
    print(f"   Distribution cible — Train: {y_train.mean():.1%} | Test: {y_test.mean():.1%}")

    return X_train, X_test, y_train, y_test, scaler, feature_names


# ─────────────────────────────────────────────
#  EXÉCUTION DIRECTE (test)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data("data/train.csv")
    show_missing(df)
    df = full_preprocessing(df)
    X_train, X_test, y_train, y_test, scaler, features = prepare_train_test(df)
    print(f"\n🎯 Prêt pour la modélisation avec {len(features)} features !")
