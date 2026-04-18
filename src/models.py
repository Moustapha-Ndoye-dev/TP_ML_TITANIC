"""
=============================================================
 src/models.py — Registre des 8+ Modèles ML
=============================================================
Définition centralisée de tous les modèles, hyperparamètres
par défaut et grilles pour Grid Search CV.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    VotingClassifier,
    StackingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


# ─────────────────────────────────────────────
#  1. LES 8 MODÈLES DE BASE
# ─────────────────────────────────────────────

def get_all_models() -> dict:
    """
    Retourne un dictionnaire de 8 modèles ML avec
    leurs hyperparamètres par défaut.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0,
            solver="lbfgs"
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            weights="uniform",
            metric="minkowski"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=42,
            criterion="gini"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ),
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=42
        ),
        "Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0
        ),
    }
    return models


# ─────────────────────────────────────────────
#  2. GRILLES POUR GRID SEARCH CV
# ─────────────────────────────────────────────

def get_param_grids() -> dict:
    """
    Retourne les grilles d'hyperparamètres pour
    l'optimisation par Grid Search CV.
    """
    grids = {
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["lbfgs", "liblinear"],
            "penalty": ["l2"],
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 9, 11, 15],
            "weights": ["uniform", "distance"],
            "metric": ["minkowski", "euclidean", "manhattan"],
        },
        "Decision Tree": {
            "max_depth": [3, 5, 7, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"],
        },
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 7, 10, 15],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "Gradient Boosting": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 4, 5],
            "subsample": [0.8, 1.0],
        },
        "SVM": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear", "poly"],
            "gamma": ["scale", "auto"],
        },
        "Naive Bayes": {
            "var_smoothing": np.logspace(-12, -6, 7),
        },
        "XGBoost": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 4, 5, 6],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
        },
    }
    return grids


# ─────────────────────────────────────────────
#  3. MODÈLES D'ENSEMBLE LEARNING
# ─────────────────────────────────────────────

def get_voting_classifier(voting: str = "soft") -> VotingClassifier:
    """
    Constructeur de Voting Classifier combinant les meilleurs modèles.
    
    Args:
        voting: 'hard' (majorité) ou 'soft' (probabilités moyennées)
    """
    estimators = [
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1)),
        ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ("xgb", XGBClassifier(n_estimators=100, random_state=42, verbosity=0,
                              use_label_encoder=False, eval_metric="logloss")),
        ("svm", SVC(probability=True, random_state=42)),
    ]
    return VotingClassifier(estimators=estimators, voting=voting)


def get_bagging_classifier() -> BaggingClassifier:
    """Constructeur de Bagging Classifier avec Decision Tree de base."""
    return BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=7, random_state=42),
        n_estimators=50,
        max_samples=0.8,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    )


def get_stacking_classifier() -> StackingClassifier:
    """
    Constructeur de Stacking Classifier :
    - Niveau 1 : RF, GB, SVM, XGBoost
    - Méta-modèle : Logistic Regression
    """
    base_estimators = [
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1)),
        ("gb", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ("svm", SVC(probability=True, random_state=42)),
        ("xgb", XGBClassifier(n_estimators=100, random_state=42, verbosity=0,
                              use_label_encoder=False, eval_metric="logloss")),
    ]
    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,
        n_jobs=-1
    )


def get_adaboost_classifier() -> AdaBoostClassifier:
    """Constructeur d'AdaBoost Classifier."""
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )


def get_ensemble_models() -> dict:
    """Retourne tous les modèles d'ensemble learning."""
    return {
        "Voting (Soft)": get_voting_classifier("soft"),
        "Voting (Hard)": get_voting_classifier("hard"),
        "Bagging": get_bagging_classifier(),
        "Stacking": get_stacking_classifier(),
        "AdaBoost": get_adaboost_classifier(),
    }


# ─────────────────────────────────────────────
#  EXÉCUTION DIRECTE (test)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("📦 Modèles de base disponibles :")
    for name, model in get_all_models().items():
        print(f"   ▸ {name}: {model.__class__.__name__}")

    print("\n📦 Modèles d'ensemble disponibles :")
    for name, model in get_ensemble_models().items():
        print(f"   ▸ {name}: {model.__class__.__name__}")

    print("\n📦 Grilles Grid Search disponibles :")
    for name, grid in get_param_grids().items():
        total = 1
        for values in grid.values():
            total *= len(values) if hasattr(values, '__len__') else 1
        print(f"   ▸ {name}: {len(grid)} params, ~{total} combinaisons")
