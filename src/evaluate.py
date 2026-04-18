"""
=============================================================
 src/evaluate.py — Métriques d'Évaluation & Visualisations
=============================================================
Fonctions pour évaluer, comparer et visualiser les performances
des modèles ML sur le dataset Titanic.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, learning_curve
import joblib

# ── Style global ──
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100

FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


# ─────────────────────────────────────────────
#  1. ENTRAÎNER & ÉVALUER UN MODÈLE
# ─────────────────────────────────────────────

def train_and_evaluate(model, X_train, X_test, y_train, y_test, name: str = "Model") -> dict:
    """
    Entraîne un modèle et retourne ses métriques sur train et test.
    
    Returns:
        dict avec accuracy, precision, recall, f1, auc pour train et test
    """
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Probabilités pour AUC (si le modèle le supporte)
    try:
        y_proba_test = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba_test)
    except (AttributeError, ValueError):
        auc = None

    results = {
        "Modèle": name,
        "Train Accuracy": accuracy_score(y_train, y_pred_train),
        "Test Accuracy": accuracy_score(y_test, y_pred_test),
        "Precision": precision_score(y_test, y_pred_test),
        "Recall": recall_score(y_test, y_pred_test),
        "F1-Score": f1_score(y_test, y_pred_test),
        "AUC-ROC": auc if auc else np.nan,
        "Overfit": accuracy_score(y_train, y_pred_train) - accuracy_score(y_test, y_pred_test),
    }

    return results


# ─────────────────────────────────────────────
#  2. COMPARER TOUS LES MODÈLES
# ─────────────────────────────────────────────

def compare_models(models: dict, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """
    Entraîne et compare tous les modèles.
    
    Args:
        models: dict {nom: modèle}
    
    Returns:
        DataFrame trié par Test Accuracy (décroissant)
    """
    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"   ▸ Entraînement de {name}...", end=" ")
        metrics = train_and_evaluate(model, X_train, X_test, y_train, y_test, name)
        results.append(metrics)
        trained_models[name] = model
        print(f"✓ Accuracy: {metrics['Test Accuracy']:.4f}")

    df = pd.DataFrame(results).set_index("Modèle")
    df = df.sort_values("Test Accuracy", ascending=False)

    # Mettre en forme
    print("\n" + "=" * 70)
    print("📊 COMPARAISON DES MODÈLES")
    print("=" * 70)
    print(df.round(4).to_string())

    return df, trained_models


# ─────────────────────────────────────────────
#  3. CROSS-VALIDATION
# ─────────────────────────────────────────────

def cross_validate_models(models: dict, X, y, cv: int = 5) -> pd.DataFrame:
    """
    Évalue les modèles avec la validation croisée (k-fold).
    
    Returns:
        DataFrame avec mean ± std pour chaque modèle
    """
    results = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        results.append({
            "Modèle": name,
            "CV Mean": scores.mean(),
            "CV Std": scores.std(),
            "CV Scores": f"{scores.mean():.4f} ± {scores.std():.4f}",
        })
        print(f"   ▸ {name}: {scores.mean():.4f} ± {scores.std():.4f}")

    df = pd.DataFrame(results).set_index("Modèle")
    return df.sort_values("CV Mean", ascending=False)


# ─────────────────────────────────────────────
#  4. VISUALISATIONS
# ─────────────────────────────────────────────

def plot_model_comparison(results_df: pd.DataFrame, save: bool = True):
    """Diagramme en barres comparant Test Accuracy de tous les modèles."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = sns.color_palette("viridis", len(results_df))
    bars = ax.barh(results_df.index, results_df["Test Accuracy"], color=colors, edgecolor="white")

    # Annotations
    for bar, acc in zip(bars, results_df["Test Accuracy"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{acc:.2%}", va="center", fontweight="bold", fontsize=11)

    ax.set_xlabel("Test Accuracy", fontsize=13)
    ax.set_title("🏆 Comparaison des Modèles — Test Accuracy", fontsize=15, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    plt.tight_layout()

    if save:
        plt.savefig(f"{FIGURES_DIR}/model_comparison.png", bbox_inches="tight")
    plt.show()


def plot_overfitting(results_df: pd.DataFrame, save: bool = True):
    """
    Diagramme comparant Train vs Test accuracy pour détecter l'overfitting.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(results_df))
    width = 0.35

    ax.bar(x - width / 2, results_df["Train Accuracy"], width, label="Train", color="#4C72B0", alpha=0.85)
    ax.bar(x + width / 2, results_df["Test Accuracy"], width, label="Test", color="#DD8452", alpha=0.85)

    # Zone d'alerte overfitting
    for i, (train, test) in enumerate(zip(results_df["Train Accuracy"], results_df["Test Accuracy"])):
        diff = train - test
        if diff > 0.05:
            ax.annotate(f"⚠ {diff:.1%}", xy=(i, max(train, test) + 0.01),
                        ha="center", fontsize=9, color="red", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(results_df.index, rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("📈 Détection du Sur-apprentissage (Train vs Test)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.set_ylim(0.5, 1.05)
    plt.tight_layout()

    if save:
        plt.savefig(f"{FIGURES_DIR}/overfitting_comparison.png", bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(model, X_test, y_test, name: str = "Model", save: bool = True):
    """Matrice de confusion pour un modèle donné."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", square=True,
                xticklabels=["Décédé", "Survécu"],
                yticklabels=["Décédé", "Survécu"], ax=ax)
    ax.set_xlabel("Prédiction", fontsize=12)
    ax.set_ylabel("Réalité", fontsize=12)
    ax.set_title(f"Matrice de Confusion — {name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        safe_name = name.replace(" ", "_").lower()
        plt.savefig(f"{FIGURES_DIR}/confusion_{safe_name}.png", bbox_inches="tight")
    plt.show()


def plot_roc_curves(trained_models: dict, X_test, y_test, save: bool = True):
    """Courbes ROC superposées pour tous les modèles."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for name, model in trained_models.items():
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)
        except (AttributeError, ValueError):
            continue

    ax.plot([0, 1], [0, 1], "k--", label="Aléatoire", alpha=0.5)
    ax.set_xlabel("Taux de Faux Positifs (FPR)", fontsize=12)
    ax.set_ylabel("Taux de Vrais Positifs (TPR)", fontsize=12)
    ax.set_title("📉 Courbes ROC — Tous les Modèles", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    if save:
        plt.savefig(f"{FIGURES_DIR}/roc_curves.png", bbox_inches="tight")
    plt.show()


def plot_feature_importance(model, feature_names: list, name: str = "Model",
                             top_n: int = 15, save: bool = True):
    """Importance des features (pour les modèles à base d'arbres)."""
    if not hasattr(model, "feature_importances_"):
        print(f"⚠ {name} ne supporte pas feature_importances_")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", top_n)
    ax.barh(
        [feature_names[i] for i in indices][::-1],
        importances[indices][::-1],
        color=colors
    )
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"🌳 Top {top_n} Features — {name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        safe_name = name.replace(" ", "_").lower()
        plt.savefig(f"{FIGURES_DIR}/importance_{safe_name}.png", bbox_inches="tight")
    plt.show()


def plot_learning_curve(model, X, y, name: str = "Model", cv: int = 5, save: bool = True):
    """Courbe d'apprentissage (train size vs score)."""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="accuracy", n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#4C72B0")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#DD8452")
    ax.plot(train_sizes, train_mean, "o-", label="Train", color="#4C72B0", linewidth=2)
    ax.plot(train_sizes, val_mean, "o-", label="Validation", color="#DD8452", linewidth=2)

    ax.set_xlabel("Taille de l'échantillon d'entraînement", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"📊 Courbe d'Apprentissage — {name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    plt.tight_layout()

    if save:
        safe_name = name.replace(" ", "_").lower()
        plt.savefig(f"{FIGURES_DIR}/learning_curve_{safe_name}.png", bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────
#  5. EDA VISUALISATIONS
# ─────────────────────────────────────────────

def plot_survival_overview(df: pd.DataFrame, save: bool = True):
    """Génère le dashboard de visualisations EDA du Titanic."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("🚢 Analyse de Survie — Titanic", fontsize=18, fontweight="bold", y=1.02)

    # 1. Taux de survie global
    ax = axes[0, 0]
    df["Survived"].value_counts().plot.pie(
        autopct="%1.1f%%", labels=["Décédé", "Survécu"],
        colors=["#e74c3c", "#2ecc71"], startangle=90, ax=ax,
        textprops={"fontsize": 12}
    )
    ax.set_title("Taux de Survie Global", fontweight="bold")
    ax.set_ylabel("")

    # 2. Survie par sexe
    ax = axes[0, 1]
    sns.barplot(data=df, x="Sex", y="Survived", ax=ax, palette=["#3498db", "#e74c3c"])
    ax.set_title("Survie par Sexe", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Taux de survie")

    # 3. Survie par classe
    ax = axes[0, 2]
    sns.barplot(data=df, x="Pclass", y="Survived", ax=ax, palette="viridis")
    ax.set_title("Survie par Classe", fontweight="bold")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Taux de survie")

    # 4. Distribution de l'âge
    ax = axes[1, 0]
    df[df["Survived"] == 0]["Age"].hist(bins=30, alpha=0.6, color="#e74c3c", label="Décédé", ax=ax)
    df[df["Survived"] == 1]["Age"].hist(bins=30, alpha=0.6, color="#2ecc71", label="Survécu", ax=ax)
    ax.set_title("Distribution de l'Âge", fontweight="bold")
    ax.set_xlabel("Âge")
    ax.legend()

    # 5. Survie par port d'embarquement
    ax = axes[1, 1]
    sns.barplot(data=df, x="Embarked", y="Survived", ax=ax, palette="Set2")
    ax.set_title("Survie par Port", fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Taux de survie")

    # 6. Boxplot des tarifs
    ax = axes[1, 2]
    sns.boxplot(data=df, x="Survived", y="Fare", ax=ax, palette=["#e74c3c", "#2ecc71"])
    ax.set_title("Tarif vs Survie", fontweight="bold")
    ax.set_xlabel("Survécu")
    ax.set_ylim(0, 300)

    plt.tight_layout()

    if save:
        plt.savefig(f"{FIGURES_DIR}/eda_survival_overview.png", bbox_inches="tight")
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True):
    """Heatmap de corrélation des variables numériques."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=ax,
                vmin=-1, vmax=1)
    ax.set_title("🔥 Matrice de Corrélation", fontsize=15, fontweight="bold")
    plt.tight_layout()

    if save:
        plt.savefig(f"{FIGURES_DIR}/correlation_heatmap.png", bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────
#  6. SAUVEGARDE DU MEILLEUR MODÈLE
# ─────────────────────────────────────────────

def save_best_model(model, name: str, scaler, feature_names: list, filepath: str = "models/best_model.pkl"):
    """
    Sauvegarde le meilleur modèle avec le scaler et les noms de features.
    Gère également le versioning en créant une copie horodatée.
    """
    import os
    from datetime import datetime
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    artifact = {
        "model": model,
        "model_name": name,
        "scaler": scaler,
        "feature_names": list(feature_names),
        "created_at": datetime.utcnow().isoformat()
    }
    
    # Remplacer model_name Logistic Regression par LogisticRegression pour le chargement s'il faut
    joblib.dump(artifact, filepath)
    
    # Sauvegarde versionnée
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = filepath.replace(".pkl", f"_{timestamp}.pkl")
    joblib.dump(artifact, versioned_path)
    
    print(f"✅ Modèle sauvegardé → {filepath} (Latest)")
    print(f"✅ Version archivée → {versioned_path}")
    return filepath


def load_model(filepath: str = "models/best_model.pkl") -> dict:
    """Charge un modèle sauvegardé."""
    artifact = joblib.load(filepath)
    print(f"✅ Modèle chargé : {artifact['model_name']} ({len(artifact['feature_names'])} features)")
    return artifact
