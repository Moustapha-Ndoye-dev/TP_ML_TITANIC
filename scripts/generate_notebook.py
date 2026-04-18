"""Script pour générer le notebook Jupyter du TP Titanic complet."""

import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.11.0"
    }
}

cells = []

def md(source):
    cells.append(nbf.v4.new_markdown_cell(source))

def code(source):
    cells.append(nbf.v4.new_code_cell(source))

# ============================================================
# TITRE
# ============================================================
md("""# 🚢 TP Complet — Data Science & Machine Learning (Titanic)

---

**Objectifs :**
- Étude exploratoire des données (EDA)
- Data visualisation
- Modélisation (minimum 8 modèles)
- Analyse du sur-apprentissage
- Optimisation (Grid Search CV)
- Ensemble Learning
- Introduction au MLOps

---""")

# ============================================================
# IMPORTS
# ============================================================
md("## ⚙️ Imports & Configuration")

code("""import sys
import os
import warnings
warnings.filterwarnings("ignore")

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.join(os.getcwd(), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Nos modules personnalisés
from src.preprocessing import (
    load_data, show_missing, extract_title, impute_age,
    impute_embarked, impute_fare, create_features,
    encode_features, drop_useless_columns,
    full_preprocessing, prepare_train_test
)
from src.models import get_all_models, get_param_grids, get_ensemble_models
from src.evaluate import (
    train_and_evaluate, compare_models, cross_validate_models,
    plot_model_comparison, plot_overfitting, plot_confusion_matrix,
    plot_roc_curves, plot_feature_importance, plot_learning_curve,
    plot_survival_overview, plot_correlation_heatmap,
    save_best_model
)

# Style
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100

print("✅ Tous les imports sont OK !")""")

# ============================================================
# PARTIE 1 — EXPLORATION
# ============================================================
md("""---
# 📊 Partie 1 — Exploration des Données

**Objectifs :**
- Comprendre les variables
- Identifier la variable cible
- Analyse univariée et bivariée""")

code("""# Chargement du dataset
df = load_data("../data/train.csv")""")

code("""# Aperçu des premières lignes
df.head(10)""")

code("""# Informations sur le dataset
df.info()""")

code("""# Statistiques descriptives
df.describe().round(2)""")

code("""# Variable cible : Survived
print("=" * 50)
print("VARIABLE CIBLE : Survived")
print("=" * 50)
print(f"\\n0 = Décédé, 1 = Survécu")
print(f"\\nDistribution :")
print(df["Survived"].value_counts())
print(f"\\nTaux de survie : {df['Survived'].mean():.1%}")""")

code("""# Valeurs manquantes
show_missing(df)""")

md("""### Analyse Bivariée""")

code("""# Survie par sexe
print("\\n📊 Survie par Sexe :")
print(df.groupby("Sex")["Survived"].agg(["count", "sum", "mean"]).round(3))
print(f"\\n→ Les femmes ont {df[df['Sex']=='female']['Survived'].mean()/df[df['Sex']=='male']['Survived'].mean():.1f}x plus de chances de survivre")""")

code("""# Survie par classe
print("\\n📊 Survie par Classe :")
print(df.groupby("Pclass")["Survived"].agg(["count", "sum", "mean"]).round(3))""")

code("""# Survie croisée Sexe × Classe
print("\\n📊 Survie par Sexe × Classe :")
print(df.groupby(["Sex", "Pclass"])["Survived"].agg(["count", "sum", "mean"]).round(3))""")

code("""# Survie par port d'embarquement
print("\\n📊 Survie par Port :")
print(df.groupby("Embarked")["Survived"].agg(["count", "sum", "mean"]).round(3))""")

code("""# Corrélation avec la survie
print("\\n📊 Corrélation avec Survived :")
numeric_cols = df.select_dtypes(include="number").columns
print(df[numeric_cols].corr()["Survived"].sort_values(ascending=False).round(3))""")

# ============================================================
# PARTIE 2 — VISUALISATION
# ============================================================
md("""---
# 📈 Partie 2 — Data Visualisation

**Objectifs :**
- Histogrammes
- Diagrammes en barres
- Boxplots
- Heatmap de corrélation""")

code("""# Dashboard complet EDA
plot_survival_overview(df)""")

code("""# Heatmap de corrélation
plot_correlation_heatmap(df)""")

code("""# Distribution de l'âge par classe et survie
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, pclass in enumerate([1, 2, 3]):
    ax = axes[i]
    subset = df[df["Pclass"] == pclass]
    subset[subset["Survived"] == 0]["Age"].hist(bins=25, alpha=0.5, color="red", label="Décédé", ax=ax)
    subset[subset["Survived"] == 1]["Age"].hist(bins=25, alpha=0.5, color="green", label="Survécu", ax=ax)
    ax.set_title(f"Classe {pclass}", fontweight="bold", fontsize=14)
    ax.set_xlabel("Âge")
    ax.legend()
fig.suptitle("Distribution de l'Âge par Classe et Survie", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()""")

code("""# Tarif par classe (boxplot)
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x="Pclass", y="Fare", hue="Survived", palette=["#e74c3c", "#2ecc71"], ax=ax)
ax.set_title("Tarif par Classe et Survie", fontsize=14, fontweight="bold")
ax.set_ylim(0, 300)
plt.tight_layout()
plt.show()""")

code("""# Taille de la famille et survie
df_temp = df.copy()
df_temp["FamilySize"] = df_temp["SibSp"] + df_temp["Parch"] + 1

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=df_temp, x="FamilySize", y="Survived", palette="viridis", ax=ax)
ax.set_title("Survie par Taille de Famille", fontsize=14, fontweight="bold")
ax.set_xlabel("Taille de la famille")
ax.set_ylabel("Taux de survie")
plt.tight_layout()
plt.show()""")

# ============================================================
# PARTIE 3 — PREPROCESSING
# ============================================================
md("""---
# 🔧 Partie 3 — Préparation des Données

**Objectifs :**
- Gestion des valeurs manquantes
- Encodage des variables catégorielles
- Feature engineering
- Normalisation""")

code("""# Recharger les données propres
df = load_data("../data/train.csv")

# Pipeline complète de preprocessing
df_clean = full_preprocessing(df)""")

code("""# Vérifier le résultat
print("\\n📋 Dataset après preprocessing :")
print(f"   Shape : {df_clean.shape}")
print(f"   Valeurs manquantes : {df_clean.isnull().sum().sum()}")
df_clean.head()""")

code("""# Séparer en train/test avec normalisation
X_train, X_test, y_train, y_test, scaler, feature_names = prepare_train_test(df_clean)

print(f"\\n📊 Features utilisées ({len(feature_names)}) :")
for i, f in enumerate(feature_names, 1):
    print(f"   {i:2d}. {f}")""")

# ============================================================
# PARTIE 4 — MODÉLISATION (8 MODÈLES)
# ============================================================
md("""---
# 🤖 Partie 4 — Modélisation (8 Modèles)

**Modèles implémentés :**
1. Régression Logistique
2. KNN (K-Nearest Neighbors)
3. Arbre de Décision
4. Random Forest
5. Gradient Boosting
6. SVM (Support Vector Machine)
7. Naive Bayes
8. XGBoost""")

code("""# Charger les 8 modèles
models = get_all_models()

print(f"📦 {len(models)} modèles chargés :")
for name in models:
    print(f"   ▸ {name}")""")

code("""# Entraîner et comparer tous les modèles
print("\\n🚀 Entraînement de tous les modèles...\\n")
results_df, trained_models = compare_models(models, X_train, X_test, y_train, y_test)""")

code("""# Visualisation de la comparaison
plot_model_comparison(results_df)""")

code("""# Rapport détaillé du meilleur modèle
best_name = results_df.index[0]
best_model = trained_models[best_name]

print(f"\\n🏆 Meilleur modèle : {best_name}")
print(f"\\n📋 Rapport de classification :\\n")
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["Décédé", "Survécu"]))""")

code("""# Matrice de confusion du meilleur modèle
plot_confusion_matrix(best_model, X_test, y_test, name=best_name)""")

code("""# Courbes ROC de tous les modèles
plot_roc_curves(trained_models, X_test, y_test)""")

code("""# Feature importance (modèles à base d'arbres)
for name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
    if name in trained_models:
        plot_feature_importance(trained_models[name], feature_names, name=name)""")

# ============================================================
# PARTIE 5 — SUR-APPRENTISSAGE
# ============================================================
md("""---
# ⚠️ Partie 5 — Détection du Sur-apprentissage

**Objectifs :**
- Comparer les performances train vs test
- Identifier l'overfitting
- Solutions : régularisation, cross-validation""")

code("""# Comparaison Train vs Test (détection overfitting)
print("📊 Analyse du Sur-apprentissage :\\n")
print(results_df[["Train Accuracy", "Test Accuracy", "Overfit"]].round(4))

print("\\n⚠️ Modèles en sur-apprentissage (écart > 5%) :")
overfit = results_df[results_df["Overfit"] > 0.05]
if len(overfit) > 0:
    for name, row in overfit.iterrows():
        print(f"   ▸ {name}: Train={row['Train Accuracy']:.4f}, Test={row['Test Accuracy']:.4f}, Écart={row['Overfit']:.4f}")
else:
    print("   ✅ Aucun modèle en sur-apprentissage significatif")""")

code("""# Diagramme overfitting
plot_overfitting(results_df)""")

code("""# Courbe d'apprentissage du meilleur modèle
plot_learning_curve(best_model, X_train, y_train, name=best_name)""")

code("""# Cross-validation (5-fold) pour validation robuste
print("\\n📊 Validation Croisée (5-fold) :\\n")
cv_results = cross_validate_models(models, X_train, y_train, cv=5)
print("\\n" + cv_results.to_string())""")

# ============================================================
# PARTIE 6 — GRID SEARCH CV
# ============================================================
md("""---
# 🔍 Partie 6 — Optimisation (Grid Search CV)

**Objectifs :**
- Optimiser les hyperparamètres de chaque modèle
- Comparer avant/après optimisation""")

code("""# Grilles d'hyperparamètres
param_grids = get_param_grids()

# On optimise les 3 meilleurs modèles pour gagner du temps
top_models = list(results_df.index[:3])
print(f"🔍 Optimisation des 3 meilleurs modèles : {top_models}\\n")

optimized_models = {}
optimization_results = []

for name in top_models:
    if name in param_grids and name in models:
        print(f"▸ Grid Search sur {name}...")
        grid = GridSearchCV(
            models[name],
            param_grids[name],
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0
        )
        grid.fit(X_train, y_train)
        
        before_acc = results_df.loc[name, "Test Accuracy"]
        after_acc = grid.score(X_test, y_test)
        
        optimized_models[name] = grid.best_estimator_
        optimization_results.append({
            "Modèle": name,
            "Avant": before_acc,
            "Après": after_acc,
            "Amélioration": after_acc - before_acc,
            "Meilleurs Params": str(grid.best_params_)
        })
        
        print(f"  ✓ Avant: {before_acc:.4f} → Après: {after_acc:.4f} (Δ={after_acc-before_acc:+.4f})")
        print(f"  Params: {grid.best_params_}\\n")

opt_df = pd.DataFrame(optimization_results).set_index("Modèle")
print("\\n" + "=" * 60)
print("📊 RÉSUMÉ GRID SEARCH CV")
print("=" * 60)
print(opt_df[["Avant", "Après", "Amélioration"]].round(4).to_string())""")

code("""# Visualisation avant/après
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(opt_df))
width = 0.35

ax.bar(x - width/2, opt_df["Avant"], width, label="Avant GridSearch", color="#3498db", alpha=0.85)
ax.bar(x + width/2, opt_df["Après"], width, label="Après GridSearch", color="#2ecc71", alpha=0.85)

for i, (avant, apres) in enumerate(zip(opt_df["Avant"], opt_df["Après"])):
    diff = apres - avant
    color = "green" if diff > 0 else "red"
    ax.annotate(f"{diff:+.2%}", xy=(i + width/2, apres + 0.005), ha="center", fontsize=11, color=color, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(opt_df.index, fontsize=12)
ax.set_ylabel("Test Accuracy", fontsize=13)
ax.set_title("🔍 Impact du Grid Search CV", fontsize=15, fontweight="bold")
ax.legend(fontsize=12)
ax.set_ylim(0.7, 1.0)
plt.tight_layout()
plt.savefig("../reports/figures/gridsearch_comparison.png", bbox_inches="tight")
plt.show()""")

# ============================================================
# PARTIE 7 — ENSEMBLE LEARNING
# ============================================================
md("""---
# 🧩 Partie 7 — Ensemble Learning

**Méthodes :**
- Bagging
- Boosting (AdaBoost)
- Voting (Hard & Soft)
- Stacking""")

code("""# Charger les modèles d'ensemble
ensemble_models = get_ensemble_models()

print(f"📦 {len(ensemble_models)} modèles d'ensemble :")
for name in ensemble_models:
    print(f"   ▸ {name}")""")

code("""# Entraîner et comparer les ensembles
print("\\n🚀 Entraînement des modèles d'ensemble...\\n")
ensemble_results_df, ensemble_trained = compare_models(
    ensemble_models, X_train, X_test, y_train, y_test
)""")

code("""# Comparaison ensembles vs modèles individuels
all_results = pd.concat([results_df, ensemble_results_df])
all_results = all_results.sort_values("Test Accuracy", ascending=False)

print("\\n" + "=" * 70)
print("🏆 CLASSEMENT GLOBAL (Individuels + Ensembles)")
print("=" * 70)
print(all_results[["Test Accuracy", "F1-Score", "AUC-ROC"]].round(4).to_string())""")

code("""# Visualisation du classement global
plot_model_comparison(all_results)""")

# ============================================================
# ASSEMBLER LE NOTEBOOK
# ============================================================
nb.cells = cells

output_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "tp_titanic_complet.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"[OK] Notebook cree : {output_path}")
print(f"   {len(cells)} cellules ({sum(1 for c in cells if c.cell_type == 'markdown')} markdown + {sum(1 for c in cells if c.cell_type == 'code')} code)")
