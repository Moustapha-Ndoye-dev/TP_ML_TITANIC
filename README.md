# 🚢 Plattform Titanic MLOps : Surveillance & Prédiction Enterprise-Grade

![Titanic MLOps Banner](https://img.shields.io/badge/MLOps-Production--Ready-blue?style=for-the-badge&logo=opsgenie)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine--Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

> **Système d'analyse prédictive et de monitoring en temps réel pour la gestion des risques maritimes transatlantiques.**

---

## 🏛️ Vision du Projet
Cette plateforme n'est pas un simple outil de prédiction ; c'est un écosystème **MLOps complet** conçu pour démontrer comment un modèle de Machine Learning peut être industrialisé, surveillé et maintenu dans un environnement de production haute disponibilité.

### 💎 Caractéristiques Premium
*   **Interface "Bureau des Archives"** : Une UI immersive inspirée de 1912 fusionnée avec le design moderne (Glassmorphism).
*   **Pipeline End-to-End** : De la donnée brute au déploiement d'API asynchrone.
*   **Pupitre Admin MLOps** : Surveillance en temps réel du CPU, RAM, Latence réseau et dérive des données (Data Drift).
*   **Sécurité Industrielle** : Protection contre les attaques DoS (limite de taille d'upload), validation stricte des schémas CSV, et authentification sécurisée.

---

## 🛠️ Stack Technique
*   **Backend** : FastAPI (Asynchrone, Haute Performance).
*   **Frontend** : Vanilla HTML5/CSS3 (Glassmorphism, Responsive).
*   **ML Engine** : Scikit-Learn (Random Forest) / Pandas / Numpy.
*   **Ops** : Psutil (Telemetry), Joblib (Versioning), JSONL (Persistence).

---

## 🚀 Installation & Lancement

### 1. Prérequis
```bash
git clone https://github.com/votre-compte/titanic-mlops.git
cd titanic-mlops
pip install -r requirements.txt
```

### 2. Exécution du Serveur
```bash
python -m uvicorn api.app:app --reload
```
L'interface sera accessible sur : `http://localhost:8000`

---

## 🔐 Accès Administrative (Démonstration)
Pour accéder au pupitre de monitoring MLOps (`/monitor`), utilisez les identifiants suivants dans la modal de connexion sécurisée :

*   **Identifiant** : `moustapha`
*   **Mot de passe** : `mlops`

*(Note : Dans un environnement Cloud de production, ce système serait couplé à un fournisseur d'identité SSO type Okta ou Auth0 via OAuth2/OIDC.)*

---

## 📊 Cycle de Vie MLOps Implémenté

1.  **Versioning** : Chaque modèle est horodaté et archivé dans `/models`. L'API charge automatiquement la version la plus performante.
2.  **Monitoring de Drift** : Le système compare en temps réel les statistiques des entrées utilisateur avec le dataset d'entraînement pour détecter tout changement de distribution.
3.  **Data Archive** : Toutes les prédictions sont loggées dans un format `JSONL` haute résistance pour analyse ultérieure.
4.  **Health-check** : Endpoint `/health` compatible avec les orchestrateurs type Kubernetes.

---

## ⚖️ Avertissement Légal
Ce projet est une **démonstration technologique** de pipeline MLOps. Les prédictions fournies sont basées sur des probabilités statistiques calculées à partir de données historiques et ne sauraient constituer une garantie de survie réelle.

© 2026 Titanic MLOps Platform — Division Gérance des Risques.
