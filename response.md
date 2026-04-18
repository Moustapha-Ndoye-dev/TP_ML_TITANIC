# 🚢 Titanic MLOps : Livrable Final — Analyse & Stratégie

Ce document complète l'analyse technique de la plateforme Titanic en détaillant les piliers du **MLOps** mis en œuvre.

---

## PARTIE 8 — MLOPS (Machine Learning Operations)

Le passage d'un modèle Notebook à un service de production nécessite une rigueur opérationnelle. Voici comment notre plateforme adresse les 5 piliers du MLOps.

### 1. Versioning du Modèle
*   **Stratégie** : Utilisation d'un registre local dans le répertoire `/models`. 
*   **Implémentation** : Chaque exportation de modèle (`.pkl`) inclut non seulement l'algorithme, mais aussi le pipeline de prétraitement et les métadonnées de performance.
*   **Gestion** : L'API identifie et charge la version la plus récente au démarrage, tout en permettant aux administrateurs de visualiser l'historique des versions via le pupitre Admin.

### 2. Pipeline de Données
*   **Automatisation** : Le preprocessing est encapsulé dans une fonction standardisée (`preprocess_passenger`) partagée entre l'entraînement et l'inférence.
*   **Reproductibilité** : Garantit que la transformation des variables (scaling, encodage) est identique, peu importe que la donnée provienne d'un formulaire UI ou d'un batch CSV.

### 3. Déploiement
*   **Architecture** : FastAPI pour une performance asynchrone maximale.
*   **Sécurité Industrielle** : 
    *   Validation Pydantic des schémas d'entrée.
    *   Filtrage de sécurité sur les uploads (Taille < 2Mo, Type MIME CSV).
    *   Authentification Administrative pour isoler les outils de monitoring de l'interface publique.

### 4. Monitoring (Suivi & Dérive)
*   **Diagnostic Serveur** : Suivi en temps réel de la charge CPU/RAM pour prévenir les pannes d'infrastructure.
*   **Data Drift** : Comparaison statistique (Z-Score) entre les données d'inférence actuelles et le dataset d'entraînement original. Une alerte est déclenchée si la distribution des passagers change radicalement (ex: passagers beaucoup plus riches ou plus vieux que la moyenne historique).

### 5. Réentraînement
*   **Cycle de vie** : Intégration d'un endpoint spécialisé permettant de lancer un script de réentraînement automatique en tâche de fond.
*   **Automatisation** : Utilisation de threads pour ne pas bloquer le service principal pendant que le modèle se met à jour avec les nouvelles données collectées dans les archives.

---

## Analyse Théorique

### ❓ Pourquoi le MLOps est-il important ?
Le MLOps est crucial car il comble le fossé entre la **science des données** (expérimentation) et l'**ingénierie logicielle** (exploitation). Sans MLOps, un modèle performant en laboratoire risque de devenir obsolète ou erroné dès sa mise au contact de la réalité, sans que personne ne s'en aperçoive. Il garantit la **fiabilité**, la **scalabilité** et la **rentabilité** des investissements en IA.

### ❓ Quelle différence entre ML et MLOps ?
*   **Machine Learning (ML)** : Se concentre sur la création de l'algorithme, l'optimisation des hyperparamètres et la précision sur un jeu de données statique. C'est une phase de R&D.
*   **MLOps** : Se concentre sur l'ensemble du cycle de vie. Il englobe l'automatisation, le déploiement continu (CD), le monitoring des performances dans le temps et la maintenance proactive. On passe d'un "code" à un "produit".

### ❓ Quels risques sans monitoring ?
L'absence de monitoring expose l'entreprise à trois risques majeurs :
1.  **Détérioration du Modèle (Model Decay)** : Le modèle perd en précision car le monde réel change (ex: inflation changeant la valeur du "Fare"), mais le système continue de donner des prédictions erronées comme si elles étaient vraies.
2.  **Biais Non Détecté** : Une dérive dans la population d'entrée peut introduire des biais discriminatoires.
3.  **Indisponibilité Silencieuse** : Le serveur peut être en ligne, mais le modèle peut échouer techniquement (ex: valeur nulle non gérée), rendant le service inutile sans que l'équipe technique ne soit alertée.

---

*Ce document fait partie intégrante de la documentation technique Titanic MLOps 2026.*
