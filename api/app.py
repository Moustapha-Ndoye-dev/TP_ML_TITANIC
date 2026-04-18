import os
import json
import io
import html
import glob
import sys
import numpy as np
import pandas as pd
from typing import List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from collections import deque

# ─────────────────────────────────────────────
#  MODÈLES DE DONNÉES ET CONFIGURATION
# ─────────────────────────────────────────────

app = FastAPI(
    title="Titanic MLOps Platform",
    description="End-to-End MLOps Pipeline for Titanic Survival Prediction",
    version="2.0.0"
)

MODELS_DIR = "models"
LOGS_PATH = os.path.join(MODELS_DIR, "prediction_logs.jsonl")

class PassengerInput(BaseModel):
    Pclass: int
    Name: str
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Cabin: Optional[str] = None
    Embarked: str

class PredictionOutput(BaseModel):
    survived: bool
    probability: float
    confidence: str
    passenger_summary: str
    influences: dict = {}

# État global (Monitoring & MLOps)
model_artifact = None
prediction_log = deque(maxlen=100)
network_logs = deque(maxlen=100) # Augmenté pour pagination
prediction_count = 0
survived_count = 0

# Statistiques de référence (extraites du training set original)
TRAINING_STATS = {
    "Age": {"mean": 29.7, "std": 14.5},
    "Fare": {"mean": 32.2, "std": 49.7},
    "Pclass": {"mean": 2.3, "std": 0.8},
    "SibSp": {"mean": 0.5, "std": 1.1},
    "Parch": {"mean": 0.4, "std": 0.8},
    "Sex_male_ratio": 0.64
}

# ─────────────────────────────────────────────
#  MIDDLEWARE & LOGGING
# ─────────────────────────────────────────────

@app.middleware("http")
async def add_network_logger(request: Request, call_next):
    """Intercepte les métriques réseau pour le monitor de prod."""
    start_time = datetime.now()
    response = await call_next(request)
    duration = (datetime.now() - start_time).total_seconds() * 1000 # ms
    
    # Enregistrer le log réseau
    network_logs.appendleft({
        "timestamp": datetime.utcnow().isoformat(),
        "method": request.method,
        "path": request.url.path,
        "ip": request.client.host,
        "status": response.status_code,
        "latency": round(duration, 2)
    })
    return response

# ─────────────────────────────────────────────
#  CHARGEMENT DU MODÈLE ET LOGS
# ─────────────────────────────────────────────

def load_model_artifact():
    """Charge le modèle le plus récent du dossier models/."""
    global model_artifact
    import joblib
    
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pkl"))
    if not model_files:
        print("[WARNING] Aucun modele .pkl trouve dans /models")
        return None
    
    latest_model = max(model_files, key=os.path.getmtime)
    print(f"[MLOps] Chargement du modele : {latest_model}")
    
    try:
        loaded = joblib.load(latest_model)
        model_artifact = {
            "model": loaded["model"],
            "feature_names": loaded["feature_names"],
            "scaler": loaded.get("scaler"),
            "model_name": loaded.get("model_name", "Random Forest Classifier"),
            "model_path": latest_model,
            "loaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return model_artifact
    except Exception as e:
        print(f"[CRITICAL] Echec chargement modele: {e}")
        return None

def load_prediction_history():
    """Charge l'historique des prédictions pour le calcul du drift au démarrage."""
    global prediction_count, survived_count
    if os.path.exists(LOGS_PATH):
        try:
            with open(LOGS_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    prediction_log.append(entry)
                    prediction_count += 1
                    if entry.get("survived"):
                        survived_count += 1
            print(f"[MLOps] {len(prediction_log)} logs recharges.")
        except Exception as e:
            print(f"[WARNING] Erreur rechargement logs: {e}")

@app.on_event("startup")
async def startup_event():
    load_model_artifact()
    load_prediction_history()

# ─────────────────────────────────────────────
#  LOGIQUE MÉTIER PRÉDICTION
# ─────────────────────────────────────────────

def preprocess_inference(pclass, name, sex, age, sibsp, parch, fare, cabin, embarked):
    """Pipeline de preprocessing complète pour l'inférence (cohérent avec src/preprocessing.py)."""
    # 1. Création du DataFrame initial
    df = pd.DataFrame([{
        'Pclass': pclass, 'Name': name, 'Sex': sex, 'Age': age,
        'SibSp': sibsp, 'Parch': parch, 'Fare': fare, 'Cabin': cabin, 'Embarked': embarked
    }])

    # 2. Extraction du Titre
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    rare_titles = ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"]
    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    df["Title"] = df["Title"].fillna("Mr")

    # 3. Imputation (fixes pour l'inférence)
    if pd.isna(df.loc[0, "Age"]): df.loc[0, "Age"] = 29.7
    if pd.isna(df.loc[0, "Fare"]): df.loc[0, "Fare"] = 32.2

    # 4. Feature Engineering
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    df["Deck"] = df["Cabin"].apply(lambda x: str(x)[0] if pd.notna(x) and str(x) != 'None' and str(x) != '' else "U")
    
    df["AgeBin"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 50, 80], labels=["Enfant", "Ado", "JeuneAdulte", "Mature", "Senior"])
    df["AgeBin"] = df["AgeBin"].fillna("JeuneAdulte")
    df["FareBin"] = pd.cut(df["Fare"], bins=[-1, 7.91, 14.45, 31.0, 600], labels=["Low", "Mid", "High", "VeryHigh"])
    df["FareBin"] = df["FareBin"].fillna("Low")

    # 5. Encodage & Nettoyage (suppression des colonnes brutes avant dummy encoding)
    df["Sex"] = df["Sex"].map({"female": 0, "male": 1}).fillna(1).astype(int)
    
    # One-Hot Encoding (drop_first=True pour matcher l'entraînement)
    categorical_cols = ["Embarked", "Title", "Deck", "AgeBin", "FareBin"]
    df = pd.get_dummies(df, columns=[c for c in categorical_cols if c in df.columns], drop_first=True, dtype=int)

    # 6. Suppression explicite des colonnes non désirées et alignement
    # PassengerId, Ticket ne sont pas dans l'input mais Name et Cabin y sont
    cols_to_drop = ["Name", "Cabin", "Embarked", "Title", "Deck", "AgeBin", "FareBin"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    expected_features = model_artifact["feature_names"]
    X = df.reindex(columns=expected_features, fill_value=0)

    # 7. Scaling
    if model_artifact.get("scaler"):
        # On s'assure que X est propre avant le transform
        X_scaled = model_artifact["scaler"].transform(X)
        return pd.DataFrame(X_scaled, columns=expected_features)
    
    return X


def make_prediction(pclass, name, sex, age, sibsp, parch, fare, cabin, embarked):
    X = preprocess_inference(pclass, name, sex, age, sibsp, parch, fare, cabin, embarked)
    model = model_artifact["model"]
    feature_names = model_artifact["feature_names"]

    prediction = model.predict(X)[0]
    try:
        proba_list = model.predict_proba(X)[0]
        survival_proba = float(proba_list[1])
    except (AttributeError, IndexError):
        survival_proba = float(prediction)

    # Calcul des influences XAI (Explainable AI)
    # On extrait les poids directement du "cerveau" du modèle (coefs ou importances)
    influences = {"social": 0.0, "profil": 0.0, "proximité": 0.0}
    
    # 1. Extraction des importances/poids
    weights = None
    if hasattr(model, "coef_"):
        weights = model.coef_[0]
    elif hasattr(model, "feature_importances_"):
        weights = model.feature_importances_
    
    if weights is not None:
        row_values = X.iloc[0].values
        for i, feat in enumerate(feature_names):
            val = row_values[i]
            # Pour un Random Forest, on multiplie l'importance par la direction de l'input
            # Pour un modèle linéaire, c'est la multiplication directe
            impact = val * weights[i]
            
            f_lower = feat.lower()
            if any(x in f_lower for x in ["pclass", "fare"]):
                influences["social"] += impact
            elif any(x in f_lower for x in ["sex", "age", "title"]):
                influences["profil"] += impact
            else:
                influences["proximité"] += impact

    # Normalisation pour l'affichage (-5 à +5)
    for k in influences:
        influences[k] = round(max(-5, min(5, influences[k])), 2)

    return bool(prediction), survival_proba, influences


# ─────────────────────────────────────────────
#  PAGE HTML — INTERFACE UTILISATEUR
# ─────────────────────────────────────────────

HTML_USER_PAGE = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PRÉDICTEUR TITANIC | BUREAU DES ARCHIVES</title>
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400;1,700&family=Space+Mono:ital,wght@0,400;0,700;1,400;1,700&display=swap" rel="stylesheet">
    <style>
        :root { --bg-paper: #f4f1ea; --ink-black: #0f1110; --ink-fade: #3f4240; --alert-red: #ff3311; --ocean-blue: #002244; --success-green: #0a6b3d; --border-thick: 4px solid var(--ink-black); --border-thin: 1px solid var(--ink-black); }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Space Mono', monospace; background-color: var(--bg-paper); color: var(--ink-black); padding: 2rem; min-height: 100vh; text-transform: uppercase; }
        h1, h2, h3 { font-family: 'Playfair Display', serif; text-transform: none; letter-spacing: -0.02em; }
        h1 { font-size: 4rem; font-weight: 900; line-height: 1.1; margin-bottom: 0.5rem; text-align: center; border-bottom: var(--border-thick); padding-bottom: 1rem; }
        h2 { font-size: 2.2rem; border-bottom: var(--border-thin); margin-bottom: 1.5rem; padding-bottom: 0.5rem; }
        p { font-size: 0.9rem; margin-bottom: 1rem; }
        
        .header-meta { display: flex; justify-content: space-between; font-size: 0.8rem; border-bottom: var(--border-thick); padding: 0.5rem 0; margin-bottom: 2rem; align-items: center; }
        .admin-btn { padding: 0.25rem 0.5rem; background: var(--ink-black); color: var(--bg-paper); cursor: pointer; border: none; font-family: 'Space Mono', monospace; font-size: 0.8rem; font-weight: bold; }
        .admin-btn:hover { background: var(--alert-red); }

        .container { max-width: 1200px; margin: 0 auto; border: var(--border-thick); background: #fff; padding: 2rem; box-shadow: 12px 12px 0px var(--ink-black); }
        
        .tabs { display: flex; border: var(--border-thick); margin-bottom: 2rem; }
        .tab { flex: 1; padding: 1rem; text-align: center; cursor: pointer; border-right: var(--border-thick); font-weight: bold; background: var(--bg-paper); transition: all 0.2s ease; }
        .tab:last-child { border-right: none; }
        .tab:hover { background: var(--ink-fade); color: var(--bg-paper); }
        .tab.active { background: var(--ink-black); color: var(--bg-paper); }
        .tab-content { display: none; }
        .tab-content.active { display: block; animation: appear 0.3s steps(4, end); }
        @keyframes appear { 0% { opacity: 0; transform: translateY(10px); } 100% { opacity: 1; transform: translateY(0); } }

        .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }
        .input-group { display: flex; flex-direction: column; border: var(--border-thin); padding: 0.5rem; background: var(--bg-paper); }
        .input-group label { font-size: 0.75rem; font-weight: 700; margin-bottom: 0.5rem; color: var(--ink-fade); }
        .input-group input, .input-group select { background: transparent; border: none; border-bottom: 2px dashed var(--ink-black); font-family: 'Space Mono', monospace; font-size: 1.1rem; padding: 0.5rem; outline: none; color: var(--ink-black); border-radius: 0; appearance: none; -webkit-appearance: none; }
        
        .upload-zone { border: 4px dashed var(--ink-black); padding: 4rem 2rem; text-align: center; background: #fff; cursor: pointer; transition: background 0.3s; margin-bottom: 2rem; }
        .upload-zone:hover, .upload-zone.dragover { background: #eee; }
        .upload-icon { font-size: 4rem; margin-bottom: 1rem; }
        #batchFile { display: none; }
        .batch-results { margin-top: 2rem; border: var(--border-thick); max-height: 400px; overflow-y: auto; }
        
        table { width: 100%; border-collapse: collapse; }
        th, td { border: var(--border-thin); padding: 0.75rem; text-align: left; font-size: 0.85rem; }
        th { background: var(--ink-black); color: var(--bg-paper); position: sticky; top: 0; z-index: 10; }
        tr:nth-child(even) { background: var(--bg-paper); }

        .btn-large { display: block; width: 100%; padding: 1.5rem; background: var(--alert-red); color: var(--bg-paper); font-family: 'Space Mono', monospace; font-size: 1.5rem; font-weight: bold; text-transform: uppercase; border: var(--border-thick); cursor: pointer; box-shadow: 6px 6px 0px var(--ink-black); transition: transform 0.1s; }
        .btn-large:hover { box-shadow: 2px 2px 0px var(--ink-black); transform: translate(4px, 4px); background: #e0290b; }
        .btn-large:active { box-shadow: none; transform: translate(6px, 6px); }

        #resultPane { margin-top: 2rem; display: none; }
        .result-dossier { border: var(--border-thick); padding: 2rem; background: var(--bg-paper); display: flex; gap: 2rem; align-items: center; }
        .stamp { font-size: 3rem; font-family: 'Playfair Display', serif; font-weight: 900; text-transform: uppercase; padding: 1rem 2rem; border: 6px solid; transform: rotate(-5deg); white-space: nowrap; }
        .stamp.survived { color: var(--success-green); border-color: var(--success-green); }
        .stamp.deceased { color: var(--alert-red); border-color: var(--alert-red); }
        .stamp.uncertain { color: #f39c12; border-color: #f39c12; }
        
        .result-details { flex: 1; }
        .proba-bar { height: 30px; background: #ccc; border: var(--border-thin); margin-top: 1rem; position: relative; }
        .proba-fill { height: 100%; background: var(--ink-black); transition: width 1s steps(20, end); }
        
        .status-badge { display: inline-block; padding: 0.25rem 0.5rem; font-size: 0.75rem; font-weight: bold; border: var(--border-thin); }
        .status-ok { background: var(--success-green); color: white; }
        .status-fail { background: var(--alert-red); color: white; }
        .status-warn { background: #f39c12; color: white; }

        /* XAI STYLES */
        .influence-section { margin-top: 1.5rem; border-top: 1px dashed var(--ink-black); padding-top: 1rem; }
        .influence-row { display: flex; align-items: center; margin-bottom: 0.75rem; font-size: 0.75rem; font-weight: bold; gap: 1rem; }
        .inf-label { width: 100px; }
        .inf-bar-bg { flex: 1; height: 12px; background: rgba(0,0,0,0.05); position: relative; border: 1px solid rgba(0,0,0,0.1); }
        .inf-bar-center { position: absolute; left: 50%; top: 0; bottom: 0; width: 2px; background: var(--ink-black); z-index: 2; }
        .inf-fill { position: absolute; height: 100%; top: 0; transition: all 0.6s ease-out; }

        .radar-loader-container { padding: 3rem 0; display: none; text-align: center; }
        .radar-loader { width: 80px; height: 80px; border: 4px solid var(--ink-black); border-radius: 50%; position: relative; overflow: hidden; margin: 0 auto 1rem auto; box-shadow: 0 0 15px rgba(0,0,0,0.1); background: var(--bg-paper); }
        .radar-loader::before { content: ''; position: absolute; top: 50%; left: 0; right: 0; height: 2px; background: rgba(0,0,0,0.1); }
        .radar-loader::after { content: ''; position: absolute; top: 0; left: 50%; width: 50%; height: 50%; background: var(--alert-red); transform-origin: bottom left; animation: radarScan 1.5s linear infinite; }
        @keyframes radarScan { 100% { transform: rotate(360deg); } }

        /* MODAL LOGIN */
        .modal-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px); z-index: 1000; align-items: center; justify-content: center; }
        .modal-card { width: 400px; background: rgba(255, 255, 255, 0.1); border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 12px; padding: 2.5rem; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); transition: transform 0.3s; transform: scale(0.9); text-transform: none; text-align: left; }
        .modal-card.active { transform: scale(1); }
        .modal-card h3 { color: #fff; margin-bottom: 1.5rem; font-size: 1.8rem; border-bottom: none; }
        .modal-card input { width: 100%; padding: 1rem; margin-bottom: 1rem; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.2); border-radius: 6px; color: #fff; font-family: 'Space Mono', monospace; outline: none; }
        .modal-card button { width: 100%; padding: 1rem; background: var(--alert-red); color: #fff; border: none; border-radius: 6px; font-weight: bold; cursor: pointer; transition: 0.3s; }
        .modal-card button:hover { background: #fff; color: var(--ink-black); }
        .close-modal { position: absolute; top: 1rem; right: 1rem; color: #fff; cursor: pointer; font-size: 1.5rem; }

        /* RESPONSIVENESS */
        @media (max-width: 768px) {
            body { padding: 1rem; }
            h1 { font-size: 2.2rem; }
            h2 { font-size: 1.6rem; }
            .header-meta { flex-direction: column; gap: 0.5rem; text-align: center; }
            .container { padding: 1.5rem; box-shadow: 6px 6px 0px var(--ink-black); }
            .tabs { flex-direction: column; }
            .tab { border-right: none; border-bottom: var(--border-thick); }
            .tab:last-child { border-bottom: none; }
            .form-grid { grid-template-columns: 1fr; gap: 1rem; }
            .result-dossier { flex-direction: column; text-align: center; gap: 1rem; }
            .stamp { transform: rotate(0); margin-bottom: 1rem; }
            .btn-large { font-size: 1.2rem; padding: 1rem; }
            .modal-card { width: 90%; padding: 1.5rem; }
        }
    </style>
</head>
<body>
    <div class="header-meta">
        <span>BUREAU DES PRÉDICTIONS</span>
        <span>ÉMISSION : 1912 / 2026</span>
        <div style="display:flex; gap:1rem;">
            <button class="admin-btn" style="background:var(--ocean-blue);" onclick="loginAdmin()">🔓 CONNEXION</button>
        </div>
    </div>

    <div class="container">
        <h1>PRÉDICTEUR DE SURVIE<br>TITANIC</h1>
        
        <div class="tabs">
            <div class="tab active" onclick="switchTab('individual')">PASSAGER INDIVIDUEL</div>
            <div class="tab" onclick="switchTab('batch')">MANIFESTE DE BORD (CSV)</div>
        </div>

        <div id="tab-individual" class="tab-content active">
            <h2>REQUÊTE PASSAGER INDIVIDUELLE</h2>
            <form id="predictForm">
                <div class="form-grid">
                    <div class="input-group">
                        <label>NOM COMPLET</label>
                        <input type="text" id="Name" placeholder="Ex: Dawson, M. Jack" required>
                    </div>
                    <div class="input-group">
                        <label>CLASSE DU BILLET</label>
                        <select id="Pclass">
                            <option value="1">1ÈRE CLASSE</option>
                            <option value="2">2ÈME CLASSE</option>
                            <option value="3" selected>3ÈME CLASSE</option>
                        </select>
                    </div>
                    <div class="input-group">
                        <label>SEXE BIOLOGIQUE</label>
                        <select id="Sex">
                            <option value="male" selected>HOMME</option>
                            <option value="female">FEMME</option>
                        </select>
                    </div>
                    <div class="input-group">
                        <label>ÂGE (ANNÉES)</label>
                        <input type="number" id="Age" placeholder="Ex: 20" min="0" step="0.5" required>
                    </div>
                    <div class="input-group">
                        <label>FRÈRES, SŒURS / CONJOINTS</label>
                        <input type="number" id="SibSp" placeholder="Ex: 0" min="0" required>
                    </div>
                    <div class="input-group">
                        <label>PARENTS / ENFANTS</label>
                        <input type="number" id="Parch" placeholder="Ex: 0" min="0" required>
                    </div>
                    <div class="input-group">
                        <label>TARIF DU BILLET (£)</label>
                        <input type="number" id="Fare" placeholder="Ex: 8.05" min="0" step="0.01" required>
                    </div>
                    <div class="input-group">
                        <label>PORT D'EMBARQUEMENT</label>
                        <select id="Embarked">
                            <option value="C">CHERBOURG</option>
                            <option value="Q">QUEENSTOWN</option>
                            <option value="S" selected>SOUTHAMPTON</option>
                        </select>
                    </div>
                    <div class="input-group">
                        <label>CABINE (OPTIONNELLE)</label>
                        <input type="text" id="Cabin" placeholder="Ex. C52">
                    </div>
                </div>
                <button type="submit" class="btn-large">EXÉCUTER LA PRÉDICTION</button>
            </form>

            <div id="resultPane">
                <div class="result-dossier">
                    <div id="stampBox" class="stamp deceased">DÉCÉDÉ</div>
                    <div class="result-details">
                        <h3 id="resName" style="margin-bottom:0.5rem;font-weight:900;">...</h3>
                        <p id="resDesc" style="color:var(--ink-fade);">...</p>
                        <div style="display:flex; justify-content:space-between; margin-top:1rem; font-weight:bold;">
                            <span>PROBABILITÉ DE SURVIE :</span>
                            <span id="resProba">0%</span>
                        </div>
                        <div class="proba-bar">
                            <div id="probaFill" class="proba-fill" style="width:0%;"></div>
                        </div>
                        <p style="margin-top:0.5rem; font-size:0.75rem;">NIVEAU DE CONFIANCE : <span id="resConf" class="status-badge status-ok">DÉTERMINATION...</span></p>
                        
                        <div class="influence-section">
                            <p style="font-size:0.7rem; color:var(--ink-fade); margin-bottom:0.5rem; font-weight:900;">ANALYSE DES FACTEURS DÉCISIFS :</p>
                            <div class="influence-row">
                                <span class="inf-label">RANG SOCIAL</span>
                                <div class="inf-bar-bg"><div class="inf-bar-center"></div><div id="inf-social" class="inf-fill"></div></div>
                            </div>
                            <div class="influence-row">
                                <span class="inf-label">PROFIL PERSO</span>
                                <div class="inf-bar-bg"><div class="inf-bar-center"></div><div id="inf-profil" class="inf-fill"></div></div>
                            </div>
                            <div class="influence-row">
                                <span class="inf-label">PROXIMITÉ</span>
                                <div class="inf-bar-bg"><div class="inf-bar-center"></div><div id="inf-proximite" class="inf-fill"></div></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div id="tab-batch" class="tab-content">
            <h2>ANALYSE DE MANIFESTE EN MASSE (.CSV)</h2>
            <div class="upload-zone" id="dropZone" onclick="document.getElementById('batchFile').click()">
                <div class="upload-icon">⚓</div>
                <h3>DÉPOSEZ LE MANIFESTE DES PASSAGERS ICI</h3>
                <p>ou cliquez pour parcourir les fichiers</p>
                <input type="file" id="batchFile" accept=".csv">
            </div>
            <button id="btnBatch" class="btn-large" style="display:none; background:var(--ocean-blue);" onclick="uploadCSV()">DÉMARRER L'ANALYSE</button>

            <div class="radar-loader-container" id="batchLoader">
                <div class="radar-loader"></div>
                <h3 style="color:var(--alert-red); animation: appear 1s infinite alternate;">ALGORITHME ENTRÉ EN ACTION...</h3>
                <p>Analyse prédictive ligne par ligne en cours</p>
            </div>

            <div id="batchStats" style="display:none; margin-top:2rem; font-weight:bold; font-size:1.2rem; border-bottom:var(--border-thick); padding-bottom:1rem;"></div>
            <div class="batch-results" id="batchResultsTable" style="display:none;"></div>
        </div>
    </div>


    <div id="loginOverlay" class="modal-overlay" onclick="if(event.target === this) closeModal()">
        <div class="modal-card" id="loginCard">
            <span class="close-modal" onclick="closeModal()">&times;</span>
            <h3 style="font-family:'Playfair Display'; text-transform:none; color:white;">Administrative Access</h3>
            <p style="color:rgba(255,255,255,0.6); font-size:0.8rem; margin-bottom:1.5rem; text-transform:none;">Veuillez entrer vos identifiants de sécurité.</p>
            <form id="loginForm" onsubmit="submitLogin(event)">
                <input type="text" id="admUser" placeholder="Username" required>
                <input type="password" id="admPass" placeholder="Password" required>
                <button type="submit">LOG IN</button>
            </form>
        </div>
    </div>

    <script>
        function loginAdmin() {
            document.getElementById('loginOverlay').style.display = 'flex';
            setTimeout(() => document.getElementById('loginCard').classList.add('active'), 10);
        }
        function closeModal() {
            document.getElementById('loginCard').classList.remove('active');
            setTimeout(() => document.getElementById('loginOverlay').style.display = 'none', 300);
        }
        function submitLogin(e) {
            if(e) e.preventDefault();
            const u = document.getElementById('admUser').value;
            const p = document.getElementById('admPass').value;
            if(u === "moustapha" && p === "mlops") {
                window.location.href = "/monitor";
            } else {
                showCustomAlert("Identifiants incorrects.");
            }
        }
        function showCustomAlert(msg) {
            alert(msg); // Placeholder, will be replaced by custom modal if needed
        }
        function switchTab(tabId) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.currentTarget.classList.add('active');
            document.getElementById('tab-' + tabId).classList.add('active');
        }

        document.getElementById('predictForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = e.target.querySelector('button');
            btn.textContent = 'TRAITEMENT EN COURS...';
            btn.style.opacity = '0.5';
            
            const payload = {
                Pclass: parseInt(document.getElementById('Pclass').value),
                Name: document.getElementById('Name').value,
                Sex: document.getElementById('Sex').value,
                Age: parseFloat(document.getElementById('Age').value),
                SibSp: parseInt(document.getElementById('SibSp').value),
                Parch: parseInt(document.getElementById('Parch').value),
                Fare: parseFloat(document.getElementById('Fare').value),
                Cabin: document.getElementById('Cabin').value || null,
                Embarked: document.getElementById('Embarked').value
            };

            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if(!res.ok) alert("Erreur: " + JSON.stringify(data));
                else {
                    document.getElementById('resultPane').style.display = 'block';
                    const stamp = document.getElementById('stampBox');
                    const proba = data.probability;
                    
                    let statusLabel, statusClass, barColor;
                    if (proba > 0.55) { statusLabel = 'SURVÉCU'; statusClass = 'survived'; barColor = 'var(--success-green)'; }
                    else if (proba < 0.45) { statusLabel = 'DÉCÉDÉ'; statusClass = 'deceased'; barColor = 'var(--alert-red)'; }
                    else { statusLabel = 'INCERTAIN'; statusClass = 'uncertain'; barColor = '#f39c12'; }

                    stamp.textContent = statusLabel;
                    stamp.className = 'stamp ' + statusClass;
                    document.getElementById('resName').textContent = payload.Name;
                    const classP = payload.Pclass === 1 ? '1ÈRE' : (payload.Pclass===2 ? '2ÈME' : '3ÈME');
                    const sexP = payload.Sex === 'male' ? 'HOMME' : 'FEMME';
                    document.getElementById('resDesc').textContent = `${sexP} | CLASSE ${classP} | ÂGE ${payload.Age}`;
                    
                    document.getElementById('resProba').textContent = (proba * 100).toFixed(1) + '%';
                    document.getElementById('probaFill').style.width = (proba * 100).toFixed(1) + '%';
                    document.getElementById('probaFill').style.background = barColor;
                    
                    const conf = document.getElementById('resConf');
                    let confText = (data.confidence || "HAUTE").toUpperCase();
                    if(confText === 'HAUTE') confText = 'ÉLEVÉE';
                    conf.textContent = confText;
                    conf.className = 'status-badge ' + (data.confidence === 'Haute' ? 'status-ok' : (data.confidence === 'Moyenne' ? 'status-warn' : 'status-fail'));

                    // UPDATE XAI BARS
                    updateInfBar('inf-social', data.influences.social);
                    updateInfBar('inf-profil', data.influences.profil);
                    updateInfBar('inf-proximite', data.influences.proximité);
                }
            } catch(err) { alert("Échec de connexion."); }
            finally { btn.textContent = 'EXÉCUTER LA PRÉDICTION'; btn.style.opacity = '1'; }
        });

        function updateInfBar(id, val) {
            const el = document.getElementById(id);
            const width = Math.abs(val) * 10; // 5 -> 50% du demi-conteneur
            el.style.width = width + '%';
            if (val >= 0) {
                el.style.left = '50%';
                el.style.right = 'auto';
                el.style.background = 'var(--success-green)';
            } else {
                el.style.left = 'auto';
                el.style.right = '50%';
                el.style.background = 'var(--alert-red)';
            }
        }

        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('batchFile');
        const btnBatch = document.getElementById('btnBatch');
        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if(e.dataTransfer.files.length) { fileInput.files = e.dataTransfer.files; handleFileSelect(); }
        });
        fileInput.addEventListener('change', handleFileSelect);
        function handleFileSelect() {
            if(fileInput.files.length > 0) {
                dropZone.querySelector('h3').textContent = fileInput.files[0].name;
                btnBatch.style.display = 'block';
            }
        }
        async function uploadCSV() {
            if(!fileInput.files.length) return;
            document.getElementById('batchStats').style.display = 'none';
            document.getElementById('batchResultsTable').style.display = 'none';
            btnBatch.style.display = 'none';
            dropZone.style.display = 'none';
            document.getElementById('batchLoader').style.display = 'block';
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            try {
                const res = await fetch('/predict/batch', { method: 'POST', body: formData });
                const data = await res.json();
                document.getElementById('batchLoader').style.display = 'none';
                dropZone.style.display = 'block';
                btnBatch.style.display = 'block';
                if(!res.ok) alert("ERREUR: " + (data.detail || JSON.stringify(data)));
                else {
                    document.getElementById('batchStats').style.display = 'block';
                    document.getElementById('batchStats').innerHTML = `DOSSIERS TRAITÉS : ${data.total_processed} <br> SURVIVANTS PRÉDITS : ${data.survivors_predicted} <span style="color:var(--success-green);">(${(data.survival_rate*100).toFixed(1)}%)</span>`;
                    window.batchData = data.predictions;
                    window.currentPage = 1;
                    renderBatchTable();
                }
            } catch(e) { alert("L'envoi a échoué."); document.getElementById('batchLoader').style.display = 'none'; }
        }
        function renderBatchTable() {
            const pageSize = 10;
            const data = window.batchData || [];
            const totalPages = Math.ceil(data.length / pageSize) || 1;
            if(window.currentPage < 1) window.currentPage = 1;
            if(window.currentPage > totalPages) window.currentPage = totalPages;
            const start = (window.currentPage - 1) * pageSize;
            const slice = data.slice(start, start + pageSize);
            let html = '<table><thead><tr><th>PASSAGER</th><th>CLASSE</th><th>STATUT</th><th>PROBABILITÉ</th></tr></thead><tbody>';
            slice.forEach(p => {
                let statusCls, statusTxt;
                if (p.probability > 0.55) { statusCls = 'status-ok'; statusTxt = 'SURVÉCU'; }
                else if (p.probability < 0.45) { statusCls = 'status-fail'; statusTxt = 'DÉCÉDÉ'; }
                else { statusCls = 'status-warn'; statusTxt = 'INCERTAIN'; }
                
                html += `<tr><td><strong>${p.name}</strong></td><td>${p.pclass}</td><td><span class="status-badge ${statusCls}">${statusTxt}</span></td><td>${(p.probability*100).toFixed(1)}%</td></tr>`;
            });
            html += '</tbody></table>';
            html += `<div style="display:flex; justify-content:space-between; align-items:center; margin-top:1rem; font-family:'Space Mono', monospace; font-size:0.9rem;">
                <button onclick="window.currentPage--; renderBatchTable()" style="padding:0.5rem; background:var(--ink-black); color:var(--bg-paper); border:none; cursor:pointer;" ${window.currentPage === 1 ? 'disabled style="opacity:0.5;"' : ''}>⬅ PRÉCÉDENT</button>
                <span>PAGE ${window.currentPage} / ${totalPages}</span>
                <button onclick="window.currentPage++; renderBatchTable()" style="padding:0.5rem; background:var(--ink-black); color:var(--bg-paper); border:none; cursor:pointer;" ${window.currentPage === totalPages ? 'disabled style="opacity:0.5;"' : ''}>SUIVANT ➡</button>
            </div>`;
            document.getElementById('batchResultsTable').style.display = 'block'; 
            document.getElementById('batchResultsTable').innerHTML = html;
        }
    </script>
</body>
</html>"""


HTML_ADMIN_PAGE = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PUPITRE ADMIN</title>
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400;1,700&family=Space+Mono:ital,wght@0,400;0,700;1,400;1,700&display=swap" rel="stylesheet">
    <style>
        :root { --bg-paper: #f4f1ea; --ink-black: #0f1110; --ink-fade: #3f4240; --alert-red: #ff3311; --ocean-blue: #002244; --success-green: #0a6b3d; --border-thick: 4px solid var(--ink-black); --border-thin: 1px solid var(--ink-black); }
        * { box-sizing: border-box; margin: 0; padding: 0; }

        /* MODAL SYSTEM (Glassmorphism & Centering) */
        .modal-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px); z-index: 1000; align-items: center; justify-content: center; }
        .modal-card { width: 450px; background: rgba(15, 17, 16, 0.95); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 2.5rem; box-shadow: 0 20px 50px rgba(0,0,0,0.5); transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); transform: scale(0.9) translateY(20px); opacity: 0; text-transform: none; text-align: center; color: white; }
        .modal-card.active { transform: scale(1) translateY(0); opacity: 1; }
        .modal-card h3 { color: #fff; margin-bottom: 1rem; font-size: 1.8rem; font-family: 'Playfair Display', serif; border-bottom: none; }
        .modal-card p { color: rgba(255,255,255,0.7); font-size: 0.95rem; margin-bottom: 2rem; line-height: 1.5; text-transform: none; font-family: 'Space Mono', monospace; }
        .modal-buttons { display: flex; gap: 1rem; }
        .modal-card button { padding: 1rem 1.5rem; font-weight: bold; border-radius: 6px; border: none; cursor: pointer; transition: 0.3s; font-family: 'Space Mono', monospace; flex: 1; }
        
        .btn-confirm { background: var(--success-green); color: #fff; }
        .btn-confirm:hover { background: #fff; color: var(--success-green); transform: translateY(-2px); }
        .btn-cancel { background: rgba(255,255,255,0.1); color: #fff; }
        .btn-cancel:hover { background: rgba(255,255,255,0.2); }
        body { font-family: 'Space Mono', monospace; background-color: var(--bg-paper); color: var(--ink-black); padding: 2rem; min-height: 100vh; text-transform: uppercase; }
        h1, h2, h3 { font-family: 'Playfair Display', serif; text-transform: none; letter-spacing: -0.02em; }
        h1 { font-size: 4rem; font-weight: 900; line-height: 1.1; margin-bottom: 0.5rem; text-align: center; border-bottom: var(--border-thick); padding-bottom: 1rem; }
        h2 { font-size: 2.2rem; border-bottom: var(--border-thin); margin-bottom: 1.5rem; padding-bottom: 0.5rem; }
        p { font-size: 0.9rem; margin-bottom: 1rem; }
        .header-meta { display: flex; justify-content: space-between; font-size: 0.8rem; border-bottom: var(--border-thick); padding: 0.5rem 0; margin-bottom: 2rem; align-items: center; }
        
        .btn-home { padding: 0.25rem 0.5rem; background: var(--bg-paper); color: var(--ink-black); cursor: pointer; border: 2px solid var(--ink-black); font-family: 'Space Mono', monospace; font-size: 0.8rem; font-weight: bold; text-decoration: none; }
        .btn-home:hover { background: var(--ink-black); color: var(--bg-paper); }

        .container { max-width: 1300px; margin: 0 auto; border: var(--border-thick); background: #fff; padding: 2rem; box-shadow: 12px 12px 0px var(--ink-black); }
        .tabs { display: flex; border: var(--border-thick); margin-bottom: 2rem; }
        .tab { flex: 1; padding: 1rem; text-align: center; cursor: pointer; border-right: var(--border-thick); font-weight: bold; background: var(--bg-paper); transition: all 0.2s ease; }
        .tab:last-child { border-right: none; }
        .tab:hover { background: var(--ink-fade); color: var(--bg-paper); }
        .tab.active { background: var(--ink-black); color: var(--bg-paper); }
        .tab-content { display: none; }
        .tab-content.active { display: block; animation: appear 0.3s steps(4, end); }
        @keyframes appear { 0% { opacity: 0; transform: translateY(10px); } 100% { opacity: 1; transform: translateY(0); } }
        
        table { width: 100%; border-collapse: collapse; }
        th, td { border: var(--border-thin); padding: 0.75rem; text-align: left; font-size: 0.85rem; }
        th { background: var(--ink-black); color: var(--bg-paper); position: sticky; top: 0; z-index: 10; }
        tr:nth-child(even) { background: var(--bg-paper); }
        
        .btn-large { display: block; width: 100%; padding: 1.5rem; background: var(--alert-red); color: var(--bg-paper); font-family: 'Space Mono', monospace; font-size: 1.5rem; font-weight: bold; text-transform: uppercase; border: var(--border-thick); cursor: pointer; box-shadow: 6px 6px 0px var(--ink-black); transition: transform 0.1s; }
        .btn-large:hover { box-shadow: 2px 2px 0px var(--ink-black); transform: translate(4px, 4px); background: #e0290b; }
        .btn-large:active { box-shadow: none; transform: translate(6px, 6px); }
        
        .status-badge { display: inline-block; padding: 0.25rem 0.5rem; font-size: 0.75rem; font-weight: bold; border: var(--border-thin); }
        .status-ok { background: var(--success-green); color: white; }
        .status-fail { background: var(--alert-red); color: white; font-weight: 900; }
        .status-warn { background: #f39c12; color: white; }
        .status-blue { background: var(--ocean-blue); color: white; }

        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }
        .metric-card { border: var(--border-thick); padding: 1.5rem; text-align: center; background: var(--ocean-blue); color: var(--bg-paper); }
        .metric-value { font-size: 3rem; font-family: 'Playfair Display', serif; font-weight: 900; margin-bottom: 0.5rem;}
        
        .marquee { width: 100%; overflow: hidden; background: var(--ink-black); color: var(--bg-paper); padding: 0.5rem; font-size: 0.8rem; margin-top: 2rem; border: var(--border-thick); white-space: nowrap; position: relative; }
        .marquee p { display: inline-block; white-space: nowrap; margin: 0; padding-left: 100%; animation: scroll 20s linear infinite; }
        @keyframes scroll { 100% { transform: translateX(-100%); } }

        /* TOAST NOTIFICATION */
        .toast { display: none; position: fixed; top: 20px; right: 20px; background: var(--success-green); color: white; padding: 1rem 2rem; border-radius: 8px; z-index: 2000; box-shadow: 0 4px 12px rgba(0,0,0,0.3); font-family:'Space Mono'; animation: slideIn 0.5s ease-out; }
        @keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity:1; } }

        .pagination-controls { display:flex; justify-content:space-between; align-items:center; margin-top:1rem; font-family:'Space Mono'; font-size:0.8rem; }
        .pagination-controls button { padding: 0.5rem; background: var(--ink-black); color: white; border:none; cursor:pointer;}
        .pagination-controls button:disabled { opacity: 0.2; cursor: not-allowed; }

        /* RESPONSIVENESS */
        @media (max-width: 768px) {
            body { padding: 1rem; }
            h1 { font-size: 2.2rem; }
            h2 { font-size: 1.6rem; }
            .header-meta { flex-direction: column; gap: 0.5rem; text-align: center; }
            .container { padding: 1.5rem; box-shadow: 6px 6px 0px var(--ink-black); }
            .tabs { flex-direction: column; }
            .tab { border-right: none; border-bottom: var(--border-thick); }
            .tab:last-child { border-bottom: none; }
            .metrics-grid { grid-template-columns: 1fr; }
            .metric-value { font-size: 2.5rem; }
            .btn-large { font-size: 1rem; padding: 1rem; }
            .modal-card { width: 95%; padding: 1.5rem; }
            .batch-results { max-height: 250px; }
            th, td { padding: 0.5rem; font-size: 0.75rem; }
        }
    </style>
</head>
<body>

    <div id="retrainToast" class="toast">RE-ENTRAÎNEMENT LANCÉ</div>
    
    <!-- MODAL RÉ-ENTRAÎNEMENT -->
    <div id="retrainModal" class="modal-overlay" onclick="if(event.target === this) closeRetrainModal()">
        <div class="modal-card" id="retrainCard">
            <h3>Confirmer la Séquence</h3>
            <p>Cette action va déclencher une nouvelle phase d'ajustement du modèle sur les dernières données archivées. Voulez-vous continuer ?</p>
            <div class="modal-buttons">
                <button onclick="confirmRetrain()" class="btn-confirm">DÉMARRER</button>
                <button onclick="closeRetrainModal()" class="btn-cancel">ANNULER</button>
            </div>
        </div>
    </div>

    <div class="header-meta">
        <span>BUREAU DES PRÉDICTIONS</span>
        <span>ÉMISSION : 1912 / 2026</span>
        <a href="/" class="btn-home">⬅ RETOUR PUBLIC</a>
    </div>

    <div class="container">
        <h1>PUPITRE ADMINISTRATEUR<br>SUPERVISION MLOPS</h1>

        <div class="tabs">
            <div class="tab active" onclick="switchTab('monitoring')">DIAGNOSTIC ET RÉSEAU</div>
            <div class="tab" onclick="switchTab('versions')">ARCHIVES DES MODÈLES</div>
        </div>

        <div id="tab-monitoring" class="tab-content active">
            <h2>MÉTRIQUES PHYSIQUES DU SERVEUR</h2>
            <div class="metrics-grid">
                <div class="metric-card" style="background:#fff; color:var(--ink-black); border-color:var(--ink-fade);">
                    <div class="metric-value" id="cpuUsage">--%</div>
                    <div>CHARGE PROCESSEUR (CPU)</div>
                </div>
                <div class="metric-card" style="background:#fff; color:var(--ink-black); border-color:var(--ink-fade);">
                    <div class="metric-value" id="ramUsage">--%</div>
                    <div>UTILISATION MÉMOIRE (RAM)</div>
                </div>
            </div>

            <h2>TRAFIC RÉSEAU ET ADRESSES IP</h2>
            <div class="batch-results" style="display:block; max-height:300px; margin-bottom:2rem; overflow-y:auto; border:var(--border-thick);">
                <table id="networkLogsTable">
                    <thead><tr><th>DATE/HEURE</th><th>ADRESSE IP</th><th> MÉTHODE </th><th>ENDPOINT</th><th>LATENCE</th></tr></thead>
                    <tbody id="netLogBody"></tbody>
                </table>
            </div>
            <div id="netPagination" class="pagination-controls" style="margin-bottom:2rem;"></div>
            
            <h3>ANALYSE DATA DRIFT EN DÉTAIL</h3>
            <div id="driftDetails" style="margin-bottom: 2rem;"></div>

            <h3>ARCHIVES LOGIQUES (REQUÊTES PRÉDICTIVES)</h3>
            <div class="batch-results" style="display:block; max-height:300px; overflow-y:auto; border:var(--border-thick);">
                <table id="predLogsTable">
                    <thead><tr><th>DATE/HEURE</th><th>NOM</th><th>STATUT</th><th>SCORE</th></tr></thead>
                    <tbody id="predLogBody"></tbody>
                </table>
            </div>
            <div id="predPagination" class="pagination-controls"></div>
        </div>

        <div id="tab-versions" class="tab-content">
            <h2>REGISTRE LOCAL DES MODÈLES</h2>
            <div id="activeModelInfo" style="margin-bottom: 2rem;"></div>
            <h3>ARCHIVES SYSTÈME (MLOps)</h3>
            <div class="batch-results" style="display:block; max-height:300px; overflow-y:auto; border:var(--border-thick);">
                <table id="versionsTable">
                    <thead><tr><th>FICHIER</th><th>TAILLE</th><th>MODIFIÉ LE</th><th>STATUT</th></tr></thead>
                    <tbody id="versionsBody"></tbody>
                </table>
            </div>
            <div id="vPagination" class="pagination-controls" style="margin-bottom:1rem;"></div>
            <button class="btn-large" style="margin-top:2rem; font-size:1rem; padding:1rem; background:var(--ink-black);" onclick="triggerRetrain()">FORCER LA SÉQUENCE DE RÉ-ENTRAÎNEMENT</button>
        </div>
    </div>


    <script>
        window.netPage = 1; window.predPage = 1; window.vPage = 1;
        const PAGE_SIZE = 8;

        function switchTab(tabId) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.currentTarget.classList.add('active');
            document.getElementById('tab-' + tabId).classList.add('active');
            if (tabId === 'monitoring') loadMonitoring();
            if (tabId === 'versions') loadVersions();
        }

        async function loadMonitoring() {
            try {
                const sHealth = await fetch('/monitoring/system');
                if(sHealth.ok) {
                    const health = await sHealth.json();
                    document.getElementById('cpuUsage').textContent = health.cpu_percent + '%';
                    document.getElementById('ramUsage').textContent = health.memory_percent + '%';
                    renderTable('netLogBody', 'netPagination', health.network_logs, (n) => `
                        <tr>
                            <td>${n.timestamp.split('T')[1].substring(0,8)}</td>
                            <td><strong>${n.ip}</strong></td>
                            <td>${n.method}</td>
                            <td>${n.path}</td>
                            <td><span class="status-badge status-blue">${n.latency} ms</span></td>
                        </tr>
                    `, 'netPage');
                }

                const dr = await fetch('/monitoring/drift'); const drift = await dr.json();
                const statusBox = document.getElementById('driftStatus');
                const dStatus = drift.status;
                statusBox.textContent = (dStatus === "no_drift" || dStatus === "insufficient_data") ? "NORMAL" : "ALERTE";
                statusBox.style.color = (dStatus === "no_drift" || dStatus === "insufficient_data") ? "var(--success-green)" : "var(--alert-red)";
                
                let dh = '<table><thead><tr><th>CARACTÉRISTIQUE</th><th>RÉFÉRENCE</th><th>ACTUEL</th><th>ÉCART (Z-SCORE)</th></tr></thead><tbody>';
                if(drift.drift_analysis) {
                    for(const [f, details] of Object.entries(drift.drift_analysis)) {
                        const dev = details.deviation || 0;
                        let statusTxt = details.status;
                        let cls = 'status-ok';
                        if(statusTxt === 'ALERT') { cls = 'status-fail'; statusTxt = 'ALERTE'; }
                        else if(statusTxt === 'WARNING') { cls = 'status-warn'; statusTxt = 'WARNING'; }
                        else { statusTxt = 'OK'; }

                        dh += `<tr><td><strong>${f}</strong></td><td>${details.reference_mean.toFixed(2)}</td><td>${(details.current_mean||0).toFixed(2)}</td><td><span class="status-badge ${cls}">${dev.toFixed(2)} (${statusTxt})</span></td></tr>`;
                    }
                }
                dh += '</tbody></table>'; document.getElementById('driftDetails').innerHTML = dh;

                const rLogs = await fetch('/monitoring/logs?limit=100'); const logsData = await rLogs.json();
                renderTable('predLogBody', 'predPagination', logsData.logs, (l) => {
                    let sCls, sTxt;
                    if (l.probability > 0.55) { sCls = 'status-ok'; sTxt = 'SURVÉCU'; }
                    else if (l.probability < 0.45) { sCls = 'status-fail'; sTxt = 'DÉCÉDÉ'; }
                    else { sCls = 'status-warn'; sTxt = 'INCERTAIN'; }
                    return `
                        <tr>
                            <td>${l.timestamp.substring(11,19)}</td>
                            <td>${l.name}</td>
                            <td><span class="status-badge ${sCls}">${sTxt}</span></td>
                            <td>${(l.probability*100).toFixed(1)}%</td>
                        </tr>
                    `;
                }, 'predPage');

            } catch(e) { console.error(e); }
        }

        async function loadVersions() {
            const rInfo = await fetch('/model/info'); const info = await rInfo.json();
            document.getElementById('activeModelInfo').innerHTML = `
                <div style="border:var(--border-thick); padding:1rem; background:var(--ink-black); color:white; display:flex; justify-content:space-between; align-items:center;">
                    <span><strong>ACTIF :</strong> ${info.model_name}</span>
                    <span><strong>FEATURES :</strong> ${info.n_features}</span>
                </div>`;
            const rVers = await fetch('/model/versions'); const vData = await rVers.json();
            renderTable('versionsBody', 'vPagination', vData.versions, (v) => `
                <tr>
                    <td><strong>${v.filename}</strong></td>
                    <td>${v.size}</td>
                    <td>${v.modified}</td>
                    <td><span class="status-badge ${v.is_active?'status-ok':'status-warn'}">${v.is_active?'ACTIF':'ARCHIVÉ'}</span></td>
                </tr>
            `, 'vPage');
        }

        function renderTable(bodyId, pagId, data, rowTpl, pageVar) {
            const container = document.getElementById(bodyId);
            const pagContainer = document.getElementById(pagId);
            const dataToUse = data || [];
            const page = window[pageVar];
            const start = (page - 1) * PAGE_SIZE;
            const slice = dataToUse.slice(start, start + PAGE_SIZE);
            const totalPages = Math.ceil(dataToUse.length / PAGE_SIZE) || 1;

            if (window[pageVar] > totalPages) window[pageVar] = totalPages;
            if (window[pageVar] < 1) window[pageVar] = 1;

            container.innerHTML = slice.length ? slice.map(rowTpl).join('') : '<tr><td colspan="5" style="text-align:center;">Aucune donnée</td></tr>';
            pagContainer.innerHTML = `
                <button onclick="window['${pageVar}']--; loadMonitoring(); loadVersions()" ${page===1?'disabled style="opacity:0.3;"':''} style="border-radius:4px;">⬅ PRÉC</button>
                <span style="font-weight:bold;">PAGE ${page} / ${totalPages}</span>
                <button onclick="window['${pageVar}']++; loadMonitoring(); loadVersions()" ${page===totalPages?'disabled style="opacity:0.3;"':''} style="border-radius:4px;">SUIV ➡</button>
            `;
        }

        function triggerRetrain() {
            document.getElementById('retrainModal').style.display = 'flex';
            setTimeout(() => document.getElementById('retrainCard').classList.add('active'), 10);
        }
        function closeRetrainModal() {
            document.getElementById('retrainCard').classList.remove('active');
            setTimeout(() => document.getElementById('retrainModal').style.display = 'none', 300);
        }
        async function confirmRetrain() {
            closeRetrainModal();
            try {
                const res = await fetch('/model/retrain', { method: 'POST' });
                const data = await res.json();
                const toast = document.getElementById('retrainToast');
                toast.textContent = data.message;
                toast.style.display = 'block';
                setTimeout(() => toast.style.display = 'none', 5000);
            } catch(e) { }
        }

        loadMonitoring();
        setInterval(loadMonitoring, 10000);
    </script>
</body>
</html>"""


# ─────────────────────────────────────────────
#  ENDPOINTS — INTERFACE WEB
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def root_ui():
    """Interface principale (prédiction seulement)."""
    return HTMLResponse(content=HTML_USER_PAGE)

@app.get("/monitor", response_class=HTMLResponse, tags=["UI Admin"])
async def monitor_ui():
    """Interface d'administration MLOps secrète."""
    return HTMLResponse(content=HTML_ADMIN_PAGE)

# ─────────────────────────────────────────────
#  ENDPOINTS — PRÉDICTION
# ─────────────────────────────────────────────

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(passenger: PassengerInput):
    """Prédit la survie d'un passager du Titanic."""
    global prediction_count, survived_count

    if model_artifact is None:
        raise HTTPException(status_code=503, detail="Modele non disponible.")

    try:
        survived, proba, influences = make_prediction(
            passenger.Pclass, passenger.Name, passenger.Sex,
            passenger.Age, passenger.SibSp, passenger.Parch,
            passenger.Fare, passenger.Cabin, passenger.Embarked,
        )

        # Monitoring : enregistrer la prédiction
        prediction_count += 1
        if survived:
            survived_count += 1

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "name": html.escape(str(passenger.Name)),
            "sex": passenger.Sex,
            "age": passenger.Age,
            "pclass": passenger.Pclass,
            "fare": passenger.Fare,
            "sibsp": passenger.SibSp,
            "parch": passenger.Parch,
            "embarked": passenger.Embarked,
            "survived": survived,
            "probability": round(proba, 4),
        }
        prediction_log.append(log_entry)
        
        # Persistance réelle (Data Archive Enterprise)
        try:
            with open(LOGS_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"[ERROR] Echec de l'archivage de la prédiction: {e}")

        if proba > 0.8 or proba < 0.2:
            confidence = "Haute"
        elif proba > 0.6 or proba < 0.4:
            confidence = "Moyenne"
        else:
            confidence = "Faible"

        return PredictionOutput(
            survived=survived,
            probability=round(proba, 4),
            confidence=confidence,
            passenger_summary=f"{passenger.Name} | {passenger.Sex} | Classe {passenger.Pclass} | Age {passenger.Age}",
            influences=influences
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(file: UploadFile = File(...)):
    """Traitement par lots d'un fichier CSV sécurisé."""
    import csv

    # SÉCURITÉ 1: Filtrage type MIME & extension
    allowed_mimes = ["text/csv", "application/vnd.ms-excel", "text/plain"]
    if not file.filename.lower().endswith('.csv') or file.content_type not in allowed_mimes:
        raise HTTPException(status_code=400, detail="Type de fichier non autorisé. Seul le format CSV est accepté.")

    try:
        # SÉCURITÉ 2: Limite de taille (2 Mo)
        MAX_FILE_SIZE = 2 * 1024 * 1024 
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
             raise HTTPException(status_code=413, detail="Fichier trop volumineux (Limite: 2Mo).")
        
        try:
            sample = content[:1024].decode('utf-8')
            dialect = csv.Sniffer().sniff(sample)
            separator = dialect.delimiter
        except Exception:
            separator = ','

        df = pd.read_csv(io.BytesIO(content), sep=separator)
        
        required_cols = {"Pclass", "Name", "Sex", "Age"}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"Structure CSV invalide. Attendu : {required_cols}")

        results = []
        survivors = 0
        total = 0

        for idx, row in df.iterrows():
            total += 1
            try:
                age = float(row.get("Age", 30))
                if np.isnan(age): age = 29.0
                fare = float(row.get("Fare", 32))
                if np.isnan(fare): fare = 32.0

                survived, proba, influences = make_prediction(
                    pclass=int(row["Pclass"]),
                    name=html.escape(str(row["Name"])),
                    sex=str(row["Sex"]),
                    age=age,
                    sibsp=int(row.get("SibSp", 0)),
                    parch=int(row.get("Parch", 0)),
                    fare=fare,
                    cabin=str(row.get("Cabin", "")),
                    embarked=str(row.get("Embarked", "S"))
                )
                if survived: survivors += 1
                results.append({"name": html.escape(str(row["Name"])), "pclass": int(row["Pclass"]), "survived": survived, "probability": round(proba, 4), "influences": influences})
            except Exception as row_e:
                results.append({"name": html.escape(str(row.get("Name", f"Ligne {idx}"))), "pclass": "?", "error": str(row_e)})

        return {"total_processed": total, "survivors_predicted": survivors, "survival_rate": round(survivors / total, 4) if total > 0 else 0, "predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur CSV: {str(e)}")

# ─────────────────────────────────────────────
#  ENDPOINTS — MONITORING
# ─────────────────────────────────────────────

@app.get("/monitoring/system", tags=["Monitoring"])
async def monitoring_system():
    import psutil
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().percent
    return {"cpu_percent": round(cpu, 1), "memory_percent": round(mem, 1), "network_logs": list(network_logs)}

@app.get("/monitoring/stats", tags=["Monitoring"])
async def monitoring_stats():
    return {"total_predictions": prediction_count, "survived_predictions": survived_count, "survival_rate": round(survived_count / prediction_count, 4) if prediction_count > 0 else None}

@app.get("/monitoring/drift", tags=["Monitoring"])
async def monitoring_drift():
    if len(prediction_log) < 5:
        return {"status": "insufficient_data", "message": "Min 5 samples", "drift_analysis": {}}

    logs = list(prediction_log)
    drift_analysis = {}
    for feature in ["Age", "Fare", "Pclass", "SibSp", "Parch"]:
        key = feature.lower()
        values = [l[key] for l in logs if key in l]
        if not values: continue
        current_mean = np.mean(values)
        ref = TRAINING_STATS.get(feature, {})
        ref_mean = ref.get("mean", current_mean)
        ref_std = ref.get("std", 1.0)
        deviation = abs(current_mean - ref_mean) / ref_std if ref_std > 0 else 0
        status = "ALERT" if deviation > 2.0 else "WARNING" if deviation > 1.0 else "OK"
        drift_analysis[feature] = {"current_mean": round(current_mean, 2), "reference_mean": round(ref_mean, 2), "deviation": round(deviation, 2), "status": status}

    male_ratio = sum(1 for l in logs if l.get("sex") == "male") / len(logs)
    sex_dev = abs(male_ratio - TRAINING_STATS["Sex_male_ratio"]) / 0.48
    drift_analysis["Sex (male ratio)"] = {"current_mean": round(male_ratio, 2), "reference_mean": TRAINING_STATS["Sex_male_ratio"], "deviation": round(sex_dev, 2), "status": "ALERT" if sex_dev > 2.0 else "WARNING" if sex_dev > 1.0 else "OK"}

    has_drift = any(v["status"] != "OK" for v in drift_analysis.values())
    return {"status": "drift_detected" if has_drift else "no_drift", "n_samples": len(logs), "drift_analysis": drift_analysis}

@app.get("/monitoring/logs", tags=["Monitoring"])
async def monitoring_logs(limit: int = 100):
    logs = list(prediction_log)
    logs.reverse()
    return {"total": len(prediction_log), "logs": logs}

@app.get("/model/versions", tags=["Versioning"])
async def list_model_versions():
    versions = []
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pkl"))
    for f in sorted(model_files, key=os.path.getmtime, reverse=True):
        stat = os.stat(f); size_kb = stat.st_size / 1024
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        is_active = model_artifact and os.path.abspath(f) == os.path.abspath(model_artifact.get("model_path", ""))
        versions.append({"filename": os.path.basename(f), "size": f"{size_kb:.1f} KB", "modified": modified, "is_active": is_active})
    return {"versions": versions, "total": len(versions)}

@app.get("/model/info", tags=["Versioning"])
async def model_info():
    if model_artifact is None: raise HTTPException(status_code=503, detail="No model")
    return {"model_name": model_artifact.get("model_name", "unknown"), "n_features": len(model_artifact.get("feature_names", [])), "loaded_at": model_artifact.get("loaded_at")}

@app.post("/model/retrain", tags=["Versioning"])
async def retrain_model():
    import subprocess, threading
    def run_validation_script():
        script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "validate.py")
        subprocess.run([sys.executable, script_path], check=False)
        load_model_artifact()
    threading.Thread(target=run_validation_script).start()
    return {"status": "processing", "message": "RÉ-ENTRAÎNEMENT LANCÉ EN ARRIÈRE-PLAN"}

@app.get("/health", tags=["Monitoring"])
async def health():
    return {"status": "healthy" if model_artifact else "degraded", "model_loaded": model_artifact is not None, "total_predictions": prediction_count, "uptime": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
