# GitHub et déploiement

## 1. Dépôt GitHub

Créer un **nouveau dépôt vide** sur GitHub, par exemple `fraud-risk-analytics` (ne pas écraser le premier projet).

Depuis le dossier `fraud-risk-dashboard` dans PowerShell :

```powershell
git init
git add .
git commit -m "Initial fraud detection dashboard"
git branch -M main
git remote add origin https://github.com/VOTRE_COMPTE/fraud-risk-analytics.git
git push -u origin main
```

Vérifier que `.env`, les CSV et `backend/artifacts/*.joblib` ne sont pas dans le dépôt.

## 2. Render : FastAPI

Créer un **Web Service** relié au nouveau dépôt.

| Paramètre | Valeur |
|---|---|
| Language | Python 3 |
| Branch | main |
| Root Directory | vide |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `python -m uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port $PORT` |
| Environment | `LOG_LEVEL=INFO` et `KAGGLE_MAX_ROWS=90000` |

Choisir Python 3.11 si proposé. Tester `https://VOTRE-SERVICE.onrender.com/health` ; réponse attendue : `{"status":"ok","service":"fraud-risk-api"}`.

Sur Render Free, les fichiers entraînés et les CSV importés disparaissent au redémarrage ou au redéploiement. Une version avec persistance nécessiterait un stockage externe.

**Pour un premier déploiement**, utiliser le jeu de démonstration. Le téléchargement Kaggle et un entraînement sur 90 000 lignes peuvent dépasser les ressources du plan gratuit. Il est possible d'importer plutôt un petit CSV Sparkov depuis l'interface, mais ce CSV ne sera pas conservé après un redémarrage de Render.

## 3. Streamlit Community Cloud

Ouvrir https://share.streamlit.io/ puis **Create app** et sélectionner le même dépôt.

| Paramètre | Valeur |
|---|---|
| Main file path | `frontend/app.py` |
| Branch | `main` |
| Python | 3.11 si disponible |

Dans **Advanced settings > Secrets**, mettre :

```toml
API_BASE_URL = "https://VOTRE-SERVICE.onrender.com"
```

Ne pas ajouter `/docs` ni `/health` à l'URL.

Déployer, puis vérifier : chargement des données -> préparation -> entraînement -> évaluation -> SHAP.

## 4. Actualisations

```powershell
git add .
git commit -m "Update fraud dashboard"
git push
```

Les plateformes redéploient les commits de la branche connectée si le déploiement automatique est activé.
