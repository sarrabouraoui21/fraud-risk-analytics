# Fraud Risk Analytics

Mini-application de détection de fraude sur les transactions Sparkov. Le travail suit le Challenge 2 de Data Mining : comparer des approches supervisées et non supervisées, classer les opérations suspectes selon un budget et expliquer les alertes.

## Choix des données

Dataset **Credit Card Transactions Fraud Detection (Sparkov)**, cible `is_fraud`.
Source Kaggle : https://www.kaggle.com/datasets/kartik2112/fraud-detection

L'application peut télécharger automatiquement un **sous-ensemble chronologique de 90 000 lignes de `fraudTrain.csv`**. Le jeu complet est plus volumineux et n'est pas conservé dans le dépôt. Un mode de démonstration synthétique est proposé pour vérifier l'installation immédiatement ; ses résultats ne doivent pas être utilisés dans le rapport comme résultats du dataset Sparkov.

L'import CSV accepte le schéma Sparkov (`amt`, `is_fraud`, `trans_date_trans_time`, `cc_num`, `category` et autres colonnes éventuelles). Sans date, le split est aléatoire stratifié et ce choix est indiqué dans l'application.

## Démarrage (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r frontend/requirements.txt
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
python -m uvicorn app.main:app --app-dir backend --reload --log-level info --access-log
```

Dans un second terminal, depuis la racine :

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run frontend/app.py
```

- Streamlit : http://localhost:8501
- FastAPI : http://localhost:8000/docs
- Santé de l'API : http://localhost:8000/health

## Pipeline

1. Visualisation du taux de fraude, des montants, du temps et des catégories.
2. Construction des variables cycliques, distances et historiques par carte : nombre d'opérations sur 1 h / 24 h, moyenne glissante 30 jours et délai entre opérations. Les variables de fréquence utilisent uniquement les transactions **antérieures** à chaque opération.
3. Split chronologique **60 % / 20 % / 20 %**, puis imputation/encodage/normalisation ajustés sur le train.
4. Régression logistique (pondérée ou non, choix sur validation), Random Forest équilibrée, Isolation Forest et LOF. Les deux méthodes d'anomalies sont entraînées sur les transactions légitimes du train uniquement.
5. PR-AUC, ROC-AUC, précision@k, rappel@k, F2, MCC. Sélection du seuil sur la validation : somme des montants des fraudes non détectées + coût unitaire de revue × alertes, dans un budget maximal. Les performances sur test sont conservées séparément.
6. Explication SHAP et importance par permutation ; waterfall et codes motifs pour une transaction.

L'API expose aussi `POST /evaluate/predict` pour un enregistrement **déjà enrichi** avec les variables comportementales. Cela ne remplace pas un calcul d'historique temps réel.

## Architecture

```text
backend/app/api/        Routes FastAPI
backend/app/services/  Données, features, modèles, métriques, XAI
frontend/pages/         Pages Streamlit 1 à 5
backend/artifacts/      Modèles et rapports (hors Git)
```

## Vérification

```powershell
python -m pytest tests -q
```

## Limites

- Sparkov est **synthétique**. Le classement d'alertes et le coût métier ne représentent pas un système bancaire opérationnel.
- Le jeu téléchargé est un **sous-échantillon de la fin de `fraudTrain.csv`**, pas le jeu intégral. La taille influe sur le taux de fraude et les scores.
- Le coût des alertes est une hypothèse paramétrable. Le seuil doit être fixé sur validation ; ne pas sélectionner le meilleur résultat du test après exploration des curseurs.
- Les scores non supervisés sont exprimés en rangs d'anomalie par rapport au train. Ils ne constituent pas des probabilités de fraude.
- Les explications SHAP décrivent le fonctionnement des modèles, sans prouver de lien causal.
- Render Free conserve les modèles uniquement tant que son disque temporaire existe ; après redémarrage/re-déploiement, il faut réentraîner.
- Le jeu Sparkov peut nécessiter beaucoup de mémoire et de temps de téléchargement sur un hébergement gratuit. Commencer par le mode démonstration.

## Sous-échantillon à importer si Render manque de mémoire

Après téléchargement du CSV Kaggle sur le PC :

```powershell
python scripts/prepare_sample.py "C:\chemin\vers\fraudTrain.csv" --rows 25000 --output sparkov_sample.csv
```

Charger `sparkov_sample.csv` dans l'onglet Données. Garder un échantillon **continu et chronologique** (pas un échantillonnage aléatoire de la cible) afin de respecter le split temporel et la proportion naturelle des fraudes.

Consulter aussi `DEPLOIEMENT.md` pour GitHub, Render et Streamlit Cloud.
