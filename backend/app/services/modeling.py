import json
import logging
import threading
import uuid
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.neighbors import LocalOutlierFactor

from app.services.datasets import dataset
from app.services.features import build_features, split_data, make_transformer
from app.services.metrics import score_metrics, money_cost, optimize_threshold

logger = logging.getLogger(__name__)
ARTIFACTS = Path(__file__).resolve().parents[2] / 'artifacts'
ARTIFACTS.mkdir(parents=True, exist_ok=True)
JOBS = {}
LOCK = threading.Lock()
MODELS = {'logistic_regression': 'Régression logistique', 'random_forest': 'Random Forest',
          'isolation_forest': 'Isolation Forest', 'lof': 'Local Outlier Factor'}


def prepare(dataset_id):
    df = dataset(dataset_id)
    engineered, numeric, categorical = build_features(df)
    parts, method = split_data(engineered)
    train, valid, test = parts
    columns = numeric + categorical
    transformer = make_transformer(numeric, categorical)
    X_train = transformer.fit_transform(train[columns])
    X_valid = transformer.transform(valid[columns])
    X_test = transformer.transform(test[columns])
    return {
        'raw': [train, valid, test], 'encoded': [X_train, X_valid, X_test],
        'transformer': transformer, 'columns': columns, 'numeric': numeric,
        'categorical': categorical, 'split_method': method,
        'features': engineered,
    }


def positive_scores(model, encoded, method, calibration=None):
    if method in ('logistic_regression', 'random_forest'):
        return model.predict_proba(encoded)[:, 1].astype(float)
    raw = -model.decision_function(encoded)
    return np.searchsorted(calibration, raw, side='right') / max(1, len(calibration))


def _save_status(job_id, state, **updates):
    with LOCK:
        state_dict = JOBS[job_id]
        state_dict.update(state=state, **updates)
        snapshot = dict(state_dict)
    if state == 'completed':
        (ARTIFACTS / f'{job_id}.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))


def status(job_id):
    with LOCK:
        if job_id in JOBS:
            return dict(JOBS[job_id])
    p = ARTIFACTS / f'{job_id}.json'
    if p.exists():
        return json.loads(p.read_text())
    raise ValueError('Entraînement introuvable sur ce serveur.')


def list_jobs():
    files = sorted(ARTIFACTS.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    return [json.loads(p.read_text()) for p in files[:15] if len(p.stem) == 32]


def train_async(job_id, dataset_id, budget, review_cost):
    try:
        _save_status(job_id, 'running', stage='Préparation des données', progress=4)
        prepared = prepare(dataset_id)
        train, valid, test = prepared['raw']
        xtr, xva, xte = prepared['encoded']
        ytr, yva, yte = [p.is_fraud.to_numpy() for p in prepared['raw']]
        if budget > len(valid) or budget > len(test):
            raise ValueError('Budget supérieur à la taille de la validation ou du test.')
        results = []
        unsupervised_indices = np.flatnonzero(ytr == 0)
        rng = np.random.default_rng(42)
        for i, name in enumerate(MODELS):
            logger.info('TRAINING | %s | start', name)
            _save_status(job_id, 'running', stage=f'Entraînement : {MODELS[name]}', progress=15 + 18 * i)
            strategy = None
            calibration = None
            if name == 'logistic_regression':
                options = {}
                for w in (None, 'balanced'):
                    estimator = LogisticRegression(max_iter=500, class_weight=w, random_state=42)
                    estimator.fit(xtr, ytr)
                    options['pondéré' if w else 'non pondéré'] = (estimator, float(average_precision_score(yva, estimator.predict_proba(xva)[:, 1])))
                strategy = {k: round(v[1], 5) for k, v in options.items()}
                best_weight = max(options, key=lambda k: options[k][1])
                model = options[best_weight][0]
            elif name == 'random_forest':
                model = RandomForestClassifier(n_estimators=85, max_depth=14, min_samples_leaf=3,
                                               class_weight='balanced_subsample', n_jobs=2, random_state=42)
                model.fit(xtr, ytr)
            elif name == 'isolation_forest':
                chosen = rng.choice(unsupervised_indices, min(14000, len(unsupervised_indices)), replace=False)
                model = IsolationForest(n_estimators=110, contamination='auto', n_jobs=2, random_state=42)
                model.fit(xtr[chosen])
                calibration = np.sort(-model.decision_function(xtr[chosen]))
            else:
                chosen = rng.choice(unsupervised_indices, min(2400, len(unsupervised_indices)), replace=False)
                model = LocalOutlierFactor(n_neighbors=28, novelty=True, n_jobs=2)
                model.fit(xtr[chosen])
                calibration = np.sort(-model.decision_function(xtr[chosen]))
            s_valid = positive_scores(model, xva, name, calibration)
            s_test = positive_scores(model, xte, name, calibration)
            threshold, val_cost = optimize_threshold(yva, valid.amt.to_numpy(), s_valid, budget, review_cost)
            val_metrics = score_metrics(yva, s_valid, threshold, budget)
            test_metrics = score_metrics(yte, s_test, threshold, budget)
            val_metrics.pop('pr_curve')
            test_metrics.pop('pr_curve')
            test_cost = money_cost(yte, test.amt.to_numpy(), s_test, threshold, budget, review_cost)
            reference_indices = rng.choice(len(xtr), min(24, len(xtr)), replace=False)
            test_columns = prepared['columns']
            raw_test = test[test_columns].copy().reset_index(drop=True)
            artifact = {
                'name': name, 'model': model, 'transformer': prepared['transformer'],
                'feature_columns': test_columns, 'encoded_names': prepared['transformer'].get_feature_names_out().tolist(),
                'numeric': prepared['numeric'], 'categorical': prepared['categorical'],
                'calibration': calibration, 'background': xtr[reference_indices],
                'x_test': xte, 'raw_test': raw_test, 'y_test': yte, 'amounts_test': test.amt.to_numpy(),
                's_test': s_test, 'threshold': threshold, 'dataset_id': dataset_id,
                'budget': budget, 'review_cost': review_cost,
            }
            joblib.dump(artifact, ARTIFACTS / f'{job_id}_{name}.joblib', compress=3)
            row = {
                'model': name, 'label': MODELS[name], 'threshold': round(float(threshold), 5),
                'val_pr_auc': val_metrics['pr_auc'], 'val_cost': val_cost,
                'test_cost': test_cost['total'], 'validation': val_metrics, 'test': test_metrics,
                'imbalance_comparison': strategy,
            }
            results.append(row)
            _save_status(job_id, 'running', progress=15 + 18 * (i + 1), results=results)
            logger.info('TRAINING | %s | val AP %.4f | test AP %.4f', name, row['val_pr_auc'], test_metrics['pr_auc'])
        winner = min(results, key=lambda r: r['val_cost'])
        _save_status(job_id, 'completed', stage='Terminé', progress=100, results=results,
                     selected_model=winner['model'], split_method=prepared['split_method'],
                     split_sizes=[len(x) for x in prepared['raw']],
                     fraud_counts=[int(x.is_fraud.sum()) for x in prepared['raw']],
                     features=prepared['columns'])
    except Exception as exc:
        logger.exception('TRAINING ERROR | job=%s', job_id)
        _save_status(job_id, 'failed', stage='Échec', error=str(exc))


def start_training(dataset_id, budget, review_cost, background_tasks):
    if review_cost < 0 or budget < 1:
        raise ValueError('Budget ou coût d’analyse invalide.')
    with LOCK:
        if any(j['state'] in ('queued', 'running') for j in JOBS.values()):
            raise ValueError('Un entraînement est déjà en cours.')
        job_id = uuid.uuid4().hex
        JOBS[job_id] = dict(job_id=job_id, dataset_id=dataset_id, budget=budget, review_cost=review_cost,
                            state='queued', stage='En attente', progress=0, results=[], error=None)
    background_tasks.add_task(train_async, job_id, dataset_id, budget, review_cost)
    return dict(JOBS[job_id])


def load_model(job_id, model):
    if not job_id.isalnum() or model not in MODELS:
        raise ValueError('Identifiant ou modèle invalide.')
    path = ARTIFACTS / f'{job_id}_{model}.joblib'
    if not path.exists():
        raise ValueError('Modèle sauvegardé introuvable sur ce serveur.')
    return joblib.load(path)


def evaluation(job_id, model, threshold, budget, review_cost):
    artifact = load_model(job_id, model)
    y, amounts, scores = artifact['y_test'], artifact['amounts_test'], artifact['s_test']
    threshold = artifact['threshold'] if threshold is None else float(threshold)
    budget = artifact['budget'] if budget is None else int(budget)
    review_cost = artifact['review_cost'] if review_cost is None else float(review_cost)
    if budget < 1 or review_cost < 0:
        raise ValueError('Budget / coût invalide.')
    m = score_metrics(y, scores, threshold, budget)
    cost = money_cost(y, amounts, scores, threshold, budget, review_cost)
    idx = np.asarray(cost['flagged_indices'])
    idx = idx[np.argsort(-scores[idx])][:120]
    top = [{'observation': int(i), 'score': round(float(scores[i]), 5), 'amount': round(float(amounts[i]), 2),
            'true_label': int(y[i])} for i in idx]
    return {'job_id': job_id, 'model': model, 'threshold': threshold, 'budget': budget,
            'review_cost': review_cost, 'metrics': m, 'cost': {k: v for k, v in cost.items() if k != 'flagged_indices'},
            'alerts': top, 'test_size': len(y)}
