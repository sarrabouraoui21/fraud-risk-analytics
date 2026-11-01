import logging
import numpy as np
from sklearn.inspection import permutation_importance
from app.services.modeling import load_model, positive_scores

logger = logging.getLogger(__name__)


def get_score_function(artifact):
    model = artifact['model']
    name = artifact['name']
    calibration = artifact['calibration']
    def score(values):
        return positive_scores(model, np.asarray(values, dtype=float), name, calibration)
    return score


def group_names(artifact):
    raw_names = artifact['feature_columns']
    transformed = artifact['encoded_names']
    mapping = {n: [] for n in raw_names}
    for i, encoded in enumerate(transformed):
        matches = [name for name in raw_names if encoded == name or encoded.startswith(name + '_')]
        if matches:
            mapping[max(matches, key=len)].append(i)
    return mapping


def explain(job_id, model, observation=0, global_count=8, background_size=12):
    import shap
    artifact = load_model(job_id, model)
    test = artifact['x_test']
    if observation < 0 or observation >= len(test):
        raise ValueError('Indice d’observation hors limites.')
    if global_count < 1 or global_count > 18 or background_size < 5 or background_size > 25:
        raise ValueError('Tailles SHAP hors limites.')
    rng = np.random.default_rng(42)
    bg = artifact['background']
    bg = bg[rng.choice(len(bg), min(background_size, len(bg)), replace=False)]
    score = get_score_function(artifact)
    explainer = shap.Explainer(score, shap.maskers.Independent(bg), algorithm='permutation')
    local = explainer(test[observation:observation + 1], max_evals=2 * test.shape[1] + 1, silent=True)
    indices = np.linspace(0, len(test)-1, min(global_count, len(test)), dtype=int)
    global_explanation = explainer(test[indices], max_evals=2 * test.shape[1] + 1, silent=True)
    mapping = group_names(artifact)
    values = np.asarray(local.values)[0]
    all_values = np.asarray(global_explanation.values)
    original = artifact['raw_test'].iloc[observation]
    contributions = []
    global_features = []
    for name, cols in mapping.items():
        if not cols:
            continue
        contribution = float(values[cols].sum())
        contributions.append({'feature': name, 'value': str(original[name]), 'contribution': round(contribution, 6)})
        global_features.append({'feature': name,
                                'importance': round(float(np.mean(np.abs(all_values[:, cols].sum(axis=1)))), 6),
                                'contributions': np.round(all_values[:, cols].sum(axis=1), 6).tolist()})
    contributions.sort(key=lambda row: abs(row['contribution']), reverse=True)
    global_features.sort(key=lambda row: row['importance'], reverse=True)
    positive = [r for r in contributions if r['contribution'] > 0]
    reasons = positive[:5] if positive else contributions[:5]
    logger.info('SHAP | model=%s | row=%d | base=%.4f', model, observation, float(local.base_values[0]))
    return {'job_id': job_id, 'model': model, 'observation': observation, 'score': round(float(artifact['s_test'][observation]), 6),
            'threshold': round(float(artifact['threshold']), 6), 'true_label': int(artifact['y_test'][observation]),
            'base_value': round(float(local.base_values[0]), 6), 'local': contributions,
            'global': global_features, 'reasons': reasons,
            'scope': 'Score de fraude : probabilité pour les modèles supervisés, rang anomalie pour les non supervisés.',
            'note': 'Les contributions SHAP reflètent le comportement du modèle, sans établir de causalité.'}


def permutation(job_id, model, sample_size=400, repeats=3):
    artifact = load_model(job_id, model)
    x = artifact['raw_test']
    y = artifact['y_test']
    count = min(max(30, int(sample_size)), len(x), 800)
    indices = np.linspace(0, len(x)-1, count, dtype=int)
    transformer = artifact['transformer']
    scorer = get_score_function(artifact)
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(21)
    baseline = average_precision_score(y[indices], scorer(transformer.transform(x.iloc[indices])))
    rows = []
    for col in x.columns:
        changes = []
        for _ in range(repeats):
            xx = x.iloc[indices].copy()
            xx[col] = rng.permutation(xx[col].to_numpy())
            sc = average_precision_score(y[indices], scorer(transformer.transform(xx)))
            changes.append(baseline - sc)
        rows.append({'feature': col, 'importance': round(float(np.mean(changes)), 6), 'std': round(float(np.std(changes)), 6)})
    rows.sort(key=lambda r: r['importance'], reverse=True)
    return {'baseline_pr_auc': round(float(baseline), 5), 'features': rows, 'sample_size': count}
