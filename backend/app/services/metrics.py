import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             fbeta_score, matthews_corrcoef, precision_score, recall_score)


def score_metrics(y, scores, threshold, budget):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    k = int(min(max(1, budget), len(y)))
    order = np.argsort(-scores, kind='stable')
    top = np.zeros(len(y), dtype=bool)
    top[order[:k]] = True
    predicted = scores >= float(threshold)
    predicted &= top
    pr, rec, _ = precision_recall_curve(y, scores)
    return {
        'pr_auc': round(float(average_precision_score(y, scores)), 5),
        'roc_auc': round(float(roc_auc_score(y, scores)), 5),
        'precision_at_k': round(float(y[top].mean()), 5),
        'recall_at_k': round(float(y[top].sum() / y.sum()), 5) if y.sum() else 0,
        'precision': round(float(precision_score(y, predicted, zero_division=0)), 5),
        'recall': round(float(recall_score(y, predicted, zero_division=0)), 5),
        'f2': round(float(fbeta_score(y, predicted, beta=2, zero_division=0)), 5),
        'mcc': round(float(matthews_corrcoef(y, predicted)), 5),
        'alerts': int(predicted.sum()),
        'tp': int(np.sum(predicted & (y == 1))),
        'fp': int(np.sum(predicted & (y == 0))),
        'fn': int(np.sum(~predicted & (y == 1))),
        'tn': int(np.sum(~predicted & (y == 0))),
        'pr_curve': [{'recall': round(float(r), 5), 'precision': round(float(p), 5)} for r, p in zip(rec[::max(1, len(rec)//160)], pr[::max(1, len(pr)//160)])],
    }


def money_cost(y, amounts, scores, threshold, budget, review_cost):
    n = len(y)
    k = min(int(budget), n)
    order = np.argsort(-np.asarray(scores), kind='stable')
    eligible = np.zeros(n, dtype=bool)
    eligible[order[:k]] = True
    flagged = (np.asarray(scores) >= threshold) & eligible
    missed = ((np.asarray(y) == 1) & ~flagged)
    return {
        'total': round(float(np.asarray(amounts)[missed].sum() + review_cost * flagged.sum()), 2),
        'missed_fraud_amount': round(float(np.asarray(amounts)[missed].sum()), 2),
        'analysis_cost': round(float(review_cost * flagged.sum()), 2),
        'alerts': int(flagged.sum()),
        'flagged_indices': np.flatnonzero(flagged).tolist(),
    }


def optimize_threshold(y, amounts, scores, budget, review_cost):
    values = np.asarray(scores, dtype=float)
    ordered = np.sort(np.unique(values))
    # Validation only, including the option of no alerts.
    candidates = np.r_[np.nextafter(ordered[-1], np.inf), ordered]
    if len(candidates) > 900:
        candidates = np.unique(np.r_[candidates[0], np.quantile(values, np.linspace(0, 1, 900))])
    best = None
    for threshold in candidates:
        result = money_cost(y, amounts, values, float(threshold), budget, review_cost)
        if best is None or result['total'] < best[1]:
            best = (float(threshold), result['total'])
    return best[0], best[1]
