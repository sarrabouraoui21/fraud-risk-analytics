import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.services.datasets import demo_dataset, normalize
from app.services.features import build_features, split_data
from app.services.metrics import money_cost, score_metrics, optimize_threshold


def test_demo_and_chronological_split():
    df = normalize(demo_dataset(2500))
    features, _, _ = build_features(df)
    parts, method = split_data(features)
    assert method == 'chronologique'
    assert [len(p) for p in parts] == [1500, 500, 500]
    assert parts[0].timestamp.max() <= parts[1].timestamp.min()


def test_history_uses_previous_transactions_only():
    df = normalize(demo_dataset(2500))
    engineered, _, _ = build_features(df)
    assert engineered['tx_24h'].min() >= 0
    assert engineered['ratio_amount_30d'].min() > 0
    first = engineered.drop_duplicates('cc_num', keep='first')
    assert first['tx_1h'].eq(0).all()
    assert first['ratio_amount_30d'].eq(1).all()


def test_cost_is_missed_amount_plus_review():
    y = np.array([1, 0, 1, 0])
    amt = np.array([100., 20., 200., 5.])
    scores = np.array([.9, .8, .2, .1])
    result = money_cost(y, amt, scores, .5, 2, 10)
    assert result['total'] == 220
    assert result['missed_fraud_amount'] == 200
    assert result['analysis_cost'] == 20
    threshold, cost = optimize_threshold(y, amt, scores, 2, 10)
    assert 0 <= cost <= 320
    m = score_metrics(y, scores, threshold, 2)
    assert 'pr_auc' in m and 'mcc' in m


def test_api_health_and_data():
    client = TestClient(app)
    assert client.get('/health').json()['status'] == 'ok'
    info = client.get('/data/summary', params={'dataset_id': 'demo'})
    assert info.status_code == 200
    assert info.json()['overview']['frauds'] > 0
