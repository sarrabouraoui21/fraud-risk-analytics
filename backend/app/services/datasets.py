import logging
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
UPLOADS = ROOT / 'data' / 'uploads'
UPLOADS.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)


def demo_dataset(n=14000):
    rng = np.random.default_rng(21)
    cards = rng.integers(100000, 100720, n)
    start = pd.Timestamp('2025-01-01 08:00:00')
    times = start + pd.to_timedelta(np.sort(rng.integers(0, 55 * 86400, n)), unit='s')
    categories = rng.choice(['grocery_pos', 'shopping_net', 'gas_transport', 'food_dining', 'entertainment'], n)
    amounts = rng.lognormal(3.35, 1.03, n).round(2)
    fraud = rng.binomial(1, 0.011, n).astype(int)
    amounts[fraud == 1] *= rng.uniform(2.2, 8, size=fraud.sum())
    return pd.DataFrame({
        'trans_date_trans_time': times,
        'cc_num': cards.astype(str),
        'category': categories,
        'amt': amounts.round(2),
        'city_pop': rng.integers(1000, 700000, n),
        'lat': rng.uniform(31, 47, n),
        'long': rng.uniform(-117, -75, n),
        'merch_lat': rng.uniform(31, 47, n),
        'merch_long': rng.uniform(-117, -75, n),
        'is_fraud': fraud,
    })


def normalize(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if 'amt' not in df or 'is_fraud' not in df:
        raise ValueError('Colonnes obligatoires : amt, is_fraud. Les autres colonnes suivent le format Sparkov.')
    df['amt'] = pd.to_numeric(df['amt'], errors='coerce')
    if df['amt'].isna().any() or (df['amt'] < 0).any():
        raise ValueError('La colonne amt doit contenir des montants positifs ou nuls.')
    target = pd.to_numeric(df['is_fraud'], errors='coerce')
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError('is_fraud doit uniquement contenir 0 et 1.')
    df['is_fraud'] = target.astype(int)
    if 'trans_date_trans_time' in df:
        df['timestamp'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
        if df['timestamp'].isna().any():
            raise ValueError('Certaines dates ne sont pas valides.')
        df = df.sort_values('timestamp', kind='stable').reset_index(drop=True)
    elif 'timestamp' in df:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        if df['timestamp'].isna().any():
            raise ValueError('Certaines dates ne sont pas valides.')
        df = df.sort_values('timestamp', kind='stable').reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    if len(df) < 300 or df['is_fraud'].sum() < 9:
        raise ValueError('Jeu insuffisant : au moins 300 transactions dont 9 fraudes sont nécessaires.')
    logger.info('DATA READY | rows=%s | frauds=%s', len(df), int(df.is_fraud.sum()))
    return df


def kaggle_dataset(max_rows=90000):
    import kagglehub
    slug = 'kartik2112/fraud-detection'
    logger.info('KAGGLE DOWNLOAD | %s', slug)
    path = Path(kagglehub.dataset_download(slug))
    files = list(path.rglob('fraudTrain.csv'))
    if not files:
        raise ValueError('fraudTrain.csv introuvable dans le dataset téléchargé.')
    chunks = []
    total = 0
    for chunk in pd.read_csv(files[0], chunksize=30000, low_memory=False):
        chunks.append(chunk)
        total += len(chunk)
        while total > max_rows + 30000 and len(chunks) > 1:
            total -= len(chunks.pop(0))
    df = pd.concat(chunks, ignore_index=True).tail(max_rows)
    return normalize(df)


_cache = {}


def dataset(dataset_id='demo'):
    if dataset_id in _cache:
        return _cache[dataset_id]
    if dataset_id == 'demo':
        df = normalize(demo_dataset())
    elif dataset_id == 'kaggle':
        df = kaggle_dataset(int(os.getenv('KAGGLE_MAX_ROWS', '90000')))
    else:
        path = UPLOADS / f'{dataset_id}.csv'
        if not path.exists() or not dataset_id.isalnum():
            raise ValueError('Dataset importé introuvable sur le serveur.')
        df = normalize(pd.read_csv(path))
    _cache[dataset_id] = df
    return df


def upload_dataset(content):
    if len(content) > 22 * 1024 * 1024:
        raise ValueError('CSV limité à 22 Mo.')
    import io
    try:
        df = normalize(pd.read_csv(io.BytesIO(content)))
    except Exception as exc:
        raise ValueError(f'CSV invalide : {exc}') from exc
    dataset_id = uuid.uuid4().hex
    df.to_csv(UPLOADS / f'{dataset_id}.csv', index=False)
    _cache[dataset_id] = df
    return dataset_id, overview(df)


def overview(df):
    by_class = df.groupby('is_fraud')['amt'].agg(['count', 'median', 'mean']).round(2)
    return {
        'rows': int(len(df)),
        'columns': int(len(df.columns)),
        'frauds': int(df.is_fraud.sum()),
        'fraud_rate': round(float(df.is_fraud.mean()), 6),
        'amount_total': round(float(df.amt.sum()), 2),
        'amount_stats': [{'class': int(k), **{c: float(v) for c, v in row.items()}} for k, row in by_class.to_dict('index').items()],
        'has_time': 'timestamp' in df,
        'has_card': 'cc_num' in df,
    }


def exploration(df):
    result = {'overview': overview(df), 'sample': df.head(12).astype(str).to_dict('records')}
    if 'timestamp' in df:
        t = df.groupby(df['timestamp'].dt.to_period('D'))['is_fraud'].agg(['sum', 'count'])
        result['daily'] = [{'date': str(k), 'rate': round(float(v['sum'] / v['count']), 5), 'frauds': int(v['sum'])} for k, v in t.iterrows()]
        h = df.groupby(df['timestamp'].dt.hour)['is_fraud'].agg(['sum', 'count'])
        result['hourly'] = [{'hour': int(k), 'rate': round(float(v['sum'] / v['count']), 5)} for k, v in h.iterrows()]
    if 'category' in df:
        x = df.groupby('category')['is_fraud'].agg(['sum', 'count']).sort_values('count', ascending=False).head(16)
        result['categories'] = [{'category': str(k), 'rate': round(float(v['sum'] / v['count']), 5), 'transactions': int(v['count'])} for k, v in x.iterrows()]
    return result
