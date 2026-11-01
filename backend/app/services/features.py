from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


NUMERIC = ['amt', 'city_pop', 'distance_km', 'hour_sin', 'hour_cos', 'day_of_week',
           'tx_1h', 'tx_24h', 'ratio_amount_30d', 'seconds_since_last']
CATEGORICAL = ['category']


def build_features(df):
    data = df.copy()
    has_time = 'timestamp' in data
    has_card = 'cc_num' in data and has_time
    if has_time:
        hour = data.timestamp.dt.hour + data.timestamp.dt.minute / 60
        data['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        data['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        data['day_of_week'] = data.timestamp.dt.dayofweek
    if all(k in data for k in ['lat', 'long', 'merch_lat', 'merch_long']):
        a, b = np.radians(pd.to_numeric(data.lat)), np.radians(pd.to_numeric(data.merch_lat))
        x = np.radians(pd.to_numeric(data.merch_lat) - pd.to_numeric(data.lat))
        y = np.radians(pd.to_numeric(data.merch_long) - pd.to_numeric(data.long))
        data['distance_km'] = 6371 * 2 * np.arcsin(np.minimum(1, np.sqrt(np.sin(x / 2)**2 + np.cos(a) * np.cos(b) * np.sin(y / 2)**2)))
    if has_card:
        per_card = defaultdict(lambda: (deque(), deque(), deque()))
        one_hour, day, ratio, elapsed = [], [], [], []
        for card, ts, amount in zip(data.cc_num.astype(str), data.timestamp, data.amt):
            recent_hour, recent_day, month = per_card[card]
            now = ts.timestamp()
            while recent_hour and recent_hour[0] < now - 3600:
                recent_hour.popleft()
            while recent_day and recent_day[0] < now - 86400:
                recent_day.popleft()
            while month and month[0][0] < now - 30 * 86400:
                month.popleft()
            one_hour.append(len(recent_hour))
            day.append(len(recent_day))
            ratio.append(float(amount / max(np.mean([v for _, v in month]), 1)) if month else 1.0)
            elapsed.append(now - recent_day[-1] if recent_day else (now - month[-1][0] if month else np.nan))
            recent_hour.append(now)
            recent_day.append(now)
            month.append((now, float(amount)))
        data['tx_1h'] = one_hour
        data['tx_24h'] = day
        data['ratio_amount_30d'] = np.minimum(ratio, 1000)
        data['seconds_since_last'] = elapsed
    numeric = [c for c in NUMERIC if c in data]
    categorical = [c for c in CATEGORICAL if c in data]
    if not categorical:
        data['category'] = 'non renseigné'
        categorical = ['category']
    return data, numeric, categorical


def split_data(df):
    n = len(df)
    if 'timestamp' in df:
        train_end = int(n * .6)
        valid_end = int(n * .8)
        groups = [df.iloc[:train_end], df.iloc[train_end:valid_end], df.iloc[valid_end:]]
        method = 'chronologique'
    else:
        train, rest = train_test_split(df, train_size=.6, stratify=df.is_fraud, random_state=42)
        valid, test = train_test_split(rest, test_size=.5, stratify=rest.is_fraud, random_state=42)
        groups = [train, valid, test]
        method = 'aléatoire stratifié : aucune colonne temporelle'
    if any(g['is_fraud'].nunique() != 2 for g in groups):
        raise ValueError('Une partition ne contient pas les deux classes. Choisissez un échantillon plus grand.')
    return groups, method


def make_transformer(numeric, categorical):
    steps = []
    if numeric:
        steps.append(('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), numeric))
    if categorical:
        steps.append(('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical))
    return ColumnTransformer(steps, remainder='drop', verbose_feature_names_out=False)
