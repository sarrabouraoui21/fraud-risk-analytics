import os
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / '.env')
BASE = os.getenv('API_BASE_URL', 'http://127.0.0.1:8000').rstrip('/')


def check(response):
    if not response.ok:
        try:
            msg = response.json().get('detail', response.text)
        except Exception:
            msg = response.text
        raise RuntimeError(f'API {response.status_code} : {msg}')
    return response.json()


def get(path, params=None):
    return check(requests.get(f'{BASE}/{path.lstrip("/")}', params=params, timeout=240))


def post(path, data):
    return check(requests.post(f'{BASE}/{path.lstrip("/")}', json=data, timeout=240))


def upload(file):
    return check(requests.post(f'{BASE}/data/upload', files={'file': (file.name, file.getvalue(), 'text/csv')}, timeout=240))
