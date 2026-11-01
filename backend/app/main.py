import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import data, training, evaluation, explain

logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO').upper(),
                    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s', force=True)
app = FastAPI(title='Fraud Risk Analytics API', version='1.0')
for router in [data.router, training.router, evaluation.router, explain.router]:
    app.include_router(router)


@app.get('/health')
def health():
    return {'status': 'ok', 'service': 'fraud-risk-api'}
