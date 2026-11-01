import logging
from fastapi import APIRouter, HTTPException
from app.schemas import EvaluationRequest, PredictRequest
from app.services.modeling import evaluation, load_model, positive_scores

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/evaluate', tags=['Évaluation et API de scoring'])


@router.post('/report')
def report(body: EvaluationRequest):
    try:
        return evaluation(body.job_id, body.model, body.threshold, body.budget, body.review_cost)
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    except Exception:
        logger.exception('EVALUATION ERROR')
        raise HTTPException(500, detail='Évaluation impossible, voir logs.')


@router.post('/predict')
def predict(body: PredictRequest):
    import pandas as pd
    try:
        a = load_model(body.job_id, body.model)
        missing = [c for c in a['feature_columns'] if c not in body.features]
        if missing:
            raise ValueError('Variables calculées manquantes : ' + ', '.join(missing))
        x = pd.DataFrame([{c: body.features[c] for c in a['feature_columns']}])
        encoded = a['transformer'].transform(x)
        value = float(positive_scores(a['model'], encoded, body.model, a['calibration'])[0])
        return {'risk_score': round(value, 6), 'threshold': float(a['threshold']), 'alert': value >= a['threshold'],
                'note': 'Cette API attend des variables déjà enrichies (historique de transactions calculé).'}
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
