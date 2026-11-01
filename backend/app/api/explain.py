import logging
from fastapi import APIRouter, HTTPException
from app.schemas import ExplainRequest, PermutationRequest
from app.services.xai import explain, permutation

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/explain', tags=['Explicabilité'])


@router.post('/shap')
def shap_explanation(body: ExplainRequest):
    try:
        return explain(body.job_id, body.model, body.observation, body.global_count, body.background_size)
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    except Exception:
        logger.exception('SHAP ERROR')
        raise HTTPException(500, detail='Calcul SHAP impossible, voir logs.')


@router.post('/permutation')
def permutation_importances(body: PermutationRequest):
    try:
        return permutation(body.job_id, body.model, body.sample_size, body.repeats)
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    except Exception:
        logger.exception('PERMUTATION ERROR')
        raise HTTPException(500, detail='Calcul impossible, voir logs.')
