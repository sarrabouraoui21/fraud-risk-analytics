import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException
from app.schemas import StartRequest
from app.services.modeling import start_training, status, list_jobs

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/train', tags=['Entraînement'])


@router.post('/start')
def start(body: StartRequest, tasks: BackgroundTasks):
    try:
        return start_training(body.dataset_id, body.budget, body.review_cost, tasks)
    except ValueError as exc:
        raise HTTPException(409, detail=str(exc)) from exc


@router.get('/status/{job_id}')
def get_status(job_id: str):
    try:
        return status(job_id)
    except ValueError as exc:
        raise HTTPException(404, detail=str(exc)) from exc


@router.get('/jobs')
def jobs():
    return {'jobs': list_jobs()}
