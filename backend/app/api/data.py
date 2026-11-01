import logging
from fastapi import APIRouter, File, HTTPException, UploadFile
from app.services.datasets import dataset, overview, exploration, upload_dataset
from app.services.modeling import prepare

router = APIRouter(prefix='/data', tags=['Données'])
logger = logging.getLogger(__name__)


@router.get('/summary')
def summary(dataset_id: str = 'demo'):
    try:
        return exploration(dataset(dataset_id))
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    except Exception:
        logger.exception('DATA ERROR')
        raise HTTPException(500, detail='Chargement impossible. Vérifier les logs FastAPI.')


@router.post('/upload')
async def upload(file: UploadFile = File(...)):
    try:
        dataset_id, stats = upload_dataset(await file.read())
        return {'dataset_id': dataset_id, 'overview': stats}
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc


@router.get('/preparation')
def preparation(dataset_id: str = 'demo'):
    try:
        p = prepare(dataset_id)
        train, valid, test = p['raw']
        return {'method': p['split_method'], 'numeric': p['numeric'], 'categorical': p['categorical'],
                'parts': [{'label': n, 'rows': len(d), 'frauds': int(d.is_fraud.sum()),
                           'rate': round(float(d.is_fraud.mean()), 5)} for n, d in zip(['Train', 'Validation', 'Test'], [train, valid, test])],
                'examples': p['features'][p['columns']].head(8).replace({float('inf'): None}).fillna('—').to_dict('records'),
                'warning': 'Les historiques de carte utilisent uniquement les transactions passées. L’encodage est ajusté sur train.'}
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    except Exception:
        logger.exception('PREPARATION ERROR')
        raise HTTPException(500, detail='Préparation impossible. Vérifier les logs.')
