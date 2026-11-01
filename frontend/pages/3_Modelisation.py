import pandas as pd
import streamlit as st
from api_client import get, post
from styles import header

header('MODULE 03 / MODÉLISATION', 'Entraînement des détecteurs',
       'Comparer des classifieurs supervisés à Isolation Forest et LOF, puis fixer le seuil sur la validation.')
with st.container(border=True):
    st.markdown('#### Paramètres métier')
    a, b = st.columns(2)
    budget = a.number_input('Budget maximal d’alertes sur la validation', 1, 10000, 60)
    review_cost = b.number_input('Coût d’analyse par alerte (unités monétaires)', 0.0, 10000.0, 15.0)
    st.caption('Le budget représente le nombre de dossiers que les analystes peuvent examiner sur la période de validation. La valeur 60 est modifiable.')
    if st.button('Entraîner les 4 modèles', type='primary', use_container_width=True):
        try:
            result = post('/train/start', {'dataset_id': st.session_state.dataset_id,
                                           'budget': budget, 'review_cost': review_cost})
            st.session_state.training_job = result['job_id']
        except Exception as exc:
            st.error(str(exc))

job_id = st.session_state.get('training_job')
if not job_id:
    st.info('Lancez l’entraînement pour comparer les modèles. Les logs sont visibles dans le terminal FastAPI.')
    st.stop()

@st.fragment(run_every='2s')
def follow():
    try:
        job = get(f'/train/status/{job_id}')
    except Exception as exc:
        st.error(str(exc))
        return
    st.progress(job['progress'], text=job['stage'])
    if job['results']:
        rows = [{'Modèle': r['label'], 'PR-AUC validation': r['val_pr_auc'],
                 'Coût validation': r['val_cost'], 'PR-AUC test': r['test']['pr_auc']} for r in job['results']]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        for r in job['results']:
            if r.get('imbalance_comparison'):
                st.caption('Régression logistique — comparaison pondérée / non pondérée (PR-AUC validation) : '
                           + str(r['imbalance_comparison']))
    if job['state'] == 'completed':
        st.success('Entraînement terminé. Seuils et modèles enregistrés côté backend.')
        st.write('Modèle retenu sur le coût de validation : **' + job['selected_model'] + '**')
        st.caption('Décision prise exclusivement sur la validation ; le test sert à la confirmation.')
    if job['state'] == 'failed':
        st.error(job.get('error', 'Échec : vérifier les logs FastAPI.'))

follow()
