import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from api_client import get, post
from styles import header

header('MODULE 05 / EXPLAINABLE AI', 'Comprendre les alertes',
       'Importance globale et contributions SHAP pour une transaction précise, avec motifs d’alerte.')
try:
    jobs = get('/train/jobs')['jobs']
except Exception as exc:
    st.error(str(exc))
    st.stop()
if not jobs:
    st.warning('Entraînez d’abord les modèles.')
    st.stop()
by_id = {j['job_id']: j for j in jobs}
job_id = st.selectbox('Entraînement', list(by_id), format_func=lambda j: 'Exécution ' + j[:10])
models = {r['model']: r['label'] for r in by_id[job_id]['results']}
model = st.selectbox('Modèle', list(models), format_func=lambda m: models[m])
with st.container(border=True):
    a, b, c = st.columns(3)
    row_id = a.number_input('Observation (jeu de test)', min_value=0, max_value=by_id[job_id]['split_sizes'][2]-1, value=0)
    bg = b.slider('Références SHAP', 5, 25, 12)
    n = c.slider('Exemples globaux', 1, 18, 8)
    if st.button('Calculer SHAP', type='primary'):
        with st.spinner('Explications en cours...'):
            try:
                st.session_state['xai'] = post('/explain/shap', {'job_id': job_id, 'model': model,
                    'observation': int(row_id), 'background_size': bg, 'global_count': n})
            except Exception as exc:
                st.error(str(exc))
                st.session_state.pop('xai', None)

result = st.session_state.get('xai')
if result and result['job_id'] == job_id and result['model'] == model and result['observation'] == row_id:
    a, b, c = st.columns(3)
    a.metric('Score de risque', f"{result['score']:.3f}")
    b.metric('Seuil validé', f"{result['threshold']:.3f}")
    c.metric('Classe réelle', 'Fraude' if result['true_label'] else 'Légitime')
    st.caption(result['scope'])
    tab1, tab2, tab3 = st.tabs(['Importance globale', 'SHAP waterfall', 'Codes motifs'])
    with tab1:
        t = pd.DataFrame(result['global'][:16]).sort_values('importance')
        fig = px.bar(t, x='importance', y='feature', orientation='h', color_discrete_sequence=['#16665C'])
        fig.update_layout(template='plotly_white', xaxis_title='Contribution SHAP absolue moyenne', yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
        points = []
        for item in result['global'][:12]:
            for x in item['contributions']:
                points.append({'Variable': item['feature'], 'Contribution SHAP': x})
        st.plotly_chart(px.strip(pd.DataFrame(points), x='Contribution SHAP', y='Variable', color_discrete_sequence=['#C46C54']), use_container_width=True)
    with tab2:
        local = result['local']
        chosen = local[:10]
        rest = sum(row['contribution'] for row in local[10:])
        labels = ['Référence'] + [r['feature'] for r in chosen]
        values = [result['base_value']] + [r['contribution'] for r in chosen]
        if len(local) > 10:
            labels.append('Autres')
            values.append(rest)
        fig = go.Figure(go.Waterfall(orientation='v', x=labels, y=values,
                                     measure=['absolute'] + ['relative'] * (len(values)-1),
                                     connector={'line': {'color':'#A5B4AC'}},
                                     increasing={'marker': {'color':'#C46C54'}},
                                     decreasing={'marker': {'color':'#16665C'}}))
        fig.update_layout(template='plotly_white', height=450, xaxis_title=None, yaxis_title='Contribution au score')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.DataFrame(local), use_container_width=True, hide_index=True)
    with tab3:
        st.markdown('#### Variables qui ont poussé le score à la hausse')
        if result['reasons']:
            for index, reason in enumerate(result['reasons'], 1):
                st.write(f"{index}. **{reason['feature']}** ({reason['value']}) · contribution : {reason['contribution']:+.4f}")
        st.caption('Les codes motifs décrivent le score du modèle, pas une cause prouvée de fraude.')
    st.info(result['note'])
else:
    st.info('Lancez SHAP pour afficher les contributions et les motifs.')

with st.expander('Importance globale par permutation (PR-AUC)'):
    if st.button('Calculer la permutation'):
        try:
            p = post('/explain/permutation', {'job_id': job_id, 'model': model, 'sample_size': 200, 'repeats': 3})
            st.session_state['permutation'] = dict(model=model, job_id=job_id, result=p)
        except Exception as exc:
            st.error(str(exc))
    cached = st.session_state.get('permutation')
    if cached and cached['model'] == model and cached['job_id'] == job_id:
        st.caption('PR-AUC de référence : ' + str(cached['result']['baseline_pr_auc']))
        t = pd.DataFrame(cached['result']['features'][:15]).sort_values('importance')
        st.plotly_chart(px.bar(t, x='importance', y='feature', orientation='h', color_discrete_sequence=['#16665C']), use_container_width=True)
