import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from api_client import get, post
from styles import header

header('MODULE 04 / ÉVALUATION', 'Évaluation et alertes',
       'Courbes précision-rappel, budget d’analyse et coût financier des fraudes manquées.')

try:
    jobs = get('/train/jobs')['jobs']
except Exception as exc:
    st.error(str(exc))
    st.stop()
if not jobs:
    st.warning('Aucun entraînement disponible sur ce backend. Commencez par la page Modélisation.')
    st.stop()
by_id = {j['job_id']: j for j in jobs}
job_id = st.selectbox('Entraînement', list(by_id), format_func=lambda j: f"Exécution {j[:8]} · {by_id[j]['dataset_id']} · {len(by_id[j]['results'])} modèles")
job = by_id[job_id]
labels = {r['model']: r['label'] for r in job['results']}
model = st.selectbox('Détecteur', list(labels), format_func=lambda m: labels[m])
saved = next(r for r in job['results'] if r['model'] == model)
with st.container(border=True):
    st.markdown('#### Choix du seuil et budget')
    st.caption('Seuil retenu sur la validation : ' + str(saved['threshold']))
    c1, c2, c3 = st.columns(3)
    custom_threshold = c1.checkbox('Explorer un autre seuil', value=False)
    displayed_threshold = min(1.0, float(saved['threshold']))
    chosen_slider = c1.slider('Seuil d’alerte', 0.0, 1.0, displayed_threshold, .005, disabled=not custom_threshold)
    threshold = chosen_slider if custom_threshold else float(saved['threshold'])
    budget = c2.slider('Budget d’alertes', 1, max(10, job['split_sizes'][2]), min(job['budget'], job['split_sizes'][2]))
    review_cost = c3.number_input('Coût par revue', min_value=0.0, value=float(job['review_cost']), step=5.0)
    st.caption('Seuil exploré sur le test : ne pas le réoptimiser pour annoncer la performance finale.')
try:
    report = post('/evaluate/report', {'job_id': job_id, 'model': model, 'threshold': threshold,
                                       'budget': budget, 'review_cost': review_cost})
except Exception as exc:
    st.error(str(exc))
    st.stop()
m, cost = report['metrics'], report['cost']
a, b, c, d = st.columns(4)
a.metric('PR-AUC', f"{m['pr_auc']:.3f}")
b.metric(f'Precision@{budget}', f"{m['precision_at_k']:.3f}")
c.metric(f'Recall@{budget}', f"{m['recall_at_k']:.3f}")
d.metric('Coût métier', f"{cost['total']:,.0f}")
e, f, g, h = st.columns(4)
e.metric('F2', f"{m['f2']:.3f}")
f.metric('MCC', f"{m['mcc']:.3f}")
g.metric('ROC-AUC', f"{m['roc_auc']:.3f}")
h.metric('Alertes déclenchées', m['alerts'])

st.markdown('### Comparaison des modèles')
rows = []
for item in job['results']:
    row = item['test']
    rows.append({'Détecteur': item['label'], 'PR-AUC': row['pr_auc'], 'ROC-AUC': row['roc_auc'],
                 'Precision@k': row['precision_at_k'], 'Recall@k': row['recall_at_k'],
                 'F2': row['f2'], 'MCC': row['mcc'], 'Coût test': item['test_cost']})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown('### Courbes précision-rappel (test)')
fig = go.Figure()
for item in job['results']:
    try:
        other = report if item['model'] == model else post('/evaluate/report',
            {'job_id': job_id, 'model': item['model']})
        points = other['metrics']['pr_curve']
        fig.add_trace(go.Scatter(x=[p['recall'] for p in points], y=[p['precision'] for p in points],
                                 mode='lines', name=item['label']))
    except Exception as exc:
        st.warning(str(exc))
fig.update_layout(template='plotly_white', xaxis_title='Rappel', yaxis_title='Précision', height=390)
st.plotly_chart(fig, use_container_width=True)

st.markdown('### Détail des alertes')
st.caption(f"Fraudes non détectées : {m['fn']} · Faux positifs : {m['fp']} · Fraudes manquées : {cost['missed_fraud_amount']:,.2f} · Coût des analyses : {cost['analysis_cost']:,.2f}")
st.dataframe(pd.DataFrame(report['alerts']), hide_index=True, use_container_width=True)
st.caption('Les montants sont exprimés dans l’unité du jeu de données. Les coûts opérationnels sont des hypothèses réglables.')
