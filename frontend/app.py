import streamlit as st
from api_client import get
from styles import apply_styles, header

st.set_page_config(page_title='Fraud Risk Analytics', page_icon='📊', layout='wide', initial_sidebar_state='expanded')
apply_styles()
st.session_state.setdefault('dataset_id', 'demo')


def home():
    header('WORKSPACE / DATA MINING', 'Détection de fraude bancaire',
           'Explorer les transactions, comparer quatre méthodes de détection et analyser le coût des alertes.')
    try:
        d = get('/data/summary', {'dataset_id': st.session_state.dataset_id})['overview']
    except Exception as exc:
        st.error(str(exc))
        return
    a, b, c, dcol = st.columns(4)
    a.metric('Transactions', f"{d['rows']:,}".replace(',', ' '))
    b.metric('Fraudes', d['frauds'])
    c.metric('Taux de fraude', f"{100 * d['fraud_rate']:.2f} %")
    dcol.metric('Montant total', f"{d['amount_total']:,.0f}".replace(',', ' '))
    st.write('')
    st.markdown('### Parcours de l’analyse')
    items = [
        ('01 · Données', 'Taux de fraude et exploration', data_page),
        ('02 · Préparation', 'Variables comportementales et split', prep_page),
        ('03 · Modélisation', 'Supervisé et détection d’anomalies', train_page),
        ('04 · Évaluation', 'Budget, PR-AUC et coût métier', eval_page),
        ('05 · Explicabilité', 'SHAP et motifs des alertes', xai_page),
    ]
    for start in [0, 2, 4]:
        cols = st.columns(2)
        for col, (name, desc, page) in zip(cols, items[start:start+2]):
            with col:
                with st.container(border=True):
                    st.markdown('#### ' + name)
                    st.caption(desc)
                    if st.button('Ouvrir', key=name, use_container_width=True):
                        st.switch_page(page)
    st.caption('Source par défaut : données simulées pour tester l’interface. Choisir Kaggle pour le vrai projet Sparkov.')


home_page = st.Page(home, title='Vue d’ensemble', icon=':material/dashboard:', default=True)
data_page = st.Page('pages/1_Donnees.py', title='Exploration des données', icon=':material/analytics:')
prep_page = st.Page('pages/2_Preparation.py', title='Préparation', icon=':material/tune:')
train_page = st.Page('pages/3_Modelisation.py', title='Modélisation', icon=':material/model_training:')
eval_page = st.Page('pages/4_Evaluation.py', title='Évaluation', icon=':material/assessment:')
xai_page = st.Page('pages/5_Explicabilite.py', title='Explicabilité', icon=':material/psychology:')

with st.sidebar:
    st.markdown('''<div class="brand"><div class="brand-mark">F</div><div>
    <div class="brand-name">Fraud Risk Analytics</div><div class="brand-subtitle">DATA MINING / FRAUD</div>
    </div></div>''', unsafe_allow_html=True)

page = st.navigation({'ANALYSE': [home_page, data_page, prep_page, train_page, eval_page, xai_page]})
with st.sidebar:
    st.divider()
    try:
        alive = get('/health').get('status') == 'ok'
        if alive:
            st.markdown('<div class="status-row"><span class="status-dot"></span>API connectée · FastAPI</div>', unsafe_allow_html=True)
        else:
            st.warning('API indisponible')
    except Exception:
        st.error('API FastAPI indisponible')
page.run()
