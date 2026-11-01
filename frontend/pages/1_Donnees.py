import pandas as pd
import plotly.express as px
import streamlit as st
from api_client import get, upload
from styles import header

header('MODULE 01 / DONNÉES', 'Exploration des transactions',
       'Données Sparkov disponibles via Kaggle, import CSV compatible ou jeu synthétique de démonstration.')

with st.container(border=True):
    st.markdown('#### Source du dataset')
    source = st.radio('Choix', ['Démonstration', 'Kaggle (Sparkov)', 'Importer un CSV'], horizontal=True)
    if source == 'Démonstration' and st.button('Utiliser le jeu de démonstration'):
        st.session_state.dataset_id = 'demo'
        st.rerun()
    if source == 'Kaggle (Sparkov)':
        st.caption('La première récupération télécharge les données publiques et conserve un sous-ensemble chronologique. Prévoir une connexion Internet.')
        if st.button('Télécharger / utiliser Kaggle', type='primary'):
            st.session_state.dataset_id = 'kaggle'
            st.rerun()
    if source == 'Importer un CSV':
        file = st.file_uploader('CSV (format Sparkov)', type=['csv'])
        if file and st.button('Importer ce fichier', type='primary'):
            try:
                response = upload(file)
                st.session_state.dataset_id = response['dataset_id']
                st.success('Dataset chargé.')
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

try:
    data = get('/data/summary', {'dataset_id': st.session_state.dataset_id})
except Exception as exc:
    st.error(str(exc))
    st.stop()

info = data['overview']
a, b, c, d = st.columns(4)
a.metric('Transactions', info['rows'])
b.metric('Fraudes', info['frauds'])
c.metric('Taux de fraude', f"{info['fraud_rate']*100:.2f} %")
d.metric('Montant total', f"{info['amount_total']:,.0f}")
if st.session_state.dataset_id == 'demo':
    st.info('Jeu de démonstration synthétique, à remplacer par Sparkov pour le rendu final.')

left, right = st.columns(2, gap='large')
with left:
    with st.container(border=True):
        st.markdown('#### Montants selon la classe')
        chart = pd.DataFrame(info['amount_stats'])
        fig = px.bar(chart, x='class', y='median', color='class',
                     labels={'class':'Fraude (0 / 1)', 'median':'Montant médian'},
                     color_discrete_sequence=['#16665C','#C46C54'])
        fig.update_layout(template='plotly_white', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
with right:
    with st.container(border=True):
        st.markdown('#### Taux de fraude par catégorie')
        if data.get('categories'):
            t = pd.DataFrame(data['categories'])
            fig = px.bar(t, x='category', y='rate', color_discrete_sequence=['#16665C'])
            fig.update_layout(template='plotly_white', xaxis_title=None, yaxis_title='Taux de fraude')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption('Catégorie non disponible.')
if data.get('daily'):
    st.markdown('#### Évolution temporelle')
    fig = px.line(pd.DataFrame(data['daily']), x='date', y='rate', color_discrete_sequence=['#16665C'])
    fig.update_layout(template='plotly_white', xaxis_title=None, yaxis_title='Taux de fraude')
    st.plotly_chart(fig, use_container_width=True)
if data.get('hourly'):
    with st.expander('Taux de fraude par heure'):
        st.bar_chart(pd.DataFrame(data['hourly']).set_index('hour'))
with st.expander('Aperçu des colonnes'):
    st.dataframe(pd.DataFrame(data['sample']), use_container_width=True, hide_index=True)
