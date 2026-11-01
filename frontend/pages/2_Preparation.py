import pandas as pd
import streamlit as st
from api_client import get
from styles import header

header('MODULE 02 / PRÉPARATION', 'Préparation des transactions',
       'Variables temporelles et comportementales, séparation train / validation / test, sans information future.')
st.caption('Le pipeline construit les historiques par carte avant le split, en utilisant uniquement les opérations antérieures.')
if st.button('Analyser la préparation', type='primary'):
    try:
        with st.spinner('Construction des variables...'):
            report = get('/data/preparation', {'dataset_id': st.session_state.dataset_id})
        st.session_state['prep_report'] = report
        st.session_state['prep_source'] = st.session_state.dataset_id
    except Exception as exc:
        st.error(str(exc))
report = st.session_state.get('prep_report') if st.session_state.get('prep_source') == st.session_state.dataset_id else None
if report:
    st.markdown('#### Séparation des données')
    st.info(report['method'])
    cols = st.columns(3)
    for col, p in zip(cols, report['parts']):
        col.metric(p['label'], p['rows'], f"{p['frauds']} fraudes")
    st.markdown('#### Variables numériques')
    st.write(', '.join(report['numeric']))
    st.markdown('#### Variables catégorielles')
    st.write(', '.join(report['categorical']))
    st.markdown('#### Aperçu après construction')
    st.dataframe(pd.DataFrame(report['examples']), use_container_width=True, hide_index=True)
    st.caption(report['warning'])
else:
    st.info('Cliquez sur « Analyser la préparation ». Le split est 60 % / 20 % / 20 %.')
