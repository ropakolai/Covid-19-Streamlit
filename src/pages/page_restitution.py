'''
Créé le 8 Avril 2024

@author: Equipe DS Jan 2024
@summary: Page restitution des résultats sur un gros volume de données
'''
import streamlit as st
import pandas as pd

from src.utils import utils_display as ud
from src.utils import utils_models as um
from src.features import build_features as bf
from config import paths,files
import os

st.cache_data(show_spinner=False)
def page_restitution(title,external_prediction_sm,internal_prediction_sm,external_prediction_mc,internal_prediction_mc): 
    st.header("Restitution sur un gros volume de données externes")
    st.subheader("Classification Sain / Malade")
    st.dataframe(external_prediction_sm,hide_index=True)
    
    st.subheader("Classification Multiple")
    st.dataframe(external_prediction_mc,hide_index=True)
    
    st.header("Restitution sur un gros volume de données internes")
    st.subheader("Classification Sain / Malade")
    st.dataframe(internal_prediction_sm,hide_index=True)
    
    
    st.subheader("Classification Multiple")
    st.dataframe(internal_prediction_mc,hide_index=True)
    