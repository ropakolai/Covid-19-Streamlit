'''
Créé le 20 mars 2024

@author: Equipe DS Jan 2024
@summary: Gestion modulaire des pages streamlit pour pouvoir facilement se répartir les tâches
'''
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import utils_display as ud
from src.utils import utils_models as um
from src.features import build_features as bf

import os
import time
def page_preprocessing(title):  # 
    """
    Fonction de la page de pré-traitement
    """
    
    st.title(title)
    st.header("Description")
    st.subheader("- Echantillonage")
    st.write("Nous appliquons un échantillonage de données aléatoires")
    st.subheader("- Redimensionnement")
    st.write("Le redimensionnement de l'image s'adapte au modèle utilisé")
    st.subheader("- Niveau de Gris / RGB")
    st.write("Le nombre de canaux s'adapte au modèle utilisé")
    
    st.header("Justification")
    st.write("Le redimensionnement est nécessaire pour l'utilisation des données dans des modèles pré-entrainés")
    st.write("La remise en niveau gris fut nécessaire pour homogénéiser le jeu de données")
    st.write("Cependant, la plupart des modèles pré-entrainés utilisés lors de ce projet ont été pré-entrainés sur des millions d'images du dataset imagenet et sont adaptés au mode RGB uniquement")
    st.write("Le mode Niveau de gris a été utilisé uniquement dans le cadre des modèles baseline (LeNet)")
    
    return 
