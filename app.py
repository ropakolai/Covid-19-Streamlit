'''
Créé le 20/03/2024

@author: Equipe DS Jan 2024
@summary: Page Streamlit
'''
## IMPORTS
import streamlit as st
from src.pages import page_intro,page_data,page_modelisation,page_preprocessing,page_demo,page_demo_externe,page_restitution,page_conclusion

import configparser
import pandas as pd
import os
from config import paths,files
 
# lecture du fichier csv de restitution des données sur le dataset
covid_ds=pd.read_csv(os.path.join(paths['main_path'],paths['data_folder'],paths['csv_path'],files['covid_csv_name']),sep=";")
#sm
external_prediction_sm=pd.read_csv(os.path.join(paths['main_path'],paths['data_folder'],paths['csv_path'],"Online_Res_VGG16_19_Effnet_SM_External.csv"),sep=",")
internal_prediction_sm=pd.read_csv(os.path.join(paths['main_path'],paths['data_folder'],paths['csv_path'],"Online_Res_VGG16_19_Effnet_SM_Internal.csv"),sep=",")

# mc
external_prediction_mc=pd.read_csv(os.path.join(paths['main_path'],paths['data_folder'],paths['csv_path'],"Online_Res_EffNet_MC_External.csv"),sep=",")
internal_prediction_mc=pd.read_csv(os.path.join(paths['main_path'],paths['data_folder'],paths['csv_path'],"Online_Res_EffNet_MC_Internal.csv"),sep=",")

###
#Fin de l'initilisation des données 
###

###
#Début de l'affichage et de la redirection vers les fonctions associées à chaque page
###
## Sidebar - Nom du Projet
st.sidebar.title("Analyse de radiographies pulmonaires Covid-19")
## Sidebar - Les pages

app_pages=["1 - Introduction", 
       "2 - Les données", 
       "3 - Pré-traitements",
       "4 - Modélisations",
       "5 - Démo",
       "6 - Démo chargement de fichiers",
       "7 - Restitution sur un gros volume de données",
       "8 - Conclusion"]

page=st.sidebar.radio("", app_pages)

if page==app_pages[0]: # intro
    page_intro.page_intro(app_pages[0])
    
elif page==app_pages[1]: # données
    page_data.page_data(app_pages[1],covid_ds)
    
elif page==app_pages[2]: # pré-traitements
    page_preprocessing.page_preprocessing(app_pages[2])
    
elif page==app_pages[3]: # démo
    page_modelisation.page_modelisation(app_pages[3])
    
elif page==app_pages[4]:
    page_demo.page_demo(app_pages[4])

elif page==app_pages[5]:
    page_demo_externe.page_demo_externe(app_pages[5]) 

elif page==app_pages[6]:
    page_restitution.page_restitution(app_pages[6],external_prediction_sm,internal_prediction_sm,external_prediction_mc,internal_prediction_mc) 
    
elif page==app_pages[7]:
    page_conclusion.page_conclusion(app_pages[7])    