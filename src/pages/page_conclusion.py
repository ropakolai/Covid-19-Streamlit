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

def page_conclusion(title):
    st.title(title)
    st.write("Notre objectif était de construire un modèle qui puisse venir en soutien au personnel médical dans la lecture de radiographies pulmonaires. L’intérêt de nos recherches a été double. D’une part, nos résultats apportent une contribution à un champs de recherche déjà établi et, d’autre part, nous avançons une solution de plus aux difficultés auxquelles les établissements sanitaires font face. Bien que la crise COVID ait été le déclencheur des investigations sur le sujet, les innovations émanant de ce champs de recherche sont toujours plus d’actualités et toujours plus pertinentes.")
    st.write("Avec des taux de précisions dépassant les 90% peu importe la configuration (classifications allant de 2 à 4 classes), nos modèles s’avèrent être un outil efficace dans les mains de spécialistes. Cependant, et nous l’évoquions ci-dessus, ils ne remplaceront jamais le regard et l’analyse de médecin d’expérience. ")
    st.write("Bien que nos recherches et résultats soient probants, nous devons toutefois reconnaitre plusieurs limites à notre projet. ")
    st.write("Premièrement, le dataset utilisé soulève quelques questionnements. Malgré sa qualité et les remarquables efforts des équipes qui ont contribué à sa construction, le déséquilibre entre les différentes classes est un élément à tenir en compte dans l’interprétation des résultats. Par conséquent, nous préférons approcher la généralisation de nos modèles pour usage clinique avec humilité et esprit critique. ")
    st.write("Deuxièmement, et toujours sur l’ensemble de données, la présence de nombreux artefacts et les erreurs éventuelles de classification (autrement dit, une image classifiée comme COVID dans le dataset alors que Normal en réalité - faux positif) nous pousse à reconnaitre une nouvelle limite quant à nos résultats.")
    st.write("Troisièmement, une analyse restreinte à de l’imagerie nous semble incomplète et requiert une intervention humaine significative malgré tout. Pour rendre notre solution encore plus pertinente, nous estimons qu’il aurait été décisif d’avoir accès à d’autres informations quant à la santé et au profil des patients. Nous l’expliquons ainsi. Pour beaucoup, le COVID-19 n’a pas entrainé d’hospitalisation et ses formes les plus sévères sont souvent couplées à des comorbidités. Un accès à des données permettant d’étoffer nos analyses sur ce point rendrait les diagnostiques d’autant plus pertinent et permettrait le tri et la focalisation sur les patients les plus à risque de développer des formes graves ou de connaitre des complications.")
