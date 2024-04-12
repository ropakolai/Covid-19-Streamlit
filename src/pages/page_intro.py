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


def page_intro(title):  # 
    """
    Fonction de la page d'Introduction
    """
    st.title(title)
    st.header("Présentation")
    st.write("La propagation rapide du coronavirus (COVID-19) a entravé la capacité des systèmes de santé à réaliser les diagnostics et tests requis dans les délais imposés par la pandémie. Ainsi, une recherche active de solutions alternatives pour le dépistage a été initiée.")
    st.write("En raison des effets significatifs du COVID-19 sur les tissus pulmonaires, l'usage de l'imagerie par radiographie thoracique s'est avéré incontournable pour le dépistage et le suivi de la maladie. Ce projet de deep learning se présente comme un apport de plus à des travaux antécédents qui ont permis de soutenir hopitaux et autres établissements sanitaires.")
    
    st.header("Problèmes")
    st.write("Les procédures de diagnostic, en particulier le diagnostic clinique, ne sont pas simples car les symptômes courants du COVID-19 ne peuvent généralement pas être distingués d'autres infections virales. L’instrument le plus utilisé pendant la pandémie fut (et est encore) le test PCR malgré un haut de taux de faux positifs.")
    st.write("Au fur et à mesure d’autres options ont émergées, notamment l’analyse de radiographies pulmonaires mais cette méthode connait également des limitations (par exemple, la difficulté que peuvent connaitre les médecins de distinguer les infections par COVID-19 des autres pneumonies virales en utilisant uniquement une radiographie du thorax).")
    
    st.header("Enjeux")
    st.subheader("Pertinence pour le métier")
    st.write("Pour que nos modèles soient pertinents et rendent de réels services au corps médical, ils se doivent d’atteindre un seuil de résultats élevé. Selon Yang et al. (2020), les modèles prouvant une précision de 89% ou plus sont tout à fait appropriés à l’usage clinique.")
    st.subheader("Résultats et efficacité opérationnelle")
    st.write("Le recours à l’intelligence artificielle a pu but d’établir un diagnostic rapide et fiable afin d’alléger le travail des effectifs hospitaliers. Ceci implique bien évidemment des modèles présentant des résultats probants mais il est également question d’établir un mode opératoire nouveau au sein des hôpitaux pour intégrer les solutions innovantes que la recherche en science de données développe.")
    st.subheader("Intelligence artificielle et éthique")
    st.write("L’IA dans le domaine de la santé présentent de multiples avantages potentiels, mais elle engendrent également des risques et posent diverses questions éthiques. Les enjeux auxquels nous faisons référence (Martineau et Romy, 2023) sont mais ne se limitent pas à : ")
    st.write("1. Le consentement à la collecte et au partage des données personnelles")
    st.write("2. La confidentialité et la protection de la vie privée des personnes")
    st.write("3. La fiabilité, la qualité et la représentativité des données")
    st.write("4. L’utilisation non éthique de l’IA")
    st.write("5. Le manque d’explicabilité des modèles")
    st.write("6. La responsabilité décisionnelle")
    st.write("7. Les biais et la discrimination algorithmique")
    
    
    st.subheader("Intelligence artificielle et régulation")
    st.write("La question éthique sous-tend la nécessité de respecter les régulations liées à l’usage médical de l’intelligence artificielle. Il y a bien sur l’obtention et la manipulation de données personnelles hautement sensibles régulées par la directive européenne RGPD. Mais cela ne s’arrête pas là puisque des modèles ne peuvent pas se substituer à l’analyse de professionnels expérimentés. La législation sur l'intelligence artificielle récemment rédigée par la Commission européenne institue cinq niveaux de risque pour les droits fondamentaux. Les systèmes d’IA utilisés dans les secteurs de la santé relèvent du niveau quatre sur cinq faisant de notre travail un projet soumis à de hautes exigences en matières de qualité, transparence et supervision humaine.")
    st.header("Objectifs")
    st.write("Dans le cadre de ce projet, nous proposons de développer un modèle de deep learning qui viendrait apporter un soutien au médecin dans la lecture des radiographies thoraciques. Notre intention est de construire un modèle via Convolutional neural networks (CNN), ou Réseau neuronal convolutif en français, qui d’adapte bien à la lecture et l’analyse d’imageries médicales. Très concrètement, l’objectif de ce modèle sera de déterminer sur base de l’analyse d’une radiographie pulmonaire si le patient est atteint de COVID-19 ou non.")
   
    st.header("Périmètre")
    st.write("Le projet est basé sur un ensemble de radiographies.")
    st.write("Il n'inclut donc pas de données textuelles basées sur les symptômes.")
    
    return 

