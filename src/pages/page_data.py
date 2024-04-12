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
@st.cache_data(show_spinner=False)
def page_data(title, covid_ds):   
    """
    Fonction de la page des données
    """
    
    st.title(title)
    
    st.subheader("Présentation / Volumétrie")
    data = {
    "Répertoire": ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"],
    "Nombre de fichiers": [3616, 10192, 6012, 1345],
    "Contenu": ["Radiographies de malades confirmées Covid", "Radiographies de patients sains", "Radiographies présentant des anomalies pulmonaires", "Radiographies de pneumonies virales"]
    }
    
    df = pd.DataFrame(data)
    
    data_info = {
    "FILE NAME": ["Nom du fichier"],
    "FORMAT": ["PNG"],
    "SIZE": ["Dimensions de l'image (masque)"],
    "URL": ["Source de l'image"]
    }
    df_info = pd.DataFrame(data_info)
    st.dataframe(df_info, hide_index=True)
    # Suppression de l'en-tête du DataFrame pour l'affichage
    df_info.to_string(index=False, header=False)
    st.subheader("Architecture")
    st.write("Les images sources sont de dimensions 299*299")
    st.write("Les images masks sont de dimensions 256*256")
    st.write("Les images masks sont toutes en mode RGB alors que les images sources sont plus hétérogènes")
    
    # Afficher le DataFrame sans index
    st.dataframe(df, hide_index=True)
    st.write("Chaque dataset fournit l'image et le masque correspondant")
    st.write("Un fichier métadata décrit les informations de chaque répertoire")   
    
    st.header("DATAVIZ")
    covid_ds['image_dimensions'] = covid_ds['file_dim_px_length'].astype(str) + "*" + covid_ds['file_dim_px_height'].astype(str)
    covid_ds_melted = pd.melt(covid_ds, id_vars=['image_type'], value_vars=['image_mode', 'image_dimensions'], var_name='origin', value_name='value')
    # Crosstab entre la colonne regroupée (value) et la colonne cible
    ct = pd.crosstab(covid_ds_melted['value'], covid_ds_melted['image_type'])
    ct
    
    covid_ds_images = covid_ds[covid_ds['image_type'] == 'main_image']
    st.write("Les types de données ayant comme image le mode RGB")
    st.write(covid_ds_images[covid_ds_images['image_mode'] == "RGB"]['health_status'].unique())
    
    st.subheader("Distribution des données par catégorie et par URL source")
    
    # Dictionnaire de remplacement
    replacement_dict = {
        'https://sirm.org/category/senza-categoria/covid-19/': 'sirm.org - covid-19',
        'https://github.com/ml-workgroup/covid-19-image-repository/tree/master/png': 'github.com - ml-workgroup',
        'https://eurorad.org': 'eurorad.org',
        'https://github.com/armiro/COVID-CXNet': 'github.com - armiro',
        'https://github.com/ieee8023/covid-chestxray-dataset': 'github.com - ieee8023',
        'https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/#1590858128006-9e640421-6711': 'bimcv.cipf.es - covid19 projects',
        'https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data': 'kaggle.com - rsna pneumonia detection challenge',
        'https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia': 'kaggle.com - chest-xray-pneumonia'
    }
    # Création de la nouvelle colonne en utilisant le dictionnaire de remplacement
    covid_ds_images['source_short'] = covid_ds_images['source_url'].replace(replacement_dict)

    # Supposons que 'covid_ds_images' est votre DataFrame
    grouped = covid_ds_images.groupby(['source_short', 'health_status']).size().reset_index(name='counts')
    grouped_sorted = grouped.sort_values(by='counts', ascending=False)  # tri des données pour améliorer l'affichage
    
    # Création d'une figure Matplotlib
    plt.figure(figsize=(12, 8))
    
    # Stockez le graphique barplot dans une variable pour y accéder plus tard
    barplot = sns.barplot(data=grouped_sorted, y='source_short', x='counts', hue='health_status', palette='pastel')
    
    plt.xlabel('Nombre de données')
    plt.ylabel('Source URL (court)')
    plt.legend(title='Etat de santé')
    
    # Récupérer les informations des barres
    bars = barplot.patches
    
    # Itération sur les données pour afficher les labels sur les barres
    for bar, label in zip(bars, grouped_sorted['counts']):
        # Positionnement du texte sur les barres
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, label,
                 ha='center', va='center', color='black', size=9)
    
    # Afficher le graphique dans Streamlit
    st.pyplot(plt)

    # ## Graphique 2
    st.subheader("Répartition des données en fonction de l'état de santé général")
    valeurs = covid_ds['health_status'].value_counts()
    # Création du camembert
    fig_b, ax = plt.subplots()
    pastel = sns.color_palette("pastel")
    ax.pie(valeurs, labels=valeurs.index, autopct='%1.2f%%', startangle=140, colors=pastel)
    # ax.set_title('Répartition des données en fonction de l"état de santé général')
    
    # Affichage du camembert dans Streamlit
    st.pyplot(fig_b)
    
    # ## Graphique 3
    st.subheader("Répartition des données en fonction de l'état de santé")
    valeurs = covid_ds_images['is_healthy'].value_counts()
    # Création du camembert
    fig_c, ax = plt.subplots()
    ax.pie(valeurs, labels=valeurs.index, autopct='%1.2f%%', startangle=140, colors=pastel)
    # ax.set_title('Répartition des données en fonction de l"état de santé')
    
    # Affichage du camembert dans Streamlit
    st.pyplot(fig_c)
    st.write("Nous considérons que seules les données du répertoire Normal concernent des patients sains")
    st.write("Nous pouvons noter un léger déséquilibre entre le volume de données par patients sains et malades en faveur de la catégorie \"malade\".")
    
    st.header("Premières remarques/conclusions")
    st.write("- Le Dataset permet de réaliser une classification Sain / Malade")
    st.write("- Cependant, la classe d'Opacité Pulmonaire manque de précision et peut altérer les performances d'un modèle dans le cas de classification binaire Sain/Malade")
    st.write("- Pour une classification multiple, nous devrons aligner les données pour chaque classe à la classe minimale, ici Pneumonie Virale")
    st.write("Une option serait de procéder à une augmentation de la taille des données.")
    
    return 

