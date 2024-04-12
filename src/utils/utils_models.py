'''
Créé le 14 mars 2024

@author: Equipe DS Jan 2024
'''
import pickle
from tensorflow.keras.models import load_model
import json
import os
import numpy as np
from src.features import build_features as bf
import tensorflow as tf
import keras
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mplt
import pandas as pd
import os
import cv2
from IPython.display import Image, display
import seaborn as sns
import streamlit as st # pour la mise en cache
from src.utils import utils_gradcam as ug
from tensorflow.keras.layers import Conv2D

## Import des modules de logging et timestamping
from datetime import datetime # timestamping
import logging
## Import du fichier de configuration pour des informations sur les Logs
from config import paths,infolog
## Import des modules de configuration

## 0 - Gestion des logs
## Gestion des logs
# récupération du chemin projet
main_path = paths["main_path"]
# récupération du chemin des logs
log_folder = infolog["logs_folder"]
# récupération du nom des fichiers logs
logfile_name = infolog["logfile_name"]
logfile_path = os.path.join(main_path,log_folder,logfile_name)
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Création d'un nouveau fichier et renommage de l'ancien fichier s'il existe
if os.path.exists(logfile_path):
    os.rename(logfile_path, f"{logfile_path}.{current_time}")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Créer un gestionnaire de logs pour un fichier
file_handler = logging.FileHandler(logfile_name)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
# Créer un logger pour le module main
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Créer un gestionnaire de logs pour la console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)
# Ajouter les gestionnaires de logs au logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)
  
## Chargement du modèle en cache
@st.cache_resource(show_spinner=False)
def load_models(load_path):
    """
    Charge un modèle Keras depuis un fichier.
    
    :param load_path: Chemin du fichier contenant le modèle à charger.
    :return: Modèle chargé.
    """
    model = load_model(load_path)
    return model 

## Les prédictions pour les images de la demo
def get_one_prediction_val_label(model, image_path, label_to_directory,modele,size=224):
    img = bf.preprocess_image(image_path, size, modele)
    predictions = model.predict(img)
    print(f"Image Path {image_path}")

    num_classes = model.outputs[0].shape[-1]

    if num_classes == 1:
        # Cas binaire
        result = int((predictions[0] > 0.5).astype(int))
        pred_directory = label_to_directory[result]
    else:
        # Classification multiple
        result_index = np.argmax(predictions)
        result = result_index
        pred_directory = label_to_directory[result_index]

    print(f"Result {result}")
    print(f"pred_directory {pred_directory}")
    pred_probability = [np.round(float(i),2) for i in predictions[0]]
    print(f"pred_probability {pred_probability}")
    return pred_directory, pred_probability

#### GRADCAM
preprocess_input = keras.applications.xception.preprocess_input

def get_img_gradcam(img_path, model, modele_prefix):
    results = []
    # Chemin de sauvegarde du fichier cam path
    file_name = os.path.basename(img_path)
    cam_path = f"{modele_prefix}_gradcam_{file_name}"
    if "gradcam_" not in img_path and img_path != '.DS_Store' and not os.path.exists(cam_path):
        # Obtenez le nom de fichier à partir du chemin complet
        
        logger.debug(f"file_name {file_name}")
        # Obtenez le tableau d'image
        img_array = preprocess_input(ug.get_img_array(img_path, size=224))
        logger.debug(f"img_array {img_array}")
        last_conv_layer_name = "block5_conv3"
        if "vgg" in modele_prefix.lower():
            last_conv_layer_name = "block5_conv3"
        elif "eff" in modele_prefix.lower():
            last_conv_layer_name = "top_conv"
        last_conv_layer_name = get_last_conv_layer(model)
        logger.debug(f"last_conv_layer_name {last_conv_layer_name}")
        # Générez la carte de chaleur Grad-CAM
        heatmap = ug.make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        logger.debug(f"heatmap {heatmap}")
        
        ug.save_and_display_gradcam(img_path, heatmap, cam_path)
    
        # Ajoutez le résultat à la liste
        cam_path
    return cam_path

def get_gradcam(test_img_path,model,modele_prefix):
    results = []
    for img_path in os.listdir(test_img_path):
        file_name = os.path.basename(img_path)
        cam_path = f"{modele_prefix}_gradcam_{file_name}"
        if "gradcam_" not in img_path and img_path != '.DS_Store' and not os.path.exists(cam_path):
            logger.debug(f"file_name {file_name}")
            # Obtenez le tableau d'image
            img_array = preprocess_input(ug.get_img_array(os.path.join(test_img_path,img_path), size=224))
            logger.debug(f"img_array {img_array}")
            last_conv_layer_name="block5_conv3"
            if "vgg" in modele_prefix.lower():
                last_conv_layer_name="block5_conv3"
            elif "eff" in modele_prefix.lower():
                last_conv_layer_name="top_conv"
            last_conv_layer_name=get_last_conv_layer(model)
            logger.debug(f"last_conv_layer_name {last_conv_layer_name}")
            # Générez la carte de chaleur Grad-CAM
            heatmap = ug.make_gradcam_heatmap(img_array, model, last_conv_layer_name)
            logger.debug(f"heatmap {heatmap}")
            # Sauvegardez Grad-CAM avec le nom de fichier associé
            ug.save_and_display_gradcam(os.path.join(test_img_path,img_path), heatmap, cam_path)
        
            # Ajoutez le résultat à la liste
            results.append((file_name, cam_path))
    return results

def get_last_conv_layer(model):
    """
    Retourne le nom de la dernière couche de convolution d'un modèle Keras.

    Args:
    model (Model): Le modèle Keras à inspecter.

    Returns:
    str: Le nom de la dernière couche de convolution. Retourne None si aucune couche de convolution n'est trouvée.
    """
    # Itérer à travers les couches du modèle en partant de la dernière
    for layer in reversed(model.layers):
        # Vérifier si la couche est une couche de convolution
        if isinstance(layer, Conv2D):
            return layer.name
    # Retourner None si aucune couche de convolution n'a été trouvée
    return None
