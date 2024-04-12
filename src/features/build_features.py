'''
Créé le 16/03/2024

@author: Equipe DS Jan 2024
@summary: Module de préparations des données: Chargement, échantillonage, classification, traitement des images et stockage dans des fichiers
@note: Cette page n'est pas utilisée pour l'instant
1- >>features/build_features.py<< Preprocessing: Utilisation de  pour le sampling et preprocessing

'''

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import keras 
##
# On importe la fonction preprocess_input pour chaque modèle pré-entrainé utilisé
from tensorflow.keras.applications.efficientnet import preprocess_input as pp_effnet
from tensorflow.keras.applications.vgg16 import preprocess_input as pp_vgg16 
from tensorflow.keras.applications.vgg19 import preprocess_input as pp_vgg19
from tensorflow.keras.applications.resnet50 import preprocess_input as pp_resnet50
##

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import TensorBoard

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

def preprocess_image(image_path,size,archi):
    logger.debug("---------------preprocess_image------------")
    img = keras.utils.load_img(image_path, target_size=(size,size))  # Redimensionner l'image à la taille attendue par VGG16
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    archi=archi.lower()
    if archi=='lenet': # Normalisation min-max
        img = img / 255.0
    elif archi=='vgg16': 
        # Effectue une normalisation en soustrayant la moyenne de chaque canal de l'image (RGB) et en inversant l'ordre des canaux de RGB à BGR
        img = pp_vgg16(img)
        
    elif archi=='vgg19':
        # Effectue une normalisation en soustrayant la moyenne de chaque canal de l'image (RGB) et en inversant l'ordre des canaux de RGB à BGR
        img = pp_vgg19(img)
        
    elif archi=='resnet50':
        # normalise les pixels de l'image en les divisant par 255 pour les amener à une échelle de 0-1, puis en effectuant une normalisation en soustrayant 0.5 et en multipliant par 2 pour obtenir une échelle de -1 à 1, sans inverser l'ordre des canaux.
        img = pp_resnet50(img)
        
    elif archi=='efficientnetb0':
        # Utilise une méthode de scaling qui amène les valeurs des pixels à l'échelle de 0-1 (en les divisant par 255)
        img = pp_effnet(img)
    else:
        logging.debug(f"Erreur - Nom d'architecture du modèle incorrecte - {archi}")
        return None
    
    return img

def preprocess_input_archi(img,archi):
    logger.debug("---------------preprocess_input_archi------------")
    archi=archi.lower()
    if archi=='lenet': # Normalisation min-max
        img = img / 255.0
    elif archi=='vgg16': 
        logger.debug(f"Archi {archi}")
        # Effectue une normalisation en soustrayant la moyenne de chaque canal de l'image (RGB) et en inversant l'ordre des canaux de RGB à BGR        
        img = pp_vgg16(img)
    elif archi=='vgg19':
        logger.debug(f"Archi {archi}")
        # Effectue une normalisation en soustrayant la moyenne de chaque canal de l'image (RGB) et en inversant l'ordre des canaux de RGB à BGR
        img = pp_vgg19(img)
    elif archi=='resnet50':
        logger.debug(f"Archi {archi}")
        # normalise les pixels de l'image en les divisant par 255 pour les amener à une échelle de 0-1, puis en effectuant une normalisation en soustrayant 0.5 et en multipliant par 2 pour obtenir une échelle de -1 à 1, sans inverser l'ordre des canaux.
        img = pp_resnet50(img)
    elif archi=='efficientnetb0':
        # Utilise une méthode de scaling qui amène les valeurs des pixels à l'échelle de 0-1 (en les divisant par 255)
        logger.debug(f"Archi {archi}")
        img = pp_effnet(img)
    else:
        logging.debug(f"Erreur - Nom d'architecture du modèle incorrecte - {archi}")
        return None
    return img

def invert_dict(d):
    logger.debug("---------------invert_dict------------")
    return {v: k for k, v in d.items()}

    
def get_data_from_parent_directory_labelled(parent_directory, labelling, size,dim,archi):
    logger.debug("---------------get_data_from_parent_directory_labelled------------")
    directories = [os.path.join(parent_directory, d) for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]
    image_paths = []
    labels = []
    directory_to_label = {directory: label for label, directory in labelling.items()}
    
    for dir_path in directories:
        directory_name = os.path.basename(dir_path)
        label = directory_to_label[directory_name]
        for img_file in os.listdir(dir_path):
            if os.path.isfile(os.path.join(dir_path, img_file)):
                image_paths.append(os.path.join(dir_path, img_file))
                labels.append(label)

    image_paths = np.array(image_paths)
    labels = np.array(labels)

    # Séparer en ensembles d'entraînement et de test
    train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=1234)

    # Prétraitement des données d'entraînement
    train_data = []
    for img_path, label in zip(train_image_paths, train_labels):
        if dim==1: # niveau de gris
            img = load_img(img_path, target_size=(size, size), color_mode='grayscale')
        else:
            img = load_img(img_path, target_size=(size, size))
        img_array = img_to_array(img)
        img_array = preprocess_input_archi(img_array,archi)
        train_data.append((img_array, label))

    X_train, y_train = zip(*train_data)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Prétraitement des données de test
    test_data = []
    for img_path, label in zip(test_image_paths, test_labels):
        if dim==1: # niveau de gris
            img = load_img(img_path, target_size=(size, size), color_mode='grayscale')
        else:
            img = load_img(img_path, target_size=(size, size))
        img_array = img_to_array(img)
        img_array = preprocess_input_archi(img_array,archi)
        test_data.append((img_array, label))

    X_test, y_test = zip(*test_data)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test, directory_to_label


