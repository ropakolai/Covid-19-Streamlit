'''
Créé le 22 mars 2024

@author: Equipe DS Jan 2024
@summary : Fonctions d'affichage utilisées dans streamlit

'''
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image



## Dictionnaires des images

def create_image_classification_dict_MC(image_directory):
    image_classification = {}

    for dirpath, _, filenames in os.walk(image_directory):
        for filename in filenames:
            if "gradcam_" not in filename and filename != '.DS_Store':
                base_name = os.path.splitext(filename)[0]
                if base_name.startswith("COVID-"):
                    classification = "COVID"
                elif base_name.startswith("Viral"):
                    classification = "Viral_Pneumonia"
                elif base_name.startswith("Lung_Opacity-"):
                    classification="Lung_Opacity"
                elif base_name.startswith("Normal-"):
                    classification = "Normal"
                else:
                    raise ValueError(f"Nom d'image inconnu : {filename}")
                image_classification[filename] = classification
    return image_classification

def create_image_classification_dict_3C(image_directory):
    image_classification = {}

    for dirpath, _, filenames in os.walk(image_directory):
        for filename in filenames:
            if "gradcam_" not in filename and filename != '.DS_Store':
                base_name = os.path.splitext(filename)[0]
                if base_name.startswith("COVID-"):
                    classification = "COVID"
                elif base_name.startswith("Viral"):
                    classification = "Viral_Pneumonia"
                elif base_name.startswith("Lung_Opacity-"):
                    classification="Non Classé"
                elif base_name.startswith("Normal-"):
                    classification = "Normal"
                else:
                    raise ValueError(f"Nom d'image inconnu : {filename}")
    
                image_classification[filename] = classification

    return image_classification

def create_image_classification_dict_SM(image_directory):
    image_classification = {}

    for dirpath, _, filenames in os.walk(image_directory):
        for filename in filenames:
            if "gradcam_" not in filename and filename != '.DS_Store':
                base_name = os.path.splitext(filename)[0]
                print(f"base_name {base_name}")
                if base_name.startswith("COVID-") or base_name.startswith("Viral") or base_name.startswith("Lung_Opacity-"):
                    classification = "Malade"
                elif base_name.startswith("Normal-"):
                    classification = "Sain"
                else:
                    raise ValueError(f"Nom d'image inconnu : {filename}")
    
                image_classification[filename] = classification

    return image_classification

def create_image_classification_dict_COV(image_directory):
    image_classification = {}

    for dirpath, _, filenames in os.walk(image_directory):
        for filename in filenames:
            if "gradcam_" not in filename and filename != '.DS_Store':
                base_name = os.path.splitext(filename)[0]
                if base_name.startswith("Normal-") or base_name.startswith("Viral"):
                    classification = "PAS_COVID"
                elif base_name.startswith("Lung_Opacity-"):
                    classification = "Non Classé"
                elif base_name.startswith("COVID-"):
                    classification = "COVID"
                else:
                    raise ValueError(f"Nom d'image inconnu : {filename}")
    
                image_classification[filename] = classification

    return image_classification

def remove_old_images_complex(dir_path):
    # ne pas suppreimer gradcam
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
        
def remove_old_images(dir_path):
    # ne pas suppreimer gradcam
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) and "gradcam_" not in filename: # on ne supprime pas les gradcam
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
