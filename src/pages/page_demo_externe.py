'''
Créé le 20 mars 2024

@author: Equipe DS Jan 2024
@summary: Page Démo
'''
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import utils_display as ud
from src.utils import utils_models as um
from src.features import build_features as bf
from config import paths,files
from PIL import Image

import os
import time
def page_demo_externe(title):
    st.title(title)
    models_path = os.path.join(paths['main_path'],paths['data_folder'],paths['models_path'])
    test_img_path=os.path.join(paths['main_path'],paths['data_folder'],paths["external_images_path"])
    liste_modeles = ["ResNet50", "EfficientNetB0", "VGG16", "VGG19", "Tous"]
    #liste_modeles = ["ResNet50", "VGG16", "VGG19", "Tous"]
    sm_path = paths[ "sm_subfolder"]
    #cv_path = paths[ "cv_subfolder"]
    mc_path = paths[ "mc_subfolder"]
    #troisc_path = paths[ "3c_subfolder"]
    
    sm_dic = {0: 'Sain', 1: 'Malade'}
    #cv_dic = {0: 'PAS_COVID', 1: 'COVID'}
    mc_dic = {0: 'COVID', 1: 'Lung_Opacity', 2: 'Normal', 3: 'Viral_Pneumonia'}
    #troisc_dic = {0: 'Viral_Pneumonia', 1: 'Normal', 2: 'COVID'}
    #troisc_dic_resnet = {0: 'COVID', 1: 'Normal', 2: 'Viral_Pneumonia'}
    sm_img_class = ud.create_image_classification_dict_SM(test_img_path)
    #cv_img_class = ud.create_image_classification_dict_COV(test_img_path)
    mc_img_class = ud.create_image_classification_dict_MC(test_img_path)
    #troisc_img_class = ud.create_image_classification_dict_3C(test_img_path)
    print(f"sm_img_class {sm_img_class}")
    #print(f"cv_img_class {cv_img_class}")
    print(f"mc_img_class {mc_img_class}")
    #print(f"troisc_img_class {troisc_img_class}")
    liste_modeles_sm = {"ResNet50":"",
                        "EfficientNetB0":"EfficientNetB0_SM_750_model_2_15_0.keras",
                        #"EfficientNetB0":"EfficientNetB0_SM_750_VA_Filentuned_model.keras",
                        "VGG16":"VGG16_finetuned_model.h5",
                        "VGG19":"VGG19_finetuned_model.h5"}

    liste_modeles_mc = {"ResNet50":"resnet50_custom3.keras",
                        "EfficientNetB0":"EfficientNetB0_MC_750_model_2_15_0.keras",
                        #"EfficientNetB0":"EfficientNetB0_MC_750_model.keras",
                        "VGG16":"",
                        "VGG19":""}
    ## pas utilisés
    liste_modeles_covid = { "ResNet50":"",
                        #"EfficientNetB0":"EfficientNetB0_CV_750_VA_Filentuned_model.keras",
                         "VGG16":"",
                        "VGG19":""}
    
    liste_modeles_3c = {"ResNet50":"",
                        #"EfficientNetB0":"EfficientNetB0_3C_750_model.keras",
                        "VGG16":"",
                        "VGG19":""}

    
    liste_classes = ["Sain / Malade", "Multiclasse"]
    
    # Création de deux colonnes
    col_modeles, col_classes = st.columns(2)
    
    # Utilisation de la première colonne
    with col_modeles:
        st.header("Modèles")
        choix_modele = st.radio("Choisissez Un modèle :", liste_modeles)
    
    # Utilisation de la deuxième colonne
    with col_classes:
        st.header("Classes")
        choix_classification = st.radio("Choisissez une classe :", liste_classes)
        
    
    if st.button('Exécuter'):
        if choix_classification == "Sain / Malade":
            path = sm_path
            class_dic = sm_dic
            model_dict = liste_modeles_sm
            new_list_modeles = [choix_modele]
            image_class_dic = sm_img_class
            prefix="sm"
        elif choix_classification == "Multiclasse":
            path = mc_path
            class_dic = mc_dic
            model_dict = liste_modeles_mc
            new_list_modeles = [choix_modele]
            image_class_dic = mc_img_class
            prefix="mc"
        if choix_modele == "Tous":
            liste_modeles.pop()  # on enleve le dernier element qui correspond à "Tous"
            new_list_modeles = liste_modeles
            
        #full_image_paths = [os.path.join(test_img_path,key) for key in image_class_dic.keys()]
        # st.write(f"new liste modele {new_list_modeles}")
        # # initialisation des 1ère colonnes du tableau à afficher
        image_class_dic["Temps d'exécution"]="N/A"
        data_result = {'Image':list(image_class_dic.keys()),
                     'Vraie classe':list(image_class_dic.values())}
        pred_dic_KO= {}
        pred_dic_OK = {}
        model_tabs = []
        print(f"new_list_modeles {new_list_modeles}")
        for modele in new_list_modeles:
            if "vgg" in modele.lower():
                model_tabs.append(modele)
            # on récupère le nom du fichier modele keras ou h5
            model_name = model_dict[modele]
            # on initialise les objets qui seront utilisés pour les affichage : tableau resultat, tableau gradcam prek KO, et pred OK
            model_result_list = [] # résultat d'un modèle qui sera affiché dans le tableau
            model_gradcams_predOK=[] # liste des Gradcam
            model_gradcams_predKO=[] # liste des Gradcam pred KO
            
            print(f"model_name {model_name}")
            if model_name != "":
                # chargement du modèle
                model_path = os.path.join(models_path, path, model_name) 
                model = um.load_models(model_path) # on le charge une fois pour réutilisation
                print("model summary")
                print(model.summary())
                # initialisation du temps de prediction
                debut = time.time()
                # traitement image par image qui ont été initialisé dans les dictionnaires image_class_dic qui contient aussi la vraie prediction
                for image in image_class_dic.keys():
                    # on ignore la clé ajoutée pour le temps d'exécution
                    if image != "Temps d'exécution":
                        image_name = image
                        img_path=os.path.join(test_img_path, image_name)
                        ## traitement pour chaque image: 
                        # 1. prediction et label
                        res, pred = um.get_one_prediction_val_label(model,img_path, class_dic,modele)
                        #pred_arrondi = round(float(pred), 2)
                        model_result_list.append(f"{res} / {pred} ")
                        ### On traite uniquement VGG
                        if "vgg" in modele.lower():  
                            # 2. Génération du Gradcam et renvoi du chemin de l'image gradcam
                            img_cam_path=um.get_img_gradcam(img_path,model,modele+"_"+prefix) # chemin image, objet modele et nom du modele pour le prefixe de l'image gradcam
                            
                            #ajout des info gradcam, vraie classe et prediction dans la liste associée (pred ok ou ko)
                            if res==image_class_dic[image]: ## prédiction OK
                                # ajout dans une liste predOK du modele : le nom de l'image, la vraie classe, la classe prédite et le chemin du gradcam
                                model_gradcams_predOK.append((os.path.splitext(image_name)[0],res,image_class_dic[image],f"{pred}",img_cam_path))
                            else:
                                # même chose mais dans la liste predKO
                                model_gradcams_predKO.append((os.path.splitext(image_name)[0],res,image_class_dic[image],f"{pred}",img_cam_path))
                fin = time.time()
                model_result_list.append(f"{fin-debut:.2f} secondes")
                data_result[f"{modele}"] = model_result_list
                pred_dic_KO[f"{modele}"] = model_gradcams_predKO
                pred_dic_OK[f"{modele}"] = model_gradcams_predOK
        df = pd.DataFrame(data_result)
        #st.dataframe(df, hide_index=True, use_container_width=True)
        #st.write("VGG grad cams")
        #st.write(f"gradcam KO {pred_dic_KO}")
        #st.write(f"gradcam OK {pred_dic_OK}")
        
        ## Affichage des Gradcam par modèles:
        # Création dynamique des tabs
        #model_tabs=list(model_dict.keys())
        if model_tabs !=[]:
            print(f"model_tabs {model_tabs}")
            print(f"model_dict.keys() {model_tabs}")
            for i,modele in enumerate(model_tabs):
                model_name = model_dict[modele]
                if model_name !="":
                    print(f"model_tabs[i] {model_tabs[i]}")
                    with st.expander(model_tabs[i]):
                        tab_names = ["Bonnes prédictions","Mauvaises prédiction"]
                        tabs = st.tabs(tab_names)
                        #st.subheader(f"Modèle {modele}")
                        model_gradcam_OK=pred_dic_OK[modele]
                        model_gradcam_KO=pred_dic_KO[modele]
                        nb_predOK=len(model_gradcam_OK)
                        nb_predKO=len(model_gradcam_KO)
                        with tabs[0]: # bonnes prédictions
                            #st.write(f"Bonnes prédictions - {(nb_predOK/(nb_predOK+nb_predKO))*100}%")
                            # Nombre de colonnes par ligne
                            nb_colonnes = 3
                            # Initialiser le compteur de colonnes
                            compteur_colonnes = 0
                            
                            # Créer une liste pour stocker temporairement les widgets à afficher dans une ligne
                            widgets_temp = []
                            
                            for image_name, prediction, classe_reelle, pred_pct,img_cam_path in model_gradcam_OK:
                                # Si le compteur de colonnes atteint le nombre de colonnes par ligne, réinitialiser le compteur et afficher la ligne
                                if compteur_colonnes == nb_colonnes:
                                    # Afficher la ligne précédente
                                    cols = st.columns(nb_colonnes)  # Crée un nombre de colonnes égal à nb_colonnes
                                    for col, widget in zip(cols, widgets_temp):
                                        with col:
                                            # widget[0] correspond au texte, widget[1] correspond à l'image
                                            st.markdown(f"""
                                                        <div style="text-align: center;"> 
                                                            {widget[1]}
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                            st.markdown(f"""
                                                            <div style="text-align: center;"> 
                                                                {widget[2]}
                                                            </div>
                                                            """, unsafe_allow_html=True)
                                            st.markdown(f"""
                                                            <div style="text-align: center;"> 
                                                                <strong>{widget[0]}</strong>
                                                            </div>
                                                            """, unsafe_allow_html=True)
                                            
                                            st.image(widget[3])
                                            
                                        
                                    # Réinitialiser les variables pour la nouvelle ligne
                                    compteur_colonnes = 0
                                    widgets_temp = []
                            
                                # Préparer le texte et l'image pour l'affichage
                                texte1= f"{image_name}"
                                texte2 = f"{classe_reelle}"
                                texte3 = f"{pred_pct}"
                                image = Image.open(img_cam_path)
                                image_redim = image.resize((224, 224))
                                
                                # Ajouter le texte et l'image à la liste temporaire
                                widgets_temp.append((texte1, texte2, texte3, image_redim))
                                compteur_colonnes += 1
                            
                            # Vérifier s'il reste des widgets à afficher après la dernière itération de la boucle
                            if widgets_temp:
                                cols = st.columns(nb_colonnes)
                                for i, widget in enumerate(widgets_temp):
                                    with cols[i]:
                                        st.markdown(f"""
                                                        <div style="text-align: center;"> 
                                                            {widget[1]}
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                        st.markdown(f"""
                                                            <div style="text-align: center;"> 
                                                                {widget[2]}
                                                            </div>
                                                            """, unsafe_allow_html=True)
                                        st.markdown(f"""
                                                            <div style="text-align: center;"> 
                                                                <strong>{widget[0]}</strong>
                                                            </div>
                                                            """, unsafe_allow_html=True)
                                            
                                        st.image(widget[3])
                                            
                                            
                        with tabs[1]: # Mauvaises predictions
                            #st.write(f"Mauvaises prédictions - {(nb_predKO/(nb_predOK+nb_predKO))*100}%")
                            # Nombre de colonnes par ligne
                            nb_colonnes = 3
                            # Initialiser le compteur de colonnes
                            compteur_colonnes = 0
                                
                                # Créer une liste pour stocker temporairement les widgets à afficher dans une ligne
                            widgets_temp = []
                            
                            for image_name, prediction, classe_reelle, pred_pct,img_cam_path in model_gradcam_KO:
                                # Si le compteur de colonnes atteint le nombre de colonnes par ligne, réinitialiser le compteur et afficher la ligne
                                if compteur_colonnes == nb_colonnes:
                                    # Afficher la ligne précédente
                                    cols = st.columns(nb_colonnes)  # Crée un nombre de colonnes égal à nb_colonnes
                                    for col, widget in zip(cols, widgets_temp):
                                        with col:
                                            # widget[0] correspond au texte, widget[1] correspond à l'image
                                            st.markdown(f"""
                                                        <div style="text-align: center;"> 
                                                            {widget[1]}
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                            st.markdown(f"""
                                                            <div style="text-align: center;"> 
                                                                {widget[2]}
                                                            </div>
                                                            """, unsafe_allow_html=True)
                                            st.markdown(f"""
                                                            <div style="text-align: center;"> 
                                                                <strong>{widget[0]}</strong>
                                                            </div>
                                                            """, unsafe_allow_html=True)
                                            
                                            st.image(widget[3])
                                            
                                    # Réinitialiser les variables pour la nouvelle ligne
                                    compteur_colonnes = 0
                                    widgets_temp = []
                            
                                # Préparer le texte et l'image pour l'affichage
                                texte1= f"{image_name}"
                                texte2 = f"{prediction}"
                                texte3 = f"{pred_pct}"
                                image = Image.open(img_cam_path) # chemin complet vers l'image gradcam
                                image_redim = image.resize((224, 224))
                                
                                # Ajouter le texte et l'image à la liste temporaire
                                widgets_temp.append((texte1, texte2, texte3, image_redim))
                                compteur_colonnes += 1
                                
                            # Vérifier s'il reste des widgets à afficher après la dernière itération de la boucle
                            if widgets_temp:
                                cols = st.columns(nb_colonnes)
                                for i, widget in enumerate(widgets_temp):
                                    with cols[i]:
                                        st.markdown(f"""
                                                        <div style="text-align: center;"> 
                                                            {widget[1]}
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                        st.markdown(f"""
                                                            <div style="text-align: center;"> 
                                                                {widget[2]}
                                                            </div>
                                                            """, unsafe_allow_html=True)
                                        st.markdown(f"""
                                                            <div style="text-align: center;"> 
                                                                <strong>{widget[0]}</strong>
                                                            </div>
                                                            """, unsafe_allow_html=True)
                                            
                                        st.image(widget[3])
                                               
                                        
        print(f"Récapitulatif {df}")
        st.write(f"Récapitulatif")
        st.dataframe(df, hide_index=True)    
    else:
        st.write('')
    ## file upload
    uploaded_files = st.file_uploader(label="Charger des images.. PNG, JPG ou JPEG, extension COVID-, Normal-, Viral_Pneumonia et Lung_Opacity pour identifier la vraie classe", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    # Supposons que uploaded_files est une liste de fichiers chargés
    if uploaded_files:
        # Supprimer les anciennes images
        ud.remove_old_images(test_img_path)
        
        for uploaded_file in uploaded_files:
            # Ici, uploaded_file est un objet fichier individuel, donc il a un attribut 'name'
            file_path = os.path.join(test_img_path, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Afficher l'image chargée
            # Note: Utilisez file_path ou créez un objet Image ici si nécessaire
           # st.image(Image.open(file_path), caption=uploaded_file.name, use_column_width=True)
       # Afficher l'image chargée
            #st.image(Image.open(uploaded_file), caption="Image chargée", use_column_width=True)
    
    
