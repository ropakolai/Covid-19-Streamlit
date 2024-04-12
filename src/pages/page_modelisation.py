'''
Créé le 20 mars 2024

@author: Equipe DS Jan 2024
@summary: Page modélisation
'''
import streamlit as st
import pandas as pd

from src.utils import utils_display as ud
from src.utils import utils_models as um
from src.features import build_features as bf
from config import paths,files
import os

@st.cache_data
def page_modelisation(title):
    '''
    >> Graphique d’entrainement
    >> Matrice de confusion
    >> Classification report
    >> Gradcam
    '''
    mc_path = os.path.join(paths['main_path'], paths['data_folder'], paths['figures_path'], paths['mc_subfolder'])
    sm_path = os.path.join(paths['main_path'], paths['data_folder'], paths['figures_path'], paths['sm_subfolder'])
    cv_path = os.path.join(paths['main_path'], paths['data_folder'], paths['figures_path'], paths['cv_subfolder'])
    troisc_path = os.path.join(paths['main_path'], paths['data_folder'], paths['figures_path'], paths['3c_subfolder'])

    st.header("Les modèles")
    # Liste des modèles utilisés
    # Stockage dans un dataframe pour affichage
    models_info = {
            'Modèle': ['VGG16', 'VGG19', 'ResNet50', 'EfficientNetB0'],
            'Année':['2014', '2014', '2015', '2019'],
            'Nombre de paramètres': ['138M+', '143M+', '25,6M+', '5,3M+'],
    }
    df_info = pd.DataFrame(models_info)
    st.dataframe(df_info, hide_index=True)
    st.write("Tous les modèles pré-entraînés sélectionnés utilisent un format d'image 224x224 RGB")
    ## VGG16 et VGG19 ##
    st.subheader("VGG16 et VGG19 - Sain/Malade")
    st.write("- Exécution sur un ensemble de données de 600 images par classe Sain / Malade")
    with st.expander("Optimisations appliquées"):
        st.markdown("""
                    - Recherche du meilleur paramètre avec Keras Tuner
                      - Taux d'apprentissage
                      - Nombre de neurone par couche dense (units)
                      - Nombre de couches de Dropout et taux de Dropout/désactivation
                      - Fonction de régularisation l2 et le taux d’apprentissage associé
                    - Dégel de couches par bloc (block5)
                    - Optimizer Adam
                    """)
    with st.expander("Code de construction"):
        st.code(build_vggX_code, language='python')
    tab_vgg16, tab_vgg19 = st.tabs(["VGG16", "VGG19"])
    class_rep_vgg16_path=os.path.join(sm_path, "VGG16_classification_report.json")
    class_rep_vgg19_path=os.path.join(sm_path, "VGG19_classification_report.json")
    ####### VGG 16 #####
    with tab_vgg16:
        st.subheader("Courbes d'entrainement ")
        col_accuracy, col_loss = st.columns(2)
        with col_accuracy:
            st.image(os.path.join(sm_path, "vgg16_entrainement_accuracy.png"))
        with col_loss:
            st.image(os.path.join(sm_path, "vgg16_entrainement_loss.png"))
        
        col_conf_matrix, col_class_report = st.columns(2)
        with col_conf_matrix:
            st.subheader("Matrice de confusion")
            st.image(os.path.join(sm_path, "VGG16_confusion_matrix.png"))
        with col_class_report:
            st.subheader("Rapport de classification")
            data = pd.read_json(class_rep_vgg16_path)
            classif_report_df = pd.DataFrame(data).T 
            st.dataframe(classif_report_df)
                       
        st.subheader("Interprétabilité avec Grad-Cam")
        col_covid, col_lo, col_normaln,col_viralp = st.columns(4)

        with col_covid:
            st.write("COVID")
            st.image(os.path.join(sm_path, "VG116_gradcam_covid_115.png"))
        with col_lo:
            st.write("Lung Opacity")
            st.image(os.path.join(sm_path, "VG116_gradcam_lung_opacity_62.png"))
        with col_normaln:
            st.write("Normal")
            st.image(os.path.join(sm_path, "VG116_gradcam_normal_228.png"))
        with col_viralp:
            st.write("Viral Pneumonia")
            st.image(os.path.join(sm_path, "VG116_gradcam_viral_pneumonia_256.png"))
            
        
    ####### VGG 19 #####
    with tab_vgg19:
        st.write("Courbe d'entrainement ")
        col_accuracy, col_loss = st.columns(2)
        with col_accuracy:
            st.image(os.path.join(sm_path, "vgg19_entrainement_accuracy.png"))
        with col_loss:
            st.image(os.path.join(sm_path, "vgg19_entrainement_loss.png"))
                     
        col_conf_matrix, col_class_report = st.columns(2)
        with col_conf_matrix:
            st.write("Matrice de confusion")
            st.image(os.path.join(sm_path, "VGG19_confusion_matrix.png"))
        with col_class_report:
            st.write("Rapport de classification")
            data = pd.read_json(class_rep_vgg19_path)
            classif_report_df = pd.DataFrame(data).T 
            st.dataframe(classif_report_df)
                       
        st.subheader("Interprétabilité avec Grad-Cam")
        col_covid, col_lo, col_normaln,col_viralp = st.columns(4)

        with col_covid:
            st.write("COVID")
            st.image(os.path.join(sm_path, "VGG19_covid_84.png"))
        with col_lo:
            st.write("Lung Opacity")
            st.image(os.path.join(sm_path, "VGG19_lung_opacity_240.png"))
        with col_normaln:
            st.write("Normal")
            st.image(os.path.join(sm_path, "VGG19_normal_63.png"))
        with col_viralp:
            st.write("Viral Pneumonia")
            st.image(os.path.join(sm_path, "VGG19_viral_pneumonia_84.png"))
            
build_vggX_code="""
# Définition de la fonction de construction de modèle pour Keras Tuner
def build_model(hp):
    # Hyperparamètres à rechercher
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    units = hp.Int('units', min_value=32, max_value=512, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    l2_lambda = hp.Choice('l2_lambda', values=[1e-3, 1e-4, 1e-5])  # Nouveau hyperparamètre pour la régularisation L2


    # Chargement du modèle VGG16 pré-entraîné
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Dégel des dernières couches de convolution
    for layer in base_model.layers:
        if 'block5' in layer.name:
            layer.trainable = True
        else:
            layer.trainable = False

    # Ajout des couches fully-connected
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(units, activation='relu', kernel_regularizer=l2(l2_lambda))(x)  # Utilisation de la régularisation L2

    # Ajouter plusieurs couches Dropout
    for _ in range(hp.Int('num_dropout_layers', min_value=1, max_value=5)):  # Ajoutez jusqu'à 5 couches de dropout
        x = Dropout(dropout_rate)(x)
    output = Dense(1, activation='sigmoid')(x)

    # Création du modèle final
    model = Model(inputs=base_model.input, outputs=output)

    # Compilation du modèle avec les hyperparamètres
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

# Configuration du tuner d'hyperparamètres
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=1,
    directory='dir11',
    project_name='vgg16_dropout_regularization'
)

# Recherche des hyperparamètres
tuner.search(x_train, y_train,
             epochs=5, batch_size=8,
             validation_data=(x_val, y_val))
""" 