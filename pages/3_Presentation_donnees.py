import matplotlib
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from PIL import Image
st.markdown('# Cadre du jeu de données')
st.markdown('---')
st.markdown(' ### Objectif ')
st.markdown(' - #### La prédiction d’anomalie cardiaque (DataSet Mitbih). ')
st.markdown(' - #### La prédiction d’infarctus du myocarde (DataSet PtbDb).')
st.markdown('---')
st.markdown(' ### Données')
st.markdown('##### [lien vers les banques de données csv](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)')
st.markdown(' #### Quatre Dataset sont disponible depuis la plateforme Kaggle')
st.markdown(' Deux pour les infarctus du myocarde')
st.markdown(' - ptbdb_normal (label 0)')
st.markdown(' - ptbdb_abnormal (label 1)')
st.markdown(' Deux pour les anomalies cardiaques (label 0 à 4)')
st.markdown(' - mitbih_test')
st.markdown(' - mitbih_train')
st.markdown(' # Volumétrie du jeu de données ')
st.markdown(' ## Ensemble des valeurs  ')
st.markdown(' ### Il s’agit de données sur une échelle de temps')
st.markdown('  - #### Découpée en 187 colonnes ')
st.markdown('  - #### Variable cible : entier en 188ème colonnes ')
st.markdown('## Volumétrie des jeux de données')
cb1 = st.checkbox('Volumétrie du Jeu de données ptbdb ')
cb2 = st.checkbox('Volumétrie du Jeu de données mitbih')
st.markdown('## Analyse exploratoire')
cb4 = st.checkbox('Exploration des données')
st.markdown('---')
st.markdown('## Conclusion de l\'Analyse exploratoire')
cb3 = st.checkbox('Conclusion')

if cb1:
    st.markdown('# ElectroCardioGramme (ECG) infarctus du myocarde')
    st.markdown('ptbdb_normal : représente dans son intégralité des individus sains ne souffrant pas d\'infarctus du myocarde. Il est composé de 4045 lignes ou individus pour 188 colonnes soit la description temporelle de son ECG sur une seconde.')
    st.markdown('ptbdb_abnormal : représente dans son intégralité des individus souffrant d\'infarctus du myocarde. Il est composé de 10505 lignes ou individus pour 188 colonnes soit la description temporelle de son ECG sur une seconde')
    st.markdown('### Categorie sain malade infarctus')
    image7 = Image.open('pages/img/sain_malade_ptbdb.png')
    st.image(image7, caption='categorie sain malade infarctus')
    image11 = Image.open('pages/img/sain_infarctus_ptbdb.png')
    st.image(image11, caption='categorie sain malade infarctus')
if cb2:
    st.markdown('# ElectroCardioGramme (ECG) arythmie cardiaque')
    st.markdown('### Jeux de données sont distingués par 5 catégories, ')
    st.markdown('### Variable cible sur la dernière colonne représentée par: ')
    st.markdown(' - #### un entier 0 (normal), ')
    st.markdown(' - #### un entier 1 (fibrillation atriale), ')
    st.markdown(' - #### un entier 2 (ventilation supra ventriculaire), ')
    st.markdown(' - #### un entier 3 (double ventilation ventriculaire), ')
    st.markdown(' - #### un entier 4 (arythmie inclassable).')
    st.markdown('---')
    st.markdown('### On remarque un très fort déséquilibre des données ')
    st.markdown(' - #### Catégorie 1: fibrillation atriale ***2.5% de l’ensemble du jeu de données***.')
    st.markdown(' - #### catégorie 3: double ventilation ventriculaire ***0.7 % de l’ensemble du jeu de données***.')
    st.markdown('### Inversement le nombre d’observation d’***individu sain représente 82,8 % du Dataset***.')
    st.markdown('### Volumétrie catégorie Arythmie Cardiaque')
    image9 = Image.open('pages/img/repartition_equilibre_mitbih.png')
    st.image(image9, caption='Volumétrie catégorie Arythmie Cardiaque')
    st.markdown('## Fouille précise de volumétrie')
    st.markdown('### Volumétrie catégorie uniquement pathologique Arythmie Cardiaque')
    image10 = Image.open('pages/img/pie_chart_arythmie_classique.png')
    st.image(image10, caption='Volumétrie catégorie uniquement pathologie Arythmie Cardiaque')
if cb4:
    st.markdown('# Exploration des données')
    st.markdown('### Complexe QRS & écart RR')
    image1 = Image.open('pages/img/qrs.jpg')
    st.image(image1, caption='Complexe QRS Sain')

    st.markdown('### Affichage simultané en transparence')
    image2 = Image.open('pages/img/alpha_def_img.png')
    st.image(image2, caption='affichage simultané en transparence')

    st.markdown('### Exemple de distribution de proéminence')
    image3 = Image.open('pages/img/exemple_proem.png')
    st.image(image3, caption='exemple de distribution de proéminence')

    st.markdown('### Distribution des medianes de proéminences')
    image4 = Image.open('pages/img/median_proem.png')
    st.image(image4, caption='distribution des medianes de proéminences')

    st.markdown('### Generation d\'image projet conv2D apparition des proéminences')
    image6 = Image.open('pages/img/proem_exemple_effet2D_v.png')
    st.image(image6, caption='generation d\'image projet conv2D')

if cb3:
    st.markdown('## Conclusion : ')
    st.markdown('#### ***Au regard du très fort déséquilibre***')
    st.markdown('#### Il apparait indispensable de faire un rééchantillonnage sur le jeu de donnée')

