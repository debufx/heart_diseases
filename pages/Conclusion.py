import streamlit as st

import streamlit as st

st.markdown('# Critiques et perspectives')
cb1 = st.checkbox('Arythmies cardiaques')
cb2 = st.checkbox('Infarctus du myocarde')
if cb1:
    st.markdown('## Critiques arythmies cardiaques')
    st.markdown('### Compréhension de l’apport des proéminences')
    st.markdown(' - #### Dataset en espace à trois dimensions de type image en niveaux de gris ')
    st.markdown(' - #### Pre-processing transformation de données numériques en images')
    st.markdown(' - #### Lots nombreux d’observations.')
    st.markdown(' - #### Manque de technicité sur le sujet en cours de projet')
    st.markdown('---')
    st.markdown('## Perspectives')
    st.markdown(' - ### Meilleure gestion du volume de données en image')
    st.markdown(' - ### L\'information des proéminences dans des Dataset pouvait servir:')
    st.markdown(' - #### les réseaux de convolutions 2D ')
    st.markdown(' - #### Eventuellement les GANs avec un discrimant capable de classer les ECG')

st.markdown('---')
if cb2:
    st.markdown('## Critiques infarctus du myocarde')
    st.markdown('### DataSet infarctus du myocarde')
    st.markdown(' - #### En l\'état le modèle est optimal')
    st.markdown(' - #### Il répond à la demande de précision et de rappel très fort')
    st.markdown(' - #### La précision (haut ratio) : une observation est placé dans sa catégorie avec une très forte fiabilité')
    st.markdown(' - #### le retour (haut ratio) : très faibles proportions de Faux Positif')
    st.markdown('---')
    st.markdown('## Perspectives')
    st.markdown(' - ### Au vue des enjeux ce modèle peut être mise en production ')
    st.markdown(' - ### Pour analyse')
    st.markdown(' - #### Pour étayage du corps médical ')
    st.markdown(' - #### Pour une aide fiable à la prise de décision')
st.markdown('# L’étude des pathologies à complication mortelle')
cb1 = st.checkbox('Pistes d\'amélioration')
cb2 = st.checkbox('Intégration Process métiers')
cb3 = st.checkbox('Remerciement')
st.markdown('---')
if cb1:
    st.header('Pistes d’amélioration pour le modèle appliqué aux anomalies cardiaque')
    st.markdown('### En cas de danger imminent pour le patient: ')
    st.markdown('#### Mon projet sur les anomalies ne répond pas à la fiabilité ')
    st.markdown(' - #### sur un environnement critique et vital à complication mortelle')
    st.markdown(' - #### et l’exigence de ce secteur médical.')

st.markdown('---')
if cb2:
    st.header('Intégration du modèle de détection d’infarctus du myocarde en process métiers')
    st.markdown(' - #### Le jeu de donnée d’infarctus du myocarde' )
    st.markdown(' - #### cas de classification binaire,')
    st.markdown('### Résultats d\'efficacité')
    st.markdown(' - #### soins immédiat à délivrer')
    st.markdown(' - #### où d’alerte à lancer ')
    st.markdown(' ### Montre connectée ou appareils médicaux dédiés:')
    st.markdown(' - #### capable d’appeler les secours en cas de danger imminent pour le patient.')

st.markdown('---')

if cb3:
    st.header('Remerciements')
    st.markdown(' ### Ce projet a été réalisé au départ pour la phase d’analyse exploratoire')
    st.markdown(' - ### avec deux autres personnes que je tiens à remercier à travers ce site,')
    st.markdown(' - ### Éric et Thomas.')
    st.markdown('---')
    st.markdown(' - ### Je tiens à remercier l’équipe de DataScientest, ')
    st.markdown(' - ### et tout particulièrement:')
    st.markdown(' - ### Laurène et Pauline pour le soutien et l’accompagnement qui m\'ont permis d’arriver au bout de ma formation')
