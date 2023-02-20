
import keras
import streamlit as st
from sklearn.metrics import plot_confusion_matrix, plot_precision_recall_curve
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.metrics import classification_report_imbalanced
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


def get_classification_report(y_test, y_pred):
    from sklearn import metrics
    report = classification_report_imbalanced(y_test, y_pred, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    return df_classification_report


st.markdown('### Deux algorithmes simples de machine Learning.')
st.markdown(' - ##### La régression logistique')
st.markdown(' - ##### Les systèmes de vecteur SVC')
st.markdown('#### Les résultats:')
st.markdown(' - ##### Très faibles en termes de précision et de rappels.')
st.markdown('#### En oversampling ')
st.markdown('#### Avec des valeurs de proéminences renseignées:')
st.markdown('##### Au regard de l’accuracy et de la matrice de confusion, ')
st.markdown(' - ##### ***Acceptable pour: les K plus proches voisins*** ')
st.markdown(' - ##### ***Acceptable pour les Forêts Aléatoires***')
st.markdown('---')
st.markdown('### Essai : réseau de convolution 2D')
st.markdown(' #### ***sur des images renseignées par des données de proéminences***:')
st.markdown(' - #### Le pre-processing s\'est avéré être trop long')
st.markdown('---')
st.markdown('### Evaluation : ')
st.markdown(' - #### En réseau neuronaux DENSE')
st.markdown(' - #### En réseau de convolution 1D. ')
cb1 = st.checkbox('Réseau dense')
cb2 = st.checkbox('Réseau de convolution 1D Arythmie')
cb3 = st.checkbox('Réseau de convolution 1D infarctus du myocarde')
if cb1:
    st.markdown(' - #### Nous observons ici un modèle pour analyse uniquement')
    df1 = pd.read_csv('pages/archive/mitbih_test.csv', header=None)
    df2 = pd.read_csv('pages/archive/mitbih_train.csv', header=None)

    df = pd.concat([df1, df2], ignore_index=True, sort=False)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=28)
    st.subheader('Présentation du problème de classification d\'arythmie cardiaque')
    st.markdown('### Cinq types d’ElectroCardioGramme')
    st.markdown(' - Catégorie saine ')
    st.markdown(' - Fibrillation atriale')
    st.markdown(' - Ventilation supra ventriculaires')
    st.markdown(' - Fusion ventriculaire ')
    st.markdown(' - Classe d’ECG inclassable mais présentant des anomalies')
    st.markdown('#### Nous sommes donc dans un problème de classification multiple')
    st.markdown('***mauvaise prédiction globale du modèle***')
    model_dir = 'pages/modele/model_dense_mitbih.h5'
    modelDense = keras.models.load_model(model_dir)
    smote = SMOTE()
    batch_size = 32
    _, mse, mae, accuracy = modelDense.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
    st.write('mse', 'mae', 'accuracy', mse, mae, accuracy)
    modelDense.summary(print_fn=lambda x: st.text(x))
    st.subheader("Matrice de confusionS")
    y_test = to_categorical(y_test)
    pred_class_a = modelDense.predict(X_test).argmax(axis=1)
    rounded_labels_a = np.argmax(y_test, axis=1)
    st.table(pd.crosstab(rounded_labels_a, pred_class_a, rownames=["reel"], colnames=["predict"]))
    metrics_df = get_classification_report(rounded_labels_a, pred_class_a)
    st.table(metrics_df)
if cb2:
    import streamlit as st
    from PIL import Image

    st.markdown('## Modèle acceptable sur le jeu de données d\'arythmie cardiaque')
    st.markdown('### Réseau de neurones convolutif 1D : ')
    st.markdown(
        ' - #### Deux premières couches de convolution avec deux filtre consecutifs de 128 et respectivement un noyau de 5 puis de 3.')
    st.markdown(' - #### Normalisation du batch et pooling')
    st.markdown(' - #### Deux couches suivantes de filtres égaux à 64 et de noyaux de 5 et 3')
    st.markdown(' - #### Normalisation du batch et pooling')
    st.markdown(' - #### Deux couches suivantes de filtres égaux à 32 et de noyaux de 5 et 3')
    st.markdown(' - #### Normalisation du batch et pooling')
    st.markdown(' - #### Deux couches suivantes de filtres égaux à 16 et de noyaux de 5 et 3')
    st.markdown(' - #### Applatissement')
    st.markdown(' - #### Réduction d\'unité de réseau Dense jusqu\'à 5 unité pour 5 catégories en sortie')
    st.markdown('---')
    st.markdown('## Fonctionnement du réseau convolutif 1D')
    image1 = Image.open('pages/img/global_understand.jpg')
    st.image(image1, caption='Architecture pour un réseau de convolution 1D')
    image1 = Image.open('pages/img/a-Simple-scheme-of-a-one-dimension-1D-convolutional-operation-b-Full.jpg')
    st.image(image1, caption='Fonctionnement pour un réseau de convolution 1D')
    import pandas as pd
    import numpy as np
    import keras
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from keras.utils import to_categorical

    df1 = pd.read_csv('pages/archive/mitbih_test.csv', header=None)
    df2 = pd.read_csv('pages/archive/mitbih_train.csv', header=None)
    df = pd.concat([df1, df2], ignore_index=True, sort=False)
    df.isnull().sum().sum()
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=28)
    smote = SMOTE()
    X_sm, y_sm = smote.fit_resample(X_train, y_train)
    scaler = StandardScaler()
    X_sm = scaler.fit_transform(X_sm)
    X_test = scaler.transform(X_test)
    y_sm = to_categorical(y_sm)
    y_test = to_categorical(y_test)
    X_train_cnn = np.reshape(X_sm, (X_sm.shape[0], X_sm.shape[1], 1))
    X_test_cnn = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


    def get_classification_report(y_test, y_pred):
        from sklearn import metrics
        report = classification_report_imbalanced(y_test, y_pred, output_dict=True)
        df_classification_report = pd.DataFrame(report).transpose()
        return df_classification_report


    model_dir = 'pages/modele/model_mycnn_mitbih_modelbest_128.h5'
    model_cnn_opt = keras.models.load_model(model_dir)
    batch_size = 32
    verbose = 1
    _, mse, accuracy = model_cnn_opt.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
    st.write('mse', 'accuracy', mse, accuracy)
    model_cnn_opt.summary(print_fn=lambda x: st.text(x))
    st.subheader("Matrice de confusion")
    pred_class_a = model_cnn_opt.predict(X_test).argmax(axis=1)
    rounded_labels_a = np.argmax(y_test, axis=1)
    get_classification_report(rounded_labels_a, pred_class_a)
    st.table(pd.crosstab(rounded_labels_a, pred_class_a, rownames=["reel"], colnames=["predict"]))
    metrics_df = get_classification_report(rounded_labels_a, pred_class_a)
    st.table(metrics_df)

    cb4 = st.checkbox('ligne  101056,101057,101058 en prediction')
    cb7 = st.checkbox('ligne  100056,100057,100058 en prediction')
    cb8 = st.checkbox('ligne  95056,95057,95058 en prediction')
    cb9 = st.checkbox('ligne  102556,102557,102558 en prediction')
    if cb4:
        file_1 = df.iloc[103056:103059, :-1]
        target_file_1 = df.iloc[103056, -1]
        target_file_2 = df.iloc[103057, -1]
        target_file_3 = df.iloc[103058, -1]
        x_file = scaler.transform(file_1)
        pred_X = model_cnn_opt.predict(x_file).argmax(axis = 1)
        st.write('ligne  103056,103057,103058 en prediction:')
        res_1 = target_file_1
        res_2 = target_file_2
        res_3 = target_file_3
        st.write('103056 prediction VS valeur réelle', pred_X[0], res_1)
        st.write('103057 prediction VS valeur réelle', pred_X[1], res_2)
        st.write('103058 prediction VS valeur réelle', pred_X[2], res_3)
    if cb7:
        file_1 = df.iloc[100056:100059, :-1]
        target_file_1 = df.iloc[100056, -1]
        target_file_2 = df.iloc[100057, -1]
        target_file_3 = df.iloc[100058, -1]
        x_file = scaler.transform(file_1)
        pred_X = model_cnn_opt.predict(x_file).argmax(axis=1)
        st.write('ligne  100056,100057,100058 en prediction:')
        res_1 = target_file_1
        res_2 = target_file_2
        res_3 = target_file_3
        st.write('100056 prediction VS valeur réelle', pred_X[0], res_1)
        st.write('100057 prediction VS valeur réelle', pred_X[1], res_2)
        st.write('100058 prediction VS valeur réelle', pred_X[2], res_3)
    if cb8:
        file_1 = df.iloc[95056:95059, :-1]
        target_file_1 = df.iloc[95056, -1]
        target_file_2 = df.iloc[95057, -1]
        target_file_3 = df.iloc[95058, -1]
        x_file = scaler.transform(file_1)
        pred_X = model_cnn_opt.predict(x_file).argmax(axis=1)
        st.write('ligne  95056,95057,95058 en prediction:')
        res_1 = target_file_1
        res_2 = target_file_2
        res_3 = target_file_3
        st.write('95056 prediction VS valeur réelle', pred_X[0], res_1)
        st.write('95057 prediction VS valeur réelle', pred_X[1], res_2)
        st.write('95058 prediction VS valeur réelle', pred_X[2], res_3)
        st.markdown( ' - Le modèle pour mitbih ne répond pas au exigences critiques nécessaires pour détecter arythmies cardiaque')
    if cb9:
        file_1 = df.iloc[102556:102559, :-1]
        target_file_1 = df.iloc[102556, -1]
        target_file_2 = df.iloc[102557, -1]
        target_file_3 = df.iloc[102558, -1]
        x_file = scaler.transform(file_1)
        pred_X = model_cnn_opt.predict(x_file).argmax(axis = 1)
        st.write('ligne  102556,102557,102558 en prediction:')
        res_1 = target_file_1
        res_2 = target_file_2
        res_3 = target_file_3
        st.write('102556 prediction VS valeur réelle', pred_X[0], res_1)
        st.write('102557 prediction VS valeur réelle', pred_X[1], res_2)
        st.write('102558 prediction VS valeur réelle', pred_X[2], res_3)
if cb3:
    from imblearn.over_sampling import SMOTE
    from imblearn.metrics import classification_report_imbalanced, geometric_mean_score
    df1=pd.read_csv('pages/archive/ptbdb_abnormal.csv',header=None)
    df2=pd.read_csv('pages/archive/ptbdb_normal.csv',header=None)


    df=pd.concat([df1,df2],ignore_index=True,sort=False)
    df.isnull().sum().sum()
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=28)
    import imblearn
    import keras
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix
    from keras.utils import to_categorical
    from PIL import Image
    # oversampling
    smote=SMOTE()
    X_sm_1, y_sm_1 = smote.fit_resample(X_train, y_train)
    scaler_1 = StandardScaler()
    X_sm_1 =scaler_1.fit_transform(X_sm_1)
    X_test_1 = scaler_1.transform(X_test)
    y_sm_1 = to_categorical(y_sm_1)
    y_test_1 = to_categorical(y_test)
    X_train_cnn = np.reshape(X_sm_1, (X_sm_1.shape[0], X_sm_1.shape[1], 1))
    X_test_cnn = np.reshape(X_test_1, (X_test_1.shape[0], X_test_1.shape[1], 1))
    def get_classification_report(y_test, y_pred):
        from sklearn import metrics
        report = classification_report_imbalanced(y_test, y_pred, output_dict=True)
        df_classification_report = pd.DataFrame(report).transpose()
        return df_classification_report
    model_dir = 'pages/modele/model_mycnn_ptdb_2.h5'
    model_cnn_opt = keras.models.load_model(model_dir)
    batch_size=32
    verbose=1
    _, mse, accuracy = model_cnn_opt.evaluate(X_test_1, y_test_1, batch_size=batch_size, verbose=verbose)
    st.markdown('# Modèle  le plus élevé qualitativement ')
    st.markdown('## Jeu de données infarctus du myocarde')
    st.markdown('### Réseau de neurones convolutif 1D ')
    st.markdown('#### Deux premières couches de convolution ')
    st.markdown(' - Deux filtre consecutif de 64 et respectivement un noyau de 5 puis de 3')
    st.markdown(' - Deux couches dense pour arriver à 2 unités en sortie de modèles, une par catégorie.')
    #st.subheader('mse:', mse.astype(float))
    #st.subheader('accuracy:', accuracy.astype(float))

    model_cnn_opt.summary(print_fn=lambda x: st.text(x))
    st.subheader("Matrice de confusion")
    pred_class_a = model_cnn_opt.predict(X_test_1).argmax(axis=1)
    rounded_labels_a = np.argmax(y_test_1, axis=1)
    get_classification_report(rounded_labels_a, pred_class_a)
    st.table(pd.crosstab(rounded_labels_a, pred_class_a, rownames=["reel"], colnames=["predict"]))
    metrics_df = get_classification_report(rounded_labels_a, pred_class_a)
    st.table(metrics_df)

    image1 = Image.open('pages/img/acc_model_ptb.png')
    st.image(image1, caption='accuracy model ptbdb')

    image1 = Image.open('pages/img/mse_model_ptb.png')
    st.image(image1, caption='mse model ptbdb')
    cb5 = st.checkbox('ligne df initial 14456,14457,14458 en prediction')
    cb6 = st.checkbox('ligne df initial 56, 57, 58 en prediction')
    if cb5:
        file_1 = df.iloc[14496:14499, :-1]
        target_file_1 = df.iloc[14496, -1]
        target_file_2 = df.iloc[14497, -1]
        target_file_3 = df.iloc[14498, -1]
        x_file = scaler_1.transform(file_1)
        pred_X = model_cnn_opt.predict(x_file).argmax(axis=1)
        st.write('ligne df initiale 14056,14057,14058 en prediction:')
        res_1 = target_file_1
        res_2 = target_file_2
        res_3 = target_file_3
        st.write('14056 prediction VS valeur réelle', pred_X[0], res_1)
        st.write('14057 prediction VS valeur réelle', pred_X[1], res_2)
        st.write('14058 prediction VS valeur réelle', pred_X[2], res_3)
    if cb6:
        file_1b = df.iloc[56:59, :-1]
        target_file_1b = df.iloc[56, -1]
        target_file_2b = df.iloc[57, -1]
        target_file_3b = df.iloc[58, -1]
        x_fileb = scaler_1.transform(file_1b)
        pred_Xb = model_cnn_opt.predict(x_fileb).argmax(axis=1)
        st.write('ligne 56,57,58 en prediction:')
        res_1b = target_file_1b
        res_2b = target_file_2b
        res_3b = target_file_3b
        st.write('56 prediction VS valeur réelle', pred_Xb[0], res_1b)
        st.write('57 prediction VS valeur réelle', pred_Xb[1], res_2b)
        st.write('58 prediction VS valeur réelle', pred_Xb[2], res_3b)
    st.markdown(' - ***Le modèle pour la detection d\'infarctus du myocarde répond aux  exigences d’un statut critique en cadre de soin ou prévention***')
