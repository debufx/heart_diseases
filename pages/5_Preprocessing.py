import streamlit as st
st.markdown('## Objectifs principaux:'
            '- Proéminences,'
            '- Rééchantillonage'
            '- Standardisation')
st.subheader('Proéminence avec pour projet d\'informer des lots d\'image')
st.code(' def distrib_proeminence(df):')
st.code('   prominences = []')
st.code('   for i in range (len(df)):')
st.code('       x=df.iloc[i,:]')
st.code('        peaks, _ = signal.find_peaks(x)')
st.code('        pro=signal.peak_prominences(x, peaks)[0]')
st.code('        prominences.append(pro)')
st.code('    pro_S = pd.DataFrame(prominences)')
st.code('return pro_S ')

st.subheader('Re-échantillonage')
st.code('smote = SMOTE()')
st.code('X_sm, y_sm = smote.fit_resample(X_train, y_train)')

st.subheader('Standardisation')
st.code('from sklearn.preprocessing import StandardScaler')

st.code('scaler = StandardScaler()')
st.code('X_sm =scaler.fit_transform(X_sm)')
st.code('X_test = scaler.transform(X_test)')
