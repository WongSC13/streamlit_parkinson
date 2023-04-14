import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
import pickle

# loading the saved models
parkinsons_model = pickle.load(open(r'parkinsons_model.pkl', 'rb'))

st.write("""
# Simple Parkinson Prediction App
This app predicts if a patient has Parkinson's Disease!
""")

# CSV file upload
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

    # predictions
    predictions = parkinsons_model.predict(input_df)
    input_df['Prediction'] = predictions

    # show predictions
    st.write(input_df)


# show form
st.title("Parkinson's Disease Prediction using ML")

col1, col2, col3, col4, col5 = st.columns(5)  

with col1:
    fo = st.slider('MDVP:Fo(Hz)', -1.0, 1.0, 1.0)
    RAP = st.slider('MDVP:RAP', -1.0, 1.0, -1.0)
    APQ3 = st.slider('Shimmer:APQ3', -1.0, 1.0, -1.0)
    RPDE = st.slider('RPDE', -1.0, 1.0, -1.0)
    D2 = st.slider('D2', -1.0, 1.0, -1.0)
    
with col2:
    fhi = st.slider('MDVP:Fhi(Hz)', -1.0, 1.0, 1.0)
    PPQ = st.slider('MDVP:PPQ', -1.0, 1.0, -1.0)
    APQ5 = st.slider('Shimmer:APQ5', -1.0, 1.0, -1.0)
    DFA = st.slider('DFE', -1.0, 1.0, -1.0)
    PPE = st.slider('PPE', -1.0, 1.0, -1.0)
    
with col3:
    flo = st.slider('MDVP:Flo(Hz)', -1.0, 1.0, 0.0)
    DDP = st.slider('Jitter:DDP', -1.0, 1.0, 0.0)
    APQ = st.slider('MDVP:APQ', -1.0, 1.0, 0.0)
    spread1 = st.slider('spread1', -1.0, 1.0, 0.0)
    
with col4:
    Jitter_percent = st.slider('MDVP:Jitter(%)', -1.0, 1.0, -1.0)
    Shimmer = st.slider('MDVP:Shimmer', -1.0, 1.0, -1.0)
    DDA = st.slider('Shimmer:DDA', -1.0, 1.0, -1.0)
    spread2 = st.slider('spread2', -1.0, 1.0, -1.0)
    
with col5:
    Jitter_Abs = st.slider('MDVP:Jitter(Abs)', -1.0, 1.0, -1.0)
    Shimmer_dB = st.slider('MDVP:Shimmer(dB)', -1.0, 1.0, -1.0)
    NHR = st.slider('NHR', -1.0, 1.0, -1.0)
    HNR = st.slider('HNR', -1.0, 1.0, 1.0)

# predict
if st.button("Parkinson's Test Result"):
    features = [[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]]
    predictions = parkinsons_model.predict(features)

    if predictions[0] == 1:
        st.success("The person has Parkinson's disease")
    else:
        st.success("The person does not have Parkinson's disease")
