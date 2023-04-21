import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
import pickle

# loading the saved models
parkinsons_model = pickle.load(open(r'parkinsons_model1.pkl', 'rb'))

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

col1, col2, col3, col4= st.columns(4)  

with col1:
    fo = st.slider('MDVP:Fo(Hz)', -1.0, 1.0, -1.0)
    RAP = st.slider('MDVP:RAP', -1.0, 1.0, -1.0)
    HNR = st.slider('HNR', -1.0, 1.0, 1.0)
    spread2 = st.slider('spread2', -1.0, 1.0, -1.0)
      
with col2:
    fhi = st.slider('MDVP:Fhi(Hz)', -1.0, 1.0, -1.0)
    APQ3 = st.slider('Shimmer:APQ3', -1.0, 1.0, -1.0)
    RPDE = st.slider('RPDE', -1.0, 1.0, -1.0)
    D2 = st.slider('D2', -1.0, 1.0, -1.0)
        
with col3:
    flo = st.slider('MDVP:Flo(Hz)', -1.0, 1.0, 1.0)
    APQ = st.slider('MDVP:APQ', -1.0, 1.0, 0.0)
    DFA = st.slider('DFE', -1.0, 1.0, 1.0)
    
with col4:
    Jitter_Abs = st.slider('MDVP:Jitter(Abs)', -1.0, 1.0, 0.0)
    NHR = st.slider('NHR', -1.0, 1.0, -1.0)
    spread1 = st.slider('spread1', -1.0, 1.0, -1.0)
    

# predict
if st.button("Parkinson's Test Result"):
    features = [[fo, fhi, flo, Jitter_Abs, RAP,APQ3,APQ,NHR,HNR,RPDE,DFA,spread1,spread2,D2]]
    predictions = parkinsons_model.predict(features)

    if predictions[0] == 1:
        st.success("The person has Parkinson's disease")
    else:
        st.success("The person does not have Parkinson's disease")
