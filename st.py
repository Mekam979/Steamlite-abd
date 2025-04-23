import streamlit as st
import pickle
import pandas as pd


st.set_page_config(page_title="OFppt", page_icon=":guardsman:", layout="wide")
model_path = "random_forest_model.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)

st.title("predection en uturisation de modèle de ML")
CreditScore = st.number_input("CreditScore", min_value=0, max_value=10000)
Geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("gender",["male","female"])
Age = st.number_input("Age", min_value=0, max_value=100)
Tenure = st.number_input("Tenure", min_value=0, max_value=100)
Balance = st.number_input("Balance", min_value=0, max_value=1000000)
NumOfProducts = st.number_input("NumOfProducts", min_value=0, max_value=10)
HasCrCard = st.number_input("HasCrCard", min_value=0, max_value=1)
IsActiveMember = st.number_input("IsActiveMember", min_value=0, max_value=1)
EstimatedSalary = st.number_input("EstimatedSalary", min_value=0, max_value=1000000)


if st.button("Predict", key="predict"):
    input_data = {
        'CreditScore': CreditScore,
        'Age': Age,
        'Tenure': Tenure,
        'Balance': Balance,
        'NumOfProducts': NumOfProducts,
        'HasCrCard': HasCrCard,
        'IsActiveMember': IsActiveMember,
        'EstimatedSalary': EstimatedSalary,
        'Gender_Male': 1 if gender == 'male' else 0,
        'Geography_France': 1 if Geography == 'France' else 0,
        'Geography_Germany': 1 if Geography == 'Germany' else 0,
        'Geography_Spain': 1 if Geography == 'Spain' else 0,
    }

    input_df = pd.DataFrame([input_data])

    input_df = input_df.reindex(columns=feature_order)

    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][int(prediction)]

    st.subheader("Résultat de la prédiction :")
    if prediction == 1:
        st.success("Le client risque de **quitter** la banque avec une probabilité de ",prediction_proba)
    else:
        st.info("Le client va **rester** avec une probabilité de ", prediction_proba)