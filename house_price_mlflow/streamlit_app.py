import streamlit as st
import pandas as pd
import mlflow.pyfunc

st.title("🏠 House Price Prediction")

features = ['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']
inputs = {f: st.number_input(f, value=5.0) for f in features}
df = pd.DataFrame([inputs])

if st.button("Predict"):
    model = mlflow.pyfunc.load_model("models:/Best Randomforest Model/Production")
    prediction = model.predict(df)[0]
    st.success(f"Predicted House Price: ${prediction * 100000:.2f}")
