import streamlit as st
import pandas as pd
import mlflow.pyfunc
import mlflow
from mlflow.tracking import MlflowClient
from project.config import TRACKING_URI, EXPERIMENT_NAME, MODEL_NAME

st.title("🏠 House Price Prediction")

features = ['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']
inputs = {f: st.number_input(f, value=5.0) for f in features}
df = pd.DataFrame([inputs])

if st.button("Predict"):
    client = MlflowClient()

    # Get all versions of the model
    latest_versions = client.get_latest_versions(MODEL_NAME)

    # Pick the highest version number
    latest = max(latest_versions, key=lambda v: int(v.version))

    # Show the latest version to the user
    st.write(f"📦 Using model version: {latest.version}")

    model_uri = f"models:/{MODEL_NAME}/{latest.version}"
    model = mlflow.pyfunc.load_model(model_uri)

    prediction = model.predict(df)[0]
    st.success(f"Predicted House Price: ${prediction * 100000:.2f}")
