import streamlit as st
import pandas as pd
from model import train_model

# Train model
model, columns = train_model()

st.title("AI Lead Scoring System")

# Inputs
industry = st.selectbox("Industry", ["Tech", "Finance", "Healthcare", "Retail"])
company_size = st.number_input("Company Size", 1, 1000)
traffic = st.number_input("Website Traffic", 0, 10000)
engagement = st.slider("Engagement Score", 0, 100)

# Create input data
input_data = pd.DataFrame([{
    "Industry": industry,
    "CompanySize": company_size,
    "WebsiteTraffic": traffic,
    "EngagementScore": engagement
}])

# Convert to model format
input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=columns, fill_value=0)

# Predict
if st.button("Predict"):
    score = model.predict_proba(input_data)[0][1] * 100
    st.success(f"Conversion Probability: {score:.2f}%")

