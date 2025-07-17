# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 13:33:26 2025

@author: Shruti
"""

import pandas as pd
import streamlit as st
import time
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Load Data
data = pd.read_excel(r"C:\Users\Shruti\OneDrive\Desktop\Project Churn Prediction\Churn.xlsx", sheet_name="Churn (1)")
data = data.drop(columns=["Unnamed: 0"], errors='ignore')

# Define Features & Target
selected_features = [
    "intl.plan", "voice.plan", "customer.calls", "day.charge", "intl.charge",
    "eve.charge", "night.charge", "day.mins", "eve.mins", "night.mins", "state", "area.code"]

X = data[selected_features]
y = data["churn"].map({"no": 0, "yes": 1})

# Encode Categorical Columns
categorical_cols = ["intl.plan", "voice.plan", "state", "area.code"]
label_encoders = {}

for col in categorical_cols:
    X[col] = X[col].astype(str)
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

X = X.astype(float)

# Handle Missing Values
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Handle Imbalance using SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_imputed, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Build Pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=150, random_state=42))
])

# Train Model
pipeline.fit(X_train, y_train)

# Streamlit UI Enhancements
st.set_page_config(page_title="Churn Prediction", layout="wide")

# Dark Mode Toggle
dark_mode = st.sidebar.checkbox("🌙 Dark Mode")

if dark_mode:
    st.markdown("""
        <style>
        body { background-color: #121212; color: white; }
        .stButton>button { background-color: #FF5733; color: white; }
        </style>
    """, unsafe_allow_html=True)

# Custom CSS Styles
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #1e1e2e, #25273c);
        color: #ffffff;
        font-family: 'Arial', sans-serif;
    }
    .big-font {
        font-size:24px !important;
        font-weight:bold;
        color: #FFD700;
        text-shadow: 2px 2px 8px rgba(255, 255, 255, 0.5);
    }
    .stButton>button {
        background: linear-gradient(to right, #FF5733, #FF8D1A);
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 12px;
        border-radius: 10px;
        box-shadow: 2px 2px 12px rgba(255, 87, 51, 0.4);
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #FF8D1A, #FF5733);
        box-shadow: 2px 2px 20px rgba(255, 87, 51, 0.6);
        transform: scale(1.05);
    }
    .prediction-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        color: white;
        margin-top: 20px;
        animation: fadeIn 1.5s ease-in;
    }
    .churn-yes {
        background-color: #dc3545;
    }
    .churn-no {
        background-color: #28a745;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Churn Prediction App")
st.markdown("**<span class='big-font'>Enter customer details to predict churn probability.</span>**", unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.header("🔧 Adjust Inputs")
user_inputs = {}

for col in selected_features:
    if col in categorical_cols:
        user_inputs[col] = st.sidebar.selectbox(f"{col}", label_encoders[col].classes_)
    else:
        default_value = float(X[col].mean()) if col in X.columns else None
        user_inputs[col] = st.sidebar.slider(f"{col}", float(X[col].min()), float(X[col].max()), default_value)

if st.button("🚀 Predict Churn"):
    with st.spinner('🔍 Analyzing customer data...'):
        time.sleep(2)

    input_df = pd.DataFrame([user_inputs])

    for col in categorical_cols:
        input_df[col] = label_encoders[col].transform([input_df[col][0]])[0]

    input_df = input_df.astype(float)
    input_df = pd.DataFrame(imputer.transform(input_df), columns=selected_features)

    prediction = pipeline.predict(input_df)[0]
    prediction_proba = pipeline.predict_proba(input_df)[0]

    churn_text = "Yes" if prediction == 1 else "No"
    churn_class = "churn-yes" if prediction == 1 else "churn-no"

    st.markdown(f"""
        <div class="prediction-box {churn_class}">
            🔥 The Predicted Churn is: <strong>{churn_text}</strong>!
        </div>
    """, unsafe_allow_html=True)

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction_proba[1] * 100,
        title={'text': "Churn Probability (%)"},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "red" if prediction == 1 else "green"},
               'steps': [
                   {'range': [0, 50], 'color': "lightgreen"},
                   {'range': [50, 80], 'color': "orange"},
                   {'range': [80, 100], 'color': "red"}
               ]}))

    st.plotly_chart(fig, use_container_width=True)

    # Customer Insights
    st.subheader("📊 Customer Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="📞 Calls Made", value=f"{user_inputs['customer.calls']} calls")
        st.metric(label="🌍 Intl Plan", value=user_inputs["intl.plan"])
        st.metric(label="🎤 Voice Plan", value=user_inputs["voice.plan"])

    with col2:
        st.metric(label="💰 Total Charges", value=f"${user_inputs['day.charge'] + user_inputs['intl.charge']:.2f}")
        st.metric(label="📍 State", value=user_inputs["state"])
        st.metric(label="🔢 Area Code", value=user_inputs["area.code"])

    # Insight Report
    st.subheader("📜 Churn Insight Report")

    if prediction == 1:
        st.warning("⚠️ **High Churn Risk!** Consider offering loyalty rewards, better pricing plans, or enhanced customer service.")
    else:
        st.success("✅ **Low Churn Risk!** This customer is likely to stay.")

st.sidebar.info("🔦 **Tip:** Use the dark mode toggle for a different experience.")