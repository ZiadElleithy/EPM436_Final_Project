# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Classifier & Regressor App", layout="wide")
st.title("🔍 Machine Learning Model Explorer – Classification & Regression")

# Sidebar selection
st.sidebar.header("🧭 Navigation")
task_type = st.sidebar.radio("Select Task Type", ["Classification", "Regression"])

# Model and pkl mapping
model_map = {
    "Classification": {
        "Naive Bayes": "naive_bayes_classifier.pkl",
        "Random Forest": "random_forest_classifier.pkl",
        "Neural Network": "neural_network_classifier.pkl"
    },
    "Regression": {
        "Linear Regression": "linear_regression_regressor.pkl",
        "KNN Regressor": "knn_regressor_regressor.pkl",
        "Neural Network": "neural_network_regressor.pkl"
    }
}

# Select model
model_choice = st.sidebar.selectbox("Choose Model", list(model_map[task_type].keys()))

# Option: Predict from form or batch CSV
mode = st.sidebar.radio("Prediction Input Method", ["Single Input", "Upload CSV"])

# Load model
model_path = model_map[task_type][model_choice]
model = joblib.load(model_path)

# Load corresponding scaler
scaler_path = "scaler_class.pkl" if task_type == "Classification" else "scaler_reg.pkl"
scaler = joblib.load(scaler_path)

# Define input schema for demonstration (customize based on your dataset)
example_inputs = {
    "Classification (in mm)": ["Banknote Length", "Banknote Height_Left", "Banknote Height_Right", "Banknote Height_Diagonal"],
    "Regression": ["Daily Screen-Time (hrs)", "Hours spent on TikTok", "The Number of Social Media Platforms Used", "Daily Sleep Hours (hrs)"]
}

feature_names = example_inputs[task_type]

st.subheader(f"🧪 {task_type} – {model_choice}")

if mode == "Single Input":
    st.markdown("### Enter feature values below:")
    user_input = []
    for feature in feature_names:
        val = st.number_input(f"{feature}", value=0.0, format="%0.2f")
        user_input.append(val)

    if st.button("Predict"):
        X = np.array(user_input).reshape(1, -1)
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)

        if task_type == "Classification":
            st.success(f"🔍 Predicted Class: {int(y_pred[0])}")
        else:
            st.success(f"🔍 Predicted Value: {y_pred[0]:.2f}")

else:
    st.markdown("### Upload a CSV file with feature columns")
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("✅ Uploaded Data Preview:", df.head())

        try:
            X = df[feature_names]
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)
            df["Prediction"] = y_pred
            st.success("✅ Prediction completed.")
            st.write(df.head())

            if task_type == "Regression":
                st.subheader("📊 Actual vs Predicted Plot")
                if "stress_level" in df.columns:
                    sns.scatterplot(x=df["stress_level"], y=df["Prediction"])
                    plt.xlabel("Actual Stress Level")
                    plt.ylabel("Predicted Stress Level")
                    plt.title("Actual vs Predicted")
                    st.pyplot(plt.gcf())
                    plt.clf()
            elif task_type == "Classification":
                st.subheader("📊 Prediction Distribution")
                pred_counts = df["Prediction"].value_counts()
                st.bar_chart(pred_counts)

        except Exception as e:
            st.error(f"Error during prediction: {e}")

st.sidebar.info("ℹ️ This app supports two modes: interactive form input or batch CSV prediction.\nMake sure your features match the trained model.")
