# Streamlit SHD Classification App

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import __main__

# Register custom transformers if used
class RemainderTransformer:
    def fit(self, X, y=None): return self
    def transform(self, X): return X
__main__.RemainderTransformer = RemainderTransformer

# Load model and metadata
model = joblib.load("final_model_pipeline.joblib")
metadata = joblib.load("metadata.joblib")

numeric_features = metadata["numeric_features"]
categorical_features = metadata["categorical_features"]
all_features = numeric_features + categorical_features

st.title("🔬 SHD Classification App")
st.markdown("Upload ECG metadata or enter values to predict SHD flag.")

# Sidebar for input
st.sidebar.header("Manual Input")

user_input = {}
for col in numeric_features:
    user_input[col] = st.sidebar.number_input(f"{col}", value=0.0)
for col in categorical_features:
    user_input[col] = st.sidebar.selectbox(f"{col}", options=["Unknown", "Normal", "Abnormal"])

input_df = pd.DataFrame([user_input])

# Prediction
if st.sidebar.button("Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    st.subheader("📊 Prediction Result")
    st.write(f"**Predicted SHD Flag:** {pred}")
    st.write(f"**Probability of SHD:** {prob:.2f}")

# Batch prediction from CSV
st.subheader("📁 Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file:
    batch_data = pd.read_csv(uploaded_file)
    batch_pred = model.predict(batch_data)
    batch_prob = model.predict_proba(batch_data)[:,1]
    result_df = batch_data.copy()
    result_df["Prediction"] = batch_pred
    result_df["Probability"] = batch_prob
    st.write(result_df.head())

    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions", data=csv, file_name="shd_predictions.csv")

# Feature importances
if hasattr(model.named_steps["classifier"], "feature_importances_"):
    st.subheader("📌 Feature Importances")
    importances = model.named_steps["classifier"].feature_importances_
    try:
        ohe = model.named_steps["preprocess"].transformers_[1][1].named_steps["encoder"]
        cat_names = ohe.get_feature_names_out(categorical_features)
    except:
        cat_names = categorical_features
    all_feature_names = numeric_features + list(cat_names)
    fi_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    st.bar_chart(fi_df.set_index("Feature").head(10))