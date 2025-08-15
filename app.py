# app.py — Gradio Web App for SHD Classifier (Render-ready)

import gradio as gr
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
import os

# --- Define and register custom class used in joblib ---
class RemainderColsList(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X

import __main__
__main__.RemainderColsList = RemainderColsList

# --- Load model and metadata safely ---
model = joblib.load("final_model_pipeline.joblib")
metadata = joblib.load("metadata.joblib")

numeric_features = metadata.get("numeric_features", [])
categorical_features = metadata.get("categorical_features", [])

def fill_missing_features(df):
    # Derive 'rate_diff' if possible
    if 'rate_diff' not in df.columns:
        if 'ventricular_rate' in df.columns and 'atrial_rate' in df.columns:
            df['rate_diff'] = df['ventricular_rate'] - df['atrial_rate']

    # Fill other missing columns with default values or NaN
    for col in numeric_features + categorical_features:
        if col not in df.columns:
            df[col] = np.nan
    return df

def predict_single(**inputs):
    df = pd.DataFrame([inputs])
    df = fill_missing_features(df)
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    return f"{'SHD' if pred == 1 else 'No SHD'} (Prob: {prob:.2f})"

def predict_batch(file):
    df = pd.read_csv(file.name)
    df = fill_missing_features(df)
    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]
    df['Prediction'] = preds
    df['Probability'] = probs
    return df

# --- Gradio UI Setup ---
single_inputs = []
for col in numeric_features:
    single_inputs.append(gr.Number(label=col))
for col in categorical_features:
    single_inputs.append(gr.Dropdown(choices=["", "Unknown", "Yes", "No"], label=col))

single_interface = gr.Interface(
    fn=lambda *args: predict_single(**dict(zip(numeric_features + categorical_features, args))),
    inputs=single_inputs,
    outputs="text",
    title="🧠 SHD Classifier (Single Input)"
)

batch_interface = gr.Interface(
    fn=predict_batch,
    inputs=gr.File(label="Upload CSV"),
    outputs=gr.Dataframe(),
    title="📁 SHD Classifier (Batch Prediction)"
)

gr.TabbedInterface(
    [single_interface, batch_interface],
    ["Single Input", "Batch Prediction"]
).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
