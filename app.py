# app.py — Gradio Web App for SHD Classifier

import gradio as gr
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# --- Define and register custom class used in joblib ---
class _RemainderColsList(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X

import sklearn.compose._column_transformer as ct
ct._RemainderColsList = _RemainderColsList

# --- Load model and metadata safely ---
model = joblib.load("final_model_pipeline.joblib")
metadata = joblib.load("metadata.joblib")

numeric_features = metadata.get("numeric_features", [])
categorical_features = metadata.get("categorical_features", [])

def predict_single(**inputs):
    df = pd.DataFrame([inputs])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    return f"{'SHD' if pred == 1 else 'No SHD'} (Prob: {prob:.2f})"

def predict_batch(file):
    df = pd.read_csv(file.name)
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
