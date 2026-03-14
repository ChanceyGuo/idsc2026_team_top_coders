import os
import streamlit as st
import pandas as pd
import numpy as np
import wfdb
import joblib


# Config
st.set_page_config(page_title="Brugada ECG Demo", layout="wide")

DEFAULT_DATA_ROOT = r"E:\idsc2026\dataset"
DEFAULT_MODEL_PATH = r"E:\idsc2026\idsc2026_team_top_coders\models\lr_balanced.pkl"


# Helper functions
def zscore_per_lead(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    x = (x - mean) / (std + 1e-8)
    return x


def clip_signal(x: np.ndarray, low: float = -5.0, high: float = 5.0) -> np.ndarray:
    x = np.clip(x, low, high)
    return x


def load_and_preprocess(patient_id: str, data_root: str):
    files_dir = os.path.join(data_root, "files", "files")
    record_path = os.path.join(files_dir, patient_id, patient_id)

    record = wfdb.rdrecord(record_path)
    signals = record.p_signal

    x = signals.T.astype(np.float32)
    x = zscore_per_lead(x)
    x = clip_signal(x)

    return x


def extract_features(x: np.ndarray) -> np.ndarray:
    feats = []

    for i in range(x.shape[0]):
        lead = x[i]
        feats.append(lead.mean())
        feats.append(lead.std())
        feats.append(lead.min())
        feats.append(lead.max())

    feats = np.array(feats, dtype=np.float32)
    feats = feats.reshape(1, -1)

    return feats


def load_metadata(data_root: str) -> pd.DataFrame:
    meta_path = os.path.join(data_root, "metadata.csv")
    meta = pd.read_csv(meta_path, dtype={"patient_id": str})
    return meta


# Main
st.title("Brugada ECG Screening Demo")

data_root = DEFAULT_DATA_ROOT
model_path = DEFAULT_MODEL_PATH

meta = load_metadata(data_root)
model = joblib.load(model_path)

patient_ids = meta["patient_id"].tolist()
selected_pid = st.selectbox("Select patient_id", patient_ids)

if st.button("Run Prediction"):
    x = load_and_preprocess(selected_pid, data_root)
    features = extract_features(x)

    prob = float(model.predict_proba(features)[0, 1])

    if prob >= 0.5:
        pred_text = "Brugada-like"
    else:
        pred_text = "Normal-like"

    st.write(f"**Patient ID:** {selected_pid}")
    st.write(f"**Predicted Probability:** {prob:.4f}")
    st.write(f"**Predicted Class:** {pred_text}")
