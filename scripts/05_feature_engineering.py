import os
import argparse
import pandas as pd
import wfdb
import numpy as np

def zscore_per_lead(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / (std + 1e-8)

def load_and_preprocess(patient_id: str, files_dir: str) -> np.ndarray:
    record = wfdb.rdrecord(f"{files_dir}/{patient_id}/{patient_id}")
    signals = record.p_signal  # (1200, 12)

    if signals.shape != (1200, 12):
        raise ValueError(f"Unexpected shape for {patient_id}: {signals.shape}")

    x = signals.T.astype(np.float32)   # (12, 1200)
    x = zscore_per_lead(x)
    return x

def extract_features(x: np.ndarray) -> np.ndarray:
    """
    x shape: (12, 1200)
    return: 48-dim feature vector
    """
    feats = []
    for i in range(x.shape[0]):
        lead = x[i]
        feats.extend([
            lead.mean(),
            lead.std(),
            lead.min(),
            lead.max()
        ])
    return np.array(feats, dtype=np.float32)

def build_feature_table(df: pd.DataFrame, files_dir: str) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        patient_id = str(row["patient_id"])
        y = int(row["brugada"])

        x = load_and_preprocess(patient_id, files_dir)
        feats = extract_features(x)

        row_dict = {
            "patient_id": patient_id,
            "brugada": y
        }

        for j, val in enumerate(feats):
            row_dict[f"f{j:02d}"] = float(val)

        rows.append(row_dict)

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Kaggle dataset root path")
    args = parser.parse_args()

    root = args.root.rstrip("/")
    files_dir = os.path.join(root, "files", "files")

    os.makedirs("outputs/results", exist_ok=True)

    summary = pd.read_csv("outputs/results/data_summary.csv")
    train_ids = pd.read_csv("outputs/splits/train.csv")
    val_ids = pd.read_csv("outputs/splits/val.csv")
    test_ids = pd.read_csv("outputs/splits/test.csv")

    summary["patient_id"] = summary["patient_id"].astype(str)
    train_ids["patient_id"] = train_ids["patient_id"].astype(str)
    val_ids["patient_id"] = val_ids["patient_id"].astype(str)
    test_ids["patient_id"] = test_ids["patient_id"].astype(str)

    train_df = train_ids.merge(summary, on="patient_id", how="left")
    val_df = val_ids.merge(summary, on="patient_id", how="left")
    test_df = test_ids.merge(summary, on="patient_id", how="left")

    print("=" * 60)
    print("05_feature_engineering.py - Build feature tables")
    print("=" * 60)
    print("Train rows:", len(train_df))
    print("Val rows:", len(val_df))
    print("Test rows:", len(test_df))
    print("-" * 60)

    X_train = build_feature_table(train_df, files_dir)
    X_val = build_feature_table(val_df, files_dir)
    X_test = build_feature_table(test_df, files_dir)

    X_train.to_csv("outputs/results/features_train.csv", index=False)
    X_val.to_csv("outputs/results/features_val.csv", index=False)
    X_test.to_csv("outputs/results/features_test.csv", index=False)

    print("Saved: outputs/results/features_train.csv")
    print("Saved: outputs/results/features_val.csv")
    print("Saved: outputs/results/features_test.csv")
    print("Feature dimension:", X_train.shape[1] - 2)  # minus patient_id + label
    print("OK: 05_feature_engineering completed.")

if __name__ == "__main__":
    main()