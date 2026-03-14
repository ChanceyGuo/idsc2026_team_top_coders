import os
import argparse
import pandas as pd
import wfdb
import numpy as np
import matplotlib.pyplot as plt

def zscore_per_lead(x: np.ndarray) -> np.ndarray:
    """
    x shape: (12, 1200)
    """
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / (std + 1e-8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Kaggle dataset root path")
    parser.add_argument("--patient_id", default=None, help="Optional patient_id to preprocess")
    args = parser.parse_args()

    root = args.root.rstrip("/")
    metadata_csv = os.path.join(root, "metadata.csv")
    files_dir = os.path.join(root, "files", "files")

    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)

    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"metadata.csv not found: {metadata_csv}")
    if not os.path.exists(files_dir):
        raise FileNotFoundError(f"files_dir not found: {files_dir}")

    meta = pd.read_csv(metadata_csv)

    if args.patient_id is None:
        patient_id = str(meta.loc[0, "patient_id"])
    else:
        patient_id = str(args.patient_id)

    print("=" * 60)
    print("04_preprocess_signal.py - Basic ECG preprocessing")
    print("=" * 60)
    print("Chosen patient_id:", patient_id)
    print("-" * 60)

    record = wfdb.rdrecord(f"{files_dir}/{patient_id}/{patient_id}")
    signals = record.p_signal   # (1200, 12)

    if signals.shape != (1200, 12):
        raise ValueError(f"Unexpected shape: {signals.shape}")

    # transpose to (12, 1200)
    x = signals.T.astype(np.float32)

    # z-score normalization per lead
    x_norm = zscore_per_lead(x)

    print("original shape:", signals.shape)
    print("transposed shape:", x.shape)
    print("normalized shape:", x_norm.shape)
    print("dtype:", x_norm.dtype)

    # save a quick plot of first 3 leads after normalization
    plt.figure(figsize=(12, 5))
    for i in range(3):
        plt.plot(x_norm[i], label=f"Lead {i+1}", linewidth=1)
    plt.title(f"Normalized ECG (first 3 leads) - patient {patient_id}")
    plt.xlabel("Time Step")
    plt.ylabel("Normalized Signal")
    plt.legend()
    plt.tight_layout()

    out_png = f"outputs/figures/preprocessed_ecg_{patient_id}.png"
    plt.savefig(out_png, dpi=150)
    plt.close()

    print("Saved:", out_png)
    print("OK: 04_preprocess_signal completed.")

if __name__ == "__main__":
    main()