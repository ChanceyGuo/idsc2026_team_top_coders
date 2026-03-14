import os
import argparse
import pandas as pd
import wfdb
import numpy as np
import matplotlib.pyplot as plt


def zscore_per_lead(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / (std + 1e-8)


def clip_signal(x: np.ndarray, low: float = -5.0, high: float = 5.0) -> np.ndarray:
    return np.clip(x, low, high)


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
    print("05_preprocess_signal.py - Basic ECG preprocessing")
    print("=" * 60)
    print("Chosen patient_id:", patient_id)
    print("-" * 60)

    record = wfdb.rdrecord(f"{files_dir}/{patient_id}/{patient_id}")
    signals = record.p_signal  # expected: (1200, 12)

    if signals.shape != (1200, 12):
        raise ValueError(f"Unexpected shape: {signals.shape}")

    # transpose to (12, 1200)
    x = signals.T.astype(np.float32)

    # simple preprocessing
    x_norm = zscore_per_lead(x)
    x_proc = clip_signal(x_norm, -5.0, 5.0)

    print("original shape:", signals.shape)
    print("transposed shape:", x.shape)
    print("normalized shape:", x_norm.shape)
    print("processed shape:", x_proc.shape)
    print("dtype:", x_proc.dtype)

    # save a quick visualization of first 3 leads after preprocessing
    plt.figure(figsize=(12, 5))
    for i in range(3):
        plt.plot(x_proc[i], label=f"Lead {i+1}", linewidth=1)
    plt.title(f"Preprocessed ECG (first 3 leads) - patient {patient_id}")
    plt.xlabel("Time Step")
    plt.ylabel("Processed Signal")
    plt.legend()
    plt.tight_layout()

    out_png = f"outputs/figures/preprocessed_ecg_{patient_id}.png"
    plt.savefig(out_png, dpi=150)
    plt.close()

    print("Saved:", out_png)
    print("OK: 05_preprocess_signal completed.")


if __name__ == "__main__":
    main()