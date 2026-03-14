import os
import argparse
import pandas as pd
import wfdb
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Kaggle dataset root path")
    parser.add_argument("--patient_id", default=None, help="Optional patient_id to visualize")
    args = parser.parse_args()

    root = args.root.rstrip("/")
    metadata_csv = os.path.join(root, "metadata.csv")
    files_dir = os.path.join(root, "files", "files")

    os.makedirs("outputs/figures", exist_ok=True)

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
    print("03_visualize_signals.py - Visualize 12-lead ECG")
    print("=" * 60)
    print("Chosen patient_id:", patient_id)
    print("-" * 60)

    record = wfdb.rdrecord(f"{files_dir}/{patient_id}/{patient_id}")
    signals = record.p_signal   # (1200, 12)
    leads = record.sig_name
    fs = record.fs

    print("fs:", fs)
    print("signals shape:", signals.shape)

    fig, axes = plt.subplots(12, 1, figsize=(14, 20), sharex=True)

    for i in range(12):
        axes[i].plot(signals[:, i], linewidth=1)
        axes[i].set_ylabel(leads[i], rotation=0, labelpad=25, fontsize=9)
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Sample")
    fig.suptitle(f"12-Lead ECG Visualization - patient {patient_id}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    out_png = f"outputs/figures/visual_ecg_{patient_id}.png"
    plt.savefig(out_png, dpi=150)
    plt.close()

    print("Saved:", out_png)
    print("OK: 03_visualize_signals completed.")

if __name__ == "__main__":
    main()