import os
import argparse
import pandas as pd
import wfdb
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Kaggle dataset root path")
    parser.add_argument("--patient_id", default=None, help="Optional patient_id to load")
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
    print("02_explore_data.py - Read one ECG record")
    print("=" * 60)
    print("ROOT:", root)
    print("FILES_DIR:", files_dir)
    print("Chosen patient_id:", patient_id)
    print("-" * 60)

    record = wfdb.rdrecord(f"{files_dir}/{patient_id}/{patient_id}")
    signals = record.p_signal
    leads = record.sig_name
    fs = record.fs

    print("fs:", fs)
    print("signals shape:", signals.shape)
    print("leads:", leads)

    plt.figure(figsize=(12, 4))
    plt.plot(signals[:, 0])
    plt.title(f"Sample ECG - patient {patient_id} - lead {leads[0]}")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    out_png = f"outputs/figures/sample_ecg_{patient_id}.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    print("-" * 60)
    print("Saved:", out_png)
    print("OK: 02_explore_data completed.")

if __name__ == "__main__":
    main()