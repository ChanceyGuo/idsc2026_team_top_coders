import os
import argparse
import json
import pandas as pd
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Kaggle dataset root path")
    args = parser.parse_args()

    root = args.root.rstrip("/")

    # expected paths
    metadata_csv = os.path.join(root, "metadata.csv")
    dict_csv = os.path.join(root, "metadata_dictionary.csv")

    # your dataset has files/files/
    files_dir = os.path.join(root, "files", "files")

    # outputs
    os.makedirs("outputs/results", exist_ok=True)

    print("=" * 60)
    print("01_load_data.py - Data path & metadata sanity check")
    print("=" * 60)
    print("ROOT:", root)
    print("METADATA:", metadata_csv)
    print("DICT:", dict_csv)
    print("FILES_DIR:", files_dir)
    print("-" * 60)

    # existence checks
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"metadata.csv not found: {metadata_csv}")
    if not os.path.exists(files_dir):
        raise FileNotFoundError(f"files_dir not found (expected files/files): {files_dir}")

    # count WFDB files
    hea_count = len(glob.glob(files_dir + "/**/*.hea", recursive=True))
    dat_count = len(glob.glob(files_dir + "/**/*.dat", recursive=True))

    # read metadata
    meta = pd.read_csv(metadata_csv)

    # label distribution (raw)
    raw_unique = sorted(meta["brugada"].dropna().unique().tolist())
    raw_counts = meta["brugada"].value_counts(dropna=False).to_dict()

    # binary mapping (for reporting later)
    brugada_bin = (meta["brugada"].astype(int) > 0).astype(int)
    bin_counts = brugada_bin.value_counts().to_dict()

    summary = {
        "n_rows": int(len(meta)),
        "hea_count": int(hea_count),
        "dat_count": int(dat_count),
        "raw_brugada_unique": raw_unique,
        "raw_brugada_counts": raw_counts,
        "binary_counts_(brugada>0)": bin_counts,
    }

    print("Rows:", summary["n_rows"])
    print("HEA count:", summary["hea_count"])
    print("DAT count:", summary["dat_count"])
    print("Raw brugada unique:", summary["raw_brugada_unique"])
    print("Raw brugada counts:", summary["raw_brugada_counts"])
    print("Binary counts (brugada>0):", summary["binary_counts_(brugada>0)"])

    out_path = "outputs/results/data_sanity_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("-" * 60)
    print("Saved:", out_path)
    print("OK: 01_load_data completed.")

if __name__ == "__main__":
    main()