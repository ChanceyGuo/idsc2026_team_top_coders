import os
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Kaggle dataset root path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible split")
    args = parser.parse_args()

    root = args.root.rstrip("/")
    metadata_csv = os.path.join(root, "metadata.csv")

    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("outputs/splits", exist_ok=True)

    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"metadata.csv not found: {metadata_csv}")

    print("=" * 60)
    print("02_make_splits.py - Build binary label summary and train/val/test splits")
    print("=" * 60)
    print("ROOT:", root)
    print("METADATA:", metadata_csv)
    print("SEED:", args.seed)
    print("-" * 60)

    metadata = pd.read_csv(metadata_csv)

    # build index table
    data_summary = metadata[["patient_id", "brugada"]].copy()
    data_summary["patient_id"] = data_summary["patient_id"].astype(str)

    # binary mapping: 0 stays 0, 1/2 -> 1
    data_summary["brugada"] = (data_summary["brugada"].astype(int) > 0).astype(int)

    # save binary summary
    data_summary.to_csv("outputs/results/data_summary.csv", index=False)

    # random split
    shuffled = data_summary.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    n = len(shuffled)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    train = shuffled.head(n_train)[["patient_id"]]
    rest = shuffled.tail(len(shuffled) - n_train)
    val = rest.head(n_val)[["patient_id"]]
    test = rest.tail(len(rest) - n_val)[["patient_id"]]

    train.to_csv("outputs/splits/train.csv", index=False)
    val.to_csv("outputs/splits/val.csv", index=False)
    test.to_csv("outputs/splits/test.csv", index=False)

    # checks
    train_ids = set(train["patient_id"])
    val_ids = set(val["patient_id"])
    test_ids = set(test["patient_id"])

    print("Saved: outputs/results/data_summary.csv")
    print("Saved: outputs/splits/train.csv")
    print("Saved: outputs/splits/val.csv")
    print("Saved: outputs/splits/test.csv")
    print("-" * 60)
    print("Split sizes:", {"train": len(train), "val": len(val), "test": len(test)})
    print("Class counts:", {
        "total": len(data_summary),
        "brugada": int((data_summary["brugada"] == 1).sum()),
        "normal": int((data_summary["brugada"] == 0).sum())
    })
    print("Overlap train&val:", len(train_ids & val_ids))
    print("Overlap train&test:", len(train_ids & test_ids))
    print("Overlap val&test:", len(val_ids & test_ids))
    print("Unique total:", len(train_ids | val_ids | test_ids))
    print("OK: 02_make_splits completed.")

if __name__ == "__main__":
    main()