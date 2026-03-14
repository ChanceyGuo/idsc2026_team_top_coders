import os
import argparse
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)

    # load feature tables
    train_df = pd.read_csv("outputs/results/features_train.csv")
    val_df = pd.read_csv("outputs/results/features_val.csv")
    test_df = pd.read_csv("outputs/results/features_test.csv")

    feature_cols = [c for c in train_df.columns if c.startswith("f")]

    X_train = train_df[feature_cols]
    y_train = train_df["brugada"]

    X_val = val_df[feature_cols]
    y_val = val_df["brugada"]

    X_test = test_df[feature_cols]
    y_test = test_df["brugada"]

    print("=" * 60)
    print("07_train_model.py - Train baseline models")
    print("=" * 60)
    print("Train shape:", X_train.shape)
    print("Val shape:", X_val.shape)
    print("Test shape:", X_test.shape)
    print("Feature count:", len(feature_cols))
    print("-" * 60)

    # 1) Logistic Regression (unweighted)
    lr_unweighted = LogisticRegression(
        random_state=args.seed,
        max_iter=1000
    )
    lr_unweighted.fit(X_train, y_train)
    joblib.dump(lr_unweighted, "models/lr_unweighted.pkl")
    print("Saved: models/lr_unweighted.pkl")

    # 2) Logistic Regression (balanced)
    lr_balanced = LogisticRegression(
        random_state=args.seed,
        max_iter=1000,
        class_weight="balanced"
    )
    lr_balanced.fit(X_train, y_train)
    joblib.dump(lr_balanced, "models/lr_balanced.pkl")
    print("Saved: models/lr_balanced.pkl")

    # 3) Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=args.seed,
        class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, "models/rf.pkl")
    print("Saved: models/rf.pkl")

    print("-" * 60)
    print("OK: 07_train_model completed.")

if __name__ == "__main__":
    main()