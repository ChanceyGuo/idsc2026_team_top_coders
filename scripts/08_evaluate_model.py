import os
import argparse
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }

    return y_pred, metrics


def save_confusion_matrix(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            current_value = str(cm[i, j])
            plt.text(j, i, current_value, ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_roc_curve(y_true, y_prob, title, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Classifier (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate (Positive label: 1)")
    plt.ylabel("True Positive Rate (Positive label: 1)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def evaluate_model(model_name, model, df, feature_cols, split_name, threshold=0.5):
    X = df[feature_cols]
    y_true = df["brugada"]

    y_prob = model.predict_proba(X)[:, 1]
    y_pred, metrics = compute_metrics(y_true, y_prob, threshold=threshold)

    metrics_row = {
        "model": model_name,
        "split": split_name,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
    }

    pred_df = pd.DataFrame({
        "patient_id": df["patient_id"],
        "true_label": y_true,
        "pred_label": y_pred,
        "pred_prob": y_prob,
    })

    wrong_mask = pred_df["true_label"] != pred_df["pred_label"]
    misclassified_df = pred_df[wrong_mask].copy()

    return metrics_row, pred_df, misclassified_df, y_true, y_pred, y_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    args = parser.parse_args()

    os.makedirs("outputs/results", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    train_df = pd.read_csv("outputs/results/features_train.csv")
    val_df = pd.read_csv("outputs/results/features_val.csv")
    test_df = pd.read_csv("outputs/results/features_test.csv")

    feature_cols = []
    for c in train_df.columns:
        if c.startswith("f"):
            feature_cols.append(c)

    models = {
        "lr_unweighted": joblib.load("models/lr_unweighted.pkl"),
        "lr_balanced": joblib.load("models/lr_balanced.pkl"),
        "rf": joblib.load("models/rf.pkl"),
    }

    summary_val = []
    summary_test = []

    print("=" * 60)
    print("08_evaluate_model.py - Evaluate baseline models")
    print("=" * 60)
    print("Threshold:", args.threshold)
    print("-" * 60)

    for model_name, model in models.items():
        print(f"Evaluating {model_name} ...")

        metrics_val, pred_val, mis_val, y_val, ypred_val, yprob_val = evaluate_model(
            model_name,
            model,
            val_df,
            feature_cols,
            "val",
            threshold=args.threshold,
        )
        summary_val.append(metrics_val)

        metrics_val_df = pd.DataFrame([metrics_val])
        metrics_val_df.to_csv(f"outputs/results/metrics_{model_name}_val.csv", index=False)

        save_confusion_matrix(
            y_val,
            ypred_val,
            f"Confusion Matrix - {model_name} - Val",
            f"outputs/figures/cm_{model_name}_val.png",
        )

        save_roc_curve(
            y_val,
            yprob_val,
            f"ROC Curve - {model_name} - Val",
            f"outputs/figures/roc_{model_name}_val.png",
        )

        metrics_test, pred_test, mis_test, y_test, ypred_test, yprob_test = evaluate_model(
            model_name,
            model,
            test_df,
            feature_cols,
            "test",
            threshold=args.threshold,
        )
        summary_test.append(metrics_test)

        metrics_test_df = pd.DataFrame([metrics_test])
        metrics_test_df.to_csv(f"outputs/results/metrics_{model_name}_test.csv", index=False)

        pred_test.to_csv(f"outputs/results/predictions_{model_name}_test.csv", index=False)
        mis_test.to_csv(f"outputs/results/misclassified_{model_name}_test.csv", index=False)

        save_confusion_matrix(
            y_test,
            ypred_test,
            f"Confusion Matrix - {model_name} - Test",
            f"outputs/figures/cm_{model_name}_test.png",
        )

        save_roc_curve(
            y_test,
            yprob_test,
            f"ROC Curve - {model_name} - Test",
            f"outputs/figures/roc_{model_name}_test.png",
        )

    summary_val_df = pd.DataFrame(summary_val)
    summary_test_df = pd.DataFrame(summary_test)

    summary_val_df.to_csv("outputs/results/experiment_summary_val.csv", index=False)
    summary_test_df.to_csv("outputs/results/experiment_summary_test.csv", index=False)

    print("-" * 60)
    print("Saved: outputs/results/experiment_summary_val.csv")
    print("Saved: outputs/results/experiment_summary_test.csv")
    print("OK: 08_evaluate_model completed.")


if __name__ == "__main__":
    main()
