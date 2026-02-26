"""
Evaluation script for ProctorGuard Hybrid 3.3
Reads gaze_log.csv and evaluates predictions.
Includes optimal threshold search, F1 vs threshold plot, and confidence distribution.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt


LOG_FILE = "gaze_log.csv"


def load_data():
    df = pd.read_csv(LOG_FILE)
    df = df[df["status"].isin(["INSIDE", "OUTSIDE"])]

    df["pred"] = df["status"].map({
        "INSIDE": 0,
        "OUTSIDE": 1
    })

    y_true = df["ground_truth"].values
    y_pred = df["pred"].values
    confidence = df["confidence"].values

    return y_true, y_pred, confidence


def find_best_threshold(y_true, confidence):
    best_f1 = 0
    best_t = 0.3
    for t in np.arange(0.01, 0.99, 0.005):
        y_pred = (confidence < t).astype(int)
        if len(np.unique(y_pred)) < 2:
            continue
        score = f1_score(y_true, y_pred)
        if score > best_f1:
            best_f1 = score
            best_t = t
    return best_t, best_f1


def evaluate(y_true, y_pred, confidence):

    print("\n===== ProctorGuard Evaluation =====")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Confidence distribution analysis
    inside_conf = confidence[y_true == 0]
    outside_conf = confidence[y_true == 1]
    print(f"\n--- Confidence Distribution ---")
    print(f"INSIDE  samples: mean={np.mean(inside_conf):.4f}, "
          f"median={np.median(inside_conf):.4f}, "
          f"std={np.std(inside_conf):.4f}, "
          f"min={np.min(inside_conf):.4f}, max={np.max(inside_conf):.4f}")
    print(f"OUTSIDE samples: mean={np.mean(outside_conf):.4f}, "
          f"median={np.median(outside_conf):.4f}, "
          f"std={np.std(outside_conf):.4f}, "
          f"min={np.min(outside_conf):.4f}, max={np.max(outside_conf):.4f}")

    fpr, tpr, _ = roc_curve(y_true, 1 - confidence)
    roc_auc = auc(fpr, tpr)
    print(f"\nAUC: {roc_auc:.4f}")

    best_t, best_f1 = find_best_threshold(y_true, confidence)
    print(f"\n--- Optimal Threshold Search ---")
    print(f"Best threshold : {best_t:.3f}")
    print(f"Best F1 at threshold: {best_f1:.4f}")

    y_pred_opt = (confidence < best_t).astype(int)
    cm_opt = confusion_matrix(y_true, y_pred_opt)
    print(f"Confusion Matrix at optimal threshold:")
    print(cm_opt)
    print(f"Accuracy at optimal: {accuracy_score(y_true, y_pred_opt):.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # ROC curve
    axes[0].plot(fpr, tpr)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Curve (AUC={roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)

    # F1 vs threshold
    thresholds = np.arange(0.01, 0.99, 0.005)
    f1_scores = []
    for t in thresholds:
        yp = (confidence < t).astype(int)
        if len(np.unique(yp)) < 2:
            f1_scores.append(0)
        else:
            f1_scores.append(f1_score(y_true, yp))

    axes[1].plot(thresholds, f1_scores)
    axes[1].axvline(x=best_t, color='r', linestyle='--', label=f'Best t={best_t:.3f}')
    axes[1].set_xlabel("Confidence Threshold")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("F1 vs Confidence Threshold")
    axes[1].legend()

    # Confidence distribution histogram
    axes[2].hist(inside_conf, bins=50, alpha=0.6, label="INSIDE (GT)", color="green", density=True)
    axes[2].hist(outside_conf, bins=50, alpha=0.6, label="OUTSIDE (GT)", color="red", density=True)
    axes[2].axvline(x=best_t, color='black', linestyle='--', label=f'Optimal t={best_t:.3f}')
    axes[2].set_xlabel("Confidence")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Confidence Distribution by Ground Truth")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    y_true, y_pred, confidence = load_data()
    evaluate(y_true, y_pred, confidence)
