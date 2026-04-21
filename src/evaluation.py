"""
evaluation.py  ── MoodMate Research Pipeline
═════════════════════════════════════════════
Full post-training evaluation:
  ✅ Confusion matrix (normalised + raw count)
  ✅ sklearn classification report
  ✅ Per-class accuracy bar chart (colour-coded ✅⚠️❌)
  ✅ Top-K accuracy (k=2, k=3)
  ✅ Weak-class spotlight (fear / disgust / sad)
  ✅ All plots saved to models/

Usage:
    python src/evaluation.py
  or automatically called by train_model.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocess import preprocess_fer_directory, EMOTION_LABELS, NUM_CLASSES

BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion_model.keras")
PLOT_DIR   = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data", "fer2013")


def plot_confusion_matrix(cm: np.ndarray, labels: list, save_dir: str):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Confusion Matrix — MoodMate Emotion Detection",
                 fontsize=15, fontweight="bold")

    for ax, data, title, fmt in zip(
        axes,
        [cm_norm, cm],
        ["Normalised (row %)", "Raw Counts"],
        [".2f", "d"],
    ):
        sns.heatmap(data, annot=True, fmt=fmt,
                    xticklabels=labels, yticklabels=labels,
                    cmap="Blues", linewidths=0.5, linecolor="white",
                    ax=ax, cbar=True)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("True Label", fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=130); plt.close()
    print(f"  📊 Confusion matrix → {path}")


def plot_per_class_accuracy(cm: np.ndarray, labels: list, save_dir: str):
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    colours = ["#2ecc71" if a >= 0.70 else "#f39c12" if a >= 0.50 else "#e74c3c"
               for a in per_class_acc]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, per_class_acc * 100, color=colours,
                  edgecolor="white", linewidth=0.8, zorder=3)

    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{acc*100:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylim(0, 108)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Per-Class Accuracy (green ≥70%, orange ≥50%, red <50%)",
                 fontsize=12, fontweight="bold")
    ax.axhline(80, color="#2ecc71", ls="-",  lw=1.5, alpha=0.7, label="80% goal")
    ax.axhline(70, color="green",   ls="--", lw=1.2, alpha=0.6, label="70% target")
    ax.axhline(50, color="orange",  ls="--", lw=1.2, alpha=0.6, label="50% baseline")
    ax.legend(fontsize=9)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path = os.path.join(save_dir, "per_class_accuracy.png")
    plt.savefig(path, dpi=130); plt.close()
    print(f"  📊 Per-class accuracy → {path}")


def run_full_evaluation(model=None, X_test=None, y_test=None):
    """
    Run the complete evaluation suite.

    Can be called:
      (a) from train_model.py — pass model, X_test, y_test directly
      (b) as standalone script — loads model + data from disk
    """
    import tensorflow as tf

    print("\n" + "═" * 60)
    print("  Full Model Evaluation — MoodMate")
    print("═" * 60)

    # ── Load model if not provided ────────────────────────────────────────────
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No model at {MODEL_PATH}. Run: python src/train_model.py"
            )
        from src.train_model import (
            StableFocalLoss, WarmUpCosineDecay,
            CBAM, ChannelAttention, SpatialAttention,
        )
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "StableFocalLoss":   StableFocalLoss,
                "WarmUpCosineDecay": WarmUpCosineDecay,
                "CBAM":              CBAM,
                "ChannelAttention":  ChannelAttention,
                "SpatialAttention":  SpatialAttention,
            },
        )
        print(f"✅ Model loaded: {MODEL_PATH}")

    # ── Load test data if not provided ───────────────────────────────────────
    if X_test is None or y_test is None:
        print("📂 Loading test data …")
        _, _, X_test, y_test = preprocess_fer_directory(DATA_DIR)
    print(f"   Test samples: {len(X_test)}")

    # ── Predictions — apply softmax to logit outputs ──────────────────────────
    print("\n⏳ Running predictions …")
    logits  = model.predict(X_test, batch_size=64, verbose=1)   # (N, 7) logits
    y_proba = tf.nn.softmax(logits).numpy()                     # (N, 7) probs
    y_pred  = y_proba.argmax(axis=1)

    # ── Accuracy metrics ──────────────────────────────────────────────────────
    acc  = float(np.mean(y_pred == y_test))
    top2 = top_k_accuracy_score(y_test, y_proba, k=2)
    top3 = top_k_accuracy_score(y_test, y_proba, k=3)

    print(f"\n{'─'*40}")
    print(f"  Overall Accuracy  : {acc*100:.2f}%")
    print(f"  Top-2  Accuracy   : {top2*100:.2f}%")
    print(f"  Top-3  Accuracy   : {top3*100:.2f}%")
    print(f"{'─'*40}")

    # ── Classification report ─────────────────────────────────────────────────
    print("\n📋 Classification Report:\n")
    report = classification_report(y_test, y_pred,
                                   target_names=EMOTION_LABELS, digits=4)
    print(report)

    report_path = os.path.join(PLOT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Overall Accuracy : {acc*100:.2f}%\n")
        f.write(f"Top-2 Accuracy   : {top2*100:.2f}%\n")
        f.write(f"Top-3 Accuracy   : {top3*100:.2f}%\n\n")
        f.write(report)
    print(f"  📄 Report saved → {report_path}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, EMOTION_LABELS, PLOT_DIR)
    plot_per_class_accuracy(cm, EMOTION_LABELS, PLOT_DIR)

    # ── Per-class breakdown with weak-class highlight ─────────────────────────
    print("\n📊 Per-class Accuracy Summary:")
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    weak_classes  = []
    for emo, acc_c in zip(EMOTION_LABELS, per_class_acc):
        if acc_c >= 0.70:
            flag = "✅"
        elif acc_c >= 0.50:
            flag = "⚠️ "
            weak_classes.append(emo)
        else:
            flag = "❌"
            weak_classes.append(emo)
        print(f"   {flag} {emo:10s} : {acc_c*100:.2f}%")

    if weak_classes:
        print(f"\n🔴 Weak classes (need attention): {', '.join(weak_classes)}")
        print("   Suggestions:")
        print("   • Increase class weight for these labels")
        print("   • Add targeted augmentation (heavier for weak classes)")
        print("   • Inspect confusion: are they mis-predicted as each other?")
    else:
        print("\n🎉 All classes above 70%!")

    print(f"\n✅ Evaluation complete → {PLOT_DIR}")
    return {
        "overall_accuracy": acc,
        "top2_accuracy":    top2,
        "top3_accuracy":    top3,
        "per_class":        dict(zip(EMOTION_LABELS, per_class_acc.tolist())),
        "confusion_matrix": cm,
        "weak_classes":     weak_classes,
    }


if __name__ == "__main__":
    results = run_full_evaluation()
    print(f"\nFinal accuracy: {results['overall_accuracy']*100:.2f}%")
