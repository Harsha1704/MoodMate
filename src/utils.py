"""
utils.py  ── MoodMate Research Pipeline
═════════════════════════════════════════
Shared utilities:
  • GPU / mixed-precision setup
  • Reproducibility seed
  • Model parameter summary
  • Prediction visualisation (single image)
  • Batch prediction with progress bar
  • Webcam demo loop (OpenCV)
"""

import os
import sys
import time
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ══════════════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        print(f"🎲 Global seed → {seed}")
    except ImportError:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# GPU / HARDWARE
# ══════════════════════════════════════════════════════════════════════════════

def configure_gpu():
    """Enable memory growth; log GPU availability."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"⚡ GPUs: {[g.name for g in gpus]}")
        else:
            print("⚠️  No GPU — CPU mode (slow for training).")
    except ImportError:
        print("TensorFlow not installed.")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════

def model_param_summary(model) -> dict:
    total     = model.count_params()
    trainable = sum(np.prod(v.shape) for v in model.trainable_weights)
    frozen    = total - trainable
    print(f"\n{'─'*40}")
    print(f"  Total      : {total:>12,}")
    print(f"  Trainable  : {trainable:>12,}")
    print(f"  Frozen     : {frozen:>12,}")
    print(f"{'─'*40}\n")
    return {"total": total, "trainable": int(trainable), "frozen": int(frozen)}


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def visualise_prediction(image_path: str, result: dict, save_path: str = None):
    """Bar chart of emotion scores alongside face image."""
    import matplotlib
    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    fig, (ax_img, ax_bar) = plt.subplots(
        1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 2]}
    )
    fig.patch.set_facecolor("#1a1a2e")

    img = PILImage.open(image_path).convert("RGB")
    ax_img.imshow(img); ax_img.axis("off")
    conf = result["confidence"] * 100
    ax_img.set_title(f"{result['emotion'].upper()}  ({conf:.1f}%)",
                     color="white", fontsize=13, fontweight="bold", pad=10)

    scores  = result["all_scores"]
    labels  = list(scores.keys())
    values  = [scores[l] * 100 for l in labels]
    colours = ["#e74c3c" if l == result["emotion"] else "#3498db" for l in labels]

    bars = ax_bar.barh(labels, values, color=colours, edgecolor="#1a1a2e",
                       linewidth=0.6)
    for bar, val in zip(bars, values):
        ax_bar.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=9, color="white")
    ax_bar.set_xlim(0, 105)
    ax_bar.set_xlabel("Confidence (%)", color="white")
    ax_bar.set_facecolor("#16213e")
    ax_bar.tick_params(colors="white")
    ax_bar.spines[["top", "right", "bottom"]].set_visible(False)
    ax_bar.spines["left"].set_color("#4a4a7a")
    ax_bar.set_title("Emotion Scores", color="white", fontsize=11)

    if result.get("low_confidence"):
        fig.text(0.5, 0.01, "⚠️  Low confidence — result may be unreliable",
                 ha="center", color="#f39c12", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  📸 Saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# BATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def batch_predict(image_paths: list, use_tta: bool = False) -> list:
    from src.emotion_predictor import predict_from_path
    n, results = len(image_paths), []
    print(f"\n🔄 Batch predicting {n} images …")
    t0 = time.time()
    for i, path in enumerate(image_paths, 1):
        try:
            res = predict_from_path(path, use_tta=use_tta)
        except Exception as exc:
            res = {"error": str(exc), "path": path}
        results.append(res)
        if i % max(1, n // 10) == 0 or i == n:
            fps = i / (time.time() - t0)
            print(f"   {i}/{n}  ({fps:.1f} img/s)")
    print(f"✅ Done in {time.time()-t0:.1f}s")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# WEBCAM DEMO
# ══════════════════════════════════════════════════════════════════════════════

def run_webcam_demo():
    """Real-time emotion detection. Press 'q' to quit."""
    import cv2
    from src.emotion_predictor import predict_from_array

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam."); return

    print("📸 Webcam started. Press 'q' to quit.")
    frame_count, result = 0, None

    while True:
        ret, frame = cap.read()
        if not ret: break
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))

        for (x, y, w, h) in faces:
            if frame_count % 5 == 0:
                result = predict_from_array(gray[y:y+h, x:x+w], use_tta=False)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 100), 2)
            if result:
                label = f"{result['emotion']} {result['confidence']*100:.0f}%"
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 100), 2)

        frame_count += 1
        cv2.imshow("MoodMate", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release(); cv2.destroyAllWindows()
    print("👋 Webcam closed.")


if __name__ == "__main__":
    set_seed(42)
    configure_gpu()
    print("✅ utils.py OK")
