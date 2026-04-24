import os
import time
import argparse
import threading
import numpy as np
from pathlib import Path
from typing import Union, List

import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageEnhance

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(CLASSES)
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_IMG_SIZE = 128


def _get_custom_objects():
    try:
        from src.train_model import FocalCategoricalCrossentropy, WarmupCosineDecay
        return {
            "FocalCategoricalCrossentropy": FocalCategoricalCrossentropy,
            "WarmupCosineDecay": WarmupCosineDecay,
        }
    except Exception:
        return {}


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr = np.clip(arr, 0.0, 255.0)
    return arr


def _load_as_rgb_array(img_input, img_size: int) -> np.ndarray:
    if isinstance(img_input, (str, Path)):
        img = Image.open(img_input)
    elif isinstance(img_input, np.ndarray):
        if img_input.ndim == 2:
            img = Image.fromarray(img_input.astype(np.uint8), "L")
        elif img_input.ndim == 3 and img_input.shape[2] == 3:
            img = Image.fromarray(img_input.astype(np.uint8))
        elif img_input.ndim == 3 and img_input.shape[2] == 1:
            img = Image.fromarray(img_input[:, :, 0].astype(np.uint8), "L")
        else:
            raise ValueError(f"Unsupported numpy input shape: {img_input.shape}")
    elif isinstance(img_input, Image.Image):
        img = img_input
    else:
        raise TypeError(f"Unsupported input type: {type(img_input)}")

    img = img.convert("RGB")
    img = img.resize((img_size, img_size), Image.BILINEAR)
    return np.array(img, dtype=np.float32)


def _tta_variants(arr: np.ndarray):
    arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
    variants = [arr.copy()]
    variants.append(arr[:, ::-1, :].copy())

    img_pil = Image.fromarray(arr_uint8)

    variants.append(np.array(ImageEnhance.Brightness(img_pil).enhance(1.12), dtype=np.float32))
    variants.append(np.array(ImageEnhance.Brightness(img_pil).enhance(0.88), dtype=np.float32))
    variants.append(np.array(ImageEnhance.Contrast(img_pil).enhance(1.12), dtype=np.float32))
    variants.append(np.array(ImageEnhance.Contrast(img_pil).enhance(0.88), dtype=np.float32))

    h, w = arr.shape[:2]
    mh, mw = max(1, int(h * 0.05)), max(1, int(w * 0.05))
    cropped = arr_uint8[mh:h - mh, mw:w - mw, :]
    cropped_pil = Image.fromarray(cropped).resize((w, h), Image.BILINEAR)
    cropped_arr = np.array(cropped_pil, dtype=np.float32)
    variants.append(cropped_arr)
    variants.append(cropped_arr[:, ::-1, :])

    return variants


class EmotionPredictor:
    def __init__(
        self,
        model_path: Union[str, List[str]],
        img_size: int = DEFAULT_IMG_SIZE,
        temperature: float = 1.0,
    ):
        self.img_size = img_size
        self.temperature = temperature
        custom_objects = _get_custom_objects()

        if isinstance(model_path, (list, tuple)):
            self.models = []
            for p in model_path:
                print(f"Loading {p} ...")
                self.models.append(
                    keras.models.load_model(p, custom_objects=custom_objects, compile=False)
                )
            print(f"✓ Ensemble of {len(self.models)} models loaded.")
        else:
            print(f"Loading {model_path} ...")
            self.models = [
                keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            ]
            print("✓ Model loaded.")

        dummy = np.zeros((1, img_size, img_size, 3), dtype=np.float32)
        for m in self.models:
            m.predict(dummy, verbose=0)

        print(f"  Input  shape : {self.models[0].input_shape}")
        print(f"  Output shape : {self.models[0].output_shape}")

    def _infer_batch(self, batch: np.ndarray) -> np.ndarray:
        all_probs = []
        for m in self.models:
            preds = m.predict(batch, verbose=0)
            all_probs.append(preds)

        probs = np.mean(all_probs, axis=0)
        probs = np.clip(probs, 1e-8, 1.0)
        probs = probs / np.sum(probs, axis=-1, keepdims=True)

        if self.temperature != 1.0:
            log_probs = np.log(probs + 1e-8) / self.temperature
            probs = np.exp(log_probs)
            probs = probs / np.sum(probs, axis=-1, keepdims=True)

        return probs

    def _format_result(self, probs_1d: np.ndarray, top_k: int):
        idx = int(np.argmax(probs_1d))
        label = CLASSES[idx]
        confidence = float(probs_1d[idx])
        scores = {c: float(probs_1d[i]) for i, c in enumerate(CLASSES)}
        top_k_list = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        entropy = float(-np.sum(probs_1d * np.log(probs_1d + 1e-8)))
        uncertain = entropy > 1.5
        return label, confidence, scores, top_k_list, uncertain

    def predict(self, img_input, top_k: int = 3, tta: bool = True, tta_n: int = 8):
        arr = _load_as_rgb_array(img_input, self.img_size)

        if tta:
            variants = _tta_variants(arr)[:tta_n]
            batch = np.stack([_normalize(v) for v in variants], axis=0)
            probs = self._infer_batch(batch)
            probs_1d = np.mean(probs, axis=0)
        else:
            batch = _normalize(arr)[np.newaxis]
            probs = self._infer_batch(batch)
            probs_1d = probs[0]

        return self._format_result(probs_1d, top_k)

    def predict_frame(
        self,
        bgr_frame: np.ndarray,
        tta: bool = True,
        use_face_detector: bool = False,
    ):
        face_bbox = None

        if use_face_detector:
            try:
                import cv2
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                detector = cv2.CascadeClassifier(cascade_path)
                gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    face_bbox = (int(x), int(y), int(w), int(h))
                    bgr_frame = bgr_frame[y:y + h, x:x + w]
            except ImportError:
                pass

        rgb_frame = bgr_frame[:, :, ::-1]
        label, conf, scores, top3, uncertain = self.predict(rgb_frame, tta=tta)
        return label, conf, scores, top3, uncertain, face_bbox

    def benchmark(self, n_runs: int = 30, tta: bool = False):
        dummy = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.predict(dummy, tta=tta)
            times.append((time.perf_counter() - t0) * 1000)

        times = sorted(times)
        print(f"\nLatency benchmark ({n_runs} runs, tta={tta}):")
        print(f"  Mean   : {np.mean(times):.1f} ms")
        print(f"  Median : {np.median(times):.1f} ms")
        print(f"  P95    : {times[int(0.95 * n_runs)]:.1f} ms")
        print(f"  Min    : {times[0]:.1f} ms")
        return np.mean(times)


def _print_result(label, conf, _, top_k_list, uncertain):
    print(f"\nPrediction : {label.upper()} ({conf * 100:.1f}%)" + ("  ⚠ uncertain" if uncertain else ""))
    for cls, prob in top_k_list:
        bar = "█" * int(prob * 40)
        print(f"  {cls:<10} {prob * 100:5.1f}% {bar}")


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────────────────────────────────────
# FIX: Changed from stage2 → stage1 to match your actual trained model file
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_MODEL_PATH = os.path.join(_BASE_DIR, "models", "emotion_model_stage1.keras")

_predictor_instance = None
_predictor_lock = threading.Lock()


def _get_default_predictor() -> EmotionPredictor:
    global _predictor_instance
    if _predictor_instance is not None:
        return _predictor_instance

    with _predictor_lock:
        if _predictor_instance is None:
            if not os.path.exists(_DEFAULT_MODEL_PATH):
                raise FileNotFoundError(
                    f"Model not found at {_DEFAULT_MODEL_PATH}\n"
                    "Please check that 'emotion_model_stage1.keras' exists in the models/ folder."
                )
            _predictor_instance = EmotionPredictor(
                _DEFAULT_MODEL_PATH,
                img_size=DEFAULT_IMG_SIZE,
                temperature=1.0,
            )
        return _predictor_instance


def predict_from_path(image_path: str) -> dict:
    predictor = _get_default_predictor()
    label, confidence, scores, top3, uncertain = predictor.predict(
        image_path,
        top_k=3,
        tta=True,
    )
    return {
        "emotion": label,
        "confidence": confidence,
        "all_scores": scores,
        "top3": top3,
        "uncertain": uncertain,
    }


def predict_from_array(face_array: np.ndarray) -> dict:
    predictor = _get_default_predictor()

    # face_array coming from OpenCV is BGR — flip to RGB before predicting
    if face_array.ndim == 3 and face_array.shape[2] == 3:
        face_array = face_array[:, :, ::-1]

    label, confidence, scores, top3, uncertain = predictor.predict(
        face_array,
        top_k=3,
        tta=True,
    )
    return {
        "emotion": label,
        "confidence": confidence,
        "all_scores": scores,
        "top3": top3,
        "uncertain": uncertain,
    }


# ─────────────────────────────────────────────────────────────────────────────
# New helper: predict directly from a raw webcam BGR frame with face detection
# Call this from your Flask /predict route or webcam live endpoint
# ─────────────────────────────────────────────────────────────────────────────
def predict_from_frame(bgr_frame: np.ndarray) -> dict:
    """
    Accepts a raw BGR frame from OpenCV / Flask.
    Runs Haar-cascade face detection internally, crops the largest face,
    then predicts emotion.

    Returns a dict with keys:
        emotion, confidence, all_scores, top3, uncertain, face_bbox
        face_bbox = (x, y, w, h) or None if no face found
    """
    import cv2

    face_bbox = None
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_bbox = (int(x), int(y), int(w), int(h))
        face_crop = bgr_frame[y:y + h, x:x + w]  # BGR crop
    else:
        # No face found — run on full frame (will be less accurate)
        face_crop = bgr_frame

    result = predict_from_array(face_crop)  # handles BGR→RGB internally
    result["face_bbox"] = face_bbox
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Emotion Predictor")
    ap.add_argument("--model", required=True, help="Path to .keras model")
    ap.add_argument("--model2", default=None, help="Optional second model for ensemble")
    ap.add_argument("--image", default=None)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    model_paths = [args.model]
    if args.model2:
        model_paths.append(args.model2)

    predictor = EmotionPredictor(
        model_paths if len(model_paths) > 1 else model_paths[0],
        img_size=args.img_size,
        temperature=args.temperature,
    )

    if args.benchmark:
        predictor.benchmark(tta=args.tta)

    if args.image:
        result = predictor.predict(args.image, tta=args.tta)
        _print_result(*result)