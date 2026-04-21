"""
helper_functions.py
───────────────────
General-purpose utilities used across the project:
  • Face detection with OpenCV Haar cascades
  • Image encoding for browser display (base64)
  • Logging helpers
  • Webcam capture
"""

import os
import cv2
import base64
import logging
import numpy as np
from datetime import datetime
from PIL import Image
import io

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

def setup_logger(name: str = "moodmate", level=logging.INFO) -> logging.Logger:
    """
    Create a pretty logger that prints timestamps.
    Usage:
        logger = setup_logger()
        logger.info("Model loaded")
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s — %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = setup_logger()


# ──────────────────────────────────────────────────────────────────────────────
# FACE DETECTION
# ──────────────────────────────────────────────────────────────────────────────

# OpenCV ships with a pre-trained Haar cascade for frontal faces.
# No training needed — it's just a detector (not a classifier).
_FACE_CASCADE = None

def _get_cascade():
    """Lazy-load the face cascade XML file."""
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _FACE_CASCADE


def detect_faces(image_bgr: np.ndarray) -> list:
    """
    Detect faces in an OpenCV BGR image.

    Returns:
        List of (x, y, w, h) bounding boxes — one per face detected.
        Empty list if no face found.
    """
    cascade = _get_cascade()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    if len(faces) == 0:
        return []
    return faces.tolist()  # list of [x, y, w, h]


def crop_face(image_bgr: np.ndarray, face_box: list) -> np.ndarray:
    """
    Crop a face region from an image.

    Args:
        image_bgr : Full OpenCV frame
        face_box  : [x, y, w, h]

    Returns:
        Cropped BGR face region as numpy array
    """
    x, y, w, h = face_box
    # Add a small margin (10%) around the detected box
    margin_x = int(w * 0.1)
    margin_y = int(h * 0.1)
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(image_bgr.shape[1], x + w + margin_x)
    y2 = min(image_bgr.shape[0], y + h + margin_y)
    return image_bgr[y1:y2, x1:x2]


def draw_face_box(image_bgr: np.ndarray, face_box: list,
                  label: str = "", color=(0, 255, 100)) -> np.ndarray:
    """
    Draw a bounding box + emotion label onto the image.
    Returns modified copy (original not changed).
    """
    img = image_bgr.copy()
    x, y, w, h = face_box
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    if label:
        cv2.putText(img, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return img


# ──────────────────────────────────────────────────────────────────────────────
# IMAGE ENCODING (for browser / Flask responses)
# ──────────────────────────────────────────────────────────────────────────────

def image_to_base64(image_bgr: np.ndarray, fmt: str = "JPEG") -> str:
    """
    Convert an OpenCV BGR image to a base64 string.
    Useful for sending images in Flask JSON responses.

    Returns:
        "data:image/jpeg;base64,<base64data>" — drop-in for <img src=…>
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    mime = "jpeg" if fmt.upper() == "JPEG" else fmt.lower()
    return f"data:image/{mime};base64,{b64}"


def file_to_base64(path: str) -> str:
    """
    Read a file from disk and return base64-encoded data URI.
    Useful for uploaded images from Flask forms.
    """
    with open(path, "rb") as f:
        data = f.read()
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext == "jpg":
        ext = "jpeg"
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/{ext};base64,{b64}"


# ──────────────────────────────────────────────────────────────────────────────
# WEBCAM UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

class WebcamStream:
    """
    Simple OpenCV webcam wrapper.

    Usage:
        cam = WebcamStream()
        with cam:
            frame = cam.read()
    """

    def __init__(self, camera_index: int = 0):
        self.index  = camera_index
        self._cap   = None

    def open(self):
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.index}")
        logger.info(f"Webcam {self.index} opened")

    def read(self) -> np.ndarray:
        """Return the latest frame (BGR numpy array)."""
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        return frame

    def release(self):
        if self._cap:
            self._cap.release()
            logger.info("Webcam released")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.release()


# ──────────────────────────────────────────────────────────────────────────────
# MISC
# ──────────────────────────────────────────────────────────────────────────────

def timestamp() -> str:
    """Return current timestamp string (for filenames, logs, etc.)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


EMOTION_EMOJIS = {
    "happy":    "😄",
    "sad":      "😢",
    "angry":    "😠",
    "fear":     "😨",
    "disgust":  "🤢",
    "surprise": "😲",
    "neutral":  "😐",
}

def emotion_emoji(emotion: str) -> str:
    """Return an emoji for the detected emotion."""
    return EMOTION_EMOJIS.get(emotion.lower(), "🎭")
