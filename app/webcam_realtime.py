"""
webcam_realtime.py
──────────────────
Standalone real-time emotion detection using your webcam.
Runs entirely locally — no browser needed.

Requirements:
  • Trained model at models/emotion_model.h5
  • Webcam connected

Controls:
  Q or ESC  → quit
  S         → save current frame to disk
  SPACE     → freeze / unfreeze

Run:
  python app/webcam_realtime.py
"""

import os
import sys
import cv2
import numpy as np
import time

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

from src.emotion_predictor  import predict_from_array
from src.music_recommender  import MusicRecommender
from src.emotion_mapping    import get_emotion_info
from utils.helper_functions import (
    detect_faces, crop_face, draw_face_box,
    emotion_emoji, logger, timestamp, ensure_dir
)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
CAMERA_INDEX   = 0      # 0 = default webcam
PREDICT_EVERY  = 10     # Run emotion prediction every N frames (performance)
NUM_SONGS      = 5      # Songs to show on sidebar

# Colours (BGR)
GREEN  = (0, 230, 100)
PURPLE = (200, 80, 255)
WHITE  = (240, 240, 255)
DARK   = (20, 20, 35)

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def put_text(img, text, pos, scale=0.55, color=WHITE, thickness=1):
    """Draw text with a dark shadow for readability."""
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, DARK, thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_confidence_bars(img, scores: dict, x_start, y_start, width=140):
    """Draw a mini bar chart of all 7 emotion confidence scores."""
    sorted_scores = sorted(scores.items(), key=lambda kv: -kv[1])
    bar_h = 12
    gap   = 18

    for i, (label, score) in enumerate(sorted_scores):
        y = y_start + i * gap
        # Background bar
        cv2.rectangle(img, (x_start, y), (x_start + width, y + bar_h),
                      (60, 60, 80), -1)
        # Filled bar
        fill_w = int(width * score)
        color  = GREEN if i == 0 else (100, 100, 180)
        cv2.rectangle(img, (x_start, y), (x_start + fill_w, y + bar_h),
                      color, -1)
        # Label
        put_text(img, f"{label[:7]:7s} {score*100:4.1f}%",
                 (x_start + width + 5, y + bar_h - 2), scale=0.38)


def draw_song_sidebar(img, songs: list, x_start, y_start, max_width=220):
    """Draw a list of recommended songs on the right side of the frame."""
    put_text(img, "Recommended Songs:", (x_start, y_start),
             scale=0.5, color=PURPLE, thickness=1)
    y = y_start + 22
    for i, song in enumerate(songs[:NUM_SONGS]):
        name   = song['name'][:22] + '…' if len(song['name']) > 22 else song['name']
        artist = song['artist'][:18] + '…' if len(song['artist']) > 18 else song['artist']
        put_text(img, f"{i+1}. {name}", (x_start, y), scale=0.42, color=WHITE)
        y += 16
        put_text(img, f"   {artist}", (x_start, y), scale=0.38, color=(160, 160, 200))
        y += 20


# ──────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────

def run():
    logger.info("Loading music recommender …")
    recommender = MusicRecommender(
        csv_path=os.path.join(BASE_DIR, "data", "music_data.csv")
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {CAMERA_INDEX}")
        return

    logger.info("Webcam opened. Press Q to quit, S to save, SPACE to freeze.")

    frame_count     = 0
    last_emotion    = "neutral"
    last_confidence = 0.0
    last_scores     = {}
    last_songs      = []
    frozen          = False
    fps_timer       = time.time()
    fps             = 0.0

    save_dir = os.path.join(BASE_DIR, "app", "saved_frames")
    ensure_dir(save_dir)

    while True:
        if not frozen:
            ret, frame = cap.read()
            if not ret:
                logger.error("Frame capture failed")
                break

            frame_count += 1

            # ── Every N frames: run full emotion pipeline ──────────
            if frame_count % PREDICT_EVERY == 0:
                faces = detect_faces(frame)

                if faces:
                    largest = max(faces, key=lambda b: b[2] * b[3])
                    face_crop = crop_face(frame, largest)

                    try:
                        pred = predict_from_array(face_crop)
                        last_emotion    = pred["emotion"]
                        last_confidence = pred["confidence"]
                        last_scores     = pred["all_scores"]

                        # Get fresh song recs when emotion changes
                        last_songs = recommender.recommend(last_emotion, n=NUM_SONGS)

                    except FileNotFoundError:
                        logger.error("Model not found. Train first: python src/train_model.py")
                        break

                # ── FPS ───────────────────────────────────────────
                now = time.time()
                fps = PREDICT_EVERY / (now - fps_timer + 1e-9)
                fps_timer = now

        # ── BUILD DISPLAY FRAME ────────────────────────────────────
        display = frame.copy()
        h, w    = display.shape[:2]

        # Expand canvas to add a right sidebar
        sidebar_w = 250
        canvas = np.full((h, w + sidebar_w, 3), (25, 25, 40), dtype=np.uint8)
        canvas[:h, :w] = display

        # Draw face boxes on the camera area
        if not frozen:
            faces = detect_faces(frame)
            for face in faces:
                emoji_str = emotion_emoji(last_emotion)
                label = f"{last_emotion} {last_confidence*100:.0f}%"
                canvas[:h, :w] = draw_face_box(canvas[:h, :w], face,
                                               label=label, color=GREEN)

        # ── SIDEBAR CONTENT ────────────────────────────────────────
        sx = w + 10

        # Title
        put_text(canvas, "MoodMate", (sx, 28), scale=0.75, color=PURPLE, thickness=2)

        # Current emotion
        emo_display = f"{emotion_emoji(last_emotion)} {last_emotion.capitalize()}"
        put_text(canvas, emo_display, (sx, 60), scale=0.65, color=GREEN, thickness=1)

        info = get_emotion_info(last_emotion)
        put_text(canvas, info["mood"], (sx, 80), scale=0.38, color=(160, 180, 255))

        # Confidence bars
        if last_scores:
            put_text(canvas, "Confidence:", (sx, 105), scale=0.42, color=WHITE)
            draw_confidence_bars(canvas, last_scores, sx, 115, width=130)

        # Song list
        if last_songs:
            draw_song_sidebar(canvas, last_songs, sx, 260)

        # FPS + controls footer
        put_text(canvas, f"FPS: {fps:.1f}  [Q]uit [S]ave [SPC]freeze",
                 (8, h - 8), scale=0.38, color=(120, 120, 160))

        if frozen:
            put_text(canvas, "⏸  FROZEN", (w // 2 - 60, h // 2),
                     scale=1.0, color=(0, 140, 255), thickness=2)

        # ── SHOW ──────────────────────────────────────────────────
        cv2.imshow("MoodMate — Emotion & Music", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):    # Q or ESC
            break
        elif key == ord("s"):
            fname = os.path.join(save_dir, f"frame_{timestamp()}.jpg")
            cv2.imwrite(fname, canvas)
            logger.info(f"Saved frame → {fname}")
        elif key == ord(" "):
            frozen = not frozen
            logger.info("Frozen" if frozen else "Resumed")

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Webcam closed. Goodbye! 👋")


if __name__ == "__main__":
    run()
