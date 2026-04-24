"""
webcam_realtime.py  —  MoodMate (Multi-User, High-Speed Edition)
─────────────────────────────────────────────────────────────────
Detects emotions for MULTIPLE faces simultaneously using:
  • Threaded prediction pipeline  → no blocking on model inference
  • Face tracking (IoU-based)     → skip re-inference on same face
  • Batch inference               → all faces in one model call
  • Async song lookup             → doesn't stall the render loop

Controls:
  Q / ESC  → quit
  S        → save frame
  SPACE    → freeze / unfreeze

Run:
  python app/webcam_realtime.py
"""

import os, sys, cv2, time, threading, queue
import numpy as np
from collections import deque

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)

from src.emotion_predictor  import predict_from_array
from src.music_recommender  import MusicRecommender
from src.emotion_mapping    import get_emotion_info
from utils.helper_functions import (
    detect_faces, crop_face,
    emotion_emoji, logger, timestamp, ensure_dir
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_INDEX    = 0
PREDICT_EVERY   = 5          # Run detection every N frames (lower = faster feedback)
NUM_SONGS       = 5
IOU_THRESHOLD   = 0.40       # IoU above this → same face (skip re-inference)
MAX_TRACK_AGE   = 15         # Frames before a track is considered stale

# Colours (BGR)
PALETTE = [
    (0, 230, 100),   # green
    (255, 100, 80),  # coral
    (80, 180, 255),  # sky blue
    (255, 200, 60),  # amber
    (180, 80, 255),  # violet
    (60, 220, 220),  # teal
]
PURPLE = (200, 80, 255)
WHITE  = (240, 240, 255)
DARK   = (15, 15, 28)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def iou(a, b):
    """Intersection-over-Union for two (x,y,w,h) boxes."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0]+a[2], a[1]+a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0]+b[2], b[1]+b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / (union + 1e-6)


def put_text(img, text, pos, scale=0.52, color=WHITE, thickness=1):
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, DARK, thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_confidence_bars(img, scores, x0, y0, width=130):
    sorted_s = sorted(scores.items(), key=lambda kv: -kv[1])
    for i, (lbl, sc) in enumerate(sorted_s):
        y = y0 + i * 18
        cv2.rectangle(img, (x0, y), (x0+width, y+11), (55,55,75), -1)
        fw = int(width * sc)
        col = (0, 230, 100) if i == 0 else (90, 90, 170)
        cv2.rectangle(img, (x0, y), (x0+fw, y+11), col, -1)
        put_text(img, f"{lbl[:6]:6s} {sc*100:4.1f}%",
                 (x0+width+5, y+10), scale=0.36)


def draw_face_overlay(canvas, face, label, color, track_id):
    """Draw bounding box + emotion label for one face."""
    x, y, w, h = face
    # Corner brackets instead of full rectangle (cleaner look)
    t = 3       # thickness
    cs = min(w, h) // 5   # corner size
    corners = [
        ((x,    y),    (x+cs, y),    (x,    y+cs)),
        ((x+w,  y),    (x+w-cs, y),  (x+w,  y+cs)),
        ((x,    y+h),  (x+cs, y+h),  (x,    y+h-cs)),
        ((x+w,  y+h),  (x+w-cs, y+h),(x+w,  y+h-cs)),
    ]
    for (p, p1, p2) in corners:
        cv2.line(canvas, p, p1, color, t, cv2.LINE_AA)
        cv2.line(canvas, p, p2, color, t, cv2.LINE_AA)

    # Label pill
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    ly = max(y - 8, th + 6)
    cv2.rectangle(canvas, (x, ly-th-5), (x+tw+10, ly+bl+2), DARK, -1)
    cv2.rectangle(canvas, (x, ly-th-5), (x+tw+10, ly+bl+2), color, 1)
    put_text(canvas, label, (x+5, ly-1), scale=0.52, color=color)

    # User badge (top-right corner of box)
    badge = f"#{track_id+1}"
    cv2.putText(canvas, badge, (x+w-28, y+16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# FACE TRACKER
# keeps emotion results alive across frames so the display never flickers
# ─────────────────────────────────────────────────────────────────────────────

class FaceTracker:
    def __init__(self):
        self.tracks = {}   # track_id → dict
        self._next_id = 0

    def _new_track(self, box):
        tid = self._next_id
        self._next_id += 1
        self.tracks[tid] = dict(
            box=box, emotion="…", confidence=0.0,
            scores={}, songs=[], age=0, color=PALETTE[tid % len(PALETTE)]
        )
        return tid

    def update(self, boxes):
        """Match new detections to existing tracks via IoU."""
        used_tracks = set()
        used_boxes  = set()
        matches = []

        for bi, box in enumerate(boxes):
            best_iou, best_tid = 0, None
            for tid, trk in self.tracks.items():
                if tid in used_tracks:
                    continue
                s = iou(box, trk["box"])
                if s > best_iou:
                    best_iou, best_tid = s, tid
            if best_iou > IOU_THRESHOLD:
                matches.append((bi, best_tid))
                used_tracks.add(best_tid)
                used_boxes.add(bi)

        # Update matched
        for bi, tid in matches:
            self.tracks[tid]["box"] = boxes[bi]
            self.tracks[tid]["age"] = 0

        # New tracks for unmatched boxes
        new_tids = []
        for bi, box in enumerate(boxes):
            if bi not in used_boxes:
                new_tids.append(self._new_track(box))

        # Age out stale tracks
        stale = [tid for tid, t in self.tracks.items() if t["age"] > MAX_TRACK_AGE]
        for tid in stale:
            del self.tracks[tid]

        # Increment age for unmatched tracks
        for tid in self.tracks:
            if tid not in used_tracks:
                self.tracks[tid]["age"] += 1

        return {bi: tid for bi, tid in matches}, new_tids

    def set_result(self, tid, emotion, confidence, scores, songs):
        if tid in self.tracks:
            t = self.tracks[tid]
            t["emotion"]    = emotion
            t["confidence"] = confidence
            t["scores"]     = scores
            t["songs"]      = songs

    def active_tracks(self):
        return list(self.tracks.items())


# ─────────────────────────────────────────────────────────────────────────────
# ASYNC PREDICTION WORKER
# Runs model inference in a background thread so the camera loop never blocks
# ─────────────────────────────────────────────────────────────────────────────

class PredictionWorker(threading.Thread):
    def __init__(self, tracker, recommender):
        super().__init__(daemon=True)
        self.tracker    = tracker
        self.recommender = recommender
        self.in_q       = queue.Queue(maxsize=4)   # (tid, face_crop)
        self.running    = True

    def enqueue(self, tid, face_crop):
        try:
            self.in_q.put_nowait((tid, face_crop))
        except queue.Full:
            pass  # drop oldest request — frame is stale anyway

    def run(self):
        while self.running:
            try:
                tid, crop = self.in_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                pred = predict_from_array(crop)
                songs = self.recommender.recommend(pred["emotion"], n=NUM_SONGS)
                self.tracker.set_result(
                    tid,
                    pred["emotion"],
                    pred["confidence"],
                    pred["all_scores"],
                    songs
                )
            except FileNotFoundError:
                logger.error("Model not found — train first: python src/train_model.py")
                self.running = False
            except Exception as e:
                logger.warning(f"Prediction error for track {tid}: {e}")

    def stop(self):
        self.running = False


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def draw_sidebar(canvas, tracker, frame_w, frame_h):
    sx = frame_w + 10

    put_text(canvas, "MoodMate", (sx, 30), scale=0.78, color=PURPLE, thickness=2)
    put_text(canvas, "Multi-User Edition", (sx, 50), scale=0.36, color=(140,140,180))

    active = tracker.active_tracks()
    if not active:
        put_text(canvas, "No faces detected", (sx, 90), scale=0.48, color=(80,80,200))
        return

    y_cursor = 75
    for tid, trk in active:
        if trk["age"] > 0:   # show only currently visible tracks
            continue
        color = trk["color"]
        emo   = trk["emotion"]
        conf  = trk["confidence"]
        info  = get_emotion_info(emo) if emo not in ("…", "none") else {}

        # User header
        header = f"#{tid+1}  {emotion_emoji(emo)} {emo.capitalize()}"
        put_text(canvas, header, (sx, y_cursor), scale=0.52, color=color, thickness=1)
        y_cursor += 18

        if info:
            put_text(canvas, info.get("mood",""), (sx, y_cursor),
                     scale=0.35, color=(150,170,255))
        y_cursor += 14

        # Confidence bar strip
        if trk["scores"]:
            draw_confidence_bars(canvas, trk["scores"], sx, y_cursor, width=120)
            y_cursor += 8 * 18 + 4   # 7 emotions × 18px + gap

        # Songs
        songs = trk["songs"]
        if songs:
            put_text(canvas, "♪ For you:", (sx, y_cursor), scale=0.40, color=PURPLE)
            y_cursor += 16
            for s in songs[:3]:
                name   = s['name'][:20]   + '…' if len(s['name'])   > 20 else s['name']
                artist = s['artist'][:16] + '…' if len(s['artist']) > 16 else s['artist']
                put_text(canvas, f"  {name}", (sx, y_cursor), scale=0.38, color=WHITE)
                y_cursor += 14
                put_text(canvas, f"  {artist}", (sx, y_cursor), scale=0.34,
                         color=(140,140,190))
                y_cursor += 16

        # Divider
        cv2.line(canvas, (sx, y_cursor+2), (sx+230, y_cursor+2), (50,50,70), 1)
        y_cursor += 12

        if y_cursor > frame_h - 30:
            break   # no space left in sidebar


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run():
    logger.info("Loading music recommender …")
    recommender = MusicRecommender(
        csv_path=os.path.join(BASE_DIR, "data", "music_data.csv")
    )

    tracker = FaceTracker()
    worker  = PredictionWorker(tracker, recommender)
    worker.start()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    # ── Camera speed tweaks ──────────────────────────────────────
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # minimal buffer → no stale frames

    if not cap.isOpened():
        logger.error(f"Cannot open camera {CAMERA_INDEX}")
        return

    logger.info("Webcam ready. Q=quit  S=save  SPACE=freeze")

    frame_count = 0
    frozen      = False
    fps_times   = deque(maxlen=20)
    frame       = None

    save_dir = os.path.join(BASE_DIR, "app", "saved_frames")
    ensure_dir(save_dir)

    SIDEBAR_W = 260

    while True:
        t0 = time.perf_counter()

        if not frozen:
            ret, frame = cap.read()
            if not ret:
                logger.error("Frame capture failed")
                break
            frame_count += 1

            # ── Detect + track every PREDICT_EVERY frames ────────
            if frame_count % PREDICT_EVERY == 0:
                small = cv2.resize(frame, (320, 240))   # detect on half-res
                sx_scale = frame.shape[1] / 320
                sy_scale = frame.shape[0] / 240

                raw_faces = detect_faces(small)
                # Scale boxes back to full resolution
                faces = [
                    (int(x*sx_scale), int(y*sy_scale),
                     int(w*sx_scale), int(h*sy_scale))
                    for (x, y, w, h) in raw_faces
                ]

                matched, new_tids = tracker.update(faces)

                # Enqueue inference only for NEW or unconfident tracks
                for bi, face in enumerate(faces):
                    tid = matched.get(bi)
                    if tid is None:
                        continue   # already sent to worker as new track
                    trk = tracker.tracks.get(tid, {})
                    if trk.get("emotion") in ("…",) or trk.get("confidence",0) < 0.5:
                        crop = crop_face(frame, face)
                        worker.enqueue(tid, crop)

                # Always enqueue new tracks
                for tid in new_tids:
                    box = tracker.tracks[tid]["box"]
                    crop = crop_face(frame, box)
                    worker.enqueue(tid, crop)

        if frame is None:
            continue

        # ── BUILD CANVAS ─────────────────────────────────────────
        h, w = frame.shape[:2]
        canvas = np.full((h, w + SIDEBAR_W, 3), (20, 20, 35), dtype=np.uint8)
        canvas[:h, :w] = frame

        # Draw each tracked face
        for tid, trk in tracker.active_tracks():
            if trk["age"] > 2:   # only draw recently seen faces
                continue
            emo  = trk["emotion"]
            conf = trk["confidence"]
            label = (f"{emotion_emoji(emo)} {emo.capitalize()} "
                     f"{conf*100:.0f}%" if emo not in ("…","none") else "Detecting…")
            draw_face_overlay(canvas[:h, :w], trk["box"], label, trk["color"], tid)

        if not tracker.active_tracks() or all(t["age"]>2 for _,t in tracker.active_tracks()):
            put_text(canvas, "No face detected — adjust angle or lighting",
                     (w//2-180, h//2), scale=0.55, color=(80,80,200), thickness=1)

        # Sidebar
        draw_sidebar(canvas, tracker, w, h)

        # FPS overlay
        fps_times.append(time.perf_counter() - t0)
        fps = 1.0 / (np.mean(fps_times) + 1e-9)
        put_text(canvas, f"FPS:{fps:.0f}  Q=quit S=save SPC=freeze",
                 (8, h-8), scale=0.36, color=(100,100,150))

        if frozen:
            put_text(canvas, "|| FROZEN", (w//2-60, h//2),
                     scale=1.0, color=(0,140,255), thickness=2)

        cv2.imshow("MoodMate — Multi-User", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            fname = os.path.join(save_dir, f"frame_{timestamp()}.jpg")
            cv2.imwrite(fname, canvas)
            logger.info(f"Saved → {fname}")
        elif key == ord(" "):
            frozen = not frozen
            logger.info("Frozen" if frozen else "Resumed")

    worker.stop()
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Done. Goodbye!")


if __name__ == "__main__":
    run()