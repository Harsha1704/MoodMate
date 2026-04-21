"""
app.py  —  MoodMate v5  (Investor-Grade Premium Dashboard)
──────────────────────────────────────────────────────────
Premium redesign inspired by Stripe, Linear, Vercel Analytics:
  • Neue Montreal + Geist Mono typography system
  • Cinematic dark void base with chromatic glow layers
  • Animated mesh gradient background (CSS-only)
  • Command-palette style sidebar with pill indicator
  • Bento-grid metric cards with live sparklines
  • Premium chart containers with gradient fills
  • Animated pipeline flow with SVG connectors
  • Glassmorphic floating music player
  • Hover-triggered glow rings, ripple states
  • Image sections for confusion matrix + feature importance
  • Fully responsive with smooth page transitions
"""

import os, sys, gc, base64, uuid, json
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
from src.emotion_predictor import predict_from_path, predict_from_array
from src.music_recommender  import MusicRecommender
from src.emotion_mapping    import get_emotion_info
from utils.helper_functions import (
    detect_faces, crop_face, draw_face_box,
    image_to_base64, emotion_emoji, logger
)

app  = Flask(__name__)
CORS(app)

recommender   = MusicRecommender(csv_path=os.path.join(BASE_DIR, "data", "music_data.csv"))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "app", "uploads")
TRAINING_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/models/<path:filename>")
def serve_model_file(filename):
    models_dir = os.path.join(BASE_DIR, "models")
    return send_from_directory(models_dir, filename)


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>MoodMate — Emotion Intelligence Platform</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,600;1,9..144,300;1,9..144,400&display=swap" rel="stylesheet"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
/* ══════════════════════════════════════════════════════
   DESIGN TOKENS + RESET
══════════════════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  /* Backgrounds */
  --void:      #03020a;
  --bg:        #06050f;
  --surface:   #0c0b18;
  --elevated:  #121024;
  --glass:     rgba(255,255,255,0.035);
  --glass2:    rgba(255,255,255,0.06);

  /* Borders */
  --line:      rgba(255,255,255,0.06);
  --line2:     rgba(255,255,255,0.11);
  --line3:     rgba(255,255,255,0.18);

  /* Brand Chromatics */
  --iris:      #6d5be8;
  --iris2:     #9b8aff;
  --iris3:     #c4bbff;
  --nova:      #e83d9e;
  --nova2:     #ff6ec7;
  --cyan:      #00d9e8;
  --cyan2:     #7eeeff;
  --emerald:   #00e5a0;
  --gold:      #f5c842;
  --crimson:   #ff4466;

  /* Text */
  --ink:       #f5f3ff;
  --ink2:      #c4bfe8;
  --ink3:      #8880b8;
  --ghost:     #4a4570;

  /* Glows */
  --glow-iris:  0 0 60px rgba(109,91,232,0.3);
  --glow-nova:  0 0 60px rgba(232,61,158,0.25);
  --glow-cyan:  0 0 60px rgba(0,217,232,0.2);

  /* Layout */
  --sidebar:   252px;
  --radius-xl: 20px;
  --radius-lg: 14px;
  --radius-md: 10px;
  --radius-sm: 7px;

  /* Typography */
  --font-display: 'Fraunces', Georgia, serif;
  --font-sans:    'Outfit', system-ui, sans-serif;
  --font-mono:    'JetBrains Mono', monospace;
}

html { scroll-behavior: smooth; }
body {
  background: var(--void);
  color: var(--ink);
  font-family: var(--font-sans);
  font-size: 14px;
  line-height: 1.55;
  min-height: 100vh;
  overflow-x: hidden;
  -webkit-font-smoothing: antialiased;
}

/* ══════════════════════════════════════════════════════
   ANIMATED BACKGROUND MESH
══════════════════════════════════════════════════════ */
.bg-mesh {
  position: fixed; inset: 0; z-index: 0; pointer-events: none; overflow: hidden;
}
.mesh-orb {
  position: absolute; border-radius: 50%;
  filter: blur(140px); mix-blend-mode: screen;
}
.orb-a {
  width: 900px; height: 900px;
  background: radial-gradient(circle, rgba(109,91,232,0.22) 0%, transparent 70%);
  top: -300px; left: -200px;
  animation: meshDrift 28s ease-in-out infinite alternate;
}
.orb-b {
  width: 700px; height: 700px;
  background: radial-gradient(circle, rgba(232,61,158,0.16) 0%, transparent 70%);
  bottom: -200px; right: -100px;
  animation: meshDrift 22s ease-in-out infinite alternate-reverse;
}
.orb-c {
  width: 500px; height: 500px;
  background: radial-gradient(circle, rgba(0,217,232,0.12) 0%, transparent 70%);
  top: 50%; left: 55%;
  animation: meshDrift 35s ease-in-out infinite alternate;
}
@keyframes meshDrift {
  0%   { transform: translate(0,0) scale(1); }
  33%  { transform: translate(60px,-40px) scale(1.08); }
  66%  { transform: translate(-30px,70px) scale(0.95); }
  100% { transform: translate(40px,20px) scale(1.04); }
}

/* scanline texture */
body::before {
  content: '';
  position: fixed; inset: 0; z-index: 1; pointer-events: none;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,0.015) 2px,
    rgba(0,0,0,0.015) 4px
  );
}

/* ══════════════════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════════════════ */
.sidebar {
  position: fixed; top: 0; left: 0;
  width: var(--sidebar); height: 100vh;
  background: rgba(6,5,15,0.88);
  backdrop-filter: blur(32px) saturate(150%);
  border-right: 1px solid var(--line);
  display: flex; flex-direction: column;
  z-index: 200;
  transition: width 0.35s cubic-bezier(0.4,0,0.2,1);
}
.sidebar::after {
  content: '';
  position: absolute; top: 0; right: 0; width: 1px; height: 100%;
  background: linear-gradient(180deg, transparent, rgba(109,91,232,0.4) 30%, rgba(232,61,158,0.3) 70%, transparent);
}

.sb-header {
  padding: 26px 20px 22px;
  border-bottom: 1px solid var(--line);
  display: flex; align-items: center; gap: 13px;
}
.sb-logo {
  width: 38px; height: 38px; flex-shrink: 0; border-radius: 11px;
  background: linear-gradient(145deg, var(--iris), var(--nova));
  display: grid; place-items: center; font-size: 1.1rem;
  box-shadow: 0 0 30px rgba(109,91,232,0.5), inset 0 1px 0 rgba(255,255,255,0.15);
  position: relative;
}
.sb-logo::after {
  content: '';
  position: absolute; inset: -1px; border-radius: 12px;
  background: linear-gradient(145deg, rgba(255,255,255,0.15), transparent);
  pointer-events: none;
}
.sb-brand { overflow: hidden; }
.sb-brand-name {
  font-family: var(--font-display);
  font-size: 1.05rem; font-weight: 600;
  color: var(--ink); letter-spacing: -0.02em;
  white-space: nowrap;
  background: linear-gradient(135deg, #fff 0%, var(--iris3) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sb-brand-tag {
  font-size: 0.68rem; color: var(--ghost);
  letter-spacing: 0.05em; text-transform: uppercase;
  white-space: nowrap; margin-top: 1px;
}

.sb-section { padding: 14px 12px 6px; }
.sb-section-label {
  font-size: 0.62rem; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--ghost); padding: 0 8px; font-weight: 600;
}

.sb-nav { flex: 1; overflow-y: auto; overflow-x: hidden; padding: 8px 12px; }
.sb-nav::-webkit-scrollbar { width: 0; }

.nav-btn {
  display: flex; align-items: center; gap: 11px;
  padding: 10px 12px; border-radius: var(--radius-md);
  cursor: pointer; color: var(--ink3);
  font-size: 0.85rem; font-weight: 500;
  transition: all 0.22s ease;
  position: relative; overflow: hidden;
  white-space: nowrap; border: 1px solid transparent;
  margin-bottom: 2px; user-select: none;
}
.nav-btn::before {
  content: ''; position: absolute; inset: 0; opacity: 0;
  background: linear-gradient(105deg, rgba(109,91,232,0.1), transparent 60%);
  transition: opacity 0.22s;
}
.nav-btn:hover { color: var(--ink2); }
.nav-btn:hover::before { opacity: 1; }
.nav-btn.active {
  color: var(--ink);
  background: rgba(109,91,232,0.12);
  border-color: rgba(109,91,232,0.22);
  box-shadow: inset 0 0 0 1px rgba(109,91,232,0.08);
}
.nav-btn.active .nav-indicator {
  opacity: 1; transform: scaleY(1);
}
.nav-indicator {
  position: absolute; left: 0; top: 20%; bottom: 20%; width: 2.5px;
  background: linear-gradient(180deg, var(--iris), var(--nova));
  border-radius: 0 2px 2px 0;
  opacity: 0; transform: scaleY(0.3);
  transition: all 0.25s cubic-bezier(0.34,1.56,0.64,1);
}
.nav-icon { font-size: 0.95rem; flex-shrink: 0; width: 20px; text-align: center; }
.nav-label { flex: 1; }
.nav-badge {
  font-size: 0.6rem; font-family: var(--font-mono);
  padding: 2px 6px; border-radius: 20px;
  background: rgba(109,91,232,0.2); color: var(--iris2);
  border: 1px solid rgba(109,91,232,0.3);
}

.sb-footer {
  padding: 14px 20px 18px;
  border-top: 1px solid var(--line);
}
.status-pill {
  display: inline-flex; align-items: center; gap: 7px;
  padding: 6px 12px; border-radius: 100px;
  background: rgba(0,229,160,0.06);
  border: 1px solid rgba(0,229,160,0.18);
  font-size: 0.72rem; color: var(--emerald);
  font-family: var(--font-mono);
}
.status-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--emerald); box-shadow: 0 0 10px var(--emerald);
  animation: statusPulse 2.5s ease-in-out infinite;
}
@keyframes statusPulse {
  0%,100% { opacity: 1; box-shadow: 0 0 10px var(--emerald); }
  50%      { opacity: 0.4; box-shadow: 0 0 4px var(--emerald); }
}

/* ══════════════════════════════════════════════════════
   MAIN LAYOUT
══════════════════════════════════════════════════════ */
.main {
  margin-left: var(--sidebar);
  padding: 36px 40px 100px;
  min-height: 100vh;
  position: relative; z-index: 2;
  transition: margin-left 0.35s cubic-bezier(0.4,0,0.2,1);
}

/* Page transitions */
.page { display: none; animation: pageReveal 0.38s cubic-bezier(0.4,0,0.2,1); }
.page.active { display: block; }
@keyframes pageReveal {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ══════════════════════════════════════════════════════
   PAGE HEADERS
══════════════════════════════════════════════════════ */
.page-hero {
  display: flex; align-items: flex-start;
  justify-content: space-between; gap: 20px;
  margin-bottom: 32px; flex-wrap: wrap;
}
.page-eyebrow {
  display: inline-flex; align-items: center; gap: 7px;
  font-size: 0.7rem; font-weight: 600; letter-spacing: 0.13em;
  text-transform: uppercase; color: var(--iris2);
  margin-bottom: 8px; font-family: var(--font-mono);
}
.page-eyebrow::before {
  content: ''; width: 16px; height: 1px;
  background: linear-gradient(90deg, var(--iris), transparent);
}
.page-title {
  font-family: var(--font-display);
  font-size: 2rem; font-weight: 600;
  letter-spacing: -0.04em; color: var(--ink);
  line-height: 1.05;
}
.page-subtitle {
  font-size: 0.875rem; color: var(--ink3);
  margin-top: 8px; max-width: 55ch; line-height: 1.7;
}
.hero-actions { display: flex; gap: 10px; flex-wrap: wrap; align-items: flex-start; }

/* ══════════════════════════════════════════════════════
   GLASS COMPONENTS
══════════════════════════════════════════════════════ */
.glass-tag {
  display: inline-flex; align-items: center; gap: 7px;
  padding: 7px 14px; border-radius: 100px;
  background: var(--glass2);
  border: 1px solid var(--line2);
  backdrop-filter: blur(8px);
  font-size: 0.78rem; color: var(--ink2); white-space: nowrap;
  font-family: var(--font-mono);
}
.glass-tag .live-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--emerald); box-shadow: 0 0 8px var(--emerald);
  animation: statusPulse 2s infinite;
}

.chip {
  padding: 3px 9px; border-radius: 20px;
  font-size: 0.68rem; font-weight: 600; letter-spacing: 0.04em;
  background: rgba(109,91,232,0.14); border: 1px solid rgba(109,91,232,0.28);
  color: var(--iris2); white-space: nowrap;
}
.chip.green { background: rgba(0,229,160,0.1); border-color: rgba(0,229,160,0.25); color: var(--emerald); }
.chip.cyan  { background: rgba(0,217,232,0.1); border-color: rgba(0,217,232,0.25); color: var(--cyan); }
.chip.gold  { background: rgba(245,200,66,0.1); border-color: rgba(245,200,66,0.25); color: var(--gold); }

/* ══════════════════════════════════════════════════════
   CARDS
══════════════════════════════════════════════════════ */
.card {
  background: var(--glass);
  border: 1px solid var(--line);
  border-radius: var(--radius-xl);
  padding: 24px;
  position: relative; overflow: hidden;
  backdrop-filter: blur(20px);
  transition: border-color 0.25s, box-shadow 0.25s, transform 0.25s;
}
.card::before {
  content: ''; position: absolute;
  top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.09) 50%, transparent 100%);
}
.card:hover {
  border-color: var(--line2);
  transform: translateY(-1px);
  box-shadow: 0 12px 40px rgba(0,0,0,0.35);
}
.card-label {
  font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--ghost); margin-bottom: 14px;
  font-family: var(--font-mono);
}

/* ══════════════════════════════════════════════════════
   BENTO METRIC CARDS
══════════════════════════════════════════════════════ */
.bento-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin-bottom: 28px; }
.bento-card {
  border-radius: var(--radius-xl);
  border: 1px solid var(--line);
  padding: 20px 22px;
  position: relative; overflow: hidden;
  transition: transform 0.25s, box-shadow 0.25s;
  cursor: default;
}
.bento-card:hover { transform: translateY(-3px); }
.bento-card::after {
  content: ''; position: absolute;
  bottom: -30px; right: -30px;
  width: 110px; height: 110px;
  border-radius: 50%;
  filter: blur(40px); opacity: 0.5;
  pointer-events: none;
}
/* Variants */
.bento-iris  { background: linear-gradient(150deg, rgba(40,28,110,0.85), rgba(18,14,52,0.95)); border-color: rgba(109,91,232,0.3); }
.bento-iris::after  { background: var(--iris); }
.bento-iris:hover   { box-shadow: 0 8px 40px rgba(109,91,232,0.2); }
.bento-nova  { background: linear-gradient(150deg, rgba(100,20,70,0.85), rgba(38,12,30,0.95)); border-color: rgba(232,61,158,0.3); }
.bento-nova::after  { background: var(--nova); }
.bento-nova:hover   { box-shadow: 0 8px 40px rgba(232,61,158,0.2); }
.bento-cyan  { background: linear-gradient(150deg, rgba(0,60,80,0.85), rgba(4,22,32,0.95)); border-color: rgba(0,217,232,0.3); }
.bento-cyan::after  { background: var(--cyan); }
.bento-cyan:hover   { box-shadow: 0 8px 40px rgba(0,217,232,0.15); }
.bento-em    { background: linear-gradient(150deg, rgba(0,70,55,0.85), rgba(4,24,20,0.95)); border-color: rgba(0,229,160,0.3); }
.bento-em::after    { background: var(--emerald); }
.bento-em:hover     { box-shadow: 0 8px 40px rgba(0,229,160,0.15); }
.bento-gold  { background: linear-gradient(150deg, rgba(80,55,0,0.85), rgba(30,22,4,0.95)); border-color: rgba(245,200,66,0.3); }
.bento-gold::after  { background: var(--gold); }
.bento-gold:hover   { box-shadow: 0 8px 40px rgba(245,200,66,0.15); }

.bento-label { font-size: 0.67rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: rgba(196,191,232,0.65); margin-bottom: 10px; font-family: var(--font-mono); }
.bento-value { font-family: var(--font-sans); font-size: 1.65rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1; }
.bento-sub   { font-size: 0.73rem; color: rgba(196,191,232,0.55); margin-top: 7px; }

/* ══════════════════════════════════════════════════════
   BUTTONS
══════════════════════════════════════════════════════ */
.btn {
  display: inline-flex; align-items: center; justify-content: center; gap: 8px;
  padding: 12px 22px; border: none; border-radius: var(--radius-lg);
  cursor: pointer; font-family: var(--font-sans);
  font-size: 0.875rem; font-weight: 600;
  transition: all 0.22s ease; position: relative; overflow: hidden;
  white-space: nowrap; letter-spacing: -0.01em;
}
.btn::after {
  content: ''; position: absolute; inset: 0;
  background: rgba(255,255,255,0); transition: background 0.18s;
}
.btn:hover::after { background: rgba(255,255,255,0.06); }
.btn:active { transform: scale(0.97); }
.btn:disabled { opacity: 0.38; cursor: not-allowed; }
.btn:disabled::after { display: none; }

.btn-primary {
  background: linear-gradient(135deg, var(--iris) 0%, var(--nova) 100%);
  color: #fff; width: 100%; margin-top: 16px;
  box-shadow: 0 4px 24px rgba(109,91,232,0.38), 0 1px 0 rgba(255,255,255,0.12) inset;
  font-size: 0.92rem;
}
.btn-primary:hover { box-shadow: 0 8px 36px rgba(109,91,232,0.55), 0 1px 0 rgba(255,255,255,0.15) inset; }
.btn-secondary {
  background: var(--glass2);
  border: 1px solid var(--line2);
  color: var(--ink2);
}
.btn-secondary:hover { border-color: var(--line3); color: var(--ink); }
.btn-sm { padding: 7px 14px; font-size: 0.78rem; border-radius: var(--radius-md); margin-top: 0; width: auto; }
.btn-play { background: rgba(29,185,84,0.1); color: #1DB954; border: 1px solid rgba(29,185,84,0.28); }
.btn-play:hover { background: rgba(29,185,84,0.18); }

/* ══════════════════════════════════════════════════════
   UPLOAD ZONE
══════════════════════════════════════════════════════ */
.upload-zone {
  border: 1.5px dashed rgba(109,91,232,0.35);
  border-radius: var(--radius-xl);
  padding: 36px 28px; text-align: center; cursor: pointer;
  position: relative; overflow: hidden;
  transition: all 0.28s ease;
  background: rgba(109,91,232,0.025);
}
.upload-zone:hover {
  border-color: rgba(109,91,232,0.6);
  background: rgba(109,91,232,0.07);
  box-shadow: 0 0 0 4px rgba(109,91,232,0.08);
}
.upload-zone.drag-over {
  border-color: var(--iris);
  background: rgba(109,91,232,0.12);
  box-shadow: 0 0 0 6px rgba(109,91,232,0.12);
}
.upload-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
.upload-icon-ring {
  width: 68px; height: 68px; border-radius: 50%; margin: 0 auto 16px;
  background: conic-gradient(from 220deg, rgba(109,91,232,0.3), rgba(232,61,158,0.2), rgba(109,91,232,0.3));
  display: grid; place-items: center; font-size: 1.5rem;
  animation: ringRotate 8s linear infinite;
  border: 1.5px solid rgba(109,91,232,0.25);
  box-shadow: 0 0 30px rgba(109,91,232,0.2);
}
@keyframes ringRotate {
  to { background: conic-gradient(from 580deg, rgba(109,91,232,0.3), rgba(232,61,158,0.2), rgba(109,91,232,0.3)); }
}
.upload-heading { font-size: 0.98rem; font-weight: 700; color: var(--ink); margin-bottom: 5px; }
.upload-hint    { font-size: 0.76rem; color: var(--ink3); font-family: var(--font-mono); }
#preview-img { max-width: 100%; max-height: 260px; border-radius: var(--radius-lg); display: none; margin-top: 16px; border: 1px solid var(--line2); }

/* ══════════════════════════════════════════════════════
   EMOTION RESULT
══════════════════════════════════════════════════════ */
.emotion-result {
  display: flex; align-items: center; gap: 18px;
  padding: 20px 22px; border-radius: var(--radius-lg);
  background: linear-gradient(135deg, rgba(109,91,232,0.1), rgba(232,61,158,0.07));
  border: 1px solid rgba(109,91,232,0.22);
  margin: 14px 0; position: relative; overflow: hidden;
}
.emotion-result::before {
  content: ''; position: absolute; inset: 0;
  background: linear-gradient(225deg, rgba(232,61,158,0.05), transparent);
}
.emo-emoji { font-size: 3rem; animation: emoPop 0.5s cubic-bezier(0.34,1.56,0.64,1); }
@keyframes emoPop {
  from { transform: scale(0.3) rotate(-15deg); opacity: 0; }
  to   { transform: scale(1) rotate(0deg); opacity: 1; }
}
.emo-label {
  font-family: var(--font-display);
  font-size: 1.7rem; font-weight: 600; text-transform: capitalize;
  letter-spacing: -0.03em; color: var(--ink);
}
.emo-conf { font-size: 0.82rem; color: var(--ink3); margin-top: 4px; font-family: var(--font-mono); }
.emo-mood { font-size: 0.82rem; color: var(--iris2); margin-top: 3px; }

/* ══════════════════════════════════════════════════════
   CONFIDENCE BARS
══════════════════════════════════════════════════════ */
.conf-row { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
.conf-name { font-size: 0.74rem; color: var(--ink2); width: 68px; text-align: right; text-transform: capitalize; font-family: var(--font-mono); }
.conf-track { flex: 1; height: 6px; background: rgba(255,255,255,0.06); border-radius: 3px; overflow: hidden; }
.conf-fill  { height: 100%; border-radius: 3px; background: linear-gradient(90deg, var(--iris), var(--nova)); transition: width 0.7s cubic-bezier(0.4,0,0.2,1); }
.conf-pct   { font-size: 0.7rem; color: var(--ghost); width: 38px; text-align: right; font-family: var(--font-mono); }

/* ══════════════════════════════════════════════════════
   SONG CARDS
══════════════════════════════════════════════════════ */
.song-list-wrap { max-height: 400px; overflow-y: auto; padding-right: 3px; }
.song-list-wrap::-webkit-scrollbar { width: 4px; }
.song-list-wrap::-webkit-scrollbar-track { background: transparent; }
.song-list-wrap::-webkit-scrollbar-thumb { background: var(--line2); border-radius: 2px; }

.song-row {
  display: flex; align-items: center; gap: 13px;
  padding: 11px 14px; border-radius: var(--radius-md);
  background: var(--glass);
  border: 1px solid var(--line);
  margin-bottom: 7px; transition: all 0.2s ease; cursor: pointer;
}
.song-row:hover { background: var(--glass2); border-color: var(--line2); transform: translateX(3px); }
.song-row.playing { background: rgba(109,91,232,0.1); border-color: rgba(109,91,232,0.28); }
.song-num { color: var(--ghost); font-size: 0.72rem; min-width: 20px; font-family: var(--font-mono); }
.song-info { flex: 1; min-width: 0; }
.song-name { font-size: 0.88rem; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.song-meta { font-size: 0.72rem; color: var(--ink3); margin-top: 2px; }
.song-tags { display: flex; gap: 4px; margin-top: 5px; flex-wrap: wrap; }
.tag { font-size: 0.62rem; padding: 2px 7px; border-radius: 20px; font-weight: 600; font-family: var(--font-mono); }
.tag-genre  { background: rgba(109,91,232,0.16); color: var(--iris2); border: 1px solid rgba(109,91,232,0.22); }
.tag-val    { background: rgba(0,229,160,0.1); color: var(--emerald); border: 1px solid rgba(0,229,160,0.2); }
.tag-energy { background: rgba(245,200,66,0.1); color: var(--gold); border: 1px solid rgba(245,200,66,0.18); }
.song-actions { display: flex; gap: 6px; flex-shrink: 0; }

/* ══════════════════════════════════════════════════════
   NOW PLAYING BAR
══════════════════════════════════════════════════════ */
.now-playing {
  position: fixed; bottom: 0; left: var(--sidebar); right: 0;
  height: 70px;
  background: rgba(6,5,15,0.96);
  backdrop-filter: blur(40px) saturate(180%);
  border-top: 1px solid var(--line);
  padding: 0 36px; display: none; align-items: center; gap: 22px; z-index: 500;
  transition: left 0.35s;
}
.now-playing::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(109,91,232,0.6) 30%, rgba(232,61,158,0.5) 70%, transparent);
}
.np-thumb {
  width: 40px; height: 40px; border-radius: 9px; flex-shrink: 0;
  background: linear-gradient(135deg, var(--iris), var(--nova));
  display: grid; place-items: center; font-size: 1rem;
  box-shadow: 0 0 18px rgba(109,91,232,0.35);
}
.np-info { flex: 1; min-width: 0; }
.np-title  { font-size: 0.875rem; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.np-artist { font-size: 0.72rem; color: var(--ink3); margin-top: 1px; font-family: var(--font-mono); }
.np-ctrl { display: flex; align-items: center; gap: 9px; }
.np-btn {
  background: var(--glass2); border: 1px solid var(--line2);
  color: var(--ink); width: 34px; height: 34px; border-radius: 50%;
  cursor: pointer; font-size: 0.85rem;
  display: grid; place-items: center; transition: all 0.18s;
}
.np-btn:hover { background: var(--glass); border-color: var(--line3); }
.np-play-btn {
  width: 42px; height: 42px; font-size: 0.95rem;
  background: linear-gradient(135deg, var(--iris), var(--nova));
  border: none; box-shadow: 0 0 20px rgba(109,91,232,0.4);
}
.np-progress { flex: 2; display: flex; align-items: center; gap: 10px; }
.np-bar {
  flex: 1; height: 3px; background: rgba(255,255,255,0.1);
  border-radius: 2px; cursor: pointer; position: relative;
}
.np-bar-fill {
  height: 100%; border-radius: 2px; width: 0%;
  background: linear-gradient(90deg, var(--iris), var(--nova));
  transition: width 0.1s linear; pointer-events: none;
}
.np-time { font-size: 0.7rem; color: var(--ghost); font-family: var(--font-mono); white-space: nowrap; }
.np-close {
  background: none; border: 1px solid var(--line2); color: var(--ghost);
  width: 28px; height: 28px; border-radius: 50%; cursor: pointer;
  font-size: 0.75rem; display: grid; place-items: center; transition: all 0.18s;
}
.np-close:hover { color: var(--ink); border-color: var(--line3); }

/* ══════════════════════════════════════════════════════
   PIPELINE FLOW (Detection Flow Page)
══════════════════════════════════════════════════════ */
.pipeline-rail {
  display: flex; align-items: flex-start; gap: 0;
  margin: 24px 0; overflow-x: auto; padding-bottom: 12px;
}
.pipeline-node {
  flex: 1; min-width: 160px;
  display: flex; flex-direction: column; align-items: center;
  position: relative; padding: 0 10px;
}
.pipeline-connector {
  position: absolute; top: 30px; left: calc(50% + 36px);
  right: calc(-50% + 36px); height: 1px;
  background: linear-gradient(90deg, rgba(109,91,232,0.4), rgba(232,61,158,0.3));
  pointer-events: none;
}
.pipeline-connector::after {
  content: ''; position: absolute;
  right: -1px; top: -3px; width: 7px; height: 7px;
  border-top: 1px solid rgba(232,61,158,0.5);
  border-right: 1px solid rgba(232,61,158,0.5);
  transform: rotate(45deg);
}
.pipeline-node:last-child .pipeline-connector { display: none; }
.pipe-badge {
  width: 58px; height: 58px; border-radius: 50%;
  background: linear-gradient(135deg, rgba(109,91,232,0.2), rgba(232,61,158,0.15));
  border: 1.5px solid rgba(109,91,232,0.3);
  display: grid; place-items: center;
  font-size: 1.4rem; margin-bottom: 12px; position: relative; z-index: 1;
  box-shadow: 0 0 28px rgba(109,91,232,0.15);
  transition: all 0.3s ease;
}
.pipe-badge:hover {
  background: linear-gradient(135deg, rgba(109,91,232,0.35), rgba(232,61,158,0.25));
  box-shadow: 0 0 40px rgba(109,91,232,0.3);
  transform: scale(1.06);
}
.pipe-num {
  position: absolute; top: -4px; right: -4px;
  width: 20px; height: 20px; border-radius: 50%;
  background: linear-gradient(135deg, var(--iris), var(--nova));
  font-size: 0.62rem; font-weight: 800; font-family: var(--font-mono);
  display: grid; place-items: center; color: #fff;
  border: 1.5px solid var(--bg);
}
.pipe-title { font-size: 0.9rem; font-weight: 700; margin-bottom: 5px; text-align: center; }
.pipe-desc  { font-size: 0.74rem; color: var(--ink3); text-align: center; line-height: 1.5; }

/* ══════════════════════════════════════════════════════
   CHART CONTAINERS
══════════════════════════════════════════════════════ */
.chart-shell { position: relative; }
.chart-shell.h200 { height: 200px; }
.chart-shell.h240 { height: 240px; }
.chart-shell.h280 { height: 280px; }
.chart-shell.h320 { height: 320px; }

/* ══════════════════════════════════════════════════════
   GRID LAYOUTS
══════════════════════════════════════════════════════ */
.grid-2  { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.grid-3  { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
.grid-4  { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; }
.grid-5  { display: grid; grid-template-columns: repeat(5,1fr); gap: 14px; }
.grid-7030 { display: grid; grid-template-columns: 1.4fr 0.9fr; gap: 20px; }
.grid-5545 { display: grid; grid-template-columns: 1.1fr 0.95fr; gap: 18px; }

/* ══════════════════════════════════════════════════════
   WEBCAM
══════════════════════════════════════════════════════ */
#webcam-canvas { border-radius: var(--radius-lg); border: 1px solid var(--line2); display: block; width: 100%; }

/* ══════════════════════════════════════════════════════
   HISTORY / DATA ROWS
══════════════════════════════════════════════════════ */
.history-item {
  display: grid; grid-template-columns: auto 1fr auto auto;
  gap: 12px; align-items: center;
  padding: 12px 16px; border-radius: var(--radius-md);
  background: var(--glass); border: 1px solid var(--line);
  margin-bottom: 7px; transition: background 0.16s;
}
.history-item:hover { background: var(--glass2); }
.history-emoji { font-size: 1.25rem; }
.history-name { font-weight: 600; font-size: 0.875rem; }
.history-meta { font-size: 0.76rem; color: var(--ink3); }

/* ══════════════════════════════════════════════════════
   TRAINING TABLES
══════════════════════════════════════════════════════ */
.data-table-wrap { overflow: auto; border-radius: var(--radius-md); border: 1px solid var(--line); background: rgba(12,11,24,0.9); }
.data-table { width: 100%; border-collapse: collapse; font-size: 0.83rem; }
.data-table th, .data-table td { padding: 11px 16px; border-bottom: 1px solid var(--line); text-align: left; }
.data-table th { color: var(--ghost); font-weight: 600; font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; background: rgba(0,0,0,0.3); position: sticky; top: 0; font-family: var(--font-mono); }
.data-table tr.best-row { background: rgba(109,91,232,0.1); }
.data-table .mono { font-family: var(--font-mono); }
.best-badge { display: inline-block; padding: 2px 8px; border-radius: 20px; background: rgba(109,91,232,0.25); color: var(--iris2); font-size: 0.67rem; font-weight: 700; font-family: var(--font-mono); }

/* ══════════════════════════════════════════════════════
   MODEL INFO GRID (Training page)
══════════════════════════════════════════════════════ */
.kv-table { display: grid; grid-template-columns: max-content 1fr; gap: 9px 18px; font-size: 0.85rem; }
.kv-k { color: var(--ink3); }
.kv-v { color: var(--ink); font-weight: 600; font-family: var(--font-mono); }

/* ══════════════════════════════════════════════════════
   SCORE RING
══════════════════════════════════════════════════════ */
.score-ring-outer {
  width: 144px; height: 144px; border-radius: 50%; position: relative;
  background: conic-gradient(var(--iris) calc(var(--pct) * 1%), rgba(255,255,255,0.06) 0);
  display: grid; place-items: center;
  box-shadow: 0 0 50px rgba(109,91,232,0.35);
}
.score-ring-inner {
  position: absolute; width: 104px; height: 104px; border-radius: 50%;
  background: var(--surface); border: 1px solid var(--line);
  display: grid; place-items: center;
}
.ring-val { font-family: var(--font-sans); font-size: 1.8rem; font-weight: 800; letter-spacing: -0.04em; display: block; text-align: center; }
.ring-sub { font-size: 0.67rem; color: var(--ghost); font-family: var(--font-mono); display: block; text-align: center; }

/* ══════════════════════════════════════════════════════
   CLASS CARDS
══════════════════════════════════════════════════════ */
.class-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(148px,1fr)); gap: 12px; }
.class-card { background: var(--glass); border: 1px solid var(--line); border-radius: var(--radius-lg); padding: 16px; }
.class-card-head { display: flex; align-items: center; gap: 9px; margin-bottom: 10px; }
.class-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.class-name { font-size: 0.95rem; font-weight: 700; }
.class-count { font-family: var(--font-sans); font-size: 1.7rem; font-weight: 800; letter-spacing: -0.04em; line-height: 1; margin-bottom: 3px; }
.class-pct  { font-size: 0.75rem; color: var(--ink3); font-family: var(--font-mono); }

/* ══════════════════════════════════════════════════════
   COMPARISON PAGE
══════════════════════════════════════════════════════ */
.comp-tabs { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 20px; }
.comp-tab {
  padding: 9px 16px; border-radius: 100px; cursor: pointer; user-select: none;
  border: 1px solid var(--line2); background: var(--glass);
  color: var(--ink3); font-size: 0.8rem; font-weight: 600;
  transition: all 0.2s ease;
}
.comp-tab:hover { color: var(--ink2); border-color: rgba(109,91,232,0.3); }
.comp-tab.active {
  color: #fff;
  background: linear-gradient(135deg, rgba(109,91,232,0.28), rgba(232,61,158,0.18));
  border-color: rgba(109,91,232,0.4);
  box-shadow: 0 0 0 1px rgba(109,91,232,0.12) inset;
}
.comp-panel { display: none; animation: pageReveal 0.25s ease; }
.comp-panel.active { display: block; }

.comp-rank-badge {
  display: inline-flex; align-items: center; justify-content: center;
  width: 24px; height: 24px; border-radius: 50%;
  background: rgba(109,91,232,0.16); color: var(--iris2);
  font-size: 0.7rem; font-weight: 700; font-family: var(--font-mono);
}
.comp-table th, .comp-table td { padding: 11px 14px; border-bottom: 1px solid var(--line); font-size: 0.84rem; }
.comp-table th { color: var(--ghost); font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase; text-align: left; font-family: var(--font-mono); }
.comp-table tr.top-row { background: rgba(109,91,232,0.09); }
.comp-table { width: 100%; border-collapse: collapse; }

/* ══════════════════════════════════════════════════════
   IMAGE FRAMES (Confusion Matrix, Feature Importance)
══════════════════════════════════════════════════════ */
.img-frame {
  border-radius: var(--radius-lg); border: 1px solid var(--line2);
  background: rgba(10,9,24,0.8); overflow: hidden;
  transition: border-color 0.25s;
}
.img-frame:hover { border-color: var(--line3); }
.img-frame img { width: 100%; display: block; filter: brightness(0.95); }
.img-frame-caption {
  padding: 10px 14px; border-top: 1px solid var(--line);
  font-size: 0.72rem; color: var(--ghost); font-family: var(--font-mono);
}

/* ══════════════════════════════════════════════════════
   INSIGHT CARDS
══════════════════════════════════════════════════════ */
.insight-card {
  padding: 16px 18px; border-radius: var(--radius-lg);
  background: var(--glass); border: 1px solid var(--line);
}
.insight-eyebrow { font-size: 0.67rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ghost); margin-bottom: 7px; font-family: var(--font-mono); }
.insight-value   { font-size: 1.05rem; font-weight: 800; margin-bottom: 4px; }
.insight-body    { font-size: 0.8rem; color: var(--ink3); line-height: 1.6; }

/* ══════════════════════════════════════════════════════
   FLOW PAGE SPECIFICS
══════════════════════════════════════════════════════ */
.stack-item { padding: 14px; border-radius: var(--radius-md); background: var(--glass); border: 1px solid var(--line); }
.stack-item .eyebrow { font-size: 0.67rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ghost); margin-bottom: 6px; font-family: var(--font-mono); }
.stack-item .big     { font-size: 0.96rem; font-weight: 700; margin-bottom: 3px; }
.stack-item .small   { font-size: 0.78rem; color: var(--ink3); line-height: 1.55; }
.tech-badge-row { display: flex; flex-wrap: wrap; gap: 7px; margin: 14px 0; }
.tech-badge {
  padding: 5px 12px; border-radius: 20px;
  border: 1px solid var(--line2); background: var(--glass);
  font-size: 0.74rem; color: var(--ink2); font-family: var(--font-mono);
}
.flow-note {
  background: rgba(109,91,232,0.05); border: 1px solid rgba(109,91,232,0.18);
  border-radius: var(--radius-lg); padding: 14px 16px;
  font-size: 0.83rem; color: var(--ink2); line-height: 1.65; margin-top: 16px;
}

/* ══════════════════════════════════════════════════════
   ANALYTICS PAGE
══════════════════════════════════════════════════════ */
.freq-bar-row   { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
.freq-label     { font-size: 0.74rem; color: var(--ink2); width: 68px; text-align: right; text-transform: capitalize; font-family: var(--font-mono); }
.freq-track     { flex: 1; height: 8px; background: rgba(255,255,255,0.06); border-radius: 4px; overflow: hidden; }
.freq-fill      { height: 100%; border-radius: 4px; transition: width 0.5s ease; }
.freq-count     { font-size: 0.72rem; color: var(--ghost); width: 22px; font-family: var(--font-mono); }

/* ══════════════════════════════════════════════════════
   DETECT WORKSPACE LAYOUT
══════════════════════════════════════════════════════ */
.detect-grid { display: grid; grid-template-columns: 1.05fr 0.95fr; gap: 20px; align-items: start; }
.detect-col  { display: grid; gap: 16px; }
.detect-pipe { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.pipe-mini { padding: 13px; border-radius: var(--radius-md); background: var(--glass); border: 1px solid var(--line); }
.pipe-mini-icon  { font-size: 1.05rem; margin-bottom: 8px; }
.pipe-mini-title { font-size: 0.875rem; font-weight: 700; margin-bottom: 3px; }
.pipe-mini-text  { font-size: 0.75rem; color: var(--ink3); line-height: 1.5; }
.kpi-list { display: grid; gap: 7px; margin-top: 8px; }
.kpi-row { display: flex; justify-content: space-between; align-items: center; padding: 9px 13px; border-radius: var(--radius-sm); background: var(--glass); border: 1px solid var(--line); }
.kpi-row .k { font-size: 0.74rem; color: var(--ink3); }
.kpi-row .v { font-size: 0.84rem; font-weight: 600; font-family: var(--font-mono); }
.badge-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 14px; }
.soft-tag { padding: 4px 10px; border-radius: 20px; border: 1px solid var(--line2); background: var(--glass); font-size: 0.72rem; color: var(--ink3); font-family: var(--font-mono); }

/* ══════════════════════════════════════════════════════
   ERROR + SPINNER
══════════════════════════════════════════════════════ */
.error-msg { background: rgba(255,68,102,0.09); border: 1px solid rgba(255,68,102,0.28); border-radius: var(--radius-sm); padding: 11px 14px; color: #ff7799; font-size: 0.82rem; margin-top: 12px; font-family: var(--font-mono); }
.spin { display: inline-block; width: 13px; height: 13px; border: 2px solid rgba(255,255,255,0.15); border-top-color: rgba(255,255,255,0.8); border-radius: 50%; animation: spin 0.65s linear infinite; vertical-align: middle; }
@keyframes spin { to { transform: rotate(360deg); } }

/* ══════════════════════════════════════════════════════
   SECTION HEADERS INSIDE CARDS
══════════════════════════════════════════════════════ */
.section-header {
  display: flex; align-items: flex-start; justify-content: space-between;
  gap: 12px; flex-wrap: wrap; margin-bottom: 18px;
}
.section-title { font-family: var(--font-sans); font-size: 1.05rem; font-weight: 700; letter-spacing: -0.02em; }
.section-sub   { font-size: 0.78rem; color: var(--ink3); margin-top: 3px; line-height: 1.55; }
.preview-shell {
  border-radius: var(--radius-lg); background: rgba(8,7,20,0.6);
  border: 1px solid var(--line); min-height: 180px;
  display: flex; align-items: center; justify-content: center; overflow: hidden;
}
.preview-empty { color: var(--ink3); font-size: 0.84rem; text-align: center; line-height: 1.8; padding: 28px; font-family: var(--font-mono); }

/* ══════════════════════════════════════════════════════
   SCROLLBAR
══════════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.18); }

/* ══════════════════════════════════════════════════════
   RESPONSIVE BREAKPOINTS
══════════════════════════════════════════════════════ */
@media(max-width:1280px) {
  .bento-grid { grid-template-columns: repeat(3,1fr); }
  .grid-7030, .grid-5545, .detect-grid { grid-template-columns: 1fr; }
  .grid-5 { grid-template-columns: repeat(3,1fr); }
}
@media(max-width:960px) {
  :root { --sidebar: 68px; }
  .sb-brand, .nav-label, .nav-badge { display: none; }
  .sb-header { padding: 20px 15px; justify-content: center; }
  .sb-logo   { margin: 0 auto; }
  .nav-btn   { padding: 12px; justify-content: center; }
  .sb-section, .sb-footer .footer-text { display: none; }
  .main { padding: 24px 20px; }
  .bento-grid { grid-template-columns: repeat(2,1fr); }
  .grid-2, .grid-3, .grid-4 { grid-template-columns: 1fr; }
  .now-playing { left: 68px; }
}
@media(max-width:600px) {
  .bento-grid { grid-template-columns: 1fr; }
  .grid-5 { grid-template-columns: repeat(2,1fr); }
  .pipeline-rail { flex-direction: column; }
  .pipeline-connector { display: none; }
}
</style>
</head>
<body>

<!-- ── BACKGROUND MESH ───────────────────────────────────────────────────── -->
<div class="bg-mesh">
  <div class="mesh-orb orb-a"></div>
  <div class="mesh-orb orb-b"></div>
  <div class="mesh-orb orb-c"></div>
</div>

<!-- ═══════════════════════════════════════════════════
     SIDEBAR
════════════════════════════════════════════════════ -->
<nav class="sidebar">
  <div class="sb-header">
    <div class="sb-logo">🎭</div>
    <div class="sb-brand">
      <div class="sb-brand-name">MoodMate</div>
      <div class="sb-brand-tag">Emotion Intelligence</div>
    </div>
  </div>

  <div class="sb-section"><span class="sb-section-label">Main</span></div>
  <div class="sb-nav">
    <div class="nav-btn active" onclick="showPage('detect')">
      <div class="nav-indicator"></div>
      <span class="nav-icon">🎭</span>
      <span class="nav-label">Detect</span>
    </div>
    <div class="nav-btn" onclick="showPage('flow')">
      <div class="nav-indicator"></div>
      <span class="nav-icon">⚡</span>
      <span class="nav-label">Detection Flow</span>
    </div>
    <div class="nav-btn" onclick="showPage('webcam')">
      <div class="nav-indicator"></div>
      <span class="nav-icon">📷</span>
      <span class="nav-label">Webcam Live</span>
    </div>
    <div class="nav-btn" onclick="showPage('analysis')">
      <div class="nav-indicator"></div>
      <span class="nav-icon">📊</span>
      <span class="nav-label">Analytics</span>
    </div>
    <div class="nav-btn" onclick="showPage('training')">
      <div class="nav-indicator"></div>
      <span class="nav-icon">🧠</span>
      <span class="nav-label">Training</span>
      <span class="nav-badge">50ep</span>
    </div>
    <div class="nav-btn" onclick="showPage('comparison')">
      <div class="nav-indicator"></div>
      <span class="nav-icon">🏆</span>
      <span class="nav-label">Comparison</span>
    </div>
  </div>

  <div class="sb-footer">
    <div class="status-pill">
      <span class="status-dot"></span>
      Model online
    </div>
  </div>
</nav>

<!-- ═══════════════════════════════════════════════════
     MAIN CONTENT
════════════════════════════════════════════════════ -->
<main class="main" id="main-content">

<!-- ─────────────────────────────────────────────────
     PAGE: DETECT
──────────────────────────────────────────────────── -->
<div class="page active" id="page-detect">
  <div class="page-hero">
    <div>
      <div class="page-eyebrow">Inference workspace</div>
      <div class="page-title">Emotion Detection</div>
      <div class="page-subtitle">Upload a face image — MoodMate detects emotion with confidence scores and instantly curates a matching playlist.</div>
    </div>
    <div class="hero-actions">
      <div class="glass-tag"><span class="live-dot"></span>Model ready</div>
      <div class="glass-tag">MobileNetV2 + CBAM</div>
    </div>
  </div>

  <!-- Bento Metrics -->
  <div class="bento-grid">
    <div class="bento-card bento-iris">
      <div class="bento-label">Input Mode</div>
      <div class="bento-value" style="font-size:1.15rem">Face Upload</div>
      <div class="bento-sub">JPG · PNG · WebP</div>
    </div>
    <div class="bento-card bento-nova">
      <div class="bento-label">Emotion Classes</div>
      <div class="bento-value">7</div>
      <div class="bento-sub">Full confidence map</div>
    </div>
    <div class="bento-card bento-cyan">
      <div class="bento-label">Latest Confidence</div>
      <div class="bento-value" id="detect-conf-stat">—</div>
      <div class="bento-sub">Updated on inference</div>
    </div>
    <div class="bento-card bento-em">
      <div class="bento-label">Top Emotion</div>
      <div class="bento-value" style="font-size:1.15rem;text-transform:capitalize" id="detect-emo-stat">—</div>
      <div class="bento-sub">Primary prediction</div>
    </div>
    <div class="bento-card bento-gold">
      <div class="bento-label">Session Runs</div>
      <div class="bento-value" id="detect-session-count">0</div>
      <div class="bento-sub">This session</div>
    </div>
  </div>

  <!-- Main workspace -->
  <div class="detect-grid">
    <!-- Left column -->
    <div class="detect-col">
      <div class="card">
        <div class="section-header">
          <div>
            <div class="card-label">Upload Face Image</div>
            <div class="section-sub">Clear front-facing photo for best results</div>
          </div>
          <span class="chip">Static detection</span>
        </div>
        <div class="upload-zone" id="drop-zone">
          <input type="file" id="fileInput" accept="image/*" onchange="onFileSelected(event)"/>
          <div class="upload-icon-ring">📸</div>
          <div class="upload-heading">Click or drag &amp; drop</div>
          <div class="upload-hint">Supports JPG, PNG, WebP · Max 10MB</div>
        </div>
        <div class="preview-shell" style="margin-top:14px">
          <img id="preview-img" alt="preview"/>
          <div class="preview-empty" id="preview-empty">Your image preview will appear here.<br/>Choose a clear face photo before detecting.</div>
        </div>
        <button class="btn btn-primary" id="detectBtn" onclick="runDetection()" disabled>🔎 Detect Emotion &amp; Play Music</button>
        <div id="detect-error"></div>
      </div>

      <div class="card" id="emotion-card" style="display:none">
        <div class="section-header">
          <div class="card-label" style="margin:0">Detection Result</div>
          <span class="chip green">Latest output</span>
        </div>
        <div class="emotion-result">
          <span class="emo-emoji" id="res-emoji"></span>
          <div>
            <div class="emo-label" id="res-name"></div>
            <div class="emo-conf" id="res-conf"></div>
            <div class="emo-mood" id="res-mood"></div>
          </div>
        </div>
        <div class="card-label" style="margin-top:16px">Confidence Scores</div>
        <div id="conf-bars"></div>
        <div class="card-label" style="margin-top:16px">Score Distribution</div>
        <div class="chart-shell h200"><canvas id="emotionChart"></canvas></div>
      </div>
    </div>

    <!-- Right column -->
    <div class="detect-col">
      <div class="card">
        <div class="section-header">
          <div>
            <div class="card-label" style="margin-bottom:4px">Inference Pipeline</div>
            <div class="section-sub">How MoodMate processes your image end-to-end.</div>
          </div>
          <span class="chip cyan">Demo ready</span>
        </div>
        <div class="detect-pipe">
          <div class="pipe-mini"><div class="pipe-mini-icon">📤</div><div class="pipe-mini-title">Upload</div><div class="pipe-mini-text">Image enters the Flask route.</div></div>
          <div class="pipe-mini"><div class="pipe-mini-icon">🧠</div><div class="pipe-mini-title">Detect</div><div class="pipe-mini-text">OpenCV isolates face region.</div></div>
          <div class="pipe-mini"><div class="pipe-mini-icon">⚙️</div><div class="pipe-mini-title">Process</div><div class="pipe-mini-text">Crop, resize, normalise.</div></div>
          <div class="pipe-mini"><div class="pipe-mini-icon">🎯</div><div class="pipe-mini-title">Predict</div><div class="pipe-mini-text">Model returns emotion + scores.</div></div>
        </div>
        <div class="kpi-list">
          <div class="kpi-row"><span class="k">Model</span><span class="v">MobileNetV2 + CBAM / FER-2013</span></div>
          <div class="kpi-row"><span class="k">Output</span><span class="v">Emotion · confidence · songs</span></div>
        </div>
        <div class="badge-row">
          <span class="soft-tag">7 emotions</span>
          <span class="soft-tag">Transfer learning</span>
          <span class="soft-tag">OpenCV crop</span>
          <span class="soft-tag">Chart.js</span>
          <span class="soft-tag">Spotify preview</span>
        </div>
      </div>

      <div class="card" id="songs-card" style="display:none">
        <div class="section-header">
          <div class="card-label" style="margin:0">Recommended Songs</div>
          <span id="songs-badge" class="chip"></span>
        </div>
        <div class="song-list-wrap">
          <div id="song-list"></div>
        </div>
      </div>
    </div>
  </div>
</div>


<!-- ─────────────────────────────────────────────────
     PAGE: FLOW
──────────────────────────────────────────────────── -->
<div class="page" id="page-flow">
  <div class="page-hero">
    <div>
      <div class="page-eyebrow">System architecture</div>
      <div class="page-title">Detection Flow</div>
      <div class="page-subtitle">A presentation-ready pipeline map — how MoodMate transforms a raw image into emotion-aware music.</div>
    </div>
    <div class="glass-tag">End-to-end inference</div>
  </div>

  <div class="grid-7030" style="align-items:start">
    <div class="card">
      <div class="card-label">Inference Journey</div>
      <div class="section-header" style="margin-bottom:0">
        <div>
          <div class="section-title">From image upload to curated music</div>
          <div class="section-sub">The exact MoodMate inference path — clear for demos, vivals, and investor presentations.</div>
        </div>
      </div>
      <div class="pipeline-rail">
        <div class="pipeline-node">
          <div class="pipeline-connector"></div>
          <div class="pipe-badge"><span class="pipe-num">1</span>📤</div>
          <div class="pipe-title">Upload</div>
          <div class="pipe-desc">User selects a face image or webcam frame.</div>
        </div>
        <div class="pipeline-node">
          <div class="pipeline-connector"></div>
          <div class="pipe-badge"><span class="pipe-num">2</span>🧠</div>
          <div class="pipe-title">Detect</div>
          <div class="pipe-desc">OpenCV Haar cascade finds strongest face.</div>
        </div>
        <div class="pipeline-node">
          <div class="pipeline-connector"></div>
          <div class="pipe-badge"><span class="pipe-num">3</span>⚙️</div>
          <div class="pipe-title">Preprocess</div>
          <div class="pipe-desc">Crop, resize to 96×96, normalize RGB.</div>
        </div>
        <div class="pipeline-node">
          <div class="pipeline-connector"></div>
          <div class="pipe-badge"><span class="pipe-num">4</span>🎯</div>
          <div class="pipe-title">Predict</div>
          <div class="pipe-desc">CNN returns top emotion + softmax scores.</div>
        </div>
        <div class="pipeline-node">
          <div class="pipe-badge"><span class="pipe-num">5</span>🎵</div>
          <div class="pipe-title">Recommend</div>
          <div class="pipe-desc">Mood mapped → matched songs surfaced.</div>
        </div>
      </div>

      <div class="tech-badge-row">
        <span class="tech-badge">Image / Webcam</span>
        <span style="color:var(--ghost)">→</span>
        <span class="tech-badge">Face detection</span>
        <span style="color:var(--ghost)">→</span>
        <span class="tech-badge">CNN prediction</span>
        <span style="color:var(--ghost)">→</span>
        <span class="tech-badge">Music</span>
      </div>

      <div class="grid-3" style="margin-top:16px">
        <div class="stack-item"><div class="eyebrow">Input</div><div class="big">Face image</div><div class="small">Image or webcam frame into Flask.</div></div>
        <div class="stack-item"><div class="eyebrow">Processing</div><div class="big">Detect → crop → resize</div><div class="small">Strongest face region only.</div></div>
        <div class="stack-item"><div class="eyebrow">Output</div><div class="big">Emotion + songs</div><div class="small">Top mood, confidence, matched tracks.</div></div>
      </div>

      <div class="flow-note"><strong>In one sentence:</strong> MoodMate takes a face image, finds the face, predicts the emotion, shows full confidence scores, and instantly surfaces a matching playlist — in a dashboard that feels like a real product.</div>
    </div>

    <div style="display:grid;gap:16px">
      <div class="card">
        <div class="card-label">Technology Stack</div>
        <div style="display:grid;gap:10px">
          <div class="stack-item"><div class="eyebrow">Frontend</div><div class="big">HTML · CSS · JS</div><div class="small">Dashboard, upload UI, charts, music player.</div></div>
          <div class="stack-item"><div class="eyebrow">Backend</div><div class="big">Flask + Python</div><div class="small">Image upload, JSON endpoints, webcam routes.</div></div>
          <div class="stack-item"><div class="eyebrow">Vision</div><div class="big">OpenCV</div><div class="small">Face detection, cropping, frame annotation.</div></div>
          <div class="stack-item"><div class="eyebrow">Model</div><div class="big">TensorFlow / Keras</div><div class="small">MobileNetV2 + CBAM on FER-2013. 7 emotions.</div></div>
        </div>
      </div>
      <div class="card">
        <div class="card-label">Dashboard sections</div>
        <div class="badge-row" style="margin-top:0">
          <span class="soft-tag">Emotion result</span>
          <span class="soft-tag">Confidence bars</span>
          <span class="soft-tag">Chart view</span>
          <span class="soft-tag">Song recommendations</span>
          <span class="soft-tag">Analytics</span>
          <span class="soft-tag">Training report</span>
          <span class="soft-tag">Model comparison</span>
        </div>
      </div>
    </div>
  </div>
</div>


<!-- ─────────────────────────────────────────────────
     PAGE: WEBCAM
──────────────────────────────────────────────────── -->
<div class="page" id="page-webcam">
  <div class="page-hero">
    <div>
      <div class="page-eyebrow">Live inference</div>
      <div class="page-title">Webcam Detection</div>
      <div class="page-subtitle">Real-time emotion detection from your webcam with live face-box overlay and instant music recommendation.</div>
    </div>
    <div class="glass-tag"><span class="live-dot" id="wc-dot" style="background:var(--ghost);box-shadow:none"></span><span id="wc-pill">Inactive</span></div>
  </div>
  <div class="grid-2" style="align-items:start">
    <div style="display:grid;gap:16px">
      <div class="card">
        <div class="card-label">Live Feed</div>
        <video id="webcam-video" autoplay playsinline style="display:none"></video>
        <canvas id="webcam-canvas" width="520" height="360"></canvas>
        <p id="webcam-status" style="font-size:.82rem;color:var(--ink3);margin:10px 0;text-align:center;font-family:var(--font-mono)">Click Start to activate your webcam.</p>
        <button class="btn btn-primary" id="webcamBtn" onclick="toggleWebcam()">▶ Start Webcam</button>
      </div>
      <div class="card" id="webcam-emotion-card" style="display:none">
        <div class="card-label">Detected Emotion</div>
        <div class="emotion-result">
          <span class="emo-emoji" id="wc-emoji"></span>
          <div>
            <div class="emo-label" id="wc-name"></div>
            <div class="emo-conf" id="wc-conf"></div>
            <div class="emo-mood" id="wc-mood"></div>
          </div>
        </div>
      </div>
    </div>
    <div class="card" id="wc-songs-card" style="display:none">
      <div class="card-label">Recommended Songs</div>
      <div class="song-list-wrap"><div id="wc-song-list"></div></div>
    </div>
  </div>
</div>


<!-- ─────────────────────────────────────────────────
     PAGE: ANALYSIS
──────────────────────────────────────────────────── -->
<div class="page" id="page-analysis">
  <div class="page-hero">
    <div>
      <div class="page-eyebrow">Session intelligence</div>
      <div class="page-title">Analytics Dashboard</div>
      <div class="page-subtitle">Live session overview plus final trained model metrics — ready for project demos and presentations.</div>
    </div>
    <div class="hero-actions">
      <div class="glass-tag">Session + model</div>
      <div class="glass-tag"><span class="live-dot"></span>Demo ready</div>
    </div>
  </div>

  <div class="bento-grid">
    <div class="bento-card bento-iris"><div class="bento-label">Detections</div><div class="bento-value" id="stat-total">0</div><div class="bento-sub">This session</div></div>
    <div class="bento-card bento-nova"><div class="bento-label">Top Emotion</div><div class="bento-value" id="stat-top">—</div><div class="bento-sub">Most frequent</div></div>
    <div class="bento-card bento-cyan"><div class="bento-label">Avg Confidence</div><div class="bento-value" id="stat-conf">—</div><div class="bento-sub">Mean across session</div></div>
    <div class="bento-card bento-em"><div class="bento-label">Songs Played</div><div class="bento-value" id="stat-played">0</div><div class="bento-sub">Previews triggered</div></div>
    <div class="bento-card bento-gold"><div class="bento-label">Best Val Acc</div><div class="bento-value">67.5%</div><div class="bento-sub">Final trained model</div></div>
  </div>

  <div class="grid-2" style="margin-bottom:16px">
    <div class="card">
      <div class="section-header">
        <div><div class="card-label" style="margin-bottom:3px">Session timeline</div><div class="section-title">Confidence over time</div></div>
        <div class="glass-tag" style="font-size:.72rem">Live</div>
      </div>
      <div class="chart-shell h240"><canvas id="timelineChart"></canvas></div>
    </div>
    <div class="card">
      <div class="section-header"><div><div class="card-label" style="margin-bottom:3px">Emotion frequency</div><div class="section-title">Session breakdown</div></div></div>
      <div id="emotion-freq" style="padding-top:8px"><p style="color:var(--ink3);font-size:.84rem;font-family:var(--font-mono)">No detections yet.</p></div>
    </div>
  </div>

  <div class="grid-2" style="margin-bottom:16px">
    <div class="card">
      <div class="section-header"><div><div class="card-label" style="margin-bottom:3px">Final model</div><div class="section-title">Accuracy comparison</div></div></div>
      <div class="chart-shell h220"><canvas id="modelAccuracyChart"></canvas></div>
    </div>
    <div class="card">
      <div class="section-header"><div><div class="card-label" style="margin-bottom:3px">Final model</div><div class="section-title">Loss comparison</div></div></div>
      <div class="chart-shell h220"><canvas id="modelLossChart"></canvas></div>
    </div>
  </div>

  <div class="grid-2" style="margin-bottom:16px">
    <div class="card">
      <div class="section-header"><div><div class="card-label" style="margin-bottom:3px">Class distribution</div><div class="section-title">FER-2013 emotion split</div></div></div>
      <div class="chart-shell h220"><canvas id="analysisClassChart"></canvas></div>
    </div>
    <div class="card">
      <div class="section-header"><div><div class="card-label" style="margin-bottom:3px">Model summary</div><div class="section-title">Final trained values</div></div></div>
      <div id="analysis-model-summary" style="display:grid;grid-template-columns:1fr 1fr;gap:10px"></div>
    </div>
  </div>

  <div class="card">
    <div class="section-header">
      <div><div class="card-label" style="margin-bottom:3px">History</div><div class="section-title">Session detection log</div></div>
    </div>
    <div id="history-list"></div>
  </div>
</div>


<!-- ─────────────────────────────────────────────────
     PAGE: TRAINING
──────────────────────────────────────────────────── -->
<div class="page" id="page-training">
  <div class="page-hero">
    <div>
      <div class="page-eyebrow">Model training</div>
      <div class="page-title">Training Report</div>
      <div class="page-subtitle">Complete training run — MobileNetV2 + CBAM attention fine-tuned on FER-2013 / FER+ over 50 epochs.</div>
    </div>
    <div class="hero-actions">
      <div class="glass-tag"><span class="live-dot"></span>Completed · 50 epochs</div>
      <button class="btn btn-secondary btn-sm" onclick="exportTrainingReport()">⬇ Export</button>
    </div>
  </div>

  <!-- Top metrics -->
  <div class="grid-5" style="margin-bottom:20px">
    <div class="bento-card bento-iris"><div class="bento-label">Training set</div><div class="bento-value" id="m-train">28,709</div><div class="bento-sub">FER-2013 images</div></div>
    <div class="bento-card bento-nova"><div class="bento-label">Test set</div><div class="bento-value" id="m-test">3,589</div><div class="bento-sub">Held-out</div></div>
    <div class="bento-card bento-cyan"><div class="bento-label">Best accuracy</div><div class="bento-value" id="m-acc">67.5%</div><div class="bento-sub" id="m-epoch">@ Epoch 27</div></div>
    <div class="bento-card bento-em"><div class="bento-label">Training time</div><div class="bento-value" id="m-time">~2h 56m</div><div class="bento-sub">Full run duration</div></div>
    <div class="bento-card bento-gold"><div class="bento-label">Status</div><div class="bento-value" style="font-size:1.1rem" id="m-status">Completed</div><div class="bento-sub">Ready for inference</div></div>
  </div>

  <div class="grid-5545" style="margin-bottom:16px">
    <!-- Actual training curve image -->
    <div class="card">
      <div class="section-header">
        <div><div class="card-label" style="margin-bottom:3px">Trained graph</div><div class="section-title">Actual training curves from your run</div></div>
        <div class="chip cyan">Epochs 1–50</div>
      </div>
      <div class="img-frame">
        <img src="/models/training_curves_v2.png" alt="Training curves"
          onerror="this.onerror=null;this.src='/models/training_curves.png';this.onerror=function(){this.closest('.img-frame').innerHTML='<div style=\'padding:40px;text-align:center;color:var(--ink3);font-family:var(--font-mono);font-size:.82rem\'>📊 Place training_curves.png in models/ folder</div>'}"/>
        <div class="img-frame-caption">accuracy_curve · loss_curve · training_curves_v2.png</div>
      </div>
    </div>
    <!-- Snapshot values -->
    <div class="card">
      <div class="section-header"><div><div class="card-label" style="margin-bottom:3px">Snapshot</div><div class="section-title">Best and final values</div></div></div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px" id="snapshot-grid"></div>
    </div>
  </div>

  <div class="grid-2" style="margin-bottom:16px">
    <div class="card">
      <div class="section-header"><div><div class="card-label" style="margin-bottom:3px">Exact metrics</div><div class="section-title">Final trained values</div></div><div class="chip">No clutter</div></div>
      <div class="kv-table" id="training-kv"></div>
    </div>
    <div class="card">
      <div class="section-header"><div><div class="card-label" style="margin-bottom:3px">Final epochs</div><div class="section-title">Epoch progress table</div></div></div>
      <div class="data-table-wrap"><table class="data-table" id="epoch-table"></table></div>
    </div>
  </div>

  <div class="grid-2" style="margin-bottom:16px">
    <div class="card">
      <div class="section-header"><div><div class="card-label" style="margin-bottom:3px">Class distribution</div><div class="section-title">FER-2013 emotion split</div></div></div>
      <div class="chart-shell h220"><canvas id="classDistChart"></canvas></div>
    </div>
    <div class="card">
      <div class="section-header">
        <div><div class="card-label" style="margin-bottom:3px">Model config + score</div><div class="section-title">Architecture overview</div></div>
      </div>
      <div style="display:grid;grid-template-columns:1fr auto;gap:20px;align-items:center">
        <div class="kv-table" id="model-kv"></div>
        <div style="display:grid;place-items:center">
          <div class="score-ring-outer" id="ring-outer">
            <div class="score-ring-inner">
              <div><span class="ring-val" id="ring-val">67.5%</span><span class="ring-sub">Best Accuracy</span><span class="ring-sub" id="ring-ep">@ Epoch 27</span></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="section-header"><div><div class="card-label" style="margin-bottom:3px">Emotion class detail</div><div class="section-title">Sample counts per class</div></div><div class="chip">7 classes</div></div>
    <div class="class-grid" id="class-detail"></div>
  </div>
</div>


<!-- ─────────────────────────────────────────────────
     PAGE: COMPARISON
──────────────────────────────────────────────────── -->
<div class="page" id="page-comparison">
  <div class="page-hero">
    <div>
      <div class="page-eyebrow">Benchmark suite</div>
      <div class="page-title">Model Comparison</div>
      <div class="page-subtitle">Evaluation workspace comparing the deep transfer-learning model against classical ML baselines — accuracy, ROC-AUC, confusion matrices, and feature importance.</div>
    </div>
    <div class="hero-actions">
      <div class="glass-tag"><span class="live-dot"></span>Final benchmark</div>
      <div class="glass-tag">5 models</div>
    </div>
  </div>

  <!-- Scoreboard -->
  <div class="bento-grid">
    <div class="bento-card bento-iris"><div class="bento-label">Best deep model</div><div class="bento-value" id="cmp-deep">67.49%</div><div class="bento-sub">MobileNetV2 + CBAM</div></div>
    <div class="bento-card bento-nova"><div class="bento-label">Best classical</div><div class="bento-value" id="cmp-class">56.97%</div><div class="bento-sub">RandomForest</div></div>
    <div class="bento-card bento-cyan"><div class="bento-label">Accuracy gap</div><div class="bento-value" id="cmp-gap">+10.52pp</div><div class="bento-sub">Deep advantage</div></div>
    <div class="bento-card bento-em"><div class="bento-label">Best ROC-AUC</div><div class="bento-value" id="cmp-roc">0.803</div><div class="bento-sub">RandomForest</div></div>
    <div class="bento-card bento-gold"><div class="bento-label">Best CV Score</div><div class="bento-value">0.542</div><div class="bento-sub">RandomForest</div></div>
  </div>

  <div class="card">
    <div class="section-header">
      <div><div class="card-label" style="margin-bottom:3px">Interactive benchmark</div><div class="section-title">Model comparison workspace</div></div>
      <div class="glass-tag" style="font-size:.72rem">Project-specific</div>
    </div>
    <div class="comp-tabs" id="comp-tabs">
      <button class="comp-tab active" data-tab="overview"  onclick="setTab('overview')">🎯 Overview</button>
      <button class="comp-tab" data-tab="accuracy"         onclick="setTab('accuracy')">📈 Accuracy & F1</button>
      <button class="comp-tab" data-tab="error"            onclick="setTab('error')">⚠️ Error & ROC</button>
      <button class="comp-tab" data-tab="confusion"        onclick="setTab('confusion')">🧩 Confusion Matrices</button>
      <button class="comp-tab" data-tab="feature"          onclick="setTab('feature')">🔥 Feature Importance</button>
      <button class="comp-tab" data-tab="curves"           onclick="setTab('curves')">🧠 Training Curves</button>
    </div>

    <!-- Overview tab -->
    <div class="comp-panel active" id="comp-panel-overview">
      <div class="grid-7030">
        <div>
          <div class="card-label">Model Leaderboard</div>
          <div class="data-table-wrap">
            <table class="comp-table" id="overview-table"></table>
          </div>
        </div>
        <div style="display:grid;gap:12px">
          <div class="insight-card"><div class="insight-eyebrow">Winner</div><div class="insight-value">Deep model leads</div><div class="insight-body">Transfer learning outperforms every classical baseline. CNN spatial representation is superior for facial emotion recognition over flattened pixel vectors.</div></div>
          <div class="insight-card"><div class="insight-eyebrow">Strongest baseline</div><div class="insight-value">RandomForest</div><div class="insight-body">Best balance of accuracy and ROC-AUC among classical models — the strongest benchmark reference.</div></div>
          <div class="insight-card"><div class="insight-eyebrow">Main difficulty</div><div class="insight-value">Subtle emotions</div><div class="insight-body">Confusions remain strongest around fear, sadness, and neutral. Consistent across both deep and classical models.</div></div>
        </div>
      </div>
    </div>

    <!-- Accuracy tab -->
    <div class="comp-panel" id="comp-panel-accuracy">
      <div class="grid-2">
        <div class="card"><div class="section-header"><div><div class="card-label" style="margin-bottom:3px">Performance</div><div class="section-title">Accuracy comparison</div></div></div><div class="chart-shell h280"><canvas id="accChart"></canvas></div></div>
        <div class="card"><div class="section-header"><div><div class="card-label" style="margin-bottom:3px">Weighted F1</div><div class="section-title">F1-score comparison</div></div></div><div class="chart-shell h280"><canvas id="f1Chart"></canvas></div></div>
      </div>
    </div>

    <!-- Error tab -->
    <div class="comp-panel" id="comp-panel-error">
      <div class="grid-2">
        <div class="card"><div class="section-header"><div><div class="card-label" style="margin-bottom:3px">Error profile</div><div class="section-title">Classification error rate</div></div></div><div class="chart-shell h280"><canvas id="errChart"></canvas></div></div>
        <div class="card"><div class="section-header"><div><div class="card-label" style="margin-bottom:3px">Separation power</div><div class="section-title">ROC-AUC and CV Score</div></div></div><div class="chart-shell h280"><canvas id="rocChart"></canvas></div></div>
      </div>
    </div>

    <!-- Confusion Matrix tab — REAL IMAGES -->
    <div class="comp-panel" id="comp-panel-confusion">
      <div class="grid-2" style="margin-bottom:16px">
        <div>
          <div class="card-label" style="margin-bottom:10px">Deep Model</div>
          <div class="img-frame">
            <img src="/models/confusion_matrix.png" alt="MobileNetV2 confusion matrix"/>
            <div class="img-frame-caption">MobileNetV2 + CBAM · Best checkpoint · confusion_matrix.png</div>
          </div>
        </div>
        <div>
          <div class="card-label" style="margin-bottom:10px">RandomForest (Best Classical)</div>
          <div class="img-frame">
            <img src="/models/classical_analysis/RandomForest_confusion_matrix.png"
              onerror="this.onerror=null;this.src='/models/RandomForest_confusion_matrix.png'" alt="RandomForest confusion matrix"/>
            <div class="img-frame-caption">RandomForest · Best classical baseline · RandomForest_confusion_matrix.png</div>
          </div>
        </div>
      </div>
      <div class="grid-2">
        <div>
          <div class="card-label" style="margin-bottom:10px">Logistic Regression</div>
          <div class="img-frame">
            <img src="/models/classical_analysis/LogisticRegression_confusion_matrix.png"
              onerror="this.onerror=null;this.src='/models/LogisticRegression_confusion_matrix.png'" alt="Logistic regression confusion matrix"/>
            <div class="img-frame-caption">LogisticRegression_confusion_matrix.png</div>
          </div>
        </div>
        <div>
          <div class="card-label" style="margin-bottom:10px">LinearSVC</div>
          <div class="img-frame">
            <img src="/models/classical_analysis/LinearSVC_confusion_matrix.png"
              onerror="this.onerror=null;this.src='/models/LinearSVC_confusion_matrix.png'" alt="LinearSVC confusion matrix"/>
            <div class="img-frame-caption">LinearSVC_confusion_matrix.png</div>
          </div>
        </div>
      </div>
      <div style="margin-top:16px">
        <div class="card-label" style="margin-bottom:10px">KNN</div>
        <div class="img-frame" style="max-width:520px">
          <img src="/models/classical_analysis/KNN_confusion_matrix.png"
            onerror="this.onerror=null;this.src='/models/KNN_confusion_matrix.png'" alt="KNN confusion matrix"/>
          <div class="img-frame-caption">KNN_confusion_matrix.png</div>
        </div>
      </div>
    </div>

    <!-- Feature Importance tab — REAL IMAGES -->
    <div class="comp-panel" id="comp-panel-feature">
      <div class="grid-2">
        <div>
          <div class="card-label" style="margin-bottom:10px">RandomForest — Feature Importance (Top 30)</div>
          <div class="img-frame">
            <img src="/models/classical_analysis/RandomForest_feature_importance.png"
              onerror="this.onerror=null;this.src='/models/RandomForest_feature_importance.png'" alt="Feature importance"/>
            <div class="img-frame-caption">Tree-based pixel relevance · Top 30 features · RandomForest_feature_importance.png</div>
          </div>
        </div>
        <div>
          <div class="card-label" style="margin-bottom:10px">Importance Heatmap (48×48)</div>
          <div class="img-frame">
            <img src="/models/classical_analysis/RandomForest_importance_heatmap.png"
              onerror="this.onerror=null;this.src='/models/RandomForest_importance_heatmap.png'" alt="Importance heatmap"/>
            <div class="img-frame-caption">Spatial pixel relevance projected on face grid · RandomForest_importance_heatmap.png</div>
          </div>
        </div>
      </div>
      <div style="margin-top:16px" class="grid-3">
        <div class="insight-card"><div class="insight-eyebrow">Key insight</div><div class="insight-value">Eye region dominates</div><div class="insight-body">Highest-importance pixels cluster around the eyes and brow area — consistent with human emotion perception research.</div></div>
        <div class="insight-card"><div class="insight-eyebrow">Mouth region</div><div class="insight-value">Second tier</div><div class="insight-body">Mouth corners contribute significantly for happy vs. sad discrimination — the feature map confirms this spatial pattern.</div></div>
        <div class="insight-card"><div class="insight-eyebrow">Background noise</div><div class="insight-value">Minimal impact</div><div class="insight-body">Background pixels show near-zero importance, validating that the face crop preprocessing step is effective.</div></div>
      </div>
    </div>

    <!-- Training Curves tab — REAL IMAGE -->
    <div class="comp-panel" id="comp-panel-curves">
      <div class="grid-2">
        <div>
          <div class="card-label" style="margin-bottom:10px">Training vs Validation Curves</div>
          <div class="img-frame">
            <img src="/models/training_curves.png" alt="Training curves"/>
            <div class="img-frame-caption">Deep model · Accuracy &amp; Loss over 50 epochs · training_curves.png</div>
          </div>
        </div>
        <div style="display:grid;gap:12px;align-content:start">
          <div class="insight-card"><div class="insight-eyebrow">Generalization</div><div class="insight-value">Healthy curve gap</div><div class="insight-body">Training and validation metrics move together, confirming the model learns meaningful facial features instead of memorising only the training set.</div></div>
          <div class="insight-card"><div class="insight-eyebrow">Deep advantage</div><div class="insight-value">Spatial feature learning</div><div class="insight-body">The CNN pipeline preserves facial structure. Classical baselines flatten the face into raw pixel vectors, losing all spatial relationships.</div></div>
          <div class="insight-card"><div class="insight-eyebrow">Dashboard takeaway</div><div class="insight-value">Deep model for production</div><div class="insight-body">Keep RandomForest as the strongest benchmark reference, but deploy the deep model as the final MoodMate inference engine.</div></div>
        </div>
      </div>
    </div>
  </div>
</div>

</main>

<!-- ═══════════════════════════════════════════════════
     NOW PLAYING BAR
════════════════════════════════════════════════════ -->
<div class="now-playing" id="now-playing">
  <div class="np-thumb">🎵</div>
  <div class="np-info">
    <div class="np-title" id="np-title">No song playing</div>
    <div class="np-artist" id="np-artist"></div>
  </div>
  <div class="np-ctrl">
    <button class="np-btn" onclick="prevTrack()">⏮</button>
    <button class="np-btn np-play-btn" id="np-play" onclick="togglePlay()">⏸</button>
    <button class="np-btn" onclick="nextTrack()">⏭</button>
  </div>
  <div class="np-progress">
    <span class="np-time" id="np-time-cur">0:00</span>
    <div class="np-bar" id="np-bar"><div class="np-bar-fill" id="np-bar-fill"></div></div>
    <span class="np-time" id="np-time-tot">0:00</span>
  </div>
  <button class="np-close" onclick="stopPlayer()">✕</button>
</div>

<audio id="audio-player"></audio>

<!-- ═══════════════════════════════════════════════════
     SCRIPTS
════════════════════════════════════════════════════ -->
<script>
/* ══════════════════════════
   CONSTANTS + STATE
══════════════════════════ */
const EmoColors = {
  angry:'#ff4466', disgust:'#ff9f4a', fear:'#a78bfa',
  happy:'#fbbf24', neutral:'#94a3b8', sad:'#60a5fa', surprise:'#34d399'
};
const CLASS_COUNTS = {
  angry:2097, disgust:163, fear:555, happy:6399,
  neutral:8762, sad:2987, surprise:3028
};
const TD = {
  architecture:'MobileNetV2', technique:'Transfer Learning + CBAM',
  trainCount:23991, testCount:7048, epochs:50,
  bestAcc:67.49, bestEpoch:27,
  finalTrainAcc:65.43, finalValAcc:67.49,
  finalTrainLoss:0.1325, finalValLoss:0.1512,
  time:'~2h 56m', lr:'0.0003 → 0.00005',
  inputShape:'96 × 96 × 3 (RGB)', classes:7,
  optimizer:'AdamW', loss:'Focal Categorical Crossentropy',
  batch:48, dataset:'FER-2013 / FER+', status:'Completed',
  epochs_data:[
    {e:26,ta:0.6217,va:0.6657,tl:0.1479,vl:0.1539,lr:'5e-5'},
    {e:27,ta:0.6315,va:0.6749,tl:0.1429,vl:0.1512,lr:'5e-5',best:true},
    {e:28,ta:0.6394,va:0.6522,tl:0.1409,vl:0.1581,lr:'5e-5'},
    {e:29,ta:0.6480,va:0.6263,tl:0.1358,vl:0.1691,lr:'5e-5'},
    {e:50,ta:0.6543,va:0.6354,tl:0.1325,vl:0.1631,lr:'2.5e-5'}
  ]
};
const CMP = {
  deep:{ name:'MobileNetV2 + CBAM', accuracy:0.6749, errorRate:0.3251, precision:0.6927, recall:0.6749, f1:0.6784, rocAuc:null, cvMean:null },
  classical:[
    {name:'RandomForest', accuracy:0.5697, errorRate:0.4303, f1:0.5304, rocAuc:0.8030, cvMean:0.5420},
    {name:'KNN',          accuracy:0.5159, errorRate:0.4841, f1:0.5020, rocAuc:0.7248, cvMean:0.4819},
    {name:'LinearSVC',    accuracy:0.3953, errorRate:0.6047, f1:0.4086, rocAuc:0.6817, cvMean:0.3817},
    {name:'LogisticRegression', accuracy:0.3726, errorRate:0.6274, f1:0.3928, rocAuc:0.6774, cvMean:0.3599}
  ]
};

const state = {
  selectedFile:null, currentSongs:[], wcSongs:[],
  sessionHistory:[], emotionCounts:{}, confidences:[], songsPlayed:0
};
const charts = {};
const audio  = document.getElementById('audio-player');
let playQueue=[], playQueueIdx=-1;

const $  = id => document.getElementById(id);
const esc = s => String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const fmt = s  => { const m=Math.floor(s/60),sec=Math.floor(s%60); return m+':'+(sec<10?'0':'')+sec; };
const pc  = v  => (v*100).toFixed(1)+'%';
function destroyChart(k){ if(charts[k]){ try{charts[k].destroy();}catch(e){} charts[k]=null; } }
function killCanvas(id){
  const c=$(id); if(!c) return;
  const ex=Chart.getChart(c); if(ex) ex.destroy();
}

/* ══════════════════════════
   CHART DEFAULTS
══════════════════════════ */
const chartDefaults = {
  responsive:true, maintainAspectRatio:false,
  plugins:{ legend:{ display:false }, tooltip:{ backgroundColor:'rgba(12,11,24,0.95)', borderColor:'rgba(255,255,255,0.1)', borderWidth:1, titleColor:'#f5f3ff', bodyColor:'#c4bfe8', padding:12, cornerRadius:8 } },
  scales:{
    x:{ grid:{ color:'rgba(255,255,255,0.04)' }, ticks:{ color:'#4a4570', font:{ family:"'JetBrains Mono',monospace", size:11 } } },
    y:{ grid:{ color:'rgba(255,255,255,0.04)' }, ticks:{ color:'#4a4570', font:{ family:"'JetBrains Mono',monospace", size:11 } } }
  }
};

/* ══════════════════════════
   PAGE NAVIGATION
══════════════════════════ */
function showPage(name) {
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(n=>n.classList.remove('active'));
  const pg=$('page-'+name); if(pg) pg.classList.add('active');
  const pages=['detect','flow','webcam','analysis','training','comparison'];
  document.querySelectorAll('.nav-btn')[pages.indexOf(name)]?.classList.add('active');
  if(name==='analysis')   renderAnalysis();
  if(name==='training')   renderTraining();
  if(name==='comparison') renderComparison();
}

/* ══════════════════════════
   FILE UPLOAD
══════════════════════════ */
function onFileSelected(e) {
  state.selectedFile = e.target.files[0]; if(!state.selectedFile) return;
  const reader = new FileReader();
  reader.onload = ev => {
    const img=$('preview-img'); img.src=ev.target.result; img.style.display='block';
    const em=$('preview-empty'); if(em) em.style.display='none';
  };
  reader.readAsDataURL(state.selectedFile);
  $('detectBtn').disabled=false;
  $('emotion-card').style.display='none';
  $('songs-card').style.display='none';
  $('detect-error').innerHTML='';
}
const dz=$('drop-zone');
dz.addEventListener('dragover', e=>{e.preventDefault();dz.classList.add('drag-over');});
dz.addEventListener('dragleave',()=>dz.classList.remove('drag-over'));
dz.addEventListener('drop',e=>{
  e.preventDefault(); dz.classList.remove('drag-over');
  const f=e.dataTransfer.files[0];
  if(f){state.selectedFile=f;const dt=new DataTransfer();dt.items.add(f);$('fileInput').files=dt.files;onFileSelected({target:{files:[f]}});}
});

/* ══════════════════════════
   DETECTION
══════════════════════════ */
async function runDetection() {
  const btn=$('detectBtn');
  btn.disabled=true; btn.innerHTML='<span class="spin"></span> Analyzing…';
  $('detect-error').innerHTML='';
  const fd=new FormData(); fd.append('image',state.selectedFile);
  try {
    const resp=await fetch('/predict',{method:'POST',body:fd});
    const text=await resp.text(); let data;
    try{data=JSON.parse(text);}catch(e){throw new Error('Server error — check terminal.');}
    if(!resp.ok||data.error) throw new Error(data.error||'Unknown error');
    showEmotionResult(data);
    showSongs(data.songs,data.emotion,data.emoji,'song-list','songs-card','songs-badge');
    recordHistory(data);
    const first=data.songs.find(s=>s.preview_url);
    if(first) playSong(first,data.songs.indexOf(first));
  } catch(err) {
    $('detect-error').innerHTML='<div class="error-msg">⚠️ '+err.message+'</div>';
  }
  btn.disabled=false; btn.innerHTML='🔎 Detect Emotion &amp; Play Music';
}

/* ══════════════════════════
   SHOW EMOTION RESULT
══════════════════════════ */
function showEmotionResult(data) {
  $('emotion-card').style.display='block';
  $('res-emoji').textContent=data.emoji||'';
  $('res-name').textContent=data.emotion;
  $('res-conf').textContent='Confidence: '+(data.confidence*100).toFixed(1)+'%';
  $('res-mood').textContent='🎧 '+(data.mood||'');

  const barsEl=$('conf-bars'); barsEl.innerHTML='';
  const sorted=Object.entries(data.all_scores).sort((a,b)=>b[1]-a[1]);
  sorted.forEach(([lbl,score])=>{
    const p=(score*100).toFixed(1);
    barsEl.innerHTML+='<div class="conf-row"><span class="conf-name">'+lbl+'</span><div class="conf-track"><div class="conf-fill" style="width:'+p+'%"></div></div><span class="conf-pct">'+p+'%</span></div>';
  });

  destroyChart('emotion'); killCanvas('emotionChart');
  const ctx=$('emotionChart').getContext('2d');
  const labels=sorted.map(([l])=>l); const values=sorted.map(([,v])=>+(v*100).toFixed(1));
  charts.emotion=new Chart(ctx,{type:'bar',data:{labels,datasets:[{data:values,backgroundColor:labels.map(l=>(EmoColors[l]||'#6d5be8')+'88'),borderColor:labels.map(l=>EmoColors[l]||'#6d5be8'),borderWidth:1.5,borderRadius:8}]},options:{...chartDefaults,plugins:{...chartDefaults.plugins},scales:{x:{...chartDefaults.scales.x},y:{...chartDefaults.scales.y,max:100}}}});

  const cS=$('detect-conf-stat'); if(cS) cS.textContent=(data.confidence*100).toFixed(1)+'%';
  const eS=$('detect-emo-stat');  if(eS) eS.textContent=data.emotion;
  const sc=$('detect-session-count'); if(sc) sc.textContent=state.sessionHistory.length+1;
}

/* ══════════════════════════
   SHOW SONGS
══════════════════════════ */
function showSongs(songs,emotion,emoji,listId,cardId,badgeId) {
  state.currentSongs=songs; $(cardId).style.display='block';
  if(badgeId) $(badgeId).textContent=(emoji||'')+' '+emotion;
  const el=$(listId); el.innerHTML='';
  songs.forEach((s,i)=>{
    const hp=!!s.preview_url;
    el.innerHTML+='<div class="song-row" id="sr-'+listId+'-'+i+'">'+
      '<span class="song-num">'+(i+1)+'</span>'+
      '<div class="song-info"><div class="song-name">'+esc(s.name)+'</div>'+
      '<div class="song-meta">'+esc(s.artist)+(s.year&&s.year!=='N/A'?' · '+s.year:'')+'</div>'+
      '<div class="song-tags"><span class="tag tag-genre">'+esc(s.genre)+'</span>'+
      '<span class="tag tag-val">❤ '+s.valence+'</span>'+
      '<span class="tag tag-energy">⚡ '+s.energy+'</span></div></div>'+
      '<div class="song-actions">'+
      (hp?'<button class="btn btn-sm btn-play" onclick=\'playSong('+JSON.stringify(s).replace(/'/g,"&#39;")+','+i+')\'>▶</button>':'<span style="font-size:.68rem;color:var(--ghost);font-family:var(--font-mono)">no preview</span>')+
      (s.spotify_id?'<a href="https://open.spotify.com/track/'+s.spotify_id+'" target="_blank" style="font-size:.68rem;color:#1DB954;text-decoration:none;padding:4px 9px;background:rgba(29,185,84,0.08);border-radius:6px;border:1px solid rgba(29,185,84,0.2);font-family:var(--font-mono)">Spotify</a>':'')+
      '</div></div>';
  });
}

/* ══════════════════════════
   AUDIO PLAYER
══════════════════════════ */
function playSong(song,idx) {
  if(!song.preview_url){alert('No preview available.');return;}
  playQueue=state.currentSongs.filter(s=>s.preview_url);
  playQueueIdx=playQueue.findIndex(s=>s.name===song.name);
  if(playQueueIdx<0){playQueue.unshift(song);playQueueIdx=0;}
  loadAndPlay(playQueue[playQueueIdx]); state.songsPlayed++;
}
function loadAndPlay(song){
  audio.src=song.preview_url; audio.play().catch(()=>{});
  $('np-title').textContent=song.name;
  $('np-artist').textContent=song.artist+' · '+song.genre;
  $('now-playing').style.display='flex';
  $('np-play').textContent='⏸';
  $('main-content').style.paddingBottom='86px';
}
function togglePlay(){ if(audio.paused){audio.play();$('np-play').textContent='⏸';}else{audio.pause();$('np-play').textContent='▶';} }
function nextTrack(){ if(!playQueue.length)return;playQueueIdx=(playQueueIdx+1)%playQueue.length;loadAndPlay(playQueue[playQueueIdx]); }
function prevTrack(){ if(!playQueue.length)return;playQueueIdx=(playQueueIdx-1+playQueue.length)%playQueue.length;loadAndPlay(playQueue[playQueueIdx]); }
function stopPlayer(){ audio.pause();audio.src='';$('now-playing').style.display='none';$('main-content').style.paddingBottom='100px'; }
audio.addEventListener('timeupdate',()=>{
  if(!audio.duration)return;
  const pct=(audio.currentTime/audio.duration)*100;
  $('np-bar-fill').style.width=pct+'%';
  $('np-time-cur').textContent=fmt(audio.currentTime);
  $('np-time-tot').textContent=fmt(audio.duration);
});
audio.addEventListener('ended',nextTrack);
$('np-bar').addEventListener('click',e=>{
  const r=e.target.getBoundingClientRect();
  audio.currentTime=((e.clientX-r.left)/r.width)*audio.duration;
});

/* ══════════════════════════
   WEBCAM
══════════════════════════ */
let webcamActive=false,webcamInterval=null,webcamStream=null;
async function toggleWebcam(){
  if(webcamActive){stopWebcam();return;}
  try{
    webcamStream=await navigator.mediaDevices.getUserMedia({video:true});
    const v=$('webcam-video'); v.srcObject=webcamStream; await v.play();
    $('webcamBtn').textContent='⏹ Stop Webcam';
    $('webcam-status').textContent='Scanning every 3 seconds…';
    const dot=$('wc-dot');
    if(dot){dot.style.background='var(--emerald)';dot.style.boxShadow='0 0 10px var(--emerald)';}
    $('wc-pill').textContent='Active';
    webcamActive=true; captureFrame();
    webcamInterval=setInterval(captureFrame,3000);
  }catch(e){$('webcam-status').textContent='❌ '+e.message;}
}
function stopWebcam(){
  webcamActive=false; clearInterval(webcamInterval);
  if(webcamStream)webcamStream.getTracks().forEach(t=>t.stop());
  webcamStream=null;
  $('webcamBtn').textContent='▶ Start Webcam';
  $('webcam-status').textContent='Webcam stopped.';
  const dot=$('wc-dot');
  if(dot){dot.style.background='var(--ghost)';dot.style.boxShadow='none';}
  $('wc-pill').textContent='Inactive';
  const c=$('webcam-canvas');c.getContext('2d').clearRect(0,0,c.width,c.height);
}
async function captureFrame(){
  const v=$('webcam-video'),c=$('webcam-canvas'),ctx=c.getContext('2d');
  ctx.save();ctx.scale(-1,1);ctx.drawImage(v,-c.width,0,c.width,c.height);ctx.restore();
  const b64=c.toDataURL('image/jpeg',.85).split(',')[1];
  try{
    const resp=await fetch('/webcam/frame',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({frame:b64})});
    const text=await resp.text();let data;try{data=JSON.parse(text);}catch(e){return;}
    if(data.error){$('webcam-status').textContent='⚠️ '+data.error;return;}
    if(data.annotated_frame){const img=new Image();img.onload=()=>ctx.drawImage(img,0,0,c.width,c.height);img.src=data.annotated_frame;}
    $('webcam-status').textContent=(data.face_detected?'✅ Face detected':'🖼 Full frame')+' — '+data.emotion+' ('+(data.confidence*100).toFixed(1)+'%)';
    $('webcam-emotion-card').style.display='block';
    $('wc-emoji').textContent=data.emoji||'';
    $('wc-name').textContent=data.emotion;
    $('wc-conf').textContent='Confidence: '+(data.confidence*100).toFixed(1)+'%';
    $('wc-mood').textContent='🎧 '+(data.mood||'');
    showSongs(data.songs,data.emotion,data.emoji,'wc-song-list','wc-songs-card',null);
    state.wcSongs=data.songs; recordHistory(data);
    const first=data.songs.find(s=>s.preview_url);if(first&&audio.paused)playSong(first,0);
  }catch(e){console.error(e);}
}

/* ══════════════════════════
   SESSION HISTORY
══════════════════════════ */
function recordHistory(data){
  state.sessionHistory.unshift({time:new Date().toLocaleTimeString(),emotion:data.emotion,confidence:data.confidence,emoji:data.emoji});
  state.emotionCounts[data.emotion]=(state.emotionCounts[data.emotion]||0)+1;
  state.confidences.push(+(data.confidence*100).toFixed(1));
  const sc=$('detect-session-count');if(sc)sc.textContent=state.sessionHistory.length;
}

/* ══════════════════════════
   ANALYSIS PAGE
══════════════════════════ */
function renderAnalysis(){
  const total=state.sessionHistory.length;
  $('stat-total').textContent=total;
  $('stat-played').textContent=state.songsPlayed;
  if(total===0){
    $('stat-top').textContent='—'; $('stat-conf').textContent='—';
    $('emotion-freq').innerHTML='<p style="color:var(--ink3);font-size:.83rem;font-family:var(--font-mono)">No detections yet. Run a prediction first.</p>';
  }else{
    const top=Object.entries(state.emotionCounts).sort((a,b)=>b[1]-a[1])[0];
    $('stat-top').textContent=top[0];
    $('stat-conf').textContent=(state.confidences.reduce((a,b)=>a+b,0)/state.confidences.length).toFixed(1)+'%';
    const freqEl=$('emotion-freq'); freqEl.innerHTML='';
    const max=Math.max(...Object.values(state.emotionCounts));
    Object.entries(state.emotionCounts).sort((a,b)=>b[1]-a[1]).forEach(([emo,cnt])=>{
      const p=(cnt/max*100).toFixed(0);
      freqEl.innerHTML+='<div class="freq-bar-row"><span class="freq-label">'+emo+'</span><div class="freq-track"><div class="freq-fill" style="width:'+p+'%;background:'+(EmoColors[emo]||'var(--iris)')+'"></div></div><span class="freq-count">'+cnt+'</span></div>';
    });
  }

  // Timeline
  destroyChart('timeline'); killCanvas('timelineChart');
  const tc=$('timelineChart'); if(!tc)return;
  charts.timeline=new Chart(tc.getContext('2d'),{type:'line',data:{labels:state.sessionHistory.map((_,i)=>'#'+(state.sessionHistory.length-i)).reverse(),datasets:[{label:'Confidence %',data:[...state.confidences],borderColor:'#6d5be8',backgroundColor:'rgba(109,91,232,0.1)',fill:true,tension:0.4,pointBackgroundColor:'#e83d9e',pointRadius:4,pointHoverRadius:6}]},options:{...chartDefaults,plugins:{...chartDefaults.plugins,legend:{display:false}},scales:{x:{...chartDefaults.scales.x},y:{...chartDefaults.scales.y,min:0,max:100}}}});

  // Model accuracy + loss bars
  killCanvas('modelAccuracyChart');
  const ac=$('modelAccuracyChart');
  if(ac) charts.modelAcc=new Chart(ac.getContext('2d'),{type:'bar',data:{labels:['Train Acc','Val Acc'],datasets:[{data:[TD.finalTrainAcc,TD.finalValAcc],backgroundColor:['rgba(109,91,232,0.65)','rgba(232,61,158,0.65)'],borderColor:['#6d5be8','#e83d9e'],borderWidth:1.5,borderRadius:10}]},options:{...chartDefaults,plugins:{...chartDefaults.plugins},scales:{x:{...chartDefaults.scales.x,grid:{display:false}},y:{...chartDefaults.scales.y,min:0,max:100}}}});

  killCanvas('modelLossChart');
  const lc=$('modelLossChart');
  if(lc) charts.modelLoss=new Chart(lc.getContext('2d'),{type:'bar',data:{labels:['Train Loss','Val Loss'],datasets:[{data:[TD.finalTrainLoss,TD.finalValLoss],backgroundColor:['rgba(0,217,232,0.65)','rgba(245,200,66,0.65)'],borderColor:['#00d9e8','#f5c842'],borderWidth:1.5,borderRadius:10}]},options:{...chartDefaults,plugins:{...chartDefaults.plugins},scales:{x:{...chartDefaults.scales.x,grid:{display:false}},y:{...chartDefaults.scales.y,min:0}}}});

  // Class chart
  killCanvas('analysisClassChart');
  const cc=$('analysisClassChart');
  if(cc){
    const keys=Object.keys(CLASS_COUNTS);
    charts.analysisCls=new Chart(cc.getContext('2d'),{type:'doughnut',data:{labels:keys,datasets:[{data:Object.values(CLASS_COUNTS),backgroundColor:keys.map(k=>(EmoColors[k]||'#6d5be8')+'cc'),borderColor:'rgba(6,5,15,0.8)',borderWidth:2,hoverOffset:8}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'right',labels:{color:'#8880b8',font:{size:11}}}}}});
  }

  // Model summary grid
  const sumEl=$('analysis-model-summary'); if(!sumEl)return;
  const pairs=[['Architecture',TD.architecture],['Technique',TD.technique],['Best Val Acc',TD.bestAcc.toFixed(1)+'%'],['Best Epoch',String(TD.bestEpoch)],['Final Train Acc',TD.finalTrainAcc.toFixed(1)+'%'],['Final Val Acc',TD.finalValAcc.toFixed(1)+'%'],['Final Train Loss',TD.finalTrainLoss.toFixed(3)],['Final Val Loss',TD.finalValLoss.toFixed(3)]];
  sumEl.innerHTML=pairs.map(([k,v])=>'<div class="insight-card"><div class="insight-eyebrow">'+k+'</div><div class="insight-value" style="font-size:.95rem">'+v+'</div></div>').join('');

  // History
  const hist=$('history-list'); if(!hist)return;
  hist.innerHTML=state.sessionHistory.length===0
    ?'<div class="insight-card"><div class="insight-eyebrow">History</div><div class="insight-value">No detections yet</div></div>'
    :state.sessionHistory.map(h=>'<div class="history-item"><span class="history-emoji">'+(h.emoji||'🎭')+'</span><div><div class="history-name">'+h.emotion.charAt(0).toUpperCase()+h.emotion.slice(1)+'</div><div class="history-meta">Session detection</div></div><span class="history-meta">'+(h.confidence*100).toFixed(1)+'%</span><span class="history-meta">'+h.time+'</span></div>').join('');
}

/* ══════════════════════════
   TRAINING PAGE
══════════════════════════ */
function renderTraining(){
  $('m-train').textContent=TD.trainCount.toLocaleString();
  $('m-test').textContent=TD.testCount.toLocaleString();
  $('m-acc').textContent=TD.bestAcc.toFixed(1)+'%';
  $('m-epoch').textContent='@ Epoch '+TD.bestEpoch;
  $('m-time').textContent=TD.time;
  $('m-status').textContent=TD.status;
  $('ring-val').textContent=TD.bestAcc.toFixed(1)+'%';
  $('ring-ep').textContent='@ Epoch '+TD.bestEpoch;
  $('ring-outer').style.setProperty('--pct',TD.bestAcc);

  // Snapshot grid
  const sg=$('snapshot-grid');
  const snaps=[['Architecture',TD.architecture,'Base model'],['Best epoch',TD.bestEpoch,'Best checkpoint'],['Best val acc',TD.bestAcc.toFixed(1)+'%','Strongest'],['Final val acc',TD.finalValAcc.toFixed(1)+'%','Final epoch'],['Final train loss',TD.finalTrainLoss.toFixed(3),'Train'],['Final val loss',TD.finalValLoss.toFixed(3),'Validation']];
  if(sg)sg.innerHTML=snaps.map(([k,v,s])=>'<div class="insight-card"><div class="insight-eyebrow">'+k+'</div><div class="insight-value" style="font-size:1.05rem">'+v+'</div><div class="insight-body">'+s+'</div></div>').join('');

  // KV table
  const kv=$('training-kv');
  const kvPairs=[['Architecture',TD.architecture+' ('+TD.technique+')'],['Input Shape',TD.inputShape],['Classes',TD.classes+' emotions'],['Optimizer',TD.optimizer],['Learning Rate',TD.lr],['Loss Function',TD.loss],['Batch Size',String(TD.batch)],['Epochs',String(TD.epochs)],['Dataset',TD.dataset],['Best Val Acc',TD.bestAcc.toFixed(1)+'% @ Ep.'+TD.bestEpoch],['Final Train Acc',TD.finalTrainAcc.toFixed(1)+'%'],['Final Val Acc',TD.finalValAcc.toFixed(1)+'%'],['Final Train Loss',TD.finalTrainLoss.toFixed(3)],['Final Val Loss',TD.finalValLoss.toFixed(3)]];
  if(kv)kv.innerHTML=kvPairs.map(([k,v])=>'<div class="kv-k">'+k+'</div><div class="kv-v">'+v+'</div>').join('');

  // Model KV (right card)
  const mk=$('model-kv');
  const mkPairs=[['Architecture',TD.architecture],['Technique',TD.technique],['Input',TD.inputShape],['Classes',TD.classes],['Optimizer',TD.optimizer],['Batch',TD.batch],['Epochs',TD.epochs],['Dataset',TD.dataset]];
  if(mk)mk.innerHTML=mkPairs.map(([k,v])=>'<div class="kv-k">'+k+'</div><div class="kv-v">'+v+'</div>').join('');

  // Epoch table
  const et=$('epoch-table'); if(et){
    et.innerHTML='<thead><tr><th>Epoch</th><th>Train Acc</th><th>Val Acc</th><th>Train Loss</th><th>Val Loss</th><th>LR</th><th></th></tr></thead><tbody>'+TD.epochs_data.map(r=>'<tr class="'+(r.best?'best-row':'')+'"><td class="mono">'+r.e+'</td><td class="mono">'+(r.ta*100).toFixed(1)+'%</td><td class="mono">'+(r.va*100).toFixed(1)+'%</td><td class="mono">'+r.tl.toFixed(3)+'</td><td class="mono">'+r.vl.toFixed(3)+'</td><td class="mono">'+r.lr+'</td><td>'+(r.best?'<span class="best-badge">Best</span>':'')+'</td></tr>').join('')+'</tbody>';
  }

  // Class doughnut
  destroyChart('trainingCls'); killCanvas('classDistChart');
  const cc=$('classDistChart');
  if(cc){
    const keys=Object.keys(CLASS_COUNTS);
    charts.trainingCls=new Chart(cc.getContext('2d'),{type:'doughnut',data:{labels:keys,datasets:[{data:Object.values(CLASS_COUNTS),backgroundColor:keys.map(k=>(EmoColors[k]||'#6d5be8')+'cc'),borderColor:'rgba(6,5,15,0.8)',borderWidth:2,hoverOffset:8}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'right',labels:{color:'#8880b8',font:{size:11}}}}}});
  }

  // Class cards
  const cd=$('class-detail'); if(!cd)return;
  const total=Object.values(CLASS_COUNTS).reduce((a,b)=>a+b,0);
  cd.innerHTML=Object.keys(CLASS_COUNTS).map(emo=>{
    const cnt=CLASS_COUNTS[emo]; const pct=(cnt/total*100).toFixed(1);
    return'<div class="class-card"><div class="class-card-head"><span class="class-dot" style="background:'+(EmoColors[emo]||'#6d5be8')+'"></span><span class="class-name">'+emo.charAt(0).toUpperCase()+emo.slice(1)+'</span></div><div class="class-count">'+cnt.toLocaleString()+'</div><div class="class-pct">'+pct+'% of dataset</div></div>';
  }).join('');
}

/* ══════════════════════════
   COMPARISON PAGE
══════════════════════════ */
function setTab(name){
  document.querySelectorAll('#comp-tabs .comp-tab').forEach(b=>b.classList.toggle('active',b.dataset.tab===name));
  document.querySelectorAll('.comp-panel').forEach(p=>p.classList.remove('active'));
  const pn=$('comp-panel-'+name); if(pn) pn.classList.add('active');
  if(name==='accuracy') buildAccCharts();
  if(name==='error')    buildErrCharts();
}

function allModels(){ return [CMP.deep,...CMP.classical].map((m,i)=>({...m,rank:i+1})); }

function buildOverviewTable(){
  const rows=[...allModels()].sort((a,b)=>b.accuracy-a.accuracy);
  const ot=$('overview-table'); if(!ot)return;
  ot.innerHTML='<thead><tr><th>#</th><th>Model</th><th>Accuracy</th><th>F1</th><th>ROC-AUC</th><th>CV Mean</th></tr></thead><tbody>'+rows.map((m,idx)=>'<tr class="'+(idx===0?'top-row':'')+'"><td><span class="comp-rank-badge">'+(idx+1)+'</span></td><td>'+m.name+'</td><td class="mono">'+(m.accuracy*100).toFixed(2)+'%</td><td class="mono">'+(m.f1*100).toFixed(2)+'%</td><td class="mono">'+(m.rocAuc!=null?m.rocAuc.toFixed(3):'—')+'</td><td class="mono">'+(m.cvMean!=null?m.cvMean.toFixed(3):'—')+'</td></tr>').join('')+'</tbody>';
}

const palettes=['rgba(109,91,232,0.72)','rgba(0,217,232,0.65)','rgba(0,229,160,0.62)','rgba(245,200,66,0.62)','rgba(232,61,158,0.62)'];
const palBorders=['#6d5be8','#00d9e8','#00e5a0','#f5c842','#e83d9e'];

function buildAccCharts(){
  const models=allModels(); const labels=models.map(m=>m.name);
  killCanvas('accChart'); killCanvas('f1Chart');
  const ac=$('accChart');
  if(ac) charts.acc=new Chart(ac.getContext('2d'),{type:'bar',data:{labels,datasets:[{label:'Accuracy',data:models.map(m=>+(m.accuracy*100).toFixed(2)),backgroundColor:palettes,borderColor:palBorders,borderWidth:1.5,borderRadius:10}]},options:{...chartDefaults,plugins:{...chartDefaults.plugins,legend:{display:false}},scales:{x:{...chartDefaults.scales.x},y:{...chartDefaults.scales.y,min:0,max:80}}}});
  const fc=$('f1Chart');
  if(fc) charts.f1=new Chart(fc.getContext('2d'),{type:'bar',data:{labels,datasets:[{label:'F1',data:models.map(m=>+(m.f1*100).toFixed(2)),backgroundColor:palettes,borderColor:palBorders,borderWidth:1.5,borderRadius:10}]},options:{...chartDefaults,plugins:{...chartDefaults.plugins,legend:{display:false}},scales:{x:{...chartDefaults.scales.x},y:{...chartDefaults.scales.y,min:0,max:80}}}});
}

function buildErrCharts(){
  const models=allModels(); const labels=models.map(m=>m.name);
  killCanvas('errChart'); killCanvas('rocChart');
  const ec=$('errChart');
  if(ec) charts.err=new Chart(ec.getContext('2d'),{type:'bar',data:{labels,datasets:[{label:'Error Rate',data:models.map(m=>+(m.errorRate*100).toFixed(2)),backgroundColor:palettes.map((_,i)=>['rgba(255,68,102,0.68)','rgba(255,145,77,0.62)','rgba(245,200,66,0.62)','rgba(0,217,232,0.62)','rgba(109,91,232,0.62)'][i]),borderRadius:10}]},options:{...chartDefaults,plugins:{...chartDefaults.plugins,legend:{display:false}},scales:{x:{...chartDefaults.scales.x},y:{...chartDefaults.scales.y,min:0,max:70}}}});
  const rc=$('rocChart');
  if(rc) charts.roc=new Chart(rc.getContext('2d'),{type:'bar',data:{labels,datasets:[{label:'ROC-AUC',data:models.map(m=>m.rocAuc!=null?+m.rocAuc.toFixed(3):null),backgroundColor:'rgba(109,91,232,0.68)',borderRadius:8},{label:'CV Mean',data:models.map(m=>m.cvMean!=null?+m.cvMean.toFixed(3):null),backgroundColor:'rgba(0,217,232,0.62)',borderRadius:8}]},options:{...chartDefaults,plugins:{...chartDefaults.plugins,legend:{labels:{color:'#8880b8',font:{size:11}}}},scales:{x:{...chartDefaults.scales.x},y:{...chartDefaults.scales.y,min:0,max:1}}}});
}

function renderComparison(){
  buildOverviewTable();
  const deep=CMP.deep; const best=[...CMP.classical].sort((a,b)=>b.accuracy-a.accuracy)[0];
  $('cmp-deep').textContent=(deep.accuracy*100).toFixed(2)+'%';
  $('cmp-class').textContent=(best.accuracy*100).toFixed(2)+'%';
  $('cmp-gap').textContent='+'+((deep.accuracy-best.accuracy)*100).toFixed(2)+'pp';
  $('cmp-roc').textContent=best.rocAuc.toFixed(3);
  buildAccCharts();
}

/* ══════════════════════════
   EXPORT TRAINING REPORT
══════════════════════════ */
function exportTrainingReport(){
  const lines=['MoodMate Training Report','='+'='.repeat(35),'Architecture: '+TD.architecture,'Technique: '+TD.technique,'Dataset: '+TD.dataset,'Training Set: '+TD.trainCount,'Test Set: '+TD.testCount,'Epochs: '+TD.epochs,'Best Val Accuracy: '+TD.bestAcc.toFixed(1)+'% at epoch '+TD.bestEpoch,'Final Train Accuracy: '+TD.finalTrainAcc.toFixed(1)+'%','Final Val Accuracy: '+TD.finalValAcc.toFixed(1)+'%','Final Train Loss: '+TD.finalTrainLoss.toFixed(3),'Final Val Loss: '+TD.finalValLoss.toFixed(3),'Training Time: '+TD.time];
  const blob=new Blob([lines.join('\n')],{type:'text/plain'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='moodmate_training_report.txt';a.click();URL.revokeObjectURL(a.href);
}

/* ══════════════════════════
   INIT
══════════════════════════ */
window.addEventListener('load',()=>{
  try{renderTraining();}catch(e){console.error(e);}
  try{renderAnalysis();}catch(e){console.error(e);}
  try{renderComparison();}catch(e){console.error(e);}
});
</script>
</body>
</html>
"""


# ──────────────────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    ext      = os.path.splitext(file.filename)[1].lower() or ".jpg"
    tmp_path = os.path.join(UPLOAD_FOLDER, f"up_{uuid.uuid4().hex}{ext}")
    file.save(tmp_path)

    try:
        result = _process_image_file(tmp_path)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        logger.exception("Prediction error")
        return jsonify({"error": str(e)}), 500
    finally:
        gc.collect()
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except PermissionError:
            pass


@app.route("/webcam/frame", methods=["POST"])
def webcam_frame():
    data = request.get_json(silent=True)
    if not data or "frame" not in data:
        return jsonify({"error": "No frame data"}), 400
    try:
        import cv2
        img_bytes = base64.b64decode(data["frame"])
        nparr     = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Could not decode frame"}), 400

        faces = detect_faces(frame)
        if faces:
            largest   = max(faces, key=lambda b: b[2]*b[3])
            face_crop = crop_face(frame, largest)
            pred      = predict_from_array(face_crop)
            emo       = pred["emotion"]
            label     = f"{emotion_emoji(emo)} {emo} {pred['confidence']*100:.0f}%"
            annotated = draw_face_box(frame, largest, label=label)
            ann_b64   = image_to_base64(annotated)
        else:
            pred    = predict_from_array(frame)
            emo     = pred["emotion"]
            ann_b64 = image_to_base64(frame)

        info  = get_emotion_info(emo)
        songs = recommender.recommend(emo, n=8)
        return jsonify({
            "emotion": emo, "confidence": pred["confidence"],
            "all_scores": pred["all_scores"], "mood": info["mood"],
            "emoji": emotion_emoji(emo), "songs": songs,
            "annotated_frame": ann_b64, "face_detected": bool(faces),
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        logger.exception("Webcam error")
        return jsonify({"error": str(e)}), 500


def _process_image_file(image_path):
    import cv2
    frame = cv2.imread(image_path)
    if frame is None:
        from PIL import Image as PILImage
        pil_img = PILImage.open(image_path).convert("RGB")
        frame   = np.array(pil_img)[:, :, ::-1]

    faces = detect_faces(frame)
    if faces:
        largest   = max(faces, key=lambda b: b[2]*b[3])
        face_crop = crop_face(frame, largest)
        pred      = predict_from_array(face_crop)
    else:
        pred = predict_from_path(image_path)

    emo   = pred["emotion"]
    info  = get_emotion_info(emo)
    songs = recommender.recommend(emo, n=10)
    return {
        "emotion": emo, "confidence": pred["confidence"],
        "all_scores": pred["all_scores"], "mood": info["mood"],
        "emoji": emotion_emoji(emo), "songs": songs,
        "face_detected": bool(faces),
    }


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  MoodMate v5 — Investor-Grade Premium Dashboard")
    print("  Open: http://localhost:5000")
    print("="*55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)