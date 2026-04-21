"""
spotify_routes.py
─────────────────
Drop this file into your project root (same level as app.py).
Then in app.py, add at the top:

    from spotify_routes import spotify_bp, SPOTIFY_HTML
    app.register_blueprint(spotify_bp)

And add this new route in app.py:

    @app.route("/spotify")
    def spotify_page():
        return render_template_string(SPOTIFY_HTML)

Also update your /predict route to accept a 'language' form field
and call recommend_for_emotion instead of the old recommender.
See the updated _process_image_file_v2() below.
"""

import os, gc, uuid, base64
import numpy as np
from flask import Blueprint, request, jsonify, render_template_string

from src.emotion_predictor import predict_from_path, predict_from_array
from app.emotion_music_recommender import recommend_for_emotion
from src.emotion_mapping import get_emotion_info

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "app", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

spotify_bp = Blueprint("spotify", __name__)


# ─────────────────────────────────────────────────────────────────────────────
# Updated image-processing helper
# ─────────────────────────────────────────────────────────────────────────────

def _process_image_file_v2(image_path: str, language: str = "English") -> dict:
    """Detect emotion then return staged playlist for chosen language."""
    try:
        import cv2
        from utils.helper_functions import detect_faces, crop_face
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError("cv2 could not read image")
        faces = detect_faces(frame)
        if faces:
            largest   = max(faces, key=lambda b: b[2] * b[3])
            face_crop = crop_face(frame, largest)
            pred      = predict_from_array(face_crop)
        else:
            pred = predict_from_path(image_path)
    except Exception:
        pred = predict_from_path(image_path)

    emotion = pred["emotion"]
    info    = get_emotion_info(emotion)
    music   = recommend_for_emotion(emotion, language=language, songs_per_stage=5)

    return {
        "emotion":     emotion,
        "confidence":  pred["confidence"],
        "all_scores":  pred["all_scores"],
        "mood":        info["mood"],
        "journey":     music["journey"],
        "stages":      music["stages"],
        "language":    language,
    }


# ─────────────────────────────────────────────────────────────────────────────
# /predict_v2  — new route that accepts language
# ─────────────────────────────────────────────────────────────────────────────

@spotify_bp.route("/predict_v2", methods=["POST"])
def predict_v2():
    if "image" not in request.files:
        return jsonify({"error": "No image file"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    language = request.form.get("language", "English")
    ext      = os.path.splitext(file.filename)[1].lower() or ".jpg"
    tmp_path = os.path.join(UPLOAD_FOLDER, f"up_{uuid.uuid4().hex}{ext}")
    file.save(tmp_path)

    try:
        result = _process_image_file_v2(tmp_path, language)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        gc.collect()
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except PermissionError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Frontend HTML (full page)
# ─────────────────────────────────────────────────────────────────────────────

SPOTIFY_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>MoodMate — Emotion Playlist</title>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --void:#03020a;--bg:#06050f;--surface:#0c0b18;--elevated:#121024;
  --line:rgba(255,255,255,0.07);--line2:rgba(255,255,255,0.13);
  --iris:#6d5be8;--iris2:#9b8aff;--nova:#e83d9e;--cyan:#00d9e8;
  --emerald:#00e5a0;--gold:#f5c842;--crimson:#ff4466;--amber:#f59e0b;
  --ink:#f5f3ff;--ink2:#c4bfe8;--ink3:#8880b8;--ghost:#4a4570;
  --font:'Outfit',system-ui,sans-serif;--mono:'JetBrains Mono',monospace;
  --r-xl:20px;--r-lg:14px;--r-md:10px;--r-sm:6px;
}
html{scroll-behavior:smooth}
body{background:var(--void);color:var(--ink);font-family:var(--font);font-size:14px;line-height:1.6;min-height:100vh;overflow-x:hidden;-webkit-font-smoothing:antialiased}

/* BG orbs */
.orbs{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden}
.orb{position:absolute;border-radius:50%;filter:blur(120px);mix-blend-mode:screen}
.orb-a{width:700px;height:700px;background:radial-gradient(circle,rgba(109,91,232,.20) 0%,transparent 70%);top:-200px;left:-150px;animation:drift 28s ease-in-out infinite alternate}
.orb-b{width:500px;height:500px;background:radial-gradient(circle,rgba(232,61,158,.14) 0%,transparent 70%);bottom:-150px;right:-80px;animation:drift 22s ease-in-out infinite alternate-reverse}
.orb-c{width:400px;height:400px;background:radial-gradient(circle,rgba(0,217,232,.10) 0%,transparent 70%);top:45%;left:52%;animation:drift 35s ease-in-out infinite alternate}
@keyframes drift{0%{transform:translate(0,0)}50%{transform:translate(50px,-30px)}100%{transform:translate(-20px,60px)}}

/* Layout */
.wrap{position:relative;z-index:1;max-width:900px;margin:0 auto;padding:32px 20px 80px}

/* Header */
.header{text-align:center;margin-bottom:40px}
.logo{font-size:13px;font-weight:600;letter-spacing:.12em;color:var(--iris2);text-transform:uppercase;margin-bottom:12px}
.title{font-size:36px;font-weight:700;background:linear-gradient(135deg,var(--iris2),var(--nova));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin-bottom:10px}
.subtitle{color:var(--ink3);font-size:15px}

/* Upload card */
.card{background:rgba(12,11,24,.7);border:1px solid var(--line);border-radius:var(--r-xl);padding:28px;backdrop-filter:blur(20px);margin-bottom:24px}
.card-title{font-size:13px;font-weight:600;letter-spacing:.08em;color:var(--ink3);text-transform:uppercase;margin-bottom:16px}

/* Upload zone */
.upload-zone{border:2px dashed var(--line2);border-radius:var(--r-lg);padding:36px;text-align:center;cursor:pointer;transition:all .25s;position:relative;background:rgba(255,255,255,.015)}
.upload-zone:hover,.upload-zone.drag{border-color:var(--iris);background:rgba(109,91,232,.06)}
.upload-icon{font-size:32px;margin-bottom:10px;opacity:.6}
.upload-text{color:var(--ink3);font-size:14px}
.upload-text b{color:var(--iris2)}
#file-input{position:absolute;inset:0;opacity:0;cursor:pointer}
#preview-img{width:100%;max-height:220px;object-fit:contain;border-radius:var(--r-md);margin-top:16px;display:none}

/* Language selector */
.lang-row{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}
.lang-btn{padding:8px 18px;border-radius:50px;border:1px solid var(--line2);background:transparent;color:var(--ink3);font-family:var(--font);font-size:13px;font-weight:500;cursor:pointer;transition:all .2s}
.lang-btn:hover{border-color:var(--iris);color:var(--iris2)}
.lang-btn.active{border-color:var(--iris);background:rgba(109,91,232,.18);color:var(--iris2)}

/* CTA button */
.btn{display:block;width:100%;margin-top:20px;padding:14px;border:none;border-radius:var(--r-lg);background:linear-gradient(135deg,var(--iris),var(--nova));color:#fff;font-family:var(--font);font-size:15px;font-weight:600;cursor:pointer;transition:opacity .2s;letter-spacing:.03em}
.btn:hover{opacity:.88}
.btn:disabled{opacity:.4;cursor:not-allowed}

/* Loading */
.loading{display:none;text-align:center;padding:40px;color:var(--ink3)}
.pulse{display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--iris);animation:pulse 1.2s ease-in-out infinite}
.pulse:nth-child(2){animation-delay:.2s;background:var(--nova)}
.pulse:nth-child(3){animation-delay:.4s;background:var(--cyan)}
@keyframes pulse{0%,80%,100%{transform:scale(.6);opacity:.4}40%{transform:scale(1);opacity:1}}

/* Emotion result banner */
.emotion-banner{display:none;border-radius:var(--r-xl);padding:24px 28px;margin-bottom:24px;border:1px solid var(--line);backdrop-filter:blur(20px)}
.emotion-name{font-size:28px;font-weight:700;margin-bottom:4px}
.emotion-journey{font-size:14px;color:var(--ink3);margin-bottom:12px}
.confidence-bar{height:6px;border-radius:3px;background:var(--line2);overflow:hidden}
.confidence-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--iris),var(--nova));transition:width .8s ease}
.confidence-label{font-size:12px;color:var(--ink3);margin-top:6px;font-family:var(--mono)}

/* Transformation map */
.transform-map{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:16px}
.stage-pill{padding:6px 14px;border-radius:50px;font-size:12px;font-weight:600;letter-spacing:.06em;border:1px solid}
.arrow-sep{color:var(--ink3);font-size:16px}
.stage-calming{background:rgba(0,217,232,.10);border-color:var(--cyan);color:var(--cyan)}
.stage-comforting{background:rgba(0,229,160,.10);border-color:var(--emerald);color:var(--emerald)}
.stage-hopeful{background:rgba(245,200,66,.10);border-color:var(--gold);color:var(--gold)}
.stage-uplifting{background:rgba(109,91,232,.15);border-color:var(--iris2);color:var(--iris2)}
.stage-happy{background:rgba(232,61,158,.10);border-color:var(--nova);color:var(--nova)}

/* Stages */
.stages-wrap{display:none}
.stage-card{margin-bottom:28px}
.stage-header{display:flex;align-items:center;gap:12px;margin-bottom:14px;padding-bottom:12px;border-bottom:1px solid var(--line)}
.stage-num{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;flex-shrink:0}
.stage-num-1{background:rgba(0,217,232,.15);color:var(--cyan)}
.stage-num-2{background:rgba(109,91,232,.18);color:var(--iris2)}
.stage-label{font-size:15px;font-weight:600}
.stage-desc{font-size:13px;color:var(--ink3)}

/* Song list */
.songs-list{display:flex;flex-direction:column;gap:12px}
.song-row{background:rgba(18,16,36,.6);border:1px solid var(--line);border-radius:var(--r-lg);overflow:hidden;transition:border-color .2s}
.song-row:hover{border-color:var(--line2)}
.song-top{display:flex;align-items:center;gap:14px;padding:14px 16px;cursor:pointer}
.song-index{font-family:var(--mono);font-size:11px;color:var(--ghost);width:18px;flex-shrink:0;text-align:right}
.song-info{flex:1;min-width:0}
.song-name{font-size:14px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.song-artist{font-size:12px;color:var(--ink3);margin-top:2px}
.song-tags{display:flex;gap:6px;flex-shrink:0}
.tag{padding:3px 9px;border-radius:50px;font-size:11px;font-weight:500;border:1px solid var(--line2);color:var(--ink3)}
.play-btn{width:34px;height:34px;border-radius:50%;border:1px solid var(--iris);background:rgba(109,91,232,.12);color:var(--iris2);font-size:14px;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all .2s}
.play-btn:hover{background:rgba(109,91,232,.28);transform:scale(1.05)}
.play-btn.playing{background:var(--iris);color:#fff}
.spotify-embed{display:none;padding:0 16px 14px}
.spotify-embed iframe{border-radius:10px;width:100%;border:none}

/* Open in Spotify link */
.open-spotify{display:inline-flex;align-items:center;gap:6px;margin-top:10px;font-size:12px;color:#1DB954;text-decoration:none;font-weight:500;transition:opacity .2s}
.open-spotify:hover{opacity:.8}
.sp-icon{width:16px;height:16px;background:#1DB954;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:8px;color:#000;font-weight:700}

/* Scores grid */
.scores-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:8px;margin-top:12px}
.score-item{background:rgba(255,255,255,.025);border:1px solid var(--line);border-radius:var(--r-md);padding:10px 12px}
.score-emo{font-size:11px;color:var(--ink3);text-transform:capitalize;margin-bottom:4px}
.score-bar-bg{height:4px;border-radius:2px;background:var(--line2)}
.score-bar-fill{height:100%;border-radius:2px;background:var(--iris)}
.score-val{font-family:var(--mono);font-size:11px;color:var(--ink3);margin-top:4px}

/* Error */
.error-box{background:rgba(255,68,102,.08);border:1px solid rgba(255,68,102,.3);border-radius:var(--r-lg);padding:16px;color:var(--crimson);font-size:14px;margin-bottom:20px;display:none}
</style>
</head>
<body>
<div class="orbs"><div class="orb orb-a"></div><div class="orb orb-b"></div><div class="orb orb-c"></div></div>

<div class="wrap">
  <!-- Header -->
  <div class="header">
    <div class="logo">MoodMate</div>
    <div class="title">Emotion Transformation Playlist</div>
    <div class="subtitle">We detect your mood and guide you toward happiness — song by song.</div>
  </div>

  <!-- Error box -->
  <div class="error-box" id="error-box"></div>

  <!-- Upload card -->
  <div class="card">
    <div class="card-title">Step 1 — Upload your photo</div>
    <div class="upload-zone" id="upload-zone">
      <div class="upload-icon">📸</div>
      <div class="upload-text">Drop a photo here or <b>click to browse</b></div>
      <input type="file" id="file-input" accept="image/*"/>
      <img id="preview-img"/>
    </div>

    <div class="card-title" style="margin-top:22px">Step 2 — Choose language</div>
    <div class="lang-row" id="lang-row">
      <button class="lang-btn active" data-lang="Telugu">🇮🇳 Telugu</button>
      <button class="lang-btn" data-lang="Hindi">🇮🇳 Hindi</button>
      <button class="lang-btn" data-lang="Punjabi">🇮🇳 Punjabi</button>
      <button class="lang-btn" data-lang="English">🌐 English</button>
    </div>

    <button class="btn" id="analyse-btn" disabled>Detect Emotion &amp; Build Playlist</button>
  </div>

  <!-- Loading -->
  <div class="loading" id="loading">
    <div style="margin-bottom:14px">Analysing your emotion...</div>
    <div class="pulse"></div><div class="pulse"></div><div class="pulse"></div>
  </div>

  <!-- Emotion banner -->
  <div class="card emotion-banner" id="emotion-banner">
    <div id="emotion-name" class="emotion-name"></div>
    <div id="emotion-journey" class="emotion-journey"></div>
    <div class="confidence-bar"><div class="confidence-fill" id="conf-fill" style="width:0%"></div></div>
    <div class="confidence-label" id="conf-label"></div>
    <div id="transform-map" class="transform-map"></div>
    <!-- All-emotion scores -->
    <div id="scores-grid" class="scores-grid" style="margin-top:16px"></div>
  </div>

  <!-- Stages -->
  <div class="stages-wrap" id="stages-wrap"></div>
</div>

<script>
// ── State ────────────────────────────────────────────────────────────────────
let selectedLang = 'Telugu';
let selectedFile = null;
let currentlyPlaying = null;  // song-row id

// ── Emoji map ─────────────────────────────────────────────────────────────────
const EMO_EMOJI = {angry:'😠',sad:'😢',fear:'😨',disgust:'🤢',neutral:'😐',happy:'😊',surprise:'😲'};
const EMO_COLORS = {angry:'#ff4466',sad:'#60a5fa',fear:'#a78bfa',disgust:'#34d399',neutral:'#94a3b8',happy:'#fbbf24',surprise:'#f472b6'};
const STAGE_CLS = {calming:'stage-calming',comforting:'stage-comforting',hopeful:'stage-hopeful',uplifting:'stage-uplifting',happy:'stage-happy'};
const STAGE_NUM_CLS = ['','stage-num-1','stage-num-2'];

// ── Language buttons ──────────────────────────────────────────────────────────
document.querySelectorAll('.lang-btn').forEach(btn=>{
  btn.addEventListener('click',()=>{
    document.querySelectorAll('.lang-btn').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    selectedLang = btn.dataset.lang;
  });
});

// ── File upload ───────────────────────────────────────────────────────────────
const uploadZone = document.getElementById('upload-zone');
const fileInput  = document.getElementById('file-input');
const previewImg = document.getElementById('preview-img');
const analyseBtn = document.getElementById('analyse-btn');

uploadZone.addEventListener('dragover', e=>{e.preventDefault();uploadZone.classList.add('drag')});
uploadZone.addEventListener('dragleave',()=>uploadZone.classList.remove('drag'));
uploadZone.addEventListener('drop', e=>{
  e.preventDefault(); uploadZone.classList.remove('drag');
  if(e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change',()=>{ if(fileInput.files[0]) handleFile(fileInput.files[0]); });

function handleFile(file){
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.style.display = 'block';
  analyseBtn.disabled = false;
  hide('emotion-banner'); hide('stages-wrap'); hide('error-box');
}

// ── Analyse button ────────────────────────────────────────────────────────────
analyseBtn.addEventListener('click', analyse);

async function analyse(){
  if(!selectedFile) return;
  analyseBtn.disabled = true;
  show('loading'); hide('emotion-banner'); hide('stages-wrap'); hide('error-box');

  const fd = new FormData();
  fd.append('image', selectedFile);
  fd.append('language', selectedLang);

  try{
    const res  = await fetch('/predict_v2', {method:'POST', body:fd});
    const data = await res.json();
    hide('loading');

    if(data.error){
      showError(data.error); analyseBtn.disabled=false; return;
    }
    renderResult(data);
  }catch(err){
    hide('loading');
    showError('Network error: '+err.message);
    analyseBtn.disabled=false;
  }
}

// ── Render result ─────────────────────────────────────────────────────────────
function renderResult(data){
  const emotion = data.emotion || 'neutral';
  const conf    = Math.round((data.confidence||0)*100);
  const color   = EMO_COLORS[emotion] || '#6d5be8';
  const emoji   = EMO_EMOJI[emotion]  || '🎵';

  // Emotion banner
  const banner = document.getElementById('emotion-banner');
  banner.style.background = `linear-gradient(135deg, ${color}18, transparent)`;
  banner.style.borderColor = color+'40';
  document.getElementById('emotion-name').innerHTML  = `${emoji} ${capitalize(emotion)}`;
  document.getElementById('emotion-name').style.color= color;
  document.getElementById('emotion-journey').textContent = data.journey || '';
  document.getElementById('conf-fill').style.width   = conf+'%';
  document.getElementById('conf-fill').style.background = `linear-gradient(90deg,${color},${color}aa)`;
  document.getElementById('conf-label').textContent  = `Confidence: ${conf}%`;
  show('emotion-banner');

  // Transformation map pills
  const map  = document.getElementById('transform-map');
  map.innerHTML = '';
  (data.stages||[]).forEach((s,i)=>{
    if(i>0){const sep=document.createElement('span');sep.className='arrow-sep';sep.textContent='→';map.appendChild(sep);}
    const pill=document.createElement('span');
    pill.className='stage-pill '+(STAGE_CLS[s.mood_tag]||'stage-hopeful');
    pill.textContent=capitalize(s.mood_tag);
    map.appendChild(pill);
  });

  // All emotion scores
  renderScores(data.all_scores||{});

  // Stages
  renderStages(data.stages||[], data.language||'English');
  show('stages-wrap');
  analyseBtn.disabled = false;
}

function renderScores(scores){
  const grid = document.getElementById('scores-grid');
  grid.innerHTML = '';
  Object.entries(scores).sort((a,b)=>b[1]-a[1]).forEach(([emo,val])=>{
    const pct = Math.round(val*100);
    const col = EMO_COLORS[emo]||'var(--iris)';
    grid.innerHTML += `
      <div class="score-item">
        <div class="score-emo">${EMO_EMOJI[emo]||''} ${emo}</div>
        <div class="score-bar-bg"><div class="score-bar-fill" style="width:${pct}%;background:${col}"></div></div>
        <div class="score-val">${pct}%</div>
      </div>`;
  });
}

function renderStages(stages, lang){
  const wrap = document.getElementById('stages-wrap');
  wrap.innerHTML = `<div style="font-size:13px;color:var(--ink3);margin-bottom:16px;letter-spacing:.05em;text-transform:uppercase;font-weight:600">🎵 Your Playlist — ${lang}</div>`;

  stages.forEach((stage, si)=>{
    const numCls = STAGE_NUM_CLS[si+1] || 'stage-num-1';
    let html = `
      <div class="stage-card">
        <div class="stage-header">
          <div class="stage-num ${numCls}">${si+1}</div>
          <div>
            <div class="stage-label">${stage.label}</div>
            <div class="stage-desc">${capitalize(stage.mood_tag)} songs · ${(stage.songs||[]).length} tracks</div>
          </div>
        </div>
        <div class="songs-list">`;

    (stage.songs||[]).forEach((song, idx)=>{
      const rowId   = `song-${si}-${idx}`;
      const embedUrl = song.embed_url || '';
      const spotUrl  = song.spotify_url || '';
      html += `
        <div class="song-row" id="${rowId}">
          <div class="song-top" onclick="toggleEmbed('${rowId}','${escHtml(embedUrl)}')">
            <div class="song-index">${idx+1}</div>
            <div class="song-info">
              <div class="song-name">${escHtml(song.name)}</div>
              <div class="song-artist">${escHtml(song.artist)}</div>
            </div>
            <div class="song-tags">
              <span class="tag">V ${(song.valence*100).toFixed(0)}</span>
              <span class="tag">E ${(song.energy*100).toFixed(0)}</span>
            </div>
            <button class="play-btn" id="play-${rowId}" title="Play on Spotify">▶</button>
          </div>
          <div class="spotify-embed" id="embed-${rowId}">
            ${embedUrl ? `<iframe src="${escHtml(embedUrl)}?utm_source=generator&theme=0" width="100%" height="80" allow="autoplay;clipboard-write;encrypted-media;fullscreen;picture-in-picture" loading="lazy"></iframe>` : '<div style="color:var(--ink3);font-size:13px;padding:8px 0">No Spotify preview available</div>'}
            ${spotUrl  ? `<a class="open-spotify" href="${escHtml(spotUrl)}" target="_blank" rel="noopener"><span class="sp-icon">♪</span>Open in Spotify</a>` : ''}
          </div>
        </div>`;
    });
    html += `</div></div>`;
    wrap.innerHTML += html;
  });
}

// ── Spotify embed toggle ──────────────────────────────────────────────────────
function toggleEmbed(rowId, embedUrl){
  const embed   = document.getElementById('embed-'+rowId);
  const playBtn = document.getElementById('play-'+rowId);
  const isOpen  = embed.style.display === 'block';

  // Close currently open embed
  if(currentlyPlaying && currentlyPlaying !== rowId){
    const prevEmbed = document.getElementById('embed-'+currentlyPlaying);
    const prevPlay  = document.getElementById('play-'+currentlyPlaying);
    if(prevEmbed) prevEmbed.style.display='none';
    if(prevPlay)  prevPlay.classList.remove('playing');
  }

  if(isOpen){
    embed.style.display='none';
    playBtn.classList.remove('playing');
    playBtn.textContent='▶';
    currentlyPlaying=null;
  } else {
    embed.style.display='block';
    playBtn.classList.add('playing');
    playBtn.textContent='⏸';
    currentlyPlaying=rowId;
    embed.scrollIntoView({behavior:'smooth',block:'nearest'});
  }
}

// ── Utils ─────────────────────────────────────────────────────────────────────
function show(id){const el=document.getElementById(id);if(el)el.style.display='block'}
function hide(id){const el=document.getElementById(id);if(el)el.style.display='none'}
function showError(msg){const el=document.getElementById('error-box');if(el){el.textContent=msg;el.style.display='block'}}
function capitalize(s){return s?s.charAt(0).toUpperCase()+s.slice(1):s}
function escHtml(s){if(!s)return'';return s.replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
</script>
</body>
</html>
"""
