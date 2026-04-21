# 🎭 MoodMate — Emotion Detection & Music Recommendation System

> Detects your facial emotion using a CNN, then recommends the perfect playlist to match your mood.

---

## 📁 Project Structure

```
emotion-music-app/
│
├── data/
│   ├── fer2013/           ← Put FER-2013 dataset here (download from Kaggle)
│   └── music_data.csv     ← Your 50 000-song dataset (already included)
│
├── models/
│   └── emotion_model.h5   ← Saved CNN model (generated after training)
│
├── src/
│   ├── preprocess.py      ← Data loading & preprocessing helpers
│   ├── train_model.py     ← CNN training script
│   ├── emotion_predictor.py ← Load model + predict from image/array
│   ├── music_recommender.py ← Content-based filtering engine
│   └── emotion_mapping.py   ← Emotion → audio feature rules
│
├── app/
│   ├── app.py             ← Flask web app (upload + webcam UI)
│   └── webcam_realtime.py ← Standalone OpenCV real-time detection
│
├── utils/
│   └── helper_functions.py ← Face detection, image utils, logging
│
├── requirements.txt
└── README.md
```

---

## ⚙️ How the System Works

```
[User Photo / Webcam]
        ↓
[Face Detection — OpenCV Haar Cascade]
        ↓
[Emotion CNN — 48×48 grayscale → 7 emotions]
        ↓
[Emotion Mapping — emotion → audio feature targets]
        ↓
[Music Recommender — cosine similarity on 50 000 songs]
        ↓
[Playlist Output — name, artist, genre, Spotify links]
```

---

## 🚀 Quick Start

### 1. Clone / set up the project

```bash
cd emotion-music-app
```

### 2. Create a Python virtual environment (recommended)

```bash
python -m venv venv

# On Linux / macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Train the Emotion Model

### Step 1 — Download FER-2013

1. Go to: https://www.kaggle.com/datasets/msambare/fer2013
2. Download and extract.
3. Arrange the files like this:

```
data/fer2013/
  train/
    angry/      *.png
    disgust/    *.png
    fear/       *.png
    happy/      *.png
    neutral/    *.png
    sad/        *.png
    surprise/   *.png
  test/
    angry/      *.png
    ...
```

### Step 2 — Run training

```bash
python src/train_model.py
```

Training details:
- Architecture : 3-block CNN → Dense → Softmax(7)
- Augmentation : flip, rotate, zoom, translate
- Callbacks    : early stopping, LR reduce, checkpoint
- Output       : `models/emotion_model.h5` + `models/training_curves.png`

Training takes ~30–60 minutes on CPU, ~5–10 min on GPU.

> **Tip:** Set `EPOCHS = 10` in `train_model.py` for a fast smoke-test.

---

## 🌐 Run the Web App

```bash
python app/app.py
```

Then open your browser at: **http://localhost:5000**

Features:
- 📷 Upload any face photo → get emotion + 10 song recommendations
- 🎥 Live webcam tab → emotion detected every 3 seconds
- 🎵 Each song shows genre, valence, energy + Spotify links

---

## 📷 Test With a Single Image (CLI)

```bash
python src/emotion_predictor.py path/to/face.jpg
```

Output example:
```
🎭 Detected emotion : HAPPY
   Confidence       : 91.4%

   All scores:
   happy      91.4% ██████████████████████████
   neutral     4.2% █
   surprise    2.8% █
   ...
```

---

## 🎥 Real-Time Webcam (OpenCV, no browser)

```bash
python app/webcam_realtime.py
```

Controls:
| Key   | Action              |
|-------|---------------------|
| Q / ESC | Quit              |
| S     | Save current frame  |
| SPACE | Freeze / unfreeze   |

---

## 🎵 Test the Recommender Alone

```bash
python src/music_recommender.py
```

---

## 🔬 How the Recommendation Works

### Content-Based Filtering

Each song in the dataset has Spotify audio features:

| Feature        | Meaning                              |
|----------------|--------------------------------------|
| `valence`      | Musical positiveness (0 = sad, 1 = joyful) |
| `energy`       | Intensity & activity                 |
| `danceability` | Rhythmic structure                   |
| `acousticness` | Acoustic vs electronic               |
| `tempo`        | Beats per minute                     |

### Emotion → Feature Mapping

| Emotion  | Valence    | Energy     | Tempo (BPM) |
|----------|-----------|------------|-------------|
| happy    | 0.6 – 1.0 | 0.6 – 1.0  | 100 – 200   |
| sad      | 0.0 – 0.4 | 0.0 – 0.4  | 40 – 100    |
| angry    | 0.3 – 0.7 | 0.1 – 0.45 | 50 – 110    |
| fear     | 0.3 – 0.65| 0.1 – 0.4  | 40 – 90     |
| disgust  | 0.3 – 0.6 | 0.2 – 0.5  | 60 – 110    |
| surprise | 0.55 – 1.0| 0.55 – 1.0 | 90 – 200    |
| neutral  | 0.35 – 0.65| 0.35 – 0.65| 70 – 130   |

Songs are first filtered by these ranges, then **cosine similarity** is computed against an ideal "target vector" for the emotion. Genre preferences give a small boost.

---

## 🛠 Troubleshooting

| Problem | Solution |
|---------|----------|
| `Model not found` | Run `python src/train_model.py` first |
| `No face detected` | Try a well-lit, front-facing photo |
| Low accuracy | Train for more epochs; use GPU |
| Webcam not opening | Change `CAMERA_INDEX = 1` in `webcam_realtime.py` |
| Import errors | Make sure virtual env is active and `pip install -r requirements.txt` ran |

---

## 📊 Expected Model Performance

| Metric          | Typical Value |
|-----------------|---------------|
| Train accuracy  | ~70 – 80%     |
| Test accuracy   | ~60 – 68%     |
| Inference speed | < 50 ms / image (CPU) |

FER-2013 is a challenging dataset — 65% accuracy is considered good. Humans achieve ~65–70% on the same test set.

---

## 🧩 Extending the Project

- **Better accuracy**: Use a pre-trained face model (e.g. MobileNetV2 + fine-tune)
- **More songs**: Plug in the Spotify API to fetch real-time recommendations
- **Text input**: Add a sentiment-analysis path (e.g. "I feel great today")
- **Mobile app**: Convert the Flask API into a React Native frontend

---

## 📄 Datasets

- **Emotion**: [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)  
- **Music**: Your `music_data.csv` (50 683 songs with Spotify audio features)

---

## 📜 License

MIT — free to use, modify, and distribute.
