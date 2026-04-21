"""
emotion_mapping.py
──────────────────
Maps FER-2013 emotion labels to music mood features.

Each emotion gets a set of audio feature thresholds that we'll use
to filter songs from the music dataset (valence, energy, tempo, etc.).

Emotion → Mood → Audio Features
"""

# ──────────────────────────────────────────────────────────────────────────────
# Core mapping: emotion label → human-readable mood description
# ──────────────────────────────────────────────────────────────────────────────
EMOTION_TO_MOOD = {
    "happy":    "energetic & uplifting",
    "sad":      "calm & melancholic",
    "angry":    "relaxing & soothing",
    "fear":     "ambient & calming",
    "disgust":  "chill & mellow",
    "surprise": "upbeat & exciting",
    "neutral":  "balanced & easy-going",
}

# ──────────────────────────────────────────────────────────────────────────────
# Audio feature filters for content-based recommendation.
# We filter songs whose features fall inside these (min, max) ranges.
#
# Features come from Spotify / the music dataset:
#   valence  : 0–1  (musical positiveness)
#   energy   : 0–1  (intensity & activity)
#   tempo    : BPM  (speed of the track)
#   acousticness : 0–1 (acoustic signal confidence)
# ──────────────────────────────────────────────────────────────────────────────
EMOTION_FEATURE_FILTERS = {
    # NOTE: tempo is normalised to 0-1 (raw BPM / ~240).
    # Use 0-1 values for ALL features here.
    #   tempo 0.42 ≈ 100 BPM  |  0.21 ≈ 50 BPM  |  0.63 ≈ 150 BPM
    "happy": {
        "valence":      (0.55, 1.0),   # bright, positive
        "energy":       (0.55, 1.0),   # high energy
        "tempo":        (0.38, 1.0),   # fast (≈ 90+ BPM)
        "acousticness": (0.0,  0.55),
    },
    "sad": {
        "valence":      (0.0,  0.42),  # low positiveness
        "energy":       (0.0,  0.42),  # calm
        "tempo":        (0.0,  0.42),  # slow (≈ < 100 BPM)
        "acousticness": (0.25, 1.0),   # acoustic / stripped back
    },
    "angry": {
        "valence":      (0.25, 0.75),  # moderate — counter-balance anger
        "energy":       (0.05, 0.50),  # low energy to relax
        "tempo":        (0.0,  0.50),  # slow to medium
        "acousticness": (0.30, 1.0),
    },
    "fear": {
        "valence":      (0.20, 0.70),
        "energy":       (0.05, 0.45),  # very calm
        "tempo":        (0.0,  0.42),
        "acousticness": (0.40, 1.0),
    },
    "disgust": {
        "valence":      (0.20, 0.65),
        "energy":       (0.15, 0.55),  # mellow
        "tempo":        (0.0,  0.50),
        "acousticness": (0.20, 0.85),
    },
    "surprise": {
        "valence":      (0.50, 1.0),   # positive surprise
        "energy":       (0.50, 1.0),   # upbeat
        "tempo":        (0.35, 1.0),   # fast
        "acousticness": (0.0,  0.55),
    },
    "neutral": {
        "valence":      (0.30, 0.70),  # middle of the road
        "energy":       (0.30, 0.70),
        "tempo":        (0.25, 0.60),  # medium pace
        "acousticness": (0.05, 0.75),
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Preferred genres per emotion (used as a soft boost, not hard filter)
# ──────────────────────────────────────────────────────────────────────────────
EMOTION_PREFERRED_GENRES = {
    "happy":    ["Pop", "Electronic", "Latin", "Reggae"],
    "sad":      ["Folk", "Blues", "Country", "New Age"],
    "angry":    ["New Age", "Jazz", "Folk", "Classical"],
    "fear":     ["New Age", "Jazz", "Ambient"],
    "disgust":  ["Jazz", "Blues", "Folk"],
    "surprise": ["Electronic", "Pop", "Rock"],
    "neutral":  ["Pop", "Rock", "RnB", "Country"],
}

# ──────────────────────────────────────────────────────────────────────────────
# Helper function
# ──────────────────────────────────────────────────────────────────────────────
def get_emotion_info(emotion: str) -> dict:
    """
    Return all mapping info for a given emotion label.
    Falls back to 'neutral' if emotion is not recognised.
    """
    emotion = emotion.lower().strip()
    if emotion not in EMOTION_TO_MOOD:
        emotion = "neutral"
    return {
        "emotion":         emotion,
        "mood":            EMOTION_TO_MOOD[emotion],
        "feature_filters": EMOTION_FEATURE_FILTERS[emotion],
        "preferred_genres": EMOTION_PREFERRED_GENRES.get(emotion, []),
    }


if __name__ == "__main__":
    # Quick self-test
    for emo in EMOTION_TO_MOOD:
        info = get_emotion_info(emo)
        print(f"{emo:10s} → {info['mood']}")
