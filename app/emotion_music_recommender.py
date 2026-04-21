"""
emotion_music_recommender.py
────────────────────────────
Emotion Transformation Music Recommender

Logic:
  current_emotion + language → staged playlist that moves user toward happy

Transformation map:
  angry   → Stage 1: calming  → Stage 2: uplifting
  sad     → Stage 1: comforting → Stage 2: hopeful
  fear    → Stage 1: calming  → Stage 2: hopeful
  disgust → Stage 1: calming  → Stage 2: uplifting
  neutral → Stage 1: light positive
  happy   → Stage 1: maintain happy energy
  surprise→ Stage 1: hopeful  → Stage 2: uplifting
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH  = os.path.join(BASE_DIR, "data", "curated_songs.csv")

# ── Emotion → staged mood targets ───────────────────────────────────────────
EMOTION_TRANSFORMATION = {
    "angry":    [("calming",    "Calm down first"),
                 ("uplifting",  "Lift your spirits")],
    "sad":      [("comforting", "You are not alone"),
                 ("hopeful",    "Better days ahead")],
    "fear":     [("calming",    "Breathe and relax"),
                 ("hopeful",    "Feel safe and strong")],
    "disgust":  [("calming",    "Reset your mind"),
                 ("uplifting",  "Fresh positive vibes")],
    "neutral":  [("hopeful",    "Light positive energy")],
    "happy":    [("happy",      "Keep the good vibes")],
    "surprise": [("hopeful",    "Channel that energy"),
                 ("uplifting",  "Ride the wave")],
}

# ── Mood tag → audio feature ranges (valence, energy) ───────────────────────
MOOD_FEATURE_RANGES = {
    "calming":    {"valence": (0.55, 0.80), "energy": (0.20, 0.55), "tempo": (60, 90)},
    "comforting": {"valence": (0.60, 0.80), "energy": (0.35, 0.60), "tempo": (65, 95)},
    "hopeful":    {"valence": (0.72, 0.92), "energy": (0.50, 0.75), "tempo": (80, 105)},
    "uplifting":  {"valence": (0.82, 0.96), "energy": (0.65, 0.90), "tempo": (90, 120)},
    "happy":      {"valence": (0.85, 1.00), "energy": (0.70, 0.95), "tempo": (95, 165)},
    "light":      {"valence": (0.70, 0.90), "energy": (0.45, 0.70), "tempo": (75, 110)},
}

SUPPORTED_LANGUAGES = ["Telugu", "Hindi", "Punjabi", "English"]


class EmotionMusicRecommender:
    def __init__(self, csv_path: str = CSV_PATH):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Curated songs CSV not found at: {csv_path}\n"
                "Please copy curated_songs.csv to your data/ folder."
            )
        self.df = pd.read_csv(csv_path)
        self._validate()
        print(f"✅ Loaded {len(self.df)} curated songs from {csv_path}")

    def _validate(self):
        required = ["song_name", "artist", "language", "emotion_tag",
                    "target_mood", "valence", "energy", "tempo",
                    "spotify_track_id", "spotify_url"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        # Normalise language casing
        self.df["language"] = self.df["language"].str.strip().str.title()
        self.df["emotion_tag"] = self.df["emotion_tag"].str.strip().str.lower()
        self.df["target_mood"] = self.df["target_mood"].str.strip().str.lower()

    # ─────────────────────────────────────────────────────────────────────────
    def recommend(
        self,
        emotion: str,
        language: str = "English",
        songs_per_stage: int = 5,
    ) -> Dict:
        """
        Returns a dict with:
          emotion          – detected emotion
          language         – chosen language
          transformation   – human-readable journey label
          stages           – list of stage dicts, each with:
                               stage_num, label, description, songs (list of song dicts)
        """
        emotion   = (emotion or "neutral").lower().strip()
        language  = (language or "English").strip().title()

        if language not in SUPPORTED_LANGUAGES:
            language = "English"

        stages_meta = EMOTION_TRANSFORMATION.get(
            emotion, EMOTION_TRANSFORMATION["neutral"]
        )

        result_stages = []
        for i, (mood_tag, description) in enumerate(stages_meta):
            songs = self._get_songs_for_stage(
                emotion, mood_tag, language, n=songs_per_stage
            )
            result_stages.append({
                "stage_num":   i + 1,
                "mood_tag":    mood_tag,
                "label":       description,
                "songs":       songs,
            })

        journey = self._journey_label(emotion)

        return {
            "emotion":        emotion,
            "language":       language,
            "journey":        journey,
            "stages":         result_stages,
            "total_songs":    sum(len(s["songs"]) for s in result_stages),
        }

    # ─────────────────────────────────────────────────────────────────────────
    def _get_songs_for_stage(
        self,
        emotion: str,
        mood_tag: str,
        language: str,
        n: int,
    ) -> List[Dict]:

        # Step 1: exact match – language + emotion_tag
        mask = (
            (self.df["language"] == language) &
            (self.df["emotion_tag"] == mood_tag)
        )
        pool = self.df[mask].copy()

        # Step 2: relax to audio-feature range if pool too small
        if len(pool) < n:
            pool = self._relax_filter(language, mood_tag)

        # Step 3: fallback to any language matching the mood
        if len(pool) < n:
            pool = self.df[self.df["emotion_tag"] == mood_tag].copy()

        # Step 4: absolute fallback
        if pool.empty:
            pool = self.df[self.df["language"] == language].copy()
        if pool.empty:
            pool = self.df.copy()

        # Score: prioritise valence + energy match to target mood
        ranges = MOOD_FEATURE_RANGES.get(mood_tag, MOOD_FEATURE_RANGES["hopeful"])
        v_mid  = np.mean(ranges["valence"])
        e_mid  = np.mean(ranges["energy"])

        pool["_score"] = (
            1.0 - np.abs(pool["valence"] - v_mid) +
            1.0 - np.abs(pool["energy"]  - e_mid)
        )
        pool = pool.sort_values("_score", ascending=False)
        pool = pool.drop_duplicates(subset=["song_name", "artist"])
        top  = pool.head(n)

        return [self._row_to_dict(r) for _, r in top.iterrows()]

    def _relax_filter(self, language: str, mood_tag: str) -> pd.DataFrame:
        ranges = MOOD_FEATURE_RANGES.get(mood_tag, {})
        mask   = self.df["language"] == language
        for feat, (lo, hi) in ranges.items():
            if feat in self.df.columns:
                margin = 0.15
                mask &= self.df[feat].between(
                    max(0.0, lo - margin), min(1.0, hi + margin)
                )
        return self.df[mask].copy()

    @staticmethod
    def _row_to_dict(row) -> Dict:
        track_id  = str(row.get("spotify_track_id", "")).strip()
        url       = str(row.get("spotify_url", "")).strip()

        # Build embed URL for the Spotify iframe player
        embed_url = (
            f"https://open.spotify.com/embed/track/{track_id}"
            if track_id and track_id != "nan" else ""
        )

        return {
            "name":         row.get("song_name", "Unknown"),
            "artist":       row.get("artist", "Unknown"),
            "language":     row.get("language", "Unknown"),
            "mood_tag":     row.get("emotion_tag", ""),
            "valence":      round(float(row.get("valence", 0.0)), 3),
            "energy":       round(float(row.get("energy",  0.0)), 3),
            "tempo":        round(float(row.get("tempo",   0.0)), 1),
            "spotify_id":   track_id,
            "spotify_url":  url,
            "embed_url":    embed_url,
        }

    @staticmethod
    def _journey_label(emotion: str) -> str:
        labels = {
            "angry":    "Angry → Calm → Happy",
            "sad":      "Sad → Comforted → Hopeful",
            "fear":     "Anxious → Calm → Confident",
            "disgust":  "Upset → Reset → Uplifted",
            "neutral":  "Neutral → Positive",
            "happy":    "Happy → Stay Happy",
            "surprise": "Surprised → Focused → Uplifted",
        }
        return labels.get(emotion, "Journey to Happy")


# ── Flask route helper ────────────────────────────────────────────────────────

_recommender_instance: Optional[EmotionMusicRecommender] = None


def get_recommender(csv_path: str = CSV_PATH) -> EmotionMusicRecommender:
    global _recommender_instance
    if _recommender_instance is None:
        _recommender_instance = EmotionMusicRecommender(csv_path)
    return _recommender_instance


def recommend_for_emotion(
    emotion: str,
    language: str = "English",
    songs_per_stage: int = 5,
    csv_path: str = CSV_PATH,
) -> Dict:
    """Convenience function for app.py routes."""
    return get_recommender(csv_path).recommend(emotion, language, songs_per_stage)
