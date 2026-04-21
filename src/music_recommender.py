import os
import sys
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocess import load_and_clean_music_data
from src.emotion_mapping import EMOTION_FEATURE_FILTERS, EMOTION_PREFERRED_GENRES

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "music_data.csv")

SIMILARITY_FEATURES = [
    "valence", "energy", "danceability",
    "acousticness", "instrumentalness", "tempo",
]


class MusicRecommender:
    def __init__(self, csv_path: str = CSV_PATH):
        self.df = load_and_clean_music_data(csv_path)
        if self.df.empty:
            raise ValueError("Music dataset is empty after cleaning.")
        self.scaler = MinMaxScaler()
        self._prepare_feature_matrix()

    def _prepare_feature_matrix(self):
        self.feature_cols = [f for f in SIMILARITY_FEATURES if f in self.df.columns]
        if not self.feature_cols:
            raise ValueError("No similarity feature columns found in music dataset.")
        feature_data = self.df[self.feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        self.feature_matrix = self.scaler.fit_transform(feature_data)
        print(f"✅ Music feature matrix built: {self.feature_matrix.shape}")

    def recommend(self, emotion: str, n: int = 10) -> List[Dict]:
        emotion = (emotion or "neutral").lower().strip()
        filtered_df = self._filter_by_emotion(emotion)
        if len(filtered_df) < max(5, n * 2):
            filtered_df = self._filter_by_emotion(emotion, relax=True)
        if len(filtered_df) < n:
            filtered_df = self.df.copy()

        target_vector = self._build_target_vector(emotion)
        filtered_idx = filtered_df.index.to_list()
        song_features = self.feature_matrix[filtered_idx]
        similarities = cosine_similarity([target_vector], song_features)[0]

        preferred = set(EMOTION_PREFERRED_GENRES.get(emotion, []))
        genre_boost = np.array([
            0.05 if str(row.get("genre", "Unknown")) in preferred else 0.0
            for _, row in filtered_df.iterrows()
        ])
        final_scores = similarities + genre_boost
        filtered_df = filtered_df.copy()
        filtered_df["_score"] = final_scores
        filtered_df = filtered_df.sort_values("_score", ascending=False)

        dedup = filtered_df.drop_duplicates(subset=[c for c in ["name", "artist"] if c in filtered_df.columns])
        top_rows = dedup.head(n)

        results = []
        for _, row in top_rows.iterrows():
            results.append({
                "name": row.get("name", "Unknown"),
                "artist": row.get("artist", "Unknown"),
                "genre": row.get("genre", "Unknown"),
                "year": int(row["year"]) if "year" in row and not pd.isna(row["year"]) else "N/A",
                "valence": round(float(row.get("valence", 0.0)), 3),
                "energy": round(float(row.get("energy", 0.0)), 3),
                "tempo": round(float(row.get("tempo", 0.0)), 3),
                "preview_url": row.get("spotify_preview_url", "") or row.get("preview_url", ""),
                "spotify_id": row.get("spotify_id", ""),
            })
        return results

    def _filter_by_emotion(self, emotion: str, relax: bool = False) -> pd.DataFrame:
        filters = EMOTION_FEATURE_FILTERS.get(emotion, EMOTION_FEATURE_FILTERS["neutral"])
        mask = pd.Series(True, index=self.df.index)
        for feature, (lo, hi) in filters.items():
            if feature not in self.df.columns:
                continue
            cur_lo, cur_hi = lo, hi
            if relax:
                cur_lo = max(0.0, lo - 0.15)
                cur_hi = min(1.0, hi + 0.15)
            mask &= self.df[feature].between(cur_lo, cur_hi)
        return self.df.loc[mask].copy()

    def _build_target_vector(self, emotion: str) -> np.ndarray:
        filters = EMOTION_FEATURE_FILTERS.get(emotion, EMOTION_FEATURE_FILTERS["neutral"])
        target = []
        for feat in self.feature_cols:
            if feat in filters:
                lo, hi = filters[feat]
                target.append((lo + hi) / 2.0)
            else:
                target.append(0.5)
        target_array = np.asarray(target, dtype=np.float32).reshape(1, -1)
        return self.scaler.transform(target_array)[0]