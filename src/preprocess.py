import os
from typing import Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMG_SIZE = 128
NUM_CLASSES = len(EMOTION_LABELS)

MUSIC_FEATURES = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]


def get_dataset_dirs(base_dir: str) -> Tuple[str, str]:
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    return train_dir, test_dir


def count_images_per_class(split_dir: str):
    counts = {}
    for cls in EMOTION_LABELS:
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            counts[cls] = 0
            continue
        counts[cls] = len([
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
        ])
    return counts


def _ensure_rgb(images, labels):
    images = tf.image.convert_image_dtype(images, tf.float32)
    return images, labels


def build_datasets_from_folders(
    base_dir: str,
    img_size: int = IMG_SIZE,
    batch_size: int = 32,
    seed: int = 42,
):
    train_dir, test_dir = get_dataset_dirs(base_dir)

    print("\nTrain counts:")
    train_counts = count_images_per_class(train_dir)
    for k, v in train_counts.items():
        print(f"  {k:<10}: {v}")

    print("\nTest counts:")
    test_counts = count_images_per_class(test_dir)
    for k, v in test_counts.items():
        print(f"  {k:<10}: {v}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=EMOTION_LABELS,
        color_mode="rgb",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=EMOTION_LABELS,
        color_mode="rgb",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(_ensure_rgb, num_parallel_calls=autotune)
    val_ds = val_ds.map(_ensure_rgb, num_parallel_calls=autotune)

    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    return train_ds, val_ds, train_counts, test_counts


def get_class_weight_dict(train_counts: dict):
    total = sum(train_counts.values())
    n_classes = len(EMOTION_LABELS)

    class_weights = {}
    for idx, cls in enumerate(EMOTION_LABELS):
        count = max(train_counts.get(cls, 0), 1)
        class_weights[idx] = total / (n_classes * count)

    return class_weights


def load_image_for_prediction(image_path: str, img_size: int = IMG_SIZE, rgb: bool = True) -> np.ndarray:
    img = Image.open(image_path)
    img = img.convert("RGB" if rgb else "L")
    img = img.resize((img_size, img_size))
    arr = np.array(img, dtype="float32")
    if not rgb:
        arr = arr[..., np.newaxis]
    arr = arr.reshape(1, img_size, img_size, 3 if rgb else 1)
    return arr


def preprocess_single_image(face_array: np.ndarray, img_size: int = IMG_SIZE, rgb: bool = True) -> np.ndarray:
    if face_array.ndim == 2:
        img = Image.fromarray(face_array.astype(np.uint8), "L")
        img = img.convert("RGB" if rgb else "L")
    elif face_array.ndim == 3:
        if face_array.shape[2] == 3:
            img = Image.fromarray(face_array.astype(np.uint8))
            img = img.convert("RGB" if rgb else "L")
        elif face_array.shape[2] == 1:
            img = Image.fromarray(face_array[:, :, 0].astype(np.uint8), "L")
            img = img.convert("RGB" if rgb else "L")
        else:
            raise ValueError("Unsupported channel shape in face_array.")
    else:
        raise ValueError("Unsupported face_array dimensions.")

    img = img.resize((img_size, img_size))
    arr = np.array(img, dtype="float32")
    if not rgb:
        arr = arr[..., np.newaxis]
    arr = arr.reshape(1, img_size, img_size, 3 if rgb else 1)
    return arr


def load_and_clean_music_data(csv_path: str) -> pd.DataFrame:
    print(f"\nLoading music data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Raw rows: {len(df)}")

    if "name" in df.columns and "artist" in df.columns:
        df.dropna(subset=["name", "artist"], inplace=True)

    for col in MUSIC_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    if "genre" in df.columns:
        df["genre"] = df["genre"].fillna("Unknown")
    else:
        df["genre"] = "Unknown"

    if "loudness" in df.columns:
        df["loudness"] = (df["loudness"] - df["loudness"].min()) / (
            df["loudness"].max() - df["loudness"].min() + 1e-8
        )

    if "tempo" in df.columns:
        df["tempo"] = (df["tempo"] - df["tempo"].min()) / (
            df["tempo"].max() - df["tempo"].min() + 1e-8
        )

    print(f"Clean rows: {len(df)}")
    return df.reset_index(drop=True)