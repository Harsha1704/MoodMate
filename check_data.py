"""
check_data.py
─────────────
Run this FIRST to check your FER-2013 dataset structure.
It will tell you exactly what it found and whether you're ready to train.

Usage:
    python check_data.py
    python check_data.py --data path/to/your/fer2013
"""

import os
import sys
import argparse

# ── Emotion labels FER-2013 uses ──────────────────────────────────────────────
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ── Accepted image extensions ─────────────────────────────────────────────────
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

BASE_DIR = os.path.dirname(__file__)


def check_subfolder_structure(split_dir: str, split_name: str) -> dict:
    """
    Check if split_dir has emotion subfolders (e.g. train/happy/*.png).
    Returns a summary dict.
    """
    found   = {}
    missing = []

    for emo in EMOTION_LABELS:
        emo_dir = os.path.join(split_dir, emo)
        if os.path.isdir(emo_dir):
            imgs = [f for f in os.listdir(emo_dir)
                    if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
            found[emo] = len(imgs)
        else:
            missing.append(emo)

    print(f"\n  📂 {split_name}/ — subfolder structure")
    if found:
        total = sum(found.values())
        for emo, count in sorted(found.items()):
            bar = "█" * min(30, count // 100)
            print(f"    ✅  {emo:10s}  {count:5d} images  {bar}")
        print(f"    ─────────────────────────")
        print(f"    Total: {total:,} images")
    if missing:
        for emo in missing:
            print(f"    ❌  {emo:10s}  — folder NOT FOUND")

    return {"type": "subfolders", "found": found, "missing": missing}


def check_flat_structure(split_dir: str, split_name: str) -> dict:
    """Check if split_dir has images directly (no subfolders)."""
    imgs = [f for f in os.listdir(split_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
    print(f"\n  📂 {split_name}/ — flat images (no subfolders)")
    print(f"    Found {len(imgs)} images directly in folder.")
    print(f"    ⚠️  These need to be sorted into emotion subfolders.")
    return {"type": "flat", "count": len(imgs)}


def check_csv_structure(data_dir: str) -> dict:
    """Check for CSV-format FER-2013 (fer2013.csv with pixel columns)."""
    candidates = ["fer2013.csv", "fer2013_with_emotions.csv",
                  "icml_face_data.csv", "data.csv"]
    for fname in candidates:
        path = os.path.join(data_dir, fname)
        if os.path.isfile(path):
            import pandas as pd
            df = pd.read_csv(path, nrows=3)
            print(f"\n  📄 Found CSV: {fname}")
            print(f"     Columns: {df.columns.tolist()}")
            return {"type": "csv", "path": path, "columns": df.columns.tolist()}
    return {"type": "none"}


def main():
    parser = argparse.ArgumentParser(description="Check FER-2013 dataset structure")
    parser.add_argument("--data", default=os.path.join(BASE_DIR, "data", "fer2013"),
                        help="Path to your fer2013 data folder")
    args = parser.parse_args()

    data_dir = args.data

    print("\n" + "═" * 60)
    print("  MoodMate — Dataset Checker")
    print("═" * 60)
    print(f"\n  Checking: {data_dir}")

    if not os.path.exists(data_dir):
        print(f"\n  ❌ Folder not found: {data_dir}")
        print("\n  📋 WHAT TO DO:")
        print(f"     1. Create the folder:  mkdir -p {data_dir}")
        print(f"     2. Copy your train/ and test/ folders inside it.")
        print(f"     Final structure should be:")
        print(f"     {data_dir}/")
        print(f"       train/")
        print(f"         angry/   *.png")
        print(f"         happy/   *.png  ...")
        print(f"       test/")
        print(f"         angry/   *.png  ...")
        return

    # ── Detect structure ──────────────────────────────────────────────────────
    train_dir = os.path.join(data_dir, "train")
    test_dir  = os.path.join(data_dir, "test")

    has_train = os.path.isdir(train_dir)
    has_test  = os.path.isdir(test_dir)

    results = {}

    if has_train:
        # Check for subfolders vs flat
        subdirs = [d for d in os.listdir(train_dir)
                   if os.path.isdir(os.path.join(train_dir, d))]
        if any(d.lower() in EMOTION_LABELS for d in subdirs):
            results["train"] = check_subfolder_structure(train_dir, "train")
        else:
            results["train"] = check_flat_structure(train_dir, "train")
    else:
        print(f"\n  ❌ No train/ folder found in {data_dir}")

    if has_test:
        subdirs = [d for d in os.listdir(test_dir)
                   if os.path.isdir(os.path.join(test_dir, d))]
        if any(d.lower() in EMOTION_LABELS for d in subdirs):
            results["test"] = check_subfolder_structure(test_dir, "test")
        else:
            results["test"] = check_flat_structure(test_dir, "test")
    else:
        print(f"\n  ❌ No test/ folder found in {data_dir}")

    # ── Check for CSV format ──────────────────────────────────────────────────
    csv_result = check_csv_structure(data_dir)

    # ── VERDICT & INSTRUCTIONS ────────────────────────────────────────────────
    print("\n" + "─" * 60)

    train_ok = (results.get("train", {}).get("type") == "subfolders"
                and not results["train"]["missing"])
    test_ok  = (results.get("test",  {}).get("type") == "subfolders"
                and not results["test"]["missing"])

    if train_ok and test_ok:
        print("\n  ✅  PERFECT! Your dataset is ready for training.")
        print("\n  👉  Run training now:")
        print("      python src/train_model.py")

    elif csv_result["type"] == "csv":
        print("\n  📄  Found CSV format. Run the converter first:")
        print("      python check_data.py --convert")
        print("      Then run: python src/train_model.py")

    elif results.get("train", {}).get("type") == "flat":
        print("\n  ⚠️   Your images are flat (not in emotion subfolders).")
        print("      FER-2013 from Kaggle should have subfolders.")
        print("      Make sure you downloaded the IMAGE version, not the CSV.")
        print("\n  📥  Correct Kaggle link:")
        print("      https://www.kaggle.com/datasets/msambare/fer2013")

    else:
        missing_train = results.get("train", {}).get("missing", EMOTION_LABELS)
        if missing_train:
            print(f"\n  ⚠️   Missing emotion folders in train/: {missing_train}")
            print("      Each emotion needs its own subfolder of .png images.")

        print("\n  📋  REQUIRED STRUCTURE:")
        print(f"      {data_dir}/")
        print(f"        train/")
        for emo in EMOTION_LABELS:
            print(f"          {emo}/    (*.png files)")
        print(f"        test/")
        for emo in EMOTION_LABELS:
            print(f"          {emo}/    (*.png files)")


if __name__ == "__main__":
    main()
