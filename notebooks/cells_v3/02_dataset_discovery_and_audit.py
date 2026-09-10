"""Cell 2: Dataset discovery, integrity checks, pHash deduplication, and class profiling."""

import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import LabelEncoder


def compute_dhash_int(image_path: str, hash_size: int = 8) -> int:
    """Computes difference hash (dHash) as a 64-bit integer."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = np.array(img, dtype=np.int32)
            diff = pixels[:, 1:] > pixels[:, :-1]
            decimal_val = 0
            for bit in diff.flatten():
                decimal_val = (decimal_val << 1) | int(bit)
            return decimal_val
    except Exception:
        return 0


def hamming_distance(h1: int, h2: int) -> int:
    """Computes bitwise Hamming distance between two integers."""
    return bin(h1 ^ h2).count("1")


def discover_dataset(root_dir: str) -> pd.DataFrame:
    """Discovers all valid images grouped by class subdirectory."""
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    records = []
    root_path = Path(root_dir)

    if not root_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

    class_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class subdirectories found in: {root_dir}")

    for class_dir in class_dirs:
        label = class_dir.name
        for filepath in class_dir.iterdir():
            if filepath.is_file() and filepath.suffix.lower() in valid_extensions:
                records.append({"filepath": str(filepath), "label": label})

    return pd.DataFrame(records)


df = discover_dataset(DATASET_DIR)
print(f"Total raw images discovered: {len(df)}")

# Filter corrupted images and sub-resolution files
corrupted, dimension_issues = [], []
for idx, row in df.iterrows():
    img = cv2.imread(row["filepath"])
    if img is None:
        corrupted.append(idx)
        continue
    h, w = img.shape[:2]
    if h < 32 or w < 32:
        dimension_issues.append(idx)

if corrupted:
    df.drop(index=corrupted, inplace=True)
if dimension_issues:
    df.drop(index=dimension_issues, inplace=True)

df.reset_index(drop=True, inplace=True)
print(f"Valid images after integrity verification: {len(df)}")
print(f"  Corrupted files removed : {len(corrupted)}")
print(f"  Sub-resolution removed  : {len(dimension_issues)}")

# Deduplication via Perceptual Hash
print("\nExecuting perceptual hash deduplication scan...")
hashes = [compute_dhash_int(fp) for fp in df["filepath"]]
df["dhash"] = hashes

# Check if audit file exists in results/ or execute local duplicate detection
duplicate_indices = set()
for i in range(len(df)):
    if i in duplicate_indices:
        continue
    h1 = df.iloc[i]["dhash"]
    c1 = df.iloc[i]["label"]
    for j in range(i + 1, len(df)):
        if j in duplicate_indices:
            continue
        h2 = df.iloc[j]["dhash"]
        if hamming_distance(h1, h2) <= 2:
            duplicate_indices.add(j)

if duplicate_indices:
    print(f"Removing {len(duplicate_indices)} duplicate/near-duplicate images to prevent leakage...")
    df.drop(index=list(duplicate_indices), inplace=True)
    df.reset_index(drop=True, inplace=True)

print(f"Deduplicated dataset pool: {len(df)} images")

label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])
CLASS_NAMES = list(label_encoder.classes_)
NUM_CLASSES = len(CLASS_NAMES)

with open(LABELS_PATH, "w") as f:
    f.write("\n".join(CLASS_NAMES))

print(f"Saved class labels ({NUM_CLASSES} classes) to: {LABELS_PATH}")

# Class distribution analysis
class_counts = (
    df.groupby("label")
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
    .reset_index(drop=True)
)

class_counts["tier"] = class_counts["count"].apply(
    lambda x: "Tier 1 (Aggressive)" if x <= TIER1_THRESHOLD
    else ("Tier 2 (Moderate)" if x <= TIER2_THRESHOLD else "Tier 3 (Light)")
)

tier_colors = {
    "Tier 1 (Aggressive)": "#d62728",
    "Tier 2 (Moderate)":   "#ff7f0e",
    "Tier 3 (Light)":      "#2ca02c",
}

fig, ax = plt.subplots(figsize=(14, 12))
sorted_counts = class_counts.sort_values("count", ascending=True)
bar_colors = [tier_colors[t] for t in sorted_counts["tier"]]
bars = ax.barh(sorted_counts["label"], sorted_counts["count"], color=bar_colors, edgecolor="white", height=0.75)
for bar, count in zip(bars, sorted_counts["count"]):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2, str(count), va="center", fontsize=8)

ax.axvline(x=TIER1_THRESHOLD, color="#d62728", linestyle="--", linewidth=1.2, label=f"Tier 1 Threshold ({TIER1_THRESHOLD})")
ax.axvline(x=TIER2_THRESHOLD, color="#ff7f0e", linestyle="--", linewidth=1.2, label=f"Tier 2 Threshold ({TIER2_THRESHOLD})")
ax.set_title("Class Distribution and Augmentation Tier Profile (Batik v3)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "v3_class_distribution.png"), dpi=150)
plt.close()
