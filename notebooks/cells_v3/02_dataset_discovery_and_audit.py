"""Cell 2: Dataset discovery, 28-class filtering, dHash deduplication, quarantine, and clustering."""

import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.preprocessing import LabelEncoder


def load_active_labels_set(labels_file: str) -> set[str] | None:
    """Load the 28 active class names from labels.txt when available."""
    p = Path(labels_file)
    if not p.exists():
        return None
    return {line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}


def compute_dhash_int(image_path: str, hash_size: int = 8) -> int | None:
    """Compute difference hash (dHash) as a 64-bit integer, or None if unreadable."""
    try:
        with Image.open(image_path) as img:
            gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = np.array(gray, dtype=np.int32)
            diff = pixels[:, 1:] > pixels[:, :-1]
            decimal_val = 0
            for bit in diff.flatten():
                decimal_val = (decimal_val << 1) | int(bit)
            return decimal_val
    except (OSError, UnidentifiedImageError, ValueError):
        return None


def hamming_distance(h1: int, h2: int) -> int:
    """Compute bitwise Hamming distance between two integers."""
    return (h1 ^ h2).bit_count()


def discover_dataset(root_dir: str, active_labels: set[str] | None = None) -> pd.DataFrame:
    """Discover all valid images grouped by class subdirectory."""
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    records = []
    root_path = Path(root_dir)

    if not root_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

    class_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
    if active_labels is not None:
        class_dirs = [d for d in class_dirs if d.name in active_labels]
    if not class_dirs:
        raise ValueError(f"No matching class subdirectories found in: {root_dir}")

    for class_dir in class_dirs:
        label = class_dir.name
        for filepath in sorted(class_dir.iterdir()):
            if filepath.is_file() and filepath.suffix.lower() in valid_extensions:
                records.append({"filepath": str(filepath), "label": label})

    return pd.DataFrame(records)


active_labels = load_active_labels_set(ACTIVE_LABELS_FILE)
if active_labels is not None:
    print(f"Filtering strictly to {len(active_labels)} active classes from {ACTIVE_LABELS_FILE}")
df = discover_dataset(DATASET_DIR, active_labels=active_labels)
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

# Perceptual hash scan with safe None handling and cross-class conflict quarantine
print("\nExecuting perceptual hash deduplication and cross-class quarantine scan...")
df["dhash"] = [compute_dhash_int(fp) for fp in df["filepath"]]
unhashable = df["dhash"].isna()
if unhashable.any():
    print(f"  Dropping {int(unhashable.sum())} unhashable images.")
    df = df.loc[~unhashable].reset_index(drop=True)

quarantined_cross_class: set[int] = set()
intra_class_drop: set[int] = set()
near_dup_edges: list[tuple[int, int]] = []

dhashes = df["dhash"].tolist()
labels_list = df["label"].tolist()
for i in range(len(df)):
    h1, c1 = dhashes[i], labels_list[i]
    for j in range(i + 1, len(df)):
        dist = hamming_distance(h1, dhashes[j])
        if dist <= 3:
            near_dup_edges.append((i, j))
        if dist <= 2:
            if c1 != labels_list[j]:
                # Quarantine BOTH images in cross-class conflicts (never bias alphabetically)
                quarantined_cross_class.add(i)
                quarantined_cross_class.add(j)
            elif i not in intra_class_drop and j not in intra_class_drop:
                intra_class_drop.add(j)

# Connected-component visual cluster IDs for GroupStratifiedKFold
parent = list(range(len(df)))


def _find(x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for u, v in near_dup_edges:
    ru, rv = _find(u), _find(v)
    if ru != rv:
        parent[max(ru, rv)] = min(ru, rv)

df["cluster_id"] = [_find(i) for i in range(len(df))]

drop_indices = quarantined_cross_class | intra_class_drop
if drop_indices:
    print(f"  Quarantined cross-class conflict images (both sides): {len(quarantined_cross_class)}")
    print(f"  Auto-deduplicated intra-class images                : {len(intra_class_drop)}")
    df.drop(index=list(drop_indices), inplace=True)
    df.reset_index(drop=True, inplace=True)

print(f"Deduplicated dataset pool: {len(df)} images")

label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])
CLASS_NAMES = list(label_encoder.classes_)
NUM_CLASSES = len(CLASS_NAMES)

with open(LABELS_PATH, "w", encoding="utf-8") as f:
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
    lambda x: "Tier 1 (Aggressive)"
    if x <= TIER1_THRESHOLD
    else ("Tier 2 (Moderate)" if x <= TIER2_THRESHOLD else "Tier 3 (Light)")
)

tier_colors = {
    "Tier 1 (Aggressive)": "#d62728",
    "Tier 2 (Moderate)": "#ff7f0e",
    "Tier 3 (Light)": "#2ca02c",
}

fig, ax = plt.subplots(figsize=(14, 12))
sorted_counts = class_counts.sort_values("count", ascending=True)
bar_colors = [tier_colors[t] for t in sorted_counts["tier"]]
bars = ax.barh(sorted_counts["label"], sorted_counts["count"], color=bar_colors, edgecolor="white", height=0.75)
for bar, count in zip(bars, sorted_counts["count"]):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2, str(count), va="center", fontsize=8)

ax.axvline(x=TIER1_THRESHOLD, color="#d62728", linestyle="--", linewidth=1.2, label=f"Tier 1 Threshold ({TIER1_THRESHOLD})")
ax.axvline(x=TIER2_THRESHOLD, color="#ff7f0e", linestyle="--", linewidth=1.2, label=f"Tier 2 Threshold ({TIER2_THRESHOLD})")
ax.set_title("Class Distribution and Augmentation Tier Profile (Batik v3/v4)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "v3_class_distribution.png"), dpi=150)
plt.close()
