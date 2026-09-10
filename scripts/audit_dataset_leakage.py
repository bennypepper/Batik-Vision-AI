import os
import argparse
import csv
from pathlib import Path
from PIL import Image
import numpy as np

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def compute_image_phash(img_path: str, hash_size: int = 8) -> str:
    """Compute 64-bit difference hash (dHash) from image file path."""
    with Image.open(img_path) as img:
        img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = np.array(img, dtype=np.float32)
        # Compare adjacent columns
        diff = pixels[:, 1:] > pixels[:, :-1]
        # Pack boolean array into hex string
        packed = 0
        for bit in diff.flatten():
            packed = (packed << 1) | int(bit)
        return f"{packed:016x}"

def compute_hamming_distance(hex_hash1: str, hex_hash2: str) -> int:
    """Compute bitwise Hamming distance between two 16-char hex hash strings."""
    val1 = int(hex_hash1, 16)
    val2 = int(hex_hash2, 16)
    xor_val = val1 ^ val2
    return bin(xor_val).count("1")

def scan_dataset_duplicates(dataset_dir: str, threshold: int = 3) -> tuple[dict, list[dict]]:
    """
    Scan dataset directory for exact and near-duplicate images.
    Returns:
        file_hashes: dict mapping relative path -> {hash, class, size_bytes}
        duplicates: list of dicts with {file_a, file_b, class_a, class_b, distance}
    """
    root_path = Path(dataset_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    file_hashes = {}
    
    # 1. Compute hash for each valid image
    for class_dir in root_path.iterdir():
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for img_file in class_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in VALID_EXTS:
                rel_path = str(img_file.relative_to(root_path)).replace("\\", "/")
                try:
                    h = compute_image_phash(str(img_file))
                    file_hashes[rel_path] = {
                        "hash": h,
                        "class": class_name,
                        "abs_path": str(img_file),
                        "size_bytes": img_file.stat().st_size
                    }
                except Exception as e:
                    print(f"Warning: Failed to hash {img_file}: {e}")

    # 2. Pairwise comparison
    items = list(file_hashes.items())
    duplicates = []
    
    # Group identical hashes first for efficiency
    hash_buckets = {}
    for rel_path, meta in items:
        hash_buckets.setdefault(meta["hash"], []).append(rel_path)

    # Identical duplicates (distance = 0)
    seen_pairs = set()
    for h, paths in hash_buckets.items():
        if len(paths) > 1:
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    p_a, p_b = paths[i], paths[j]
                    pair_key = tuple(sorted([p_a, p_b]))
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        duplicates.append({
                            "file_a": p_a,
                            "file_b": p_b,
                            "class_a": file_hashes[p_a]["class"],
                            "class_b": file_hashes[p_b]["class"],
                            "distance": 0,
                            "is_cross_class": file_hashes[p_a]["class"] != file_hashes[p_b]["class"]
                        })

    # Near-duplicates (distance > 0 and <= threshold)
    if threshold > 0:
        n = len(items)
        for i in range(n):
            p_a, meta_a = items[i]
            h_a = meta_a["hash"]
            for j in range(i + 1, n):
                p_b, meta_b = items[j]
                h_b = meta_b["hash"]
                pair_key = tuple(sorted([p_a, p_b]))
                if pair_key in seen_pairs:
                    continue
                dist = compute_hamming_distance(h_a, h_b)
                if dist <= threshold:
                    seen_pairs.add(pair_key)
                    duplicates.append({
                        "file_a": p_a,
                        "file_b": p_b,
                        "class_a": meta_a["class"],
                        "class_b": meta_b["class"],
                        "distance": dist,
                        "is_cross_class": meta_a["class"] != meta_b["class"]
                    })

    return file_hashes, duplicates

def run_audit(dataset_dir: str, output_csv: str = None, threshold: int = 3):
    print(f"Scanning dataset: {dataset_dir}")
    file_hashes, duplicates = scan_dataset_duplicates(dataset_dir, threshold=threshold)
    print(f"Hashed {len(file_hashes)} total images.")
    print(f"Found {len(duplicates)} duplicate/near-duplicate pairs (threshold <= {threshold}).")

    cross_class = [d for d in duplicates if d.get("is_cross_class")]
    intra_class = [d for d in duplicates if not d.get("is_cross_class")]
    print(f"  - Intra-class duplicates (redundant samples): {len(intra_class)}")
    print(f"  - Cross-class duplicates (potential labeling errors): {len(cross_class)}")

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_a", "file_b", "class_a", "class_b", "distance", "is_cross_class"])
            writer.writeheader()
            writer.writerows(duplicates)
        print(f"Saved duplicate report to {output_csv}")

    return file_hashes, duplicates

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit dataset for leakage and duplicates via perceptual hashing.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to master dataset folder")
    parser.add_argument("--output_csv", type=str, default=None, help="Path to output duplicates CSV")
    parser.add_argument("--threshold", type=int, default=3, help="Hamming distance threshold")
    args = parser.parse_args()

    run_audit(args.dataset_dir, args.output_csv, args.threshold)
