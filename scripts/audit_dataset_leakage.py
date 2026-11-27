"""Dataset leakage and duplicate image auditor using 64-bit perceptual dHash."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load_active_labels(labels_path: str) -> set[str]:
    """Load the active class names from a ``labels.txt`` or ``labels.json`` file."""
    path = Path(labels_path)
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "classes" in payload:
            return {item["slug"] for item in payload["classes"] if "slug" in item}
        if isinstance(payload, list):
            return {str(item).strip() for item in payload if str(item).strip()}
        raise ValueError(f"Unsupported JSON labels structure in {labels_path}")

    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def compute_dhash_int(img_path: str, hash_size: int = 8) -> int | None:
    """Compute 64-bit difference hash (dHash) as an integer, or ``None`` if unreadable."""
    try:
        with Image.open(img_path) as img:
            gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = np.array(gray, dtype=np.float32)
            diff = pixels[:, 1:] > pixels[:, :-1]
            packed = 0
            for bit in diff.flatten():
                packed = (packed << 1) | int(bit)
            return packed
    except (OSError, UnidentifiedImageError, ValueError):
        return None


def compute_image_phash(img_path: str, hash_size: int = 8) -> str:
    """Compute 64-bit difference hash (dHash) as a 16-char lowercase hex string."""
    dhash = compute_dhash_int(img_path, hash_size=hash_size)
    if dhash is None:
        raise OSError(f"Unable to decode image for dHash: {img_path}")
    return f"{dhash:016x}"


def compute_hamming_distance(hex_hash1: str, hex_hash2: str) -> int:
    """Compute bitwise Hamming distance between two 16-char hex hash strings."""
    val1 = int(hex_hash1, 16)
    val2 = int(hex_hash2, 16)
    return (val1 ^ val2).bit_count()


def scan_dataset_duplicates(
    dataset_dir: str,
    threshold: int = 3,
    active_labels: set[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Scan dataset directory for exact and near-duplicate images.

    Args:
        dataset_dir: Root directory containing class subdirectories.
        threshold: Maximum Hamming distance to treat as duplicate/near-duplicate.
        active_labels: Optional set of allowed class folder names (e.g. the 28
            active classes from ``models/labels.txt``). Extra folders are ignored.

    Returns:
        ``(file_hashes, duplicates)`` where unreadable/corrupt images are skipped
        cleanly without causing zero-hash false collisions.
    """
    root_path = Path(dataset_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    file_hashes: dict[str, dict[str, Any]] = {}

    for class_dir in sorted(root_path.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        if active_labels is not None and class_name not in active_labels:
            continue
        for img_file in sorted(class_dir.iterdir()):
            if img_file.is_file() and img_file.suffix.lower() in VALID_EXTS:
                dhash = compute_dhash_int(str(img_file))
                if dhash is None:
                    print(f"Warning: Skipping corrupt/unreadable image: {img_file}")
                    continue
                rel_path = str(img_file.relative_to(root_path)).replace("\\", "/")
                file_hashes[rel_path] = {
                    "hash": f"{dhash:016x}",
                    "dhash_int": dhash,
                    "class": class_name,
                    "abs_path": str(img_file),
                    "size_bytes": img_file.stat().st_size,
                }

    items = list(file_hashes.items())
    duplicates: list[dict[str, Any]] = []

    hash_buckets: dict[str, list[str]] = {}
    for rel_path, meta in items:
        hash_buckets.setdefault(meta["hash"], []).append(rel_path)

    seen_pairs: set[tuple[str, str]] = set()
    for paths in hash_buckets.values():
        if len(paths) > 1:
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    p_a, p_b = paths[i], paths[j]
                    pair_key = (p_a, p_b) if p_a < p_b else (p_b, p_a)
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        duplicates.append({
                            "file_a": p_a,
                            "file_b": p_b,
                            "class_a": file_hashes[p_a]["class"],
                            "class_b": file_hashes[p_b]["class"],
                            "distance": 0,
                            "is_cross_class": file_hashes[p_a]["class"] != file_hashes[p_b]["class"],
                        })

    if threshold > 0:
        n = len(items)
        for i in range(n):
            p_a, meta_a = items[i]
            h_a = meta_a["dhash_int"]
            for j in range(i + 1, n):
                p_b, meta_b = items[j]
                pair_key = (p_a, p_b) if p_a < p_b else (p_b, p_a)
                if pair_key in seen_pairs:
                    continue
                dist = (h_a ^ meta_b["dhash_int"]).bit_count()
                if dist <= threshold:
                    seen_pairs.add(pair_key)
                    duplicates.append({
                        "file_a": p_a,
                        "file_b": p_b,
                        "class_a": meta_a["class"],
                        "class_b": meta_b["class"],
                        "distance": dist,
                        "is_cross_class": meta_a["class"] != meta_b["class"],
                    })

    return file_hashes, duplicates


def partition_duplicate_actions(duplicates: Sequence[dict[str, Any]]) -> dict[str, set[str]]:
    """Partition duplicate pairs into safe intra-class drops vs. cross-class quarantine.

    Cross-class conflicts quarantine BOTH images for manual adjudication so we
    never greedily drop whichever class sorts later alphabetically.
    """
    quarantined_cross_class: set[str] = set()
    auto_drop_intra_class: set[str] = set()

    for dup in duplicates:
        file_a = dup["file_a"]
        file_b = dup["file_b"]
        if dup.get("is_cross_class") or dup.get("class_a") != dup.get("class_b"):
            quarantined_cross_class.add(file_a)
            quarantined_cross_class.add(file_b)

    for dup in duplicates:
        if dup.get("is_cross_class") or dup.get("class_a") != dup.get("class_b"):
            continue
        file_a, file_b = sorted((dup["file_a"], dup["file_b"]))
        if file_a in quarantined_cross_class or file_b in quarantined_cross_class:
            continue
        if file_a not in auto_drop_intra_class and file_b not in auto_drop_intra_class:
            auto_drop_intra_class.add(file_b)

    return {
        "quarantined_cross_class": quarantined_cross_class,
        "auto_drop_intra_class": auto_drop_intra_class,
    }


def build_duplicate_clusters(
    file_paths: Sequence[str],
    duplicates: Sequence[dict[str, Any]],
) -> dict[str, int]:
    """Assign connected-component cluster IDs for ``GroupStratifiedKFold`` leakage prevention."""
    parent: dict[str, str] = {fp: fp for fp in file_paths}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: str, b: str) -> None:
        if a not in parent or b not in parent:
            return
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for dup in duplicates:
        union(dup["file_a"], dup["file_b"])

    root_to_id: dict[str, int] = {}
    clusters: dict[str, int] = {}
    for fp in file_paths:
        root = find(fp)
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id)
        clusters[fp] = root_to_id[root]
    return clusters


def run_audit(
    dataset_dir: str,
    output_csv: str | None = None,
    threshold: int = 3,
    labels_path: str | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Execute the dataset leakage audit and optionally export a CSV report."""
    active_labels = load_active_labels(labels_path) if labels_path else None
    print(f"Scanning dataset: {dataset_dir}")
    if active_labels is not None:
        print(f"Filtering strictly to {len(active_labels)} active classes from {labels_path}")

    file_hashes, duplicates = scan_dataset_duplicates(
        dataset_dir,
        threshold=threshold,
        active_labels=active_labels,
    )
    actions = partition_duplicate_actions(duplicates)

    print(f"Hashed {len(file_hashes)} valid images.")
    print(f"Found {len(duplicates)} duplicate/near-duplicate pairs (threshold <= {threshold}).")
    print(f"  - Auto-deduplicate intra-class images : {len(actions['auto_drop_intra_class'])}")
    print(f"  - Quarantined cross-class images      : {len(actions['quarantined_cross_class'])}")

    if output_csv:
        parent_dir = os.path.dirname(output_csv)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["file_a", "file_b", "class_a", "class_b", "distance", "is_cross_class"],
            )
            writer.writeheader()
            writer.writerows(duplicates)
        print(f"Saved duplicate report to {output_csv}")

    return file_hashes, duplicates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audit dataset for leakage and duplicates via perceptual dHash.",
    )
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to master dataset folder")
    parser.add_argument("--output_csv", type=str, default=None, help="Path to output duplicates CSV")
    parser.add_argument("--threshold", type=int, default=3, help="Hamming distance threshold")
    parser.add_argument("--labels", type=str, default=None, help="Optional path to active labels.txt/json")
    args = parser.parse_args()

    run_audit(args.dataset_dir, args.output_csv, args.threshold, labels_path=args.labels)
