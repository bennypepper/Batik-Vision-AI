"""Extract and optimize authentic exemplar swatches for all 28 batik motifs.

Picks clear, high-resolution representative images from the dataset directory,
center-crops them into square swatches, and saves them to ``web/public/motifs/<slug>.jpg``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS_PATH = REPO_ROOT / "models" / "labels.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "web" / "public" / "motifs"


def extract_swatches(dataset_dir: Path, labels_path: Path, output_dir: Path) -> list[dict]:
    """Extract 480x480 exemplar JPEG swatches for every motif in ``labels_path``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))["classes"]

    extracted = []
    for item in labels:
        slug = item["slug"]
        folder = dataset_dir / slug
        if not folder.exists():
            print(f"Error: missing directory {folder}")
            continue

        images = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))
        if not images:
            print(f"Error: no images in {folder}")
            continue

        scored_imgs = []
        for img_path in images:
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
                    aspect = max(w, h) / max(min(w, h), 1)
                    score = min(w, h) - (aspect - 1.0) * 200.0
                    scored_imgs.append((score, img_path, w, h))
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                print(f"Warning: skipping unreadable candidate {img_path}: {exc}")

        if not scored_imgs:
            print(f"Error: no readable images for {slug}")
            continue

        scored_imgs.sort(key=lambda x: x[0], reverse=True)
        src_path = scored_imgs[0][1]

        with Image.open(src_path) as im:
            rgb = im.convert("RGB")
            w, h = rgb.size
            crop_size = min(w, h)
            left = (w - crop_size) // 2
            top = (h - crop_size) // 2
            cropped = rgb.crop((left, top, left + crop_size, top + crop_size))
            resized = cropped.resize((480, 480), Image.Resampling.LANCZOS)

            dest_path = output_dir / f"{slug}.jpg"
            resized.save(dest_path, "JPEG", quality=88, optimize=True)

            extracted.append({
                "id": item["id"],
                "slug": slug,
                "name": item["name_id"],
                "src_file": src_path.name,
                "orig_size": f"{w}x{h}",
                "dest": f"/motifs/{slug}.jpg",
                "bytes": dest_path.stat().st_size,
            })

    print(f"Successfully extracted {len(extracted)}/{len(labels)} authentic motif swatches into {output_dir}:")
    for r in extracted:
        print(f"[{r['id']:2d}] {r['slug']:28s} <- {r['src_file']:25s} ({r['orig_size']}) -> {r['bytes'] // 1024} KB")
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 480x480 exemplar motif swatches.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Root dataset directory")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_PATH, help="Path to labels.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output swatch directory")
    args = parser.parse_args()
    extract_swatches(args.dataset_dir, args.labels, args.output_dir)


if __name__ == "__main__":
    main()
