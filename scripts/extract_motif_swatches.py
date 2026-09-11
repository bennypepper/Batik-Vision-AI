"""Extracts and optimizes authentic exemplar swatches for all 28 batik motifs.

Picks clear, high-resolution representative images from the master dataset,
center-crops them into square swatches, and saves them to web/public/motifs/<slug>.jpg.
"""

import json
from pathlib import Path
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(r"D:\Personal Projects\Computer Vision-ML Projects\Batik Motifs Detector v2\Batik_dataset\dataset")
LABELS_PATH = REPO_ROOT / "models" / "labels.json"
OUTPUT_DIR = REPO_ROOT / "web" / "public" / "motifs"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))["classes"]

    extracted = []

    for item in labels:
        slug = item["slug"]
        folder = DATASET_DIR / slug
        if not folder.exists():
            print(f"Error: missing directory {folder}")
            continue

        images = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))
        if not images:
            print(f"Error: no images in {folder}")
            continue

        # Score candidate images: prefer resolutions >= 600px and aspect ratios close to 1.0
        scored_imgs = []
        for img_path in images:
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
                    aspect = max(w, h) / max(min(w, h), 1)
                    score = min(w, h) - (aspect - 1.0) * 200.0
                    scored_imgs.append((score, img_path, w, h))
            except Exception:
                continue

        if not scored_imgs:
            print(f"Error: no readable images for {slug}")
            continue

        scored_imgs.sort(key=lambda x: x[0], reverse=True)
        best_candidate = scored_imgs[0]
        src_path = best_candidate[1]

        # Open and crop to square from center
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            crop_size = min(w, h)
            left = (w - crop_size) // 2
            top = (h - crop_size) // 2
            right = left + crop_size
            bottom = top + crop_size
            cropped = im.crop((left, top, right, bottom))
            resized = cropped.resize((480, 480), Image.Resampling.LANCZOS)

            dest_path = OUTPUT_DIR / f"{slug}.jpg"
            resized.save(dest_path, "JPEG", quality=88, optimize=True)

            extracted.append({
                "id": item["id"],
                "slug": slug,
                "name": item["name_id"],
                "src_file": src_path.name,
                "orig_size": f"{w}x{h}",
                "dest": f"/motifs/{slug}.jpg",
                "bytes": dest_path.stat().st_size
            })

    print(f"Successfully extracted {len(extracted)}/28 authentic motif swatches into {OUTPUT_DIR}:")
    for r in extracted:
        print(f"[{r['id']:2d}] {r['slug']:28s} <- {r['src_file']:25s} ({r['orig_size']}) -> {r['bytes'] // 1024} KB")


if __name__ == "__main__":
    main()
