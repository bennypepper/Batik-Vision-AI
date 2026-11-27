"""Targeted authentic image downloader for Priangan Merak Ngibing dataset balancing."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import requests
from duckduckgo_search import DDGS
from PIL import Image


def compute_dhash(image_arr: np.ndarray, hash_size: int = 8) -> int | None:
    """Compute difference hash (dHash) as a 64-bit integer, or None on decode error."""
    try:
        pil_img = Image.fromarray(cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB))
        pil_img = pil_img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = np.array(pil_img, dtype=np.int32)
        diff = pixels[:, 1:] > pixels[:, :-1]
        decimal_val = 0
        for bit in diff.flatten():
            decimal_val = (decimal_val << 1) | int(bit)
        return decimal_val
    except (cv2.error, OSError, ValueError):
        return None


def hamming_distance(h1: int, h2: int) -> int:
    return (h1 ^ h2).bit_count()


def download_merak_images(target_dir: Path, max_new: int = 60) -> None:
    """Download deduplicated Priangan Merak Ngibing images into ``target_dir``."""
    target_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(target_dir.glob("*.jpg")) + list(target_dir.glob("*.jpeg")) + list(target_dir.glob("*.png"))
    print(f"Initial files in {target_dir.name}: {len(existing_files)}")

    existing_hashes: set[int] = set()
    for fp in existing_files:
        img = cv2.imread(str(fp))
        if img is not None:
            h = compute_dhash(img)
            if h is not None:
                existing_hashes.add(h)

    queries = [
        "batik priangan merak ngibing",
        "batik garut merak ngibing",
        "kain batik merak ngibing",
        "motif batik merak ngibing",
        "batik cap merak ngibing",
        "batik tulis merak ngibing",
        "pola batik merak ngibing garutan",
    ]

    candidate_urls: list[tuple[str, str]] = []
    seen_urls: set[str] = set()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    print("Fetching image URLs via DuckDuckGo...")
    with DDGS() as ddgs:
        for q in queries:
            print(f"  Searching query: '{q}'...")
            try:
                results = list(ddgs.images(q, max_results=35))
                for r in results:
                    url = r.get("image")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        candidate_urls.append((url, r.get("title", "")))
            except (OSError, RuntimeError, ValueError) as exc:
                print(f"  Error on query '{q}': {exc}")

    print(f"\nTotal candidate URLs found: {len(candidate_urls)}")

    accepted = 0
    duplicates = 0
    failed_downloads = 0
    next_idx = len(existing_files) + 1

    for url, title in candidate_urls:
        try:
            resp = requests.get(url, headers=headers, timeout=8)
            if resp.status_code != 200 or not resp.content:
                failed_downloads += 1
                continue

            arr = np.frombuffer(resp.content, np.uint8)
            cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if cv_img is None:
                failed_downloads += 1
                continue

            h, w = cv_img.shape[:2]
            if h < 120 or w < 120:
                continue

            img_hash = compute_dhash(cv_img)
            if img_hash is None:
                continue

            if any(hamming_distance(img_hash, eh) <= 3 for eh in existing_hashes):
                duplicates += 1
                continue

            out_filename = f"merak_v3_{next_idx:04d}.jpg"
            out_path = target_dir / out_filename
            cv2.imwrite(str(out_path), cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            existing_hashes.add(img_hash)
            next_idx += 1
            accepted += 1
            print(f"  [+] Saved {out_filename} ({w}x{h}): {title[:40]}")

            if accepted >= max_new:
                print(f"  Target count reached ({max_new} new images). Stopping.")
                break

        except (requests.RequestException, OSError, cv2.error, ValueError):
            failed_downloads += 1

    final_count = len(list(target_dir.glob("*.jpg")) + list(target_dir.glob("*.jpeg")) + list(target_dir.glob("*.png")))
    print("\n" + "=" * 50)
    print("DOWNLOAD AND INGESTION SUMMARY")
    print("=" * 50)
    print(f"  Accepted new authentic images: {accepted}")
    print(f"  Duplicates rejected          : {duplicates}")
    print(f"  Failed downloads/low-res     : {failed_downloads}")
    print(f"  Total images in {target_dir.name}: {final_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download deduplicated Priangan Merak Ngibing images.")
    parser.add_argument("--target-dir", type=Path, required=True, help="Output folder for downloaded images")
    parser.add_argument("--max-new", type=int, default=60, help="Maximum number of new images to save")
    args = parser.parse_args()
    download_merak_images(args.target_dir, max_new=args.max_new)


if __name__ == "__main__":
    main()
