"""Filter downloaded image folder by extension, minimum size, and top-N quality.

Safe by default: runs in ``--dry-run`` mode unless ``--execute`` is explicitly passed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def filter_directory(
    target_dir: Path,
    min_size_kb: float = 20.0,
    keep_count: int = 30,
    dry_run: bool = True,
) -> dict[str, int]:
    """Filter non-images, tiny thumbnails, and excess files from ``target_dir``."""
    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory not found: {target_dir}")

    print(f"Scanning directory: {target_dir} (dry_run={dry_run})")
    files_info: list[tuple[Path, str, float]] = []
    removed_non_image = 0
    removed_tiny = 0

    for file_path in sorted(target_dir.iterdir()):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        if ext not in VALID_EXTS:
            if not dry_run:
                file_path.unlink()
            removed_non_image += 1
            print(f"{'[DRY-RUN] Would delete' if dry_run else 'Deleted'} non-image: {file_path.name}")
            continue

        size_kb = file_path.stat().st_size / 1024.0
        if size_kb < min_size_kb:
            if not dry_run:
                file_path.unlink()
            removed_tiny += 1
            print(
                f"{'[DRY-RUN] Would delete' if dry_run else 'Deleted'} tiny thumbnail: "
                f"{file_path.name} ({size_kb:.1f} KB)"
            )
            continue

        files_info.append((file_path, file_path.name, size_kb))

    files_info.sort(key=lambda x: x[2], reverse=True)
    kept_files = files_info[:keep_count]
    excess_files = files_info[keep_count:]

    for file_path, _, _ in excess_files:
        if not dry_run:
            file_path.unlink()

    print("\n--- Summary ---")
    print(f"Kept {len(kept_files)} largest images.")
    print(f"{'Would delete' if dry_run else 'Deleted'} {len(excess_files)} excess images.")
    return {
        "kept": len(kept_files),
        "removed_non_image": removed_non_image,
        "removed_tiny": removed_tiny,
        "removed_excess": len(excess_files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter low-quality or excess downloaded images.")
    parser.add_argument("--target-dir", type=Path, required=True, help="Directory of images to filter")
    parser.add_argument("--min-size-kb", type=float, default=20.0, help="Minimum file size in KB")
    parser.add_argument("--keep-count", type=int, default=30, help="Maximum images to keep")
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preview deletions without modifying disk (default: True; pass --no-dry-run to delete).",
    )
    args = parser.parse_args()
    filter_directory(
        target_dir=args.target_dir,
        min_size_kb=args.min_size_kb,
        keep_count=args.keep_count,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
