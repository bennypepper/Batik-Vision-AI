"""Batch rename image files in a target directory with a sequential prefix."""

from __future__ import annotations

import argparse
from pathlib import Path


def rename_files(target_dir: Path, prefix: str = "merak_ngibing", dry_run: bool = True) -> int:
    """Sequentially rename files in ``target_dir`` using ``prefix``."""
    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory not found: {target_dir}")

    print(f"Renaming files in: {target_dir} (dry_run={dry_run})")
    count = 1
    renamed = 0
    for file_path in sorted(target_dir.iterdir()):
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        new_path = target_dir / f"{prefix}_{count}{ext}"
        while new_path.exists():
            count += 1
            new_path = target_dir / f"{prefix}_{count}{ext}"
        if not dry_run:
            file_path.rename(new_path)
        print(f"{'[DRY-RUN] Would rename' if dry_run else 'Renamed'}: {file_path.name} -> {new_path.name}")
        count += 1
        renamed += 1

    print(f"\nProcessed {renamed} files.")
    return renamed


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch rename files in a directory.")
    parser.add_argument("--target-dir", type=Path, required=True, help="Directory of files to rename")
    parser.add_argument("--prefix", type=str, default="merak_ngibing", help="Filename prefix")
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preview renames without modifying disk (default: True; pass --no-dry-run to rename).",
    )
    args = parser.parse_args()
    rename_files(args.target_dir, prefix=args.prefix, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
