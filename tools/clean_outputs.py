#!/usr/bin/env python3
"""Safe cleanup of outputs/ directory.

Usage:
    python tools/clean_outputs.py              # dry-run
    python tools/clean_outputs.py --apply      # actually delete

Preserves outputs/profiling/*.md documentation files if present.
"""

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.resolve()
OUTPUTS_DIR = REPO_ROOT / "outputs"


def clean_outputs(dry_run=True):
    """Remove outputs/ directory (preserving documentation)."""
    if not OUTPUTS_DIR.exists():
        print("✓ outputs/ directory does not exist")
        return

    # Find all items in outputs/
    all_items = list(OUTPUTS_DIR.rglob("*"))

    # Preserve certain documentation files
    preserve = set()
    preserve_patterns = ["outputs/profiling/*.md", "outputs/profiling/README.md"]
    for pattern in preserve_patterns:
        for path in REPO_ROOT.glob(pattern):
            preserve.add(path)
            # Also preserve parent dirs needed for these files
            for parent in path.parents:
                if parent.is_relative_to(OUTPUTS_DIR):
                    preserve.add(parent)

    to_delete = [p for p in all_items if p not in preserve and p.is_file()]
    dirs_to_delete = [p for p in all_items if p not in preserve and p.is_dir()]

    # Sort dirs deepest first
    dirs_to_delete.sort(key=lambda p: len(p.parts), reverse=True)

    total = len(to_delete) + len(dirs_to_delete)

    if total == 0:
        print("✓ outputs/ is already clean")
        return

    print(f"{'DRY RUN - ' if dry_run else ''}Found {total} items in outputs/:\n")
    print(f"  Files: {len(to_delete)}")
    print(f"  Directories: {len(dirs_to_delete)}")

    if len(preserve) > 0:
        print(f"\nPreserving {len(preserve)} documentation files:")
        for p in sorted(preserve):
            if p.is_file():
                print(f"  - {p.relative_to(REPO_ROOT)}")

    if dry_run:
        print("\nRun with --apply to delete")
    else:
        for path in to_delete:
            path.unlink()
        for path in dirs_to_delete:
            if path.exists() and not any(path.iterdir()):  # only if empty
                path.rmdir()
        print(f"\n✓ Deleted {total} items from outputs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean outputs directory")
    parser.add_argument("--apply", action="store_true", help="Actually delete (default is dry-run)")
    args = parser.parse_args()

    clean_outputs(dry_run=not args.apply)
