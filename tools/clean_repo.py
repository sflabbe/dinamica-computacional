#!/usr/bin/env python3
"""Safe repository cleanup script - removes only known generated artifacts.

Usage:
    python tools/clean_repo.py              # dry-run (shows what would be deleted)
    python tools/clean_repo.py --apply      # actually delete files

NEVER deletes anything under src/ or tests/ (except __pycache__ dirs).
"""

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.resolve()

# Patterns to delete (relative to repo root)
CLEAN_PATTERNS = {
    "pycache": ["**/__pycache__"],
    "test_cache": [".pytest_cache", ".mypy_cache", ".ruff_cache", ".coverage", "htmlcov"],
    "build_artifacts": ["build", "dist", "**/*.egg-info"],
    "profiling": ["prof.out", "scalene*.json", "pstats_*.txt", "*.prof"],
    "os_junk": [".DS_Store", "**/Thumbs.db"],
}


def find_targets():
    """Find all files/directories matching cleanup patterns."""
    targets = {}
    for category, patterns in CLEAN_PATTERNS.items():
        targets[category] = []
        for pattern in patterns:
            for path in REPO_ROOT.glob(pattern):
                # Safety: never delete anything directly under src/ or tests/ (except __pycache__)
                if path.is_relative_to(REPO_ROOT / "src") or path.is_relative_to(REPO_ROOT / "tests"):
                    if "__pycache__" not in str(path):
                        continue
                targets[category].append(path)
    return targets


def clean(dry_run=True):
    """Remove generated artifacts."""
    targets = find_targets()
    total = sum(len(v) for v in targets.values())

    if total == 0:
        print("✓ Repository is clean (no generated artifacts found)")
        return

    print(f"{'DRY RUN - ' if dry_run else ''}Found {total} items to clean:\n")

    for category, paths in targets.items():
        if not paths:
            continue
        print(f"{category} ({len(paths)} items):")
        for path in sorted(paths):
            rel_path = path.relative_to(REPO_ROOT)
            print(f"  - {rel_path}")
        print()

    if dry_run:
        print("Run with --apply to delete these files/directories")
    else:
        for category, paths in targets.items():
            for path in paths:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        print(f"✓ Deleted {total} items")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean repository artifacts")
    parser.add_argument("--apply", action="store_true", help="Actually delete files (default is dry-run)")
    args = parser.parse_args()

    clean(dry_run=not args.apply)
