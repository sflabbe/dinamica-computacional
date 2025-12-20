"""File tracking utilities to snapshot and diff output directories."""

from __future__ import annotations

from pathlib import Path
from typing import List, Set


def snapshot_files(directory: Path, recursive: bool = True) -> Set[Path]:
    """Create a snapshot of all files in a directory.

    Parameters
    ----------
    directory : Path
        Directory to snapshot
    recursive : bool, default=True
        Whether to recursively scan subdirectories

    Returns
    -------
    Set[Path]
        Set of relative paths (relative to directory) of all files found
    """
    directory = Path(directory)
    if not directory.exists():
        return set()

    files: Set[Path] = set()
    if recursive:
        for item in directory.rglob("*"):
            if item.is_file():
                try:
                    rel_path = item.relative_to(directory)
                    files.add(rel_path)
                except ValueError:
                    # Skip files that can't be made relative
                    pass
    else:
        for item in directory.iterdir():
            if item.is_file():
                files.add(item.name)

    return files


def compute_new_files(
    directory: Path,
    before_snapshot: Set[Path],
    recursive: bool = True,
) -> List[Path]:
    """Compute files that were created after a snapshot.

    Parameters
    ----------
    directory : Path
        Directory to check
    before_snapshot : Set[Path]
        Snapshot taken before (from snapshot_files)
    recursive : bool, default=True
        Whether to recursively scan subdirectories

    Returns
    -------
    List[Path]
        Sorted list of relative paths of new files
    """
    after_snapshot = snapshot_files(directory, recursive=recursive)
    new_files = after_snapshot - before_snapshot
    return sorted(new_files)
