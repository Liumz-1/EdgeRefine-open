from __future__ import annotations

import argparse
import os
from pathlib import Path
import stat
from zipfile import ZipFile


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = BASE_DIR / "EdgeRefine-datasets.zip"
DEFAULT_OUTPUT = BASE_DIR / "dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract the canonical EdgeRefine dataset archive.")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def validate_members(archive: ZipFile, output_dir: Path) -> None:
    output_root = output_dir.resolve()
    for member in archive.infolist():
        target = (output_root / member.filename).resolve()
        if output_root != target and output_root not in target.parents:
            raise ValueError(f"Archive member escapes the dataset directory: {member.filename}")


def ensure_user_access(path: Path) -> None:
    """Apply the user portion of `chmod -R u+rwX` on POSIX systems."""
    if os.name == "nt":
        return
    paths = [path, *path.rglob("*")]
    for item in paths:
        mode = item.stat().st_mode | stat.S_IRUSR | stat.S_IWUSR
        if item.is_dir() or mode & (stat.S_IXGRP | stat.S_IXOTH):
            mode |= stat.S_IXUSR
        item.chmod(mode)


def main() -> None:
    args = parse_args()
    archive_path = args.archive.resolve()
    output_dir = args.output.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {archive_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path) as archive:
        validate_members(archive, output_dir)
        archive.extractall(output_dir)
    ensure_user_access(output_dir)
    print(f"Datasets extracted to {output_dir}")


if __name__ == "__main__":
    main()
