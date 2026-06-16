#!/usr/bin/env python3
"""Create or version Kaggle datasets for the pig-counting project."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


DATASETS = {
    "pig-hf-full-archive": "Full Hugging Face archive for jonaden/pig-detection-and-tracking",
    "yolo-pigdetect": "YOLO-format PigDetect dataset for Ultralytics training",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload PigDetect datasets to Kaggle.")
    parser.add_argument("--username", required=True, help="Kaggle username or organization slug.")
    parser.add_argument("--hf-full-path", required=True, help="Path to the full Hugging Face archive folder.")
    parser.add_argument("--yolo-path", required=True, help="Path to yolo_pigdetect folder or yolo_pigdetect.zip.")
    parser.add_argument("--version-message", default="Update dataset files", help="Message for Kaggle version updates.")
    return parser.parse_args()


def write_metadata(dataset_dir: Path, username: str, slug: str, title: str) -> None:
    metadata = {
        "title": title,
        "id": f"{username}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def prepare_dataset_dir(source: Path, username: str, slug: str, title: str) -> tempfile.TemporaryDirectory[str]:
    temp_dir = tempfile.TemporaryDirectory()
    dataset_dir = Path(temp_dir.name)
    write_metadata(dataset_dir, username, slug, title)

    if source.is_dir():
        for child in source.iterdir():
            target = dataset_dir / child.name
            if child.is_dir():
                import shutil

                shutil.copytree(child, target)
            else:
                import shutil

                shutil.copy2(child, target)
    elif source.is_file():
        import shutil

        shutil.copy2(source, dataset_dir / source.name)
    else:
        temp_dir.cleanup()
        raise FileNotFoundError(f"Source path does not exist: {source}")

    return temp_dir


def dataset_exists(api: KaggleApi, username: str, slug: str) -> bool:
    try:
        api.dataset_view(f"{username}/{slug}")
        return True
    except Exception:
        return False


def create_or_version(api: KaggleApi, username: str, slug: str, source: Path, title: str, version_message: str) -> None:
    print(f"[prepare] {slug}: {source}")
    temp_dir = prepare_dataset_dir(source, username, slug, title)
    dataset_dir = Path(temp_dir.name)
    try:
        if dataset_exists(api, username, slug):
            print(f"[version] Updating Kaggle dataset {username}/{slug}")
            api.dataset_create_version(str(dataset_dir), version_message, delete_old_versions=False)
        else:
            print(f"[create] Creating Kaggle dataset {username}/{slug}")
            api.dataset_create_new(str(dataset_dir), public=False)
        print(f"[done] {username}/{slug}")
    finally:
        temp_dir.cleanup()


def main() -> None:
    args = parse_args()
    api = KaggleApi()
    api.authenticate()

    sources = {
        "pig-hf-full-archive": Path(args.hf_full_path).expanduser().resolve(),
        "yolo-pigdetect": Path(args.yolo_path).expanduser().resolve(),
    }

    for slug, title in DATASETS.items():
        create_or_version(api, args.username, slug, sources[slug], title, args.version_message)

    print("[complete] Kaggle dataset upload finished.")


if __name__ == "__main__":
    main()
