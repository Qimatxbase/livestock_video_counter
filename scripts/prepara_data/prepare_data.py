#!/usr/bin/env python3
"""Download PigDetect data from Hugging Face and convert COCO labels to YOLO."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from tqdm import tqdm


HF_REPO_ID = "jonaden/pig-detection-and-tracking"
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the PigDetect YOLO dataset.")
    parser.add_argument(
        "--work-dir",
        default="/content/drive/MyDrive/Deep_learning/YOLO",
        help="Directory where archives and converted data will be stored.",
    )
    parser.add_argument(
        "--download-full",
        action="store_true",
        help="Download the full Hugging Face dataset archive snapshot.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-convert even when outputs already exist.",
    )
    return parser.parse_args()


def find_file(root: Path, name: str) -> Path | None:
    matches = list(root.rglob(name))
    return matches[0] if matches else None


def download_hf_dataset(work_dir: Path, force: bool) -> Path:
    archive_dir = work_dir / "hf_full_archive"
    if archive_dir.exists() and any(archive_dir.iterdir()) and not force:
        print(f"[skip] Hugging Face archive already exists: {archive_dir}")
        return archive_dir

    if archive_dir.exists() and force:
        shutil.rmtree(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download] Downloading {HF_REPO_ID} to {archive_dir}")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(archive_dir),
        local_dir_use_symlinks=False,
    )
    print("[done] Full Hugging Face dataset downloaded.")
    return archive_dir


def extract_pigdetect_zip(archive_dir: Path, work_dir: Path, force: bool) -> Path:
    extract_dir = work_dir / "PigDetect"
    if extract_dir.exists() and not force:
        print(f"[skip] PigDetect already extracted: {extract_dir}")
        return extract_dir

    zip_path = find_file(archive_dir, "PigDetect.zip")
    if zip_path is None:
        raise FileNotFoundError(
            f"Could not find PigDetect.zip under {archive_dir}. "
            "Run with --download-full or verify the Hugging Face snapshot."
        )

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract] Extracting {zip_path} to {extract_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    return extract_dir


def locate_detection_root(extract_dir: Path) -> Path:
    candidates = [extract_dir / "detection", *extract_dir.rglob("detection")]
    for candidate in candidates:
        if (candidate / "coco_annotations").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not locate detection/coco_annotations under {extract_dir}")


def coco_bbox_to_yolo(bbox: list[float], image_width: float, image_height: float) -> tuple[float, float, float, float]:
    x_min, y_min, width, height = bbox
    x_center = (x_min + width / 2.0) / image_width
    y_center = (y_min + height / 2.0) / image_height
    return (
        max(0.0, min(1.0, x_center)),
        max(0.0, min(1.0, y_center)),
        max(0.0, min(1.0, width / image_width)),
        max(0.0, min(1.0, height / image_height)),
    )


def image_source_path(detection_root: Path, split: str, file_name: str) -> Path:
    candidates = [
        detection_root / "images" / split / file_name,
        detection_root / split / file_name,
        detection_root / file_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list(detection_root.rglob(file_name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find image file for split={split}: {file_name}")


def convert_split(detection_root: Path, yolo_dir: Path, split: str) -> None:
    annotation_path = detection_root / "coco_annotations" / f"{split}.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing COCO annotation file: {annotation_path}")

    with annotation_path.open("r", encoding="utf-8") as f:
        coco: dict[str, Any] = json.load(f)

    images = {image["id"]: image for image in coco.get("images", [])}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in images}
    for annotation in coco.get("annotations", []):
        image_id = annotation.get("image_id")
        if image_id in annotations_by_image:
            annotations_by_image[image_id].append(annotation)

    image_out_dir = yolo_dir / "images" / split
    label_out_dir = yolo_dir / "labels" / split
    image_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    for image_id, image in tqdm(images.items(), desc=f"convert {split}"):
        file_name = image["file_name"]
        src = image_source_path(detection_root, split, file_name)
        dst = image_out_dir / Path(file_name).name
        shutil.copy2(src, dst)

        width = float(image["width"])
        height = float(image["height"])
        label_lines = []
        for annotation in annotations_by_image.get(image_id, []):
            bbox = annotation.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x_center, y_center, box_width, box_height = coco_bbox_to_yolo(bbox, width, height)
            label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

        label_path = label_out_dir / f"{Path(file_name).stem}.txt"
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")


def write_yaml(yolo_dir: Path) -> None:
    yaml_text = "\n".join(
        [
            f"path: {yolo_dir.resolve()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: pig",
            "",
        ]
    )
    (yolo_dir / "pig.yaml").write_text(yaml_text, encoding="utf-8")


def zip_yolo_dataset(yolo_dir: Path, zip_path: Path, force: bool) -> None:
    if zip_path.exists() and not force:
        print(f"[skip] YOLO zip already exists: {zip_path}")
        return
    if zip_path.exists():
        zip_path.unlink()

    print(f"[zip] Creating {zip_path}")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in yolo_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(yolo_dir.parent))
    print("[done] YOLO dataset zip created.")


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    archive_dir = work_dir / "hf_full_archive"
    if args.download_full or not archive_dir.exists():
        archive_dir = download_hf_dataset(work_dir, args.force)
    else:
        print(f"[skip] Using existing Hugging Face archive: {archive_dir}")

    yolo_dir = work_dir / "yolo_pigdetect"
    if yolo_dir.exists() and (yolo_dir / "pig.yaml").exists() and not args.force:
        print(f"[skip] YOLO dataset already prepared: {yolo_dir}")
    else:
        if yolo_dir.exists():
            shutil.rmtree(yolo_dir)
        extract_dir = extract_pigdetect_zip(archive_dir, work_dir, args.force)
        detection_root = locate_detection_root(extract_dir)
        print(f"[convert] Using detection root: {detection_root}")
        for split in SPLITS:
            convert_split(detection_root, yolo_dir, split)
        write_yaml(yolo_dir)
        print(f"[done] YOLO dataset prepared: {yolo_dir}")

    zip_yolo_dataset(yolo_dir, work_dir / "yolo_pigdetect.zip", args.force)
    print("[complete] Data preparation finished.")


if __name__ == "__main__":
    main()
