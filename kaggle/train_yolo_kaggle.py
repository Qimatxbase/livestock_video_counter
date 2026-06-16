#!/usr/bin/env python3
"""Train YOLOv8 for PigDetect on Kaggle and optionally push artifacts to GitHub."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO


KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
DATASET_HINT = Path("/kaggle/input/yolo-pigdetect/yolo_pigdetect")


def run(command: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("[cmd]", " ".join(command))
    return subprocess.run(command, cwd=cwd, text=True, check=check, capture_output=False)


def find_yolo_dataset() -> Path:
    candidates = [DATASET_HINT]
    candidates.extend(KAGGLE_INPUT.glob("**/yolo_pigdetect"))
    candidates.extend(path.parent for path in KAGGLE_INPUT.glob("**/pig.yaml"))

    for candidate in candidates:
        if (candidate / "images" / "train").is_dir() and (candidate / "labels" / "train").is_dir():
            print(f"[data] Found YOLO dataset: {candidate}")
            return candidate
    raise FileNotFoundError("Could not find YOLO dataset under /kaggle/input.")


def write_kaggle_yaml(dataset_dir: Path) -> Path:
    yaml_path = KAGGLE_WORKING / "pig.yaml"
    yaml_text = "\n".join(
        [
            f"path: {dataset_dir}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: pig",
            "",
        ]
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"[data] Wrote Kaggle training YAML: {yaml_path}")
    return yaml_path


def copy_if_exists(src: Path, dst: Path) -> bool:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[save] {dst}")
        return True
    print(f"[warn] Missing expected artifact: {src}")
    return False


def summarize(result, run_dir: Path, best_path: Path, last_path: Path) -> Path:
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": "yolov8n.pt",
        "epochs": 50,
        "imgsz": 640,
        "batch": 8,
        "run_dir": str(run_dir),
        "best_model": str(best_path),
        "last_model": str(last_path),
        "metrics": getattr(result, "results_dict", None),
    }
    summary_path = KAGGLE_WORKING / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[save] {summary_path}")
    return summary_path


def push_to_github(best_path: Path, summary_path: Path, results_csv: Path) -> None:
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPO")
    user_name = os.getenv("GITHUB_USER_NAME", "kaggle-bot")
    user_email = os.getenv("GITHUB_USER_EMAIL", "kaggle-bot@example.com")

    if not token or not repo:
        print("[skip] GITHUB_TOKEN or GITHUB_REPO missing. Skipping GitHub push.")
        return

    repo_url = f"https://x-access-token:{token}@github.com/{repo}.git"
    clone_dir = KAGGLE_WORKING / "github_repo"
    if clone_dir.exists():
        shutil.rmtree(clone_dir)

    try:
        run(["git", "clone", repo_url, str(clone_dir)])
        run(["git", "config", "user.name", user_name], cwd=clone_dir)
        run(["git", "config", "user.email", user_email], cwd=clone_dir)

        copy_if_exists(best_path, clone_dir / "models" / "pigdetect_best.pt")
        copy_if_exists(summary_path, clone_dir / "training_artifacts" / "training_summary.json")
        copy_if_exists(results_csv, clone_dir / "training_artifacts" / "results.csv")

        run(["git", "add", "models/pigdetect_best.pt", "training_artifacts/training_summary.json", "training_artifacts/results.csv"], cwd=clone_dir)
        diff = run(["git", "diff", "--cached", "--quiet"], cwd=clone_dir, check=False)
        if diff.returncode == 0:
            print("[skip] No GitHub artifact changes to commit.")
            return
        run(["git", "commit", "-m", "Update PigDetect YOLO training artifacts"], cwd=clone_dir)
        run(["git", "push", "origin", "main"], cwd=clone_dir)
        print("[done] Training artifacts pushed to GitHub.")
    except Exception as exc:
        print(f"[warn] GitHub push failed but training completed: {exc}")


def main() -> None:
    KAGGLE_WORKING.mkdir(parents=True, exist_ok=True)
    dataset_dir = find_yolo_dataset()
    yaml_path = write_kaggle_yaml(dataset_dir)

    model = YOLO("yolov8n.pt")
    train_results = model.train(
        data=str(yaml_path),
        epochs=50,
        imgsz=640,
        batch=8,
        workers=0,
        patience=10,
        project=str(KAGGLE_WORKING / "runs" / "detect"),
        name="pigdetect_yolo",
        exist_ok=True,
    )

    run_dir = Path(model.trainer.save_dir)
    weights_dir = run_dir / "weights"
    best_path = KAGGLE_WORKING / "pigdetect_best.pt"
    last_path = KAGGLE_WORKING / "pigdetect_last.pt"
    artifacts_dir = KAGGLE_WORKING / "training_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    copy_if_exists(weights_dir / "best.pt", best_path)
    copy_if_exists(weights_dir / "last.pt", last_path)
    results_csv = artifacts_dir / "results.csv"
    copy_if_exists(run_dir / "results.csv", results_csv)
    summary_path = summarize(train_results, run_dir, best_path, last_path)

    for artifact in run_dir.glob("*"):
        if artifact.is_file() and artifact.suffix.lower() in {".png", ".jpg", ".jpeg", ".csv", ".txt"}:
            copy_if_exists(artifact, artifacts_dir / artifact.name)

    push_to_github(best_path, summary_path, results_csv)

    print("\nTraining complete.")
    print(f"Best model: {best_path}")
    print(f"Last model: {last_path}")
    print(f"Artifacts: {artifacts_dir}")
    print("If GitHub push was skipped, download these Kaggle outputs and place best.pt at models/pigdetect_best.pt.")


if __name__ == "__main__":
    main()
