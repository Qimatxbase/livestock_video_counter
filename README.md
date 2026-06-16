# Pig Count

MLOps-style project for preparing PigDetect data, training YOLOv8 on Kaggle, and running a Streamlit app that counts pigs in uploaded videos with tracking and a configurable counting line.

## Project Structure

```text
pig_count/
  app.py
  requirements.txt
  requirements-kaggle.txt
  .streamlit/config.toml
  kaggle/train_yolo_kaggle.py
  kaggle/kernel-metadata.json
  models/pigdetect_best.pt
  outputs/
  scripts/prepara_data/prepare_data.py
  scripts/upload_datakaggle/upload_datakaggle.py
  scripts/setup_git_repo.sh
  scripts/update_best_from_git.sh
  scripts/run_streamlit_local.sh
  training_artifacts/
```

Raw datasets, videos, zip archives, credentials, and secrets are ignored by Git.

## 1. Prepare Data

Use the local or Colab working directory:

```bash
python scripts/prepara_data/prepare_data.py \
  --work-dir /content/drive/MyDrive/Deep_learning/YOLO \
  --download-full
```

This downloads `jonaden/pig-detection-and-tracking`, extracts `PigDetect.zip`, converts the COCO detection subset to YOLO format, writes `yolo_pigdetect/pig.yaml`, and creates `yolo_pigdetect.zip`.

Use `--force` to rebuild existing outputs:

```bash
python scripts/prepara_data/prepare_data.py \
  --work-dir /content/drive/MyDrive/Deep_learning/YOLO \
  --download-full \
  --force
```

## 2. Upload Kaggle Datasets

Install Kaggle dependencies and make sure `~/.kaggle/kaggle.json` already exists.

```bash
python -m pip install -r requirements-kaggle.txt
python scripts/upload_datakaggle/upload_datakaggle.py \
  --username YOUR_KAGGLE_USERNAME \
  --hf-full-path /content/drive/MyDrive/Deep_learning/YOLO/hf_full_archive \
  --yolo-path /content/drive/MyDrive/Deep_learning/YOLO/yolo_pigdetect
```

The script creates or versions:

- `pig-hf-full-archive`
- `yolo-pigdetect`

No Kaggle credentials are hard-coded.

## 3. Train on Kaggle

Upload or attach the `yolo-pigdetect` Kaggle Dataset to a GPU notebook/script. The expected input path is:

```text
/kaggle/input/yolo-pigdetect/yolo_pigdetect
```

Run:

```bash
python kaggle/train_yolo_kaggle.py
```

The training script uses:

- `yolov8n.pt`
- `epochs=50`
- `imgsz=640`
- `batch=8`
- `workers=0`
- `patience=10`

Kaggle outputs include:

- `/kaggle/working/pigdetect_best.pt`
- `/kaggle/working/pigdetect_last.pt`
- `/kaggle/working/training_summary.json`
- `/kaggle/working/training_artifacts/results.csv`
- copied training plots and result files

## 4. Push Artifacts from Kaggle to GitHub

Set Kaggle secrets or environment variables:

```bash
GITHUB_TOKEN
GITHUB_REPO
GITHUB_USER_NAME
GITHUB_USER_EMAIL
```

`GITHUB_REPO` should look like:

```text
owner/repository
```

When these values are available, the Kaggle script pushes:

- `models/pigdetect_best.pt`
- `training_artifacts/training_summary.json`
- `training_artifacts/results.csv`

If the variables are missing, training still completes and the push is skipped.

## 5. Pull the Best Model Locally

After Kaggle pushes the trained model:

```bash
bash scripts/update_best_from_git.sh
```

This pulls the latest GitHub state and checks that `models/pigdetect_best.pt` exists.

## 6. Run the Streamlit App Locally

```bash
bash scripts/run_streamlit_local.sh
```

Or run manually:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The app loads:

```text
models/pigdetect_best.pt
```

It accepts uploaded `.mp4`, `.mov`, `.avi`, and `.mkv` videos. For each uploaded video it runs YOLO tracking frame by frame, draws bounding boxes, track IDs, center points, the counting line, and the total count, then saves an annotated video under `outputs/` with a download button.

## 7. Deploy to Streamlit Community Cloud

Push this repository to GitHub with:

```bash
bash scripts/setup_git_repo.sh https://github.com/Qimatxbase/pigs_count
```

Then deploy the GitHub repo in Streamlit Community Cloud with `app.py` as the entrypoint. Make sure `models/pigdetect_best.pt` is present in the repo. If the model is larger than 100MB, use Git LFS because GitHub blocks normal files above that limit.

## Notes

- Do not commit raw datasets, raw videos, `.env`, `kaggle.json`, or secrets.
- `opencv-python-headless` is used for the Streamlit app.
- The counting line can be vertical or horizontal, and each track ID is counted only once when it crosses the line.
