# Livestock Video Counter

YOLO-based pig detection, tracking and line-crossing counts in uploaded videos.

**Technologies:** Python · YOLOv8 · OpenCV · Streamlit

## Project provenance

A livestock-monitoring prototype preserved as portfolio work. Course and assessment provenance have not been established from the retained materials, and the original source history and contributor credits are preserved.

The saved training summary reports detector validation metrics. Those metrics do not measure video-counting accuracy or establish performance in commercial farm operations.

## What is included

- Configurable line position, count direction and detection thresholds.
- ByteTrack or BoT-SORT tracking with annotated video downloads.
- Kaggle training notebooks and a saved PigDetect model.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The app uses models/pigdetect_best.pt. See docs/original_readme.md for the Kaggle training workflow. A second full-workflow notebook is preserved in notebooks/pig_train_kaggle_full_workflow.ipynb. Uploaded videos and generated outputs remain local.

## Source and credits

This portfolio snapshot preserves work from the following repositories. Existing author credits, citations and license files are retained; team and upstream code are not presented as solely authored work.

- https://github.com/Qimatxbase/pigs_count
