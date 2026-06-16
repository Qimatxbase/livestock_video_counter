# Pig Count

MLOps-style project for training YOLOv8 on the PigDetect dataset on Kaggle and running a Streamlit app that counts pigs in uploaded videos with automatic tracking and a configurable counting line.

## Project Structure

```
pig_count/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── models/
│   ├── .gitkeep
│   └── pigdetect_best.pt
├── outputs/
│   └── .gitkeep
├── notebooks/
│   └── pig_train_kaggle.ipynb
└── utils/
    └── video_counting.py
```

## Workflow

### 1. Train on Kaggle

1. Open [Kaggle](https://kaggle.com) and create a new notebook
2. Upload or import `notebooks/pig_train_kaggle.ipynb`
3. **Enable Internet** in notebook settings
4. **Enable GPU** (recommended)
5. Add Kaggle Secret:
   - Name: `HF_TOKEN`
   - Value: Your Hugging Face token (get it from [hf.co/settings/tokens](https://huggingface.co/settings/tokens))
6. Run all cells
7. Download these files from the notebook output:
   - `pigdetect_best.pt` (required)
   - `pigdetect_last.pt` (optional)
   - `training_summary.json` (optional)
   - `pig_training_outputs.zip` (optional)

### 2. Add Model to Project

Place the downloaded `pigdetect_best.pt` into:

```bash
models/pigdetect_best.pt
```

### 3. Run Streamlit Locally

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

Optional: create a `.env` file to store environment secrets (recommended).
Example `.env`:

```
HF_TOKEN=your_huggingface_token
GITHUB_TOKEN=your_github_token  # optional, for Kaggle -> GitHub pushes
```

**Using the app:**
1. Adjust detection, tracking, and counting settings in the sidebar
2. Upload a pig video (`.mp4`, `.mov`, `.avi`, `.mkv`)
3. Wait for processing
4. View the annotated video with pig counts
5. Download the output video

### 4. Deploy to Streamlit Community Cloud

1. Push this repository to GitHub:

```bash
git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
git branch -M main
git push -u origin main
```

2. Make sure `models/pigdetect_best.pt` is committed to the repository

3. Go to [Streamlit Community Cloud](https://share.streamlit.io)

4. Click "New app" and deploy from your GitHub repo

5. Set entrypoint to `app.py`

## App Features

- **YOLO Detection**: Real-time pig detection using YOLOv8n
- **Tracking**: ByteTrack or BoT-SORT for persistent tracking across frames
- **Automatic Counting**: Count pigs only once when they cross a configurable line
- **Adjustable Settings**:
  - Confidence and IOU thresholds
  - Tracking algorithm selection
  - Counting line position and direction (x/y axis, horizontal/vertical)
  - Bounding box filtering (width, height, area)
  - Display options (boxes, IDs, centers, confidence)
  - Performance options (frame stride, resize)
- **Video Output**: Annotated videos with bounding boxes, track IDs, center points, counting line, and total count
- **Download**: Download the annotated video directly from the app

## Training Details

The Kaggle notebook:

1. Installs required dependencies
2. Loads Hugging Face token from Kaggle secrets
3. Downloads the full `jonaden/pig-detection-and-tracking` dataset
4. Extracts and converts PigDetect COCO annotations to YOLO format
5. Trains YOLOv8n on the dataset with:
   - 50 epochs
   - 640x640 image size
   - Batch size 8
   - Early stopping (patience=10)
6. Validates the best model
7. Saves training outputs for download

## Notes

- Do not commit raw datasets, videos, `.env`, `kaggle.json`, or credentials to Git
- If your model file is larger than 100MB, consider using Git LFS or excluding it from the repo
- The counting line can be vertical (x-axis) or horizontal (y-axis)
- Each tracked object is counted only once when it crosses the line
- Use the settings sidebar to fine-tune detection and counting parameters for your use case

