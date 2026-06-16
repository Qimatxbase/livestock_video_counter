from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "pigdetect_best.pt"
OUTPUT_DIR = ROOT / "outputs"
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv"]


@st.cache_resource
def load_model() -> YOLO:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Train on Kaggle first or pull the model from GitHub."
        )
    return YOLO(str(MODEL_PATH))


def crossed_line(
    previous: tuple[float, float],
    current: tuple[float, float],
    axis: str,
    line_position: int,
    direction: str,
) -> bool:
    prev_value = previous[0] if axis == "x" else previous[1]
    curr_value = current[0] if axis == "x" else current[1]

    forward = prev_value < line_position <= curr_value
    backward = prev_value > line_position >= curr_value

    if direction == "both":
        return forward or backward
    if axis == "x" and direction == "left_to_right":
        return forward
    if axis == "x" and direction == "right_to_left":
        return backward
    if axis == "y" and direction == "top_to_bottom":
        return forward
    if axis == "y" and direction == "bottom_to_top":
        return backward
    return False


def draw_line(frame: np.ndarray, axis: str, line_position: int) -> None:
    height, width = frame.shape[:2]
    if axis == "x":
        cv2.line(frame, (line_position, 0), (line_position, height), (0, 255, 255), 2)
    else:
        cv2.line(frame, (0, line_position), (width, line_position), (0, 255, 255), 2)


def draw_total(frame: np.ndarray, total: int) -> None:
    cv2.rectangle(frame, (12, 12), (260, 62), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Pig count: {total}",
        (24, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def process_video(
    input_path: Path,
    model: YOLO,
    confidence: float,
    iou: float,
    tracker: str,
    axis: str,
    line_ratio: float,
    direction: str,
) -> tuple[Path, int]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"annotated_{int(time.time())}_{input_path.stem}.mp4"

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open uploaded video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    line_position = int((width if axis == "x" else height) * line_ratio)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    track_centers: dict[int, tuple[float, float]] = {}
    counted_ids: set[int] = set()
    progress = st.progress(0)
    status = st.empty()

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.track(
                frame,
                persist=True,
                conf=confidence,
                iou=iou,
                tracker=tracker,
                verbose=False,
            )
            annotated = frame.copy()
            draw_line(annotated, axis, line_position)

            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
                ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else []

                for box, track_id in zip(xyxy, ids):
                    x1, y1, x2, y2 = [int(v) for v in box[:4]]
                    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    previous = track_centers.get(track_id)
                    if previous and track_id not in counted_ids:
                        if crossed_line(previous, center, axis, line_position, direction):
                            counted_ids.add(track_id)
                    track_centers[track_id] = center

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (36, 180, 80), 2)
                    cv2.circle(annotated, (int(center[0]), int(center[1])), 4, (255, 80, 80), -1)
                    cv2.putText(
                        annotated,
                        f"ID {track_id}",
                        (x1, max(24, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (36, 255, 120),
                        2,
                        cv2.LINE_AA,
                    )

            draw_total(annotated, len(counted_ids))
            writer.write(annotated)

            frame_index += 1
            progress.progress(min(frame_index / frame_count, 1.0))
            status.caption(f"Processing frame {frame_index} of about {frame_count}")
    finally:
        cap.release()
        writer.release()
        progress.empty()
        status.empty()

    return output_path, len(counted_ids)


def main() -> None:
    st.set_page_config(page_title="Pig Video Counter", layout="wide")
    st.title("Pig Video Counter")

    with st.sidebar:
        st.header("Tracking")
        confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
        iou = st.slider("IOU threshold", 0.10, 0.95, 0.50, 0.05)
        tracker = st.selectbox("Tracker", ["bytetrack.yaml", "botsort.yaml"])
        axis = st.radio("Line axis", ["x", "y"], horizontal=True)
        line_ratio = st.slider("Line position", 0.05, 0.95, 0.50, 0.01)
        direction = st.selectbox(
            "Direction",
            ["both", "left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"],
        )

    uploaded = st.file_uploader("Upload a pig video", type=VIDEO_TYPES)
    if uploaded is None:
        st.info("Upload a video to start counting.")
        return

    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded.getbuffer())
        input_path = Path(temp_file.name)

    st.subheader("Original video")
    st.video(str(input_path))

    try:
        model = load_model()
        with st.spinner("Running YOLO tracking..."):
            output_path, total = process_video(
                input_path,
                model,
                confidence,
                iou,
                tracker,
                axis,
                line_ratio,
                direction,
            )
    except Exception as exc:
        st.error(str(exc))
        return

    st.metric("Final pig count", total)
    st.subheader("Annotated video")
    st.video(str(output_path))
    st.download_button(
        "Download annotated video",
        data=output_path.read_bytes(),
        file_name=output_path.name,
        mime="video/mp4",
    )


if __name__ == "__main__":
    main()
