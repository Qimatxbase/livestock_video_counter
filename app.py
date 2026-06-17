"""
Streamlit app for pig detection, tracking, and counting.
Upload videos, adjust tracking and counting parameters, download annotated output.
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import tempfile
import shutil
import subprocess
from utils.video_counting import process_video

# Page configuration
st.set_page_config(
    page_title="Pig Counter",
    page_icon="🐷",
    layout="wide"
)

# Model path
MODEL_PATH = Path("models/pigdetect_best.pt")


def get_ffmpeg_binary():
    """Return an ffmpeg executable path if one is available."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def make_browser_playable_video(input_path: Path, output_path: Path) -> bool:
    """Transcode a video to a browser-friendly H.264 MP4 for Streamlit playback."""
    ffmpeg_binary = get_ffmpeg_binary()
    if not ffmpeg_binary:
        return False

    temp_path = output_path.with_suffix(".streamlit.mp4")
    command = [
        ffmpeg_binary,
        "-y",
        "-i",
        str(input_path),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(temp_path),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not temp_path.exists() or temp_path.stat().st_size == 0:
        temp_path.unlink(missing_ok=True)
        return False

    temp_path.replace(output_path)
    return True

def check_model_exists():
    """Check if the model file exists."""
    if not MODEL_PATH.exists():
        st.error(
            f"❌ Model not found. Please place `pigdetect_best.pt` inside the `models/` folder.\n\n"
            f"Expected path: `{MODEL_PATH}`\n\n"
            f"Steps:\n"
            f"1. Train the model using `notebooks/pig_train_kaggle.ipynb` on Kaggle\n"
            f"2. Download `pigdetect_best.pt` from Kaggle\n"
            f"3. Place it in: `models/pigdetect_best.pt`"
        )
        return False
    return True

def get_sidebar_settings():
    """Get all settings from sidebar."""
    st.sidebar.header("⚙️ Detection Settings")
    
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="YOLO confidence threshold for detection"
    )
    
    iou_threshold = st.sidebar.slider(
        "IOU Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="IOU threshold for NMS"
    )
    
    st.sidebar.header("🎯 Tracking Settings")
    
    tracker_type = st.sidebar.selectbox(
        "Tracker Type",
        ["bytetrack.yaml", "botsort.yaml"],
        help="Tracking algorithm to use"
    )
    
    st.sidebar.header("📍 Counting Line Settings")
    
    line_axis = st.sidebar.radio(
        "Counting Line Axis",
        ["x (vertical)", "y (horizontal)"],
        help="Direction of the counting line"
    )
    line_axis = "x" if line_axis == "x (vertical)" else "y"
    
    max_coord = 1280 if line_axis == "x" else 720  # Approximate max dimensions
    
    line_pos = st.sidebar.slider(
        "Counting Line Position",
        min_value=0,
        max_value=max_coord,
        value=max_coord // 2,
        step=10,
        help="Position of the counting line"
    )
    
    count_direction = st.sidebar.selectbox(
        "Counting Direction",
        [
            "both",
            "left_to_right" if line_axis == "x" else "top_to_bottom",
            "right_to_left" if line_axis == "x" else "bottom_to_top"
        ],
        help="Direction to count object crossings"
    )
    
    st.sidebar.header("📦 Bounding Box Filters")
    
    min_width = st.sidebar.slider(
        "Minimum Box Width",
        min_value=1,
        max_value=100,
        value=10,
        step=1
    )
    
    min_height = st.sidebar.slider(
        "Minimum Box Height",
        min_value=1,
        max_value=100,
        value=10,
        step=1
    )
    
    min_area = st.sidebar.slider(
        "Minimum Box Area",
        min_value=1,
        max_value=10000,
        value=100,
        step=50
    )
    
    max_area = st.sidebar.slider(
        "Maximum Box Area",
        min_value=1000,
        max_value=1000000,
        value=500000,
        step=10000
    )
    
    st.sidebar.header("🎨 Display Options")
    
    show_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)
    show_ids = st.sidebar.checkbox("Show Track IDs", value=True)
    show_centers = st.sidebar.checkbox("Show Center Points", value=True)
    show_conf = st.sidebar.checkbox("Show Confidence Scores", value=False)
    
    st.sidebar.header("⚡ Performance Settings")
    
    frame_stride = st.sidebar.slider(
        "Frame Stride",
        min_value=1,
        max_value=5,
        value=1,
        step=1,
        help="Process every Nth frame for faster inference"
    )
    
    resize_width = st.sidebar.slider(
        "Resize Width (0 = no resize)",
        min_value=0,
        max_value=1280,
        value=0,
        step=64,
        help="Resize input frames for faster processing. 0 = original size"
    )
    
    return {
        'conf': conf_threshold,
        'iou': iou_threshold,
        'tracker': tracker_type,
        'line_axis': line_axis,
        'line_pos': line_pos,
        'count_direction': count_direction,
        'min_width': min_width,
        'min_height': min_height,
        'min_area': min_area,
        'max_area': max_area,
        'show_boxes': show_boxes,
        'show_ids': show_ids,
        'show_centers': show_centers,
        'show_conf': show_conf,
        'frame_stride': frame_stride,
        'resize_width': resize_width if resize_width > 0 else None
    }

def main():
    """Main Streamlit app."""
    st.title("🐷 Pig Counter")
    st.markdown("**Detect, track, and count pigs in videos using YOLOv8**")
    
    # Check model exists
    if not check_model_exists():
        return
    
    # Get sidebar settings
    settings = get_sidebar_settings()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📹 Upload Video")
        uploaded_file = st.file_uploader(
            "Select a video file",
            type=["mp4", "mov", "avi", "mkv"],
            help="Upload a video to detect and count pigs"
        )
    
    with col2:
        st.subheader("ℹ️ Settings Summary")
        st.markdown(f"""
        - **Confidence:** {settings['conf']:.2f}
        - **IOU:** {settings['iou']:.2f}
        - **Tracker:** {settings['tracker']}
        - **Line:** {settings['line_axis'].upper()} axis at {settings['line_pos']}
        - **Count Direction:** {settings['count_direction']}
        """)
    
    if uploaded_file is not None:
        # Create output directory
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
            tmp_input.write(uploaded_file.read())
            input_path = tmp_input.name
        
        # Generate output filename
        output_filename = f"annotated_{uploaded_file.name.rsplit('.', 1)[0]}.mp4"
        output_path = str(output_dir / output_filename)
        output_file = Path(output_path)
        
        st.info(f"Processing: {uploaded_file.name}")
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(current, total):
            """Update progress bar."""
            progress = current / total if total > 0 else 0
            progress_bar.progress(min(progress, 0.99))
            status_text.text(f"Processing: {current}/{total} frames")
        
        try:
            # Process video
            with st.spinner("🔄 Processing video..."):
                results = process_video(
                    input_path=input_path,
                    output_path=output_path,
                    model_path=str(MODEL_PATH),
                    conf=settings['conf'],
                    iou=settings['iou'],
                    tracker=settings['tracker'],
                    line_axis=settings['line_axis'],
                    line_pos=settings['line_pos'],
                    count_direction=settings['count_direction'],
                    min_width=settings['min_width'],
                    min_height=settings['min_height'],
                    min_area=settings['min_area'],
                    max_area=settings['max_area'],
                    show_boxes=settings['show_boxes'],
                    show_ids=settings['show_ids'],
                    show_centers=settings['show_centers'],
                    show_conf=settings['show_conf'],
                    frame_stride=settings['frame_stride'],
                    resize_width=settings['resize_width'],
                    progress_callback=progress_callback
                )
            
            progress_bar.progress(1.0)
            status_text.text("✓ Processing complete!")

            if output_file.exists() and output_file.stat().st_size > 0:
                converted = make_browser_playable_video(output_file, output_file)
                if not converted:
                    st.info(
                        "Annotated video was created, but ffmpeg was not available to convert it "
                        "for browser playback. Download should still work."
                    )
            
            # Display results
            st.success("✅ Video processed successfully!")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📊 Results")
                st.markdown(f"""
                - **Total Count:** {results['total_count']}
                - **Tracked IDs:** {len(results['tracked_ids'])}
                - **Frames Processed:** {results['processed_frames']}
                - **Output Resolution:** {results['resolution'][0]}x{results['resolution'][1]}
                """)
            
            with col2:
                st.subheader("📥 Download Output")
                if output_file.exists():
                    with open(output_file, "rb") as f:
                        video_bytes = f.read()

                    st.download_button(
                        label="Download Annotated Video",
                        data=video_bytes,
                        file_name=output_filename,
                        mime="video/mp4",
                        use_container_width=True
                    )
                else:
                    st.warning("Annotated video file was not created, so download is unavailable.")
            
            # Display video preview of the annotated output
            st.subheader("🎬 Video Preview")
            if output_file.exists() and output_file.stat().st_size > 0:
                try:
                    st.video(video_bytes, format="video/mp4")
                except Exception as err:
                    st.error(f"Unable to play annotated video in Streamlit: {err}")
                    cap = cv2.VideoCapture(str(output_file))
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, caption="First frame preview")
            else:
                if output_file.exists():
                    st.warning("Annotated video file is empty.")
                else:
                    st.warning("Annotated video file was not created, so preview is unavailable.")
            
        except Exception as e:
            st.error(f"❌ Error processing video: {str(e)}")
        
        finally:
            # Cleanup
            Path(input_path).unlink(missing_ok=True)
    
    # Footer
    st.markdown("---")
    st.markdown("📚 **How to use:**")
    st.markdown("""
    1. Adjust tracking and counting settings in the sidebar
    2. Upload a video file
    3. Wait for processing to complete
    4. Download the annotated video with pig counts
    """)

if __name__ == "__main__":
    main()
