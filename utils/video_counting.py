"""
Video processing and pig counting utilities for Streamlit app.
Handles YOLO inference, tracking, counting line detection, and video annotation.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from ultralytics import YOLO


class PigCounter:
    """Main class for pig detection, tracking, and counting in videos."""
    
    def __init__(self, model_path: str):
        """
        Initialize the PigCounter with a YOLO model.
        
        Args:
            model_path: Path to the YOLO model file (e.g., pigdetect_best.pt)
        """
        self.model = YOLO(model_path)
        self.track_history = {}  # Store center positions for each track
        self.counted_tracks = set()  # Track IDs that have been counted
        
    def bbox_filter(self, bbox: np.ndarray, frame_shape: Tuple,
                   min_width: int, min_height: int,
                   min_area: int, max_area: int) -> bool:
        """
        Check if bounding box meets filtering criteria.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            frame_shape: Frame dimensions (height, width, channels)
            min_width: Minimum bounding box width
            min_height: Minimum bounding box height
            min_area: Minimum bounding box area
            max_area: Maximum bounding box area
            
        Returns:
            True if bbox passes all filters, False otherwise
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        if width < min_width or height < min_height:
            return False
        if area < min_area or area > max_area:
            return False
        
        return True
    
    def get_bbox_center(self, bbox: np.ndarray) -> Tuple[float, float]:
        """
        Get the center point of a bounding box.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Tuple of (center_x, center_y)
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return center_x, center_y
    
    def crossed_line(self, track_id: int, center: Tuple[float, float],
                    line_axis: str, line_pos: int,
                    count_direction: str) -> bool:
        """
        Check if an object crossed the counting line.
        
        Args:
            track_id: Unique track identifier
            center: Current center position (x, y)
            line_axis: 'x' for vertical line, 'y' for horizontal line
            line_pos: Position of the counting line
            count_direction: 'both', 'left_to_right', 'right_to_left',
                           'top_to_bottom', 'bottom_to_top'
            
        Returns:
            True if object crossed the line, False otherwise
        """
        if track_id not in self.track_history:
            self.track_history[track_id] = []
        
        if not self.track_history[track_id]:
            self.track_history[track_id].append(center)
            return False
        
        prev_center = self.track_history[track_id][-1]
        self.track_history[track_id].append(center)
        
        # Only keep last 5 positions to avoid memory bloat
        if len(self.track_history[track_id]) > 5:
            self.track_history[track_id].pop(0)
        
        if line_axis == 'x':  # Vertical line
            prev_x = prev_center[0]
            curr_x = center[0]
            
            if count_direction == 'both':
                crossed = (prev_x - line_pos) * (curr_x - line_pos) < 0
            elif count_direction == 'left_to_right':
                crossed = prev_x < line_pos <= curr_x
            elif count_direction == 'right_to_left':
                crossed = prev_x > line_pos >= curr_x
            else:
                crossed = False
                
        elif line_axis == 'y':  # Horizontal line
            prev_y = prev_center[1]
            curr_y = center[1]
            
            if count_direction == 'both':
                crossed = (prev_y - line_pos) * (curr_y - line_pos) < 0
            elif count_direction == 'top_to_bottom':
                crossed = prev_y < line_pos <= curr_y
            elif count_direction == 'bottom_to_top':
                crossed = prev_y > line_pos >= curr_y
            else:
                crossed = False
        else:
            crossed = False
        
        return crossed
    
    def draw_counting_line(self, frame: np.ndarray, line_axis: str,
                          line_pos: int, color: Tuple[int, int, int] = (0, 255, 0),
                          thickness: int = 2) -> np.ndarray:
        """
        Draw the counting line on the frame.
        
        Args:
            frame: Video frame
            line_axis: 'x' for vertical line, 'y' for horizontal line
            line_pos: Position of the counting line
            color: RGB color tuple
            thickness: Line thickness
            
        Returns:
            Frame with drawn counting line
        """
        height, width = frame.shape[:2]
        
        if line_axis == 'x':  # Vertical line
            cv2.line(frame, (int(line_pos), 0), (int(line_pos), height),
                    color, thickness)
        elif line_axis == 'y':  # Horizontal line
            cv2.line(frame, (0, int(line_pos)), (width, int(line_pos)),
                    color, thickness)
        
        return frame
    
    def process_frame(self, frame: np.ndarray, conf: float, iou: float,
                     tracker: str, line_axis: str, line_pos: int,
                     count_direction: str, min_width: int, min_height: int,
                     min_area: int, max_area: int,
                     show_boxes: bool, show_ids: bool,
                     show_centers: bool, show_conf: bool) -> Tuple[np.ndarray, int]:
        """
        Process a single frame: detect, track, count, and annotate.
        
        Args:
            frame: Input video frame
            conf: YOLO confidence threshold
            iou: YOLO IOU threshold
            tracker: 'bytetrack.yaml' or 'botsort.yaml'
            line_axis: 'x' or 'y'
            line_pos: Position of counting line
            count_direction: Direction to count crossings
            min_width: Minimum bbox width filter
            min_height: Minimum bbox height filter
            min_area: Minimum bbox area filter
            max_area: Maximum bbox area filter
            show_boxes: Whether to draw bounding boxes
            show_ids: Whether to draw track IDs
            show_centers: Whether to draw center points
            show_conf: Whether to draw confidence scores
            
        Returns:
            Tuple of (annotated_frame, current_total_count)
        """
        # Run YOLO tracking
        results = self.model.track(
            frame,
            persist=True,
            tracker=tracker,
            conf=conf,
            iou=iou,
            verbose=False
        )
        
        current_count = 0
        annotated_frame = frame.copy()
        
        # Process detections
        if results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i, box in enumerate(boxes):
                bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                track_id = int(box.id[0]) if box.id is not None else -1
                confidence = float(box.conf[0])
                
                # Apply bbox filters
                if not self.bbox_filter(bbox, frame.shape, min_width,
                                       min_height, min_area, max_area):
                    continue
                
                # Check if crossed counting line
                center = self.get_bbox_center(bbox)
                if track_id not in self.counted_tracks:
                    if self.crossed_line(track_id, center, line_axis,
                                        line_pos, count_direction):
                        self.counted_tracks.add(track_id)
                        current_count += 1
                
                # Draw bounding box
                if show_boxes:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)),
                                (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Draw track ID
                if show_ids and track_id >= 0:
                    cv2.putText(annotated_frame, f"ID: {track_id}",
                              (int(bbox[0]), int(bbox[1] - 10)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw center point
                if show_centers:
                    cv2.circle(annotated_frame, (int(center[0]), int(center[1])),
                             3, (255, 0, 0), -1)
                
                # Draw confidence score
                if show_conf:
                    cv2.putText(annotated_frame, f"Conf: {confidence:.2f}",
                              (int(bbox[0]), int(bbox[3] + 20)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw counting line
        annotated_frame = self.draw_counting_line(annotated_frame, line_axis,
                                                 line_pos)
        
        # Draw total count
        total_count = len(self.counted_tracks)
        cv2.putText(annotated_frame, f"Total Count: {total_count}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return annotated_frame, current_count
    
    def reset_count(self):
        """Reset the counting state for a new video."""
        self.track_history.clear()
        self.counted_tracks.clear()


def process_video(input_path: str, output_path: str, model_path: str,
                 conf: float = 0.5, iou: float = 0.5,
                 tracker: str = 'bytetrack.yaml', line_axis: str = 'y',
                 line_pos: float = 0.5, count_direction: str = 'both',
                 min_width: int = 10, min_height: int = 10,
                 min_area: int = 100, max_area: int = 500000,
                 show_boxes: bool = True, show_ids: bool = True,
                 show_centers: bool = True, show_conf: bool = True,
                 frame_stride: int = 1, resize_width: Optional[int] = None,
                 progress_callback: Optional[callable] = None) -> Dict:
    """
    Process a video file: detect, track, count pigs, and save output.
    
    Args:
        input_path: Path to input video
        output_path: Path to save output video
        model_path: Path to YOLO model
        conf: Confidence threshold
        iou: IOU threshold
        tracker: Tracker type
        line_axis: 'x' or 'y'
        line_pos: Counting line position (as fraction of axis if < 1, else absolute)
        count_direction: Direction to count
        min_width: Minimum bbox width
        min_height: Minimum bbox height
        min_area: Minimum bbox area
        max_area: Maximum bbox area
        show_boxes: Draw bounding boxes
        show_ids: Draw track IDs
        show_centers: Draw center points
        show_conf: Draw confidence
        frame_stride: Process every Nth frame
        resize_width: Resize frame to this width for faster processing
        progress_callback: Callback function for progress updates
        
    Returns:
        Dictionary with processing results and statistics
    """
    counter = PigCounter(model_path)
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Convert line_pos from fraction to absolute if needed
    if line_axis == 'x':
        if line_pos < 1:
            line_pos = int(line_pos * width)
        else:
            line_pos = int(line_pos)
    else:  # y axis
        if line_pos < 1:
            line_pos = int(line_pos * height)
        else:
            line_pos = int(line_pos)
    
    # Setup video writer
    if resize_width:
        output_width = resize_width
        output_height = int(resize_width * height / width)
    else:
        output_width = width
        output_height = height
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps,
                         (output_width, output_height))
    
    if not out.isOpened():
        raise ValueError(f"Cannot open video writer for: {output_path}")
    
    frame_count = 0
    total_detections = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames based on stride
            if frame_count % frame_stride != 0:
                frame_count += 1
                if progress_callback:
                    progress_callback(frame_count, total_frames)
                continue
            
            # Resize if needed
            if resize_width:
                frame = cv2.resize(frame, (output_width, output_height))
            
            # Process frame
            annotated_frame, current_count = counter.process_frame(
                frame, conf, iou, tracker, line_axis, line_pos,
                count_direction, min_width, min_height, min_area, max_area,
                show_boxes, show_ids, show_centers, show_conf
            )
            
            total_detections += current_count
            out.write(annotated_frame)
            
            frame_count += 1
            if progress_callback:
                progress_callback(frame_count, total_frames)
    
    finally:
        cap.release()
        out.release()
    
    return {
        'output_path': output_path,
        'total_frames': total_frames,
        'processed_frames': frame_count,
        'fps': fps,
        'resolution': (output_width, output_height),
        'total_count': len(counter.counted_tracks),
        'tracked_ids': list(counter.counted_tracks)
    }
