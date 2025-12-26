"""
Video Inference Script for SAR Human Detection
Processes video files with optional tracking
"""

import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO
import json
from datetime import datetime
import sys
import numpy as np

# Add parent directory to path for tracking import
sys.path.append(str(Path(__file__).parent.parent))
from tracking.tracker import SORTTracker


def _iou(box1, box2):
    """Calculate IoU between two boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    inter_area = (x2_i - x1_i) * (y2_i - y1_i)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def _remove_duplicate_detections(detections, iou_threshold=0.5):
    """Remove duplicate detections with high IoU overlap."""
    if len(detections) <= 1:
        return detections
    
    # Sort by confidence (highest first)
    sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    filtered = []
    used = [False] * len(sorted_dets)
    
    for i, det1 in enumerate(sorted_dets):
        if used[i]:
            continue
        
        filtered.append(det1)
        used[i] = True
        
        # Mark overlapping detections as used
        for j, det2 in enumerate(sorted_dets[i+1:], start=i+1):
            if used[j]:
                continue
            
            iou = _iou(det1['bbox'], det2['bbox'])
            if iou > iou_threshold:
                # Keep the one with higher confidence (already added)
                used[j] = True
    
    return filtered


def inference_video(
    model_path,
    video_path,
    output_dir='outputs/inference',
    conf_threshold=0.25,
    iou_threshold=0.45,
    use_tracking=False,
    save_video=True,
    save_json=True,
    display=False,
    fps_output=None,
    line_thickness=2,
    min_box_area=100,
    max_box_area=None,
    aspect_ratio_range=(0.2, 5.0)
):
    """
    Run inference on a video file.
    
    Args:
        model_path: Path to YOLO model weights (.pt file)
        video_path: Path to video file
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        use_tracking: Whether to use object tracking
        save_video: Whether to save annotated video
        save_json: Whether to save detection results as JSON
        display: Whether to display results (requires GUI)
        fps_output: Output FPS (None = same as input)
        line_thickness: Thickness of bounding box lines
    """
    # Load model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Initialize tracker if requested
    tracker = None
    if use_tracking:
        tracker = SORTTracker()
        print("Tracking enabled: SORT")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Open video
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer
    video_writer = None
    if save_video:
        output_video_path = output_path / f"detected_{video_path_obj.stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_fps = fps_output if fps_output else fps
        video_writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            output_fps,
            (width, height)
        )
        print(f"Output video will be saved to: {output_video_path}")
    
    # Process video
    frame_results = []
    frame_count = 0
    
    print("\nProcessing video frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count}/{total_frames} frames")
        
        # Run inference
        results = model.predict(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            save=False,
            verbose=False
        )
        
        result = results[0]
        detections = []
        
        # Extract detections
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            # Prepare detections for tracking or direct use with filtering
            detection_list = []
            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                area = w * h
                aspect_ratio = w / h if h > 0 else 0
                
                # Filter by size and aspect ratio
                if area < min_box_area:
                    continue  # Too small, likely false positive
                
                if max_box_area and area > max_box_area:
                    continue  # Too large, likely false positive
                
                if aspect_ratio < aspect_ratio_range[0] or aspect_ratio > aspect_ratio_range[1]:
                    continue  # Unrealistic aspect ratio for person
                
                # Additional filters for UAV perspective
                # Minimum width and height (very small boxes are likely noise)
                if w < 10 or h < 10:
                    continue
                
                # Maximum width and height (unrealistic for person from UAV)
                if w > width * 0.5 or h > height * 0.5:
                    continue
                
                detection_list.append({
                    'bbox': box.tolist(),
                    'confidence': float(conf),
                    'class': int(cls),
                    'class_name': model.names[cls]
                })
            
            # Remove duplicate/overlapping detections (same person detected multiple times)
            # YOLO does NMS, but we add extra filtering for edge cases where same person has multiple boxes
            if len(detection_list) > 1:
                detection_list = _remove_duplicate_detections(detection_list, iou_threshold=0.5)
            
            # Apply tracking if enabled
            if tracker:
                tracked_detections = tracker.update(detection_list, frame)
                detections = tracked_detections
            else:
                detections = detection_list
        
        # Store frame results
        frame_result = {
            'frame_number': frame_count,
            'timestamp': frame_count / fps if fps > 0 else 0,
            'detections': detections,
            'num_detections': len(detections)
        }
        frame_results.append(frame_result)
        
        # Draw detections on frame
        annotated_frame = result.plot(line_width=line_thickness)
        
        # Draw track IDs if tracking is enabled
        if tracker and detections:
            for det in detections:
                if 'track_id' in det:
                    x1, y1, x2, y2 = det['bbox']
                    # Validate bbox coordinates
                    if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
                        continue
                    track_id = det['track_id']
                    # Ensure coordinates are valid integers
                    x1_int = max(0, int(x1))
                    y1_int = max(0, int(y1) - 10)
                    cv2.putText(
                        annotated_frame,
                        f"ID: {track_id}",
                        (x1_int, y1_int),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
        
        # Save frame to video
        if video_writer:
            video_writer.write(annotated_frame)
        
        # Display if requested
        if display:
            cv2.imshow('Video Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Stopped by user")
                break
    
    # Cleanup
    cap.release()
    if video_writer:
        video_writer.release()
    if display:
        cv2.destroyAllWindows()
    
    # Save JSON results
    if save_json:
        json_path = output_path / f"detections_{video_path_obj.stem}.json"
        video_summary = {
            'video_path': str(video_path),
            'video_name': video_path_obj.name,
            'timestamp': datetime.now().isoformat(),
            'fps': fps,
            'width': width,
            'height': height,
            'total_frames': total_frames,
            'frames': frame_results
        }
        with open(json_path, 'w') as f:
            json.dump(video_summary, f, indent=2)
        print(f"\nSaved detection results to: {json_path}")
    
    # Print summary
    total_detections = sum(f['num_detections'] for f in frame_results)
    frames_with_detections = sum(1 for f in frame_results if f['num_detections'] > 0)
    
    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  Total frames: {frame_count}")
    print(f"  Frames with detections: {frames_with_detections}")
    print(f"  Total detections: {total_detections}")
    print(f"  Average detections per frame: {total_detections / frame_count if frame_count > 0 else 0:.2f}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description='Run YOLO inference on video')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to YOLO model weights (.pt file)')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to video file')
    parser.add_argument('--output', type=str, default='outputs/inference',
                        help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--track', action='store_true',
                        help='Enable object tracking')
    parser.add_argument('--no-save-video', action='store_true',
                        help='Do not save annotated video')
    parser.add_argument('--no-save-json', action='store_true',
                        help='Do not save JSON results')
    parser.add_argument('--display', action='store_true',
                        help='Display results (requires GUI)')
    parser.add_argument('--fps-output', type=int, default=None,
                        help='Output video FPS (default: same as input)')
    parser.add_argument('--line-thickness', type=int, default=2,
                        help='Bounding box line thickness')
    parser.add_argument('--min-box-area', type=int, default=100,
                        help='Minimum bounding box area (pixels) to filter small false positives')
    parser.add_argument('--max-box-area', type=int, default=None,
                        help='Maximum bounding box area (pixels) to filter large false positives')
    parser.add_argument('--aspect-ratio-min', type=float, default=0.2,
                        help='Minimum aspect ratio (width/height) for person detection')
    parser.add_argument('--aspect-ratio-max', type=float, default=5.0,
                        help='Maximum aspect ratio (width/height) for person detection')
    parser.add_argument('--remove-duplicates', action='store_true', default=True,
                        help='Remove duplicate/overlapping detections (default: True)')
    parser.add_argument('--duplicate-iou', type=float, default=0.5,
                        help='IoU threshold for duplicate detection removal')
    
    args = parser.parse_args()
    
    inference_video(
        model_path=args.model,
        video_path=args.source,
        output_dir=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        use_tracking=args.track,
        save_video=not args.no_save_video,
        save_json=not args.no_save_json,
        display=args.display,
        fps_output=args.fps_output,
        line_thickness=args.line_thickness,
        min_box_area=args.min_box_area,
        max_box_area=args.max_box_area,
        aspect_ratio_range=(args.aspect_ratio_min, args.aspect_ratio_max)
    )


if __name__ == '__main__':
    main()

