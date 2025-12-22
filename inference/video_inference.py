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

# Add parent directory to path for tracking import
sys.path.append(str(Path(__file__).parent.parent))
from tracking.tracker import SORTTracker


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
    line_thickness=2
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
            
            # Prepare detections for tracking or direct use
            detection_list = []
            for box, conf, cls in zip(boxes, confidences, classes):
                detection_list.append({
                    'bbox': box.tolist(),
                    'confidence': float(conf),
                    'class': int(cls),
                    'class_name': model.names[cls]
                })
            
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
                    track_id = det['track_id']
                    cv2.putText(
                        annotated_frame,
                        f"ID: {track_id}",
                        (int(x1), int(y1) - 10),
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
        line_thickness=args.line_thickness
    )


if __name__ == '__main__':
    main()

