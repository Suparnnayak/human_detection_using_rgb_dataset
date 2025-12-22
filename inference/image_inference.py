"""
Image Inference Script for SAR Human Detection
Processes single images or image directories
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json
from datetime import datetime


def inference_image(
    model_path,
    image_path,
    output_dir='outputs/inference',
    conf_threshold=0.25,
    iou_threshold=0.45,
    save_images=True,
    save_json=True,
    display=False,
    line_thickness=2
):
    """
    Run inference on a single image or directory of images.
    
    Args:
        model_path: Path to YOLO model weights (.pt file)
        image_path: Path to image file or directory
        output_dir: Directory to save results
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        save_images: Whether to save annotated images
        save_json: Whether to save detection results as JSON
        display: Whether to display results (requires GUI)
        line_thickness: Thickness of bounding box lines
    """
    # Load model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine if input is file or directory
    input_path = Path(image_path)
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        # Support common image formats
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        image_files = sorted(image_files)
    else:
        raise FileNotFoundError(f"Image path not found: {image_path}")
    
    if not image_files:
        print(f"No image files found in: {image_path}")
        return
    
    print(f"Found {len(image_files)} image(s) to process")
    
    all_results = []
    
    for img_file in image_files:
        print(f"Processing: {img_file.name}")
        
        # Run inference
        results = model.predict(
            source=str(img_file),
            conf=conf_threshold,
            iou=iou_threshold,
            save=False,  # We'll save manually
            verbose=False
        )
        
        # Process results
        result = results[0]
        detections = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                detection = {
                    'bbox': box.tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(conf),
                    'class': int(cls),
                    'class_name': model.names[cls]
                }
                detections.append(detection)
        
        # Store results
        result_data = {
            'image_path': str(img_file),
            'image_name': img_file.name,
            'timestamp': datetime.now().isoformat(),
            'detections': detections,
            'num_detections': len(detections)
        }
        all_results.append(result_data)
        
        # Save annotated image
        if save_images:
            annotated_img = result.plot(line_width=line_thickness)
            output_img_path = output_path / f"detected_{img_file.stem}.jpg"
            cv2.imwrite(str(output_img_path), annotated_img)
            print(f"  Saved annotated image: {output_img_path}")
        
        # Display if requested
        if display:
            cv2.imshow('Detection', annotated_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        print(f"  Detections: {len(detections)}")
    
    # Save JSON results
    if save_json:
        json_path = output_path / 'detections.json'
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved detection results to: {json_path}")
    
    # Print summary
    total_detections = sum(r['num_detections'] for r in all_results)
    print(f"\n{'='*50}")
    print(f"Summary: {len(image_files)} images processed, {total_detections} total detections")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description='Run YOLO inference on images')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to YOLO model weights (.pt file)')
    parser.add_argument('--source', type=str, required=True,
                        help='Path to image file or directory')
    parser.add_argument('--output', type=str, default='outputs/inference',
                        help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--no-save-images', action='store_true',
                        help='Do not save annotated images')
    parser.add_argument('--no-save-json', action='store_true',
                        help='Do not save JSON results')
    parser.add_argument('--display', action='store_true',
                        help='Display results (requires GUI)')
    parser.add_argument('--line-thickness', type=int, default=2,
                        help='Bounding box line thickness')
    
    args = parser.parse_args()
    
    inference_image(
        model_path=args.model,
        image_path=args.source,
        output_dir=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        save_images=not args.no_save_images,
        save_json=not args.no_save_json,
        display=args.display,
        line_thickness=args.line_thickness
    )


if __name__ == '__main__':
    main()

