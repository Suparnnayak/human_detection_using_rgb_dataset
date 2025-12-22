"""
Visualization utilities for SAR detections
"""

import cv2
import numpy as np
from typing import List, Dict, Any


def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    show_confidence: bool = True,
    show_class: bool = True
) -> np.ndarray:
    """
    Draw bounding boxes and labels on image.
    
    Args:
        image: Input image (BGR format)
        detections: List of detection dicts with 'bbox', 'confidence', 'class'
        color: Bounding box color (B, G, R)
        thickness: Line thickness
        show_confidence: Whether to show confidence score
        show_class: Whether to show class name
    
    Returns:
        Annotated image
    """
    annotated = image.copy()
    
    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label
        label_parts = []
        if show_class and 'class_name' in det:
            label_parts.append(det['class_name'])
        if show_confidence and 'confidence' in det:
            label_parts.append(f"{det['confidence']:.2f}")
        
        if label_parts:
            label = ' '.join(label_parts)
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            cv2.rectangle(
                annotated,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
    
    return annotated


def draw_tracks(
    image: np.ndarray,
    tracks: List[Dict[str, Any]],
    color_map: Dict[int, tuple] = None,
    thickness: int = 2,
    show_track_id: bool = True
) -> np.ndarray:
    """
    Draw tracked objects with unique colors per track ID.
    
    Args:
        image: Input image (BGR format)
        tracks: List of track dicts with 'bbox', 'track_id'
        color_map: Optional dict mapping track_id to color
        thickness: Line thickness
        show_track_id: Whether to show track ID
    
    Returns:
        Annotated image
    """
    annotated = image.copy()
    
    if color_map is None:
        # Generate colors for tracks
        color_map = {}
        for track in tracks:
            track_id = track.get('track_id', 0)
            if track_id not in color_map:
                # Generate distinct color
                hue = (track_id * 137.508) % 360  # Golden angle
                color_map[track_id] = tuple(map(int, cv2.cvtColor(
                    np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR
                )[0][0]))
    
    for track in tracks:
        bbox = track['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        track_id = track.get('track_id', 0)
        
        color = color_map.get(track_id, (0, 255, 0))
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        
        # Draw track ID
        if show_track_id:
            label = f"ID: {track_id}"
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
    
    return annotated

