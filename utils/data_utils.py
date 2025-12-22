"""
Data utilities for dataset preparation and validation
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2


def convert_to_yolo_format(
    annotation_path: str,
    output_dir: str,
    image_width: int,
    image_height: int
):
    """
    Convert annotations from other formats to YOLO format.
    
    YOLO format: class_id center_x center_y width height
    (all normalized 0-1)
    
    Args:
        annotation_path: Path to input annotation file
        output_dir: Directory to save YOLO format labels
        image_width: Image width in pixels
        image_height: Image height in pixels
    """
    # This is a template - implement based on your annotation format
    # Example: COCO, Pascal VOC, custom JSON, etc.
    pass


def validate_dataset(data_yaml_path: str) -> Tuple[bool, List[str]]:
    """
    Validate YOLO dataset structure.
    
    Args:
        data_yaml_path: Path to dataset YAML config
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Load YAML
    try:
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
    except Exception as e:
        return False, [f"Failed to load YAML: {e}"]
    
    # Check required fields
    required_fields = ['train', 'val', 'nc', 'names']
    for field in required_fields:
        if field not in data_config:
            errors.append(f"Missing required field: {field}")
    
    # Check paths
    for split in ['train', 'val']:
        if split in data_config:
            path = Path(data_config[split])
            if not path.exists():
                errors.append(f"Path does not exist: {data_config[split]}")
            else:
                # Check for images and labels
                image_dir = path
                label_dir = path.parent.parent / 'labels' / path.name
                
                if not image_dir.exists():
                    errors.append(f"Image directory not found: {image_dir}")
                
                if not label_dir.exists():
                    errors.append(f"Label directory not found: {label_dir}")
                else:
                    # Check if number of images matches labels
                    image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
                    label_files = list(label_dir.glob('*.txt'))
                    
                    if len(image_files) != len(label_files):
                        errors.append(
                            f"Mismatch in {split}: {len(image_files)} images vs {len(label_files)} labels"
                        )
    
    # Check class count
    if 'nc' in data_config and 'names' in data_config:
        if data_config['nc'] != len(data_config['names']):
            errors.append(
                f"Class count mismatch: nc={data_config['nc']}, names={len(data_config['names'])}"
            )
    
    return len(errors) == 0, errors


def count_dataset_statistics(data_yaml_path: str) -> Dict[str, Any]:
    """
    Count statistics about the dataset.
    
    Args:
        data_yaml_path: Path to dataset YAML config
    
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'train_images': 0,
        'val_images': 0,
        'train_labels': 0,
        'val_labels': 0,
        'classes': {}
    }
    
    try:
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        for split in ['train', 'val']:
            if split in data_config:
                path = Path(data_config[split])
                if path.exists():
                    image_dir = path
                    label_dir = path.parent.parent / 'labels' / path.name
                    
                    # Count images
                    images = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
                    stats[f'{split}_images'] = len(images)
                    
                    # Count labels and classes
                    if label_dir.exists():
                        labels = list(label_dir.glob('*.txt'))
                        stats[f'{split}_labels'] = len(labels)
                        
                        # Count class occurrences
                        for label_file in labels:
                            with open(label_file, 'r') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if parts:
                                        class_id = int(parts[0])
                                        stats['classes'][class_id] = stats['classes'].get(class_id, 0) + 1
    except Exception as e:
        print(f"Error counting statistics: {e}")
    
    return stats

