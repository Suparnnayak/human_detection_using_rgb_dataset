"""
Evaluation metrics for SAR human detection
Includes mAP, recall, precision, and video-level evaluation
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between two boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU value (0-1)
    """
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


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Compute Average Precision (AP) using 11-point interpolation.
    
    Args:
        recalls: Array of recall values
        precisions: Array of precision values
    
    Returns:
        Average Precision value
    """
    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap


def compute_map(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
    class_id: int = 0
) -> Dict[str, float]:
    """
    Compute mean Average Precision (mAP) for detections.
    
    Args:
        predictions: List of prediction dicts with 'bbox', 'confidence', 'class'
        ground_truth: List of ground truth dicts with 'bbox', 'class'
        iou_threshold: IoU threshold for positive match
        class_id: Class ID to evaluate (default: 0 for person)
    
    Returns:
        Dictionary with mAP, AP, precision, recall metrics
    """
    # Filter by class
    preds = [p for p in predictions if p.get('class', 0) == class_id]
    gts = [g for g in ground_truth if g.get('class', 0) == class_id]
    
    if len(preds) == 0 and len(gts) == 0:
        return {'mAP': 1.0, 'AP': 1.0, 'precision': 1.0, 'recall': 1.0}
    
    if len(preds) == 0:
        return {'mAP': 0.0, 'AP': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    if len(gts) == 0:
        return {'mAP': 0.0, 'AP': 0.0, 'precision': 0.0, 'recall': 0.0}
    
    # Sort predictions by confidence (descending)
    preds_sorted = sorted(preds, key=lambda x: x.get('confidence', 0.0), reverse=True)
    
    # Match predictions to ground truth
    gt_matched = [False] * len(gts)
    tp = []  # True positives
    fp = []  # False positives
    
    for pred in preds_sorted:
        best_iou = 0.0
        best_gt_idx = -1
        
        for i, gt in enumerate(gts):
            if gt_matched[i]:
                continue
            
            iou = compute_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp.append(1)
            fp.append(0)
            gt_matched[best_gt_idx] = True
        else:
            tp.append(0)
            fp.append(1)
    
    # Compute cumulative TP and FP
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # Compute precision and recall
    recalls = tp_cumsum / len(gts) if len(gts) > 0 else np.zeros(len(tp))
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum) if len(tp_cumsum + fp_cumsum) > 0 else np.zeros(len(tp))
    
    # Compute AP
    ap = compute_ap(recalls, precisions)
    
    # Overall precision and recall
    total_tp = sum(tp)
    total_fp = sum(fp)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / len(gts) if len(gts) > 0 else 0.0
    
    return {
        'mAP': ap,  # For single class, mAP = AP
        'AP': ap,
        'precision': precision,
        'recall': recall,
        'TP': total_tp,
        'FP': total_fp,
        'FN': len(gts) - total_tp
    }


def compute_recall(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
    class_id: int = 0
) -> float:
    """
    Compute recall for detections.
    
    Args:
        predictions: List of prediction dicts
        ground_truth: List of ground truth dicts
        iou_threshold: IoU threshold for positive match
        class_id: Class ID to evaluate
    
    Returns:
        Recall value (0-1)
    """
    metrics = compute_map(predictions, ground_truth, iou_threshold, class_id)
    return metrics['recall']


def evaluate_video(
    predictions_path: str,
    ground_truth_path: str,
    output_path: Optional[str] = None,
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate video-level detection results.
    
    Args:
        predictions_path: Path to JSON file with predictions
        ground_truth_path: Path to JSON file with ground truth
        output_path: Optional path to save evaluation results
        iou_threshold: IoU threshold for matching
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Load predictions and ground truth
    with open(predictions_path, 'r') as f:
        pred_data = json.load(f)
    
    with open(ground_truth_path, 'r') as f:
        gt_data = json.load(f)
    
    # Aggregate all detections
    all_preds = []
    all_gts = []
    
    # Handle different JSON structures
    if 'frames' in pred_data:
        for frame in pred_data['frames']:
            all_preds.extend(frame.get('detections', []))
    else:
        all_preds = pred_data.get('detections', [])
    
    if 'frames' in gt_data:
        for frame in gt_data['frames']:
            all_gts.extend(frame.get('detections', []))
    else:
        all_gts = gt_data.get('detections', [])
    
    # Compute metrics
    metrics = compute_map(all_preds, all_gts, iou_threshold)
    
    # Add frame-level statistics
    if 'frames' in pred_data:
        frames_with_detections = sum(1 for f in pred_data['frames'] if len(f.get('detections', [])) > 0)
        total_frames = len(pred_data['frames'])
        metrics['frames_with_detections'] = frames_with_detections
        metrics['total_frames'] = total_frames
        metrics['detection_rate'] = frames_with_detections / total_frames if total_frames > 0 else 0.0
    
    # Save results if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Evaluation results saved to: {output_path}")
    
    return metrics


# Example usage
if __name__ == '__main__':
    # Example predictions and ground truth
    predictions = [
        {'bbox': [100, 100, 200, 200], 'confidence': 0.9, 'class': 0},
        {'bbox': [300, 300, 400, 400], 'confidence': 0.8, 'class': 0},
    ]
    
    ground_truth = [
        {'bbox': [105, 105, 195, 195], 'class': 0},
        {'bbox': [310, 310, 390, 390], 'class': 0},
    ]
    
    metrics = compute_map(predictions, ground_truth)
    print(f"mAP: {metrics['mAP']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")

