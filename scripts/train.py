"""
YOLO Training Script for SAR Human Detection
Supports YOLOv8 training with RGB or thermal datasets
"""

import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_yolo(
    model_size='yolov8n',  # n, s, m, l, x
    data_yaml='configs/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='auto',
    hyp_yaml=None,
    project='runs/train',
    name='sar_detection',
    resume=False,
    stage=1,
    **kwargs
):
    """
    Train YOLO model for human detection.
    
    Args:
        model_size: YOLO model size (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        data_yaml: Path to dataset configuration YAML
        epochs: Number of training epochs
        imgsz: Image size (640, 1280, etc.)
        batch: Batch size
        device: Device ('cpu', 'cuda', '0', '1', or 'auto')
        hyp_yaml: Path to hyperparameter YAML (optional)
        project: Project directory for outputs
        name: Experiment name
        resume: Resume from last checkpoint
        stage: Training stage (1 or 2, for future multi-stage training)
        **kwargs: Additional YOLO training arguments
    """
    # Validate paths
    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")
    
    # Load hyperparameters if provided
    hyp = None
    if hyp_yaml:
        hyp_path = Path(hyp_yaml)
        if hyp_path.exists():
            hyp = load_config(hyp_path)
            print(f"Loaded hyperparameters from: {hyp_yaml}")
        else:
            print(f"Warning: Hyperparameter file not found: {hyp_yaml}, using defaults")
    
    # Auto-detect device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Training on device: {device}")
    print(f"Model: {model_size}")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}, Batch: {batch}, Image size: {imgsz}")
    
    # Initialize model
    model = YOLO(f"{model_size}.pt")  # Load pretrained weights
    
    # Prepare training arguments
    train_args = {
        'data': str(data_path),
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': device,
        'project': project,
        'name': name,
        'resume': resume,
        **kwargs
    }
    
    # Add hyperparameters if provided
    if hyp:
        train_args['hyp'] = hyp
    
    # Stage-specific configurations
    if stage == 2:
        # Stage 2: Fine-tuning (example - can be customized)
        print("Stage 2 training: Fine-tuning with lower learning rate")
        if hyp:
            hyp['lr0'] = hyp.get('lr0', 0.01) * 0.1  # Reduce LR by 10x
        train_args['epochs'] = epochs // 2  # Typically fewer epochs in stage 2
    
    # Train model
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    
    results = model.train(**train_args)
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    print(f"Best model saved to: {Path(project) / name / 'weights' / 'best.pt'}")
    
    return model, results


def main():
    parser = argparse.ArgumentParser(description='Train YOLO for SAR Human Detection')
    parser.add_argument('--model', type=str, default='yolov8n',
                        choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                        help='YOLO model size')
    parser.add_argument('--data', type=str, default='configs/data.yaml',
                        help='Path to dataset YAML config')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size (640, 1280, etc.)')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cpu, cuda, 0, 1, or auto)')
    parser.add_argument('--hyp', type=str, default=None,
                        help='Path to hyperparameter YAML (optional)')
    parser.add_argument('--project', type=str, default='runs/train',
                        help='Project directory')
    parser.add_argument('--name', type=str, default='sar_detection',
                        help='Experiment name')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2],
                        help='Training stage (1 or 2)')
    
    args = parser.parse_args()
    
    train_yolo(
        model_size=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        hyp_yaml=args.hyp,
        project=args.project,
        name=args.name,
        resume=args.resume,
        stage=args.stage
    )


if __name__ == '__main__':
    main()

