# SAR UAV Human Detection System

A complete, production-ready codebase for Search & Rescue (SAR) human detection using UAV-mounted cameras and YOLO-based object detection. Designed for deployment on companion computers connected to Pixhawk 2.4.8 flight controllers.

## 🎯 Project Overview

This system enables autonomous or semi-autonomous human detection during SAR operations using:
- **Detection Model**: YOLOv8 (Ultralytics) - supports RGB and thermal imagery
- **Flight Controller**: Pixhawk 2.4.8 (PX4/ArduPilot compatible)
- **Companion Computer**: Laptop, Jetson, or Single Board Computer (SBC)
- **Camera**: RGB or Thermal (UAV-mounted)
- **Tracking**: SORT-based object tracking for video streams
- **Telemetry Integration**: MAVLink-based GPS and attitude data

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    UAV System                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐         ┌──────────────────────┐    │
│  │   Pixhawk    │◄────────►│  Companion Computer  │    │
│  │   2.4.8      │ MAVLink  │  (Laptop/Jetson/SBC) │    │
│  │              │          │                      │    │
│  │ - GPS        │          │  ┌────────────────┐  │    │
│  │ - IMU        │          │  │  YOLO Model    │  │    │
│  │ - Flight     │          │  │  Inference     │  │    │
│  │   Control    │          │  └────────────────┘  │    │
│  └──────────────┘          │  ┌────────────────┐  │    │
│                            │  │  Tracking      │  │    │
│  ┌──────────────┐          │  └────────────────┘  │    │
│  │   Camera     │─────────►│  ┌────────────────┐  │    │
│  │ (RGB/Thermal)│          │  │  Telemetry     │  │    │
│  └──────────────┘          │  │  Integration   │  │    │
│                            │  └────────────────┘  │    │
│                            └──────────────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Key Components

1. **Training Pipeline**: YOLOv8 training with SAR-optimized hyperparameters
2. **Inference Engine**: Real-time image and video processing
3. **Tracking Module**: SORT-based multi-object tracking
4. **Telemetry Interface**: MAVLink integration for GPS/attitude data
5. **Evaluation Tools**: mAP, recall, precision metrics

## 📁 Repository Structure

```
SAR-using-rgb-dataset/
├── datasets/              # Dataset storage (empty, ready for your data)
│   └── .gitkeep
├── configs/               # Configuration files
│   ├── data.yaml          # Dataset configuration template
│   ├── hyp_default.yaml   # Default YOLO hyperparameters
│   └── hyp_sar.yaml       # SAR-optimized hyperparameters
├── scripts/               # Training and preprocessing scripts
│   └── train.py           # Main training script
├── inference/             # Inference pipeline
│   ├── image_inference.py # Single image / batch image inference
│   └── video_inference.py # Video inference with optional tracking
├── tracking/              # Object tracking
│   ├── __init__.py
│   └── tracker.py         # SORT tracker implementation
├── telemetry/             # Pixhawk / MAVLink integration
│   ├── __init__.py
│   └── mavlink_interface.py # MAVLink communication
├── evaluation/            # Evaluation and metrics
│   ├── __init__.py
│   └── metrics.py         # mAP, recall, precision computation
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── visualization.py   # Drawing and visualization
│   └── data_utils.py      # Dataset utilities
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

### Directory Purposes

- **`datasets/`**: Place your YOLO-formatted datasets here. Structure: `datasets/your_dataset/images/train`, `images/val`, `labels/train`, `labels/val`
- **`configs/`**: YOLO configuration files, hyperparameters, and dataset definitions
- **`scripts/`**: Training scripts and data preprocessing utilities
- **`inference/`**: Standalone inference scripts for images and videos
- **`tracking/`**: Object tracking algorithms (SORT) for video streams
- **`telemetry/`**: MAVLink interface for Pixhawk 2.4.8 integration
- **`evaluation/`**: Metrics computation (mAP, recall, precision)
- **`utils/`**: Common helper functions for visualization and data handling

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher (3.9+ recommended)
- CUDA-capable GPU (recommended for training, optional for inference)
- For MAVLink: Serial port access (Linux) or USB drivers (Windows)

### Step 1: Clone and Setup

```bash
# Navigate to project directory
cd "SAR using rgb dataset"

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
# Test YOLO import
python -c "from ultralytics import YOLO; print('YOLO OK')"

# Test OpenCV
python -c "import cv2; print(f'OpenCV {cv2.__version__}')"

# Test PyTorch (if GPU available)
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### GPU vs CPU Usage

- **Training**: GPU highly recommended (10-100x faster). Minimum 4GB VRAM for YOLOv8n, 8GB+ for larger models.
- **Inference**: GPU recommended for real-time video. CPU works but slower (~1-5 FPS vs 30+ FPS on GPU).
- **Jetson**: Optimize with TensorRT for best performance.

## 📊 Adding a Dataset

### YOLO Dataset Format

Your dataset should follow this structure:

```
datasets/
└── your_dataset/
    ├── images/
    │   ├── train/
    │   │   ├── img001.jpg
    │   │   ├── img002.jpg
    │   │   └── ...
    │   └── val/
    │       ├── img101.jpg
    │       └── ...
    └── labels/
        ├── train/
        │   ├── img001.txt
        │   ├── img002.txt
        │   └── ...
        └── val/
            ├── img101.txt
            └── ...
```

### Label Format

Each `.txt` file contains one line per object:
```
class_id center_x center_y width height
```
All values are normalized (0-1). Example:
```
0 0.5 0.5 0.2 0.3
```
This represents a person (class 0) at image center with 20% width and 30% height.

### Configure Dataset

Edit `configs/data.yaml`:

```yaml
train: datasets/your_dataset/images/train
val: datasets/your_dataset/images/val
nc: 1
names:
  0: person
```

### Validate Dataset

```bash
python -c "from utils.data_utils import validate_dataset; valid, errors = validate_dataset('configs/data.yaml'); print('Valid:', valid); print('Errors:', errors)"
```

## 🎓 Training

### Basic Training

```bash
python scripts/train.py \
    --model yolov8n \
    --data configs/data.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --device cuda
```

### SAR-Optimized Training

```bash
python scripts/train.py \
    --model yolov8s \
    --data configs/data.yaml \
    --hyp configs/hyp_sar.yaml \
    --epochs 200 \
    --batch 16 \
    --imgsz 1280 \
    --device cuda \
    --name sar_thermal_v1
```

### Training Arguments

- `--model`: Model size (`yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`)
- `--data`: Path to dataset YAML
- `--epochs`: Number of training epochs
- `--batch`: Batch size (adjust based on GPU memory)
- `--imgsz`: Image size (640, 1280, etc. - larger for small objects)
- `--device`: `cpu`, `cuda`, `0`, `1`, or `auto`
- `--hyp`: Path to hyperparameter YAML (optional)
- `--name`: Experiment name
- `--resume`: Resume from last checkpoint
- `--stage`: Training stage (1 or 2, for multi-stage training)

### Training Outputs

Results are saved to `runs/train/{name}/`:
- `weights/best.pt`: Best model weights
- `weights/last.pt`: Last checkpoint
- `results.png`: Training curves
- `confusion_matrix.png`: Confusion matrix

## 🔍 Inference

### Image Inference

```bash
# Single image
python inference/image_inference.py \
    --model runs/train/sar_detection/weights/best.pt \
    --source path/to/image.jpg \
    --output outputs/inference \
    --conf 0.25

# Directory of images
python inference/image_inference.py \
    --model runs/train/sar_detection/weights/best.pt \
    --source path/to/images/ \
    --output outputs/inference
```

### Video Inference

```bash
# Without tracking
python inference/video_inference.py \
    --model runs/train/sar_detection/weights/best.pt \
    --source path/to/video.mp4 \
    --output outputs/inference \
    --conf 0.25

# With tracking
python inference/video_inference.py \
    --model runs/train/sar_detection/weights/best.pt \
    --source path/to/video.mp4 \
    --output outputs/inference \
    --track \
    --conf 0.25
```

### Inference Arguments

- `--model`: Path to model weights (.pt file)
- `--source`: Image file, image directory, or video file
- `--output`: Output directory
- `--conf`: Confidence threshold (default: 0.25)
- `--iou`: IoU threshold for NMS (default: 0.45)
- `--track`: Enable object tracking (video only)
- `--display`: Display results in window (requires GUI)

## 🎯 Evaluation

### Compute Metrics

```python
from evaluation.metrics import compute_map, evaluate_video

# For image dataset
predictions = [...]  # Your predictions
ground_truth = [...]  # Your ground truth
metrics = compute_map(predictions, ground_truth, iou_threshold=0.5)
print(f"mAP: {metrics['mAP']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")

# For video
metrics = evaluate_video(
    'outputs/inference/detections.json',
    'path/to/ground_truth.json',
    'outputs/evaluation_results.json'
)
```

## 🚁 Pixhawk 2.4.8 Integration

### Overview

The telemetry module (`telemetry/mavlink_interface.py`) provides MAVLink communication with Pixhawk 2.4.8. **This code runs on the companion computer, NOT on the Pixhawk itself.**

### Connection Methods

1. **USB Serial**: Connect Pixhawk via USB cable
2. **UART**: Connect via telemetry radio or wired UART
3. **Network**: UDP/TCP connection (e.g., `udp:127.0.0.1:14550`)

### Usage Example

```python
from telemetry.mavlink_interface import MAVLinkInterface, TelemetryData

# Initialize interface
# For USB (Linux): '/dev/ttyUSB0' or '/dev/ttyACM0'
# For USB (Windows): 'COM3' or similar
# For Network: 'udp:127.0.0.1:14550'
interface = MAVLinkInterface('/dev/ttyUSB0', baudrate=57600)

# Connect
if interface.connect():
    # Get telemetry
    telemetry = interface.get_telemetry()
    if telemetry:
        print(f"GPS: ({telemetry.latitude}, {telemetry.longitude})")
        print(f"Altitude: {telemetry.altitude}m")
        print(f"Heading: {telemetry.heading}°")
    
    # Attach telemetry to detection
    detection = {'bbox': [100, 100, 200, 200], 'confidence': 0.9}
    detection_with_gps = interface.attach_telemetry_to_detection(detection)
    print(detection_with_gps['gps_coordinates'])
    
    # Disconnect
    interface.disconnect()
```

### Integration with Inference

To add telemetry to video inference:

```python
from telemetry.mavlink_interface import MAVLinkInterface

interface = MAVLinkInterface('udp:127.0.0.1:14550')
interface.connect()

# In your inference loop:
for frame in video:
    detections = model.predict(frame)
    for det in detections:
        det = interface.attach_telemetry_to_detection(det)
        # Now det contains GPS coordinates, altitude, heading, etc.
```

### MAVLink Setup

1. **PX4 Firmware**: Default MAVLink on USB and TELEM2
2. **ArduPilot**: Configure SERIAL ports for MAVLink
3. **Companion Computer**: Install `pymavlink` (included in requirements.txt)

### Testing Without Hardware

The interface supports mock mode for testing:

```python
interface = MAVLinkInterface()  # No connection string = mock mode
interface.connect()
telemetry = interface.get_telemetry()  # Returns mock data
```

## 🔧 Configuration

### Hyperparameters

- **`configs/hyp_default.yaml`**: Standard YOLO hyperparameters
- **`configs/hyp_sar.yaml`**: Optimized for:
  - Small object detection (larger image size, enhanced augmentation)
  - High recall (important for SAR)
  - Aerial imagery (perspective, scale variation)

### Model Selection

- **YOLOv8n**: Fastest, smallest (best for Jetson/edge devices)
- **YOLOv8s**: Balanced speed/accuracy
- **YOLOv8m**: Higher accuracy, slower
- **YOLOv8l/x**: Best accuracy, requires powerful GPU

## 🗺️ Future Extensions

### Planned Features

1. **RGB + Thermal Fusion**: Multi-modal detection combining RGB and thermal cameras
2. **GPS Mapping**: Plot detections on map with GPS coordinates
3. **Real-time Streaming**: Live video processing from UAV camera
4. **Advanced Tracking**: DeepSORT or ByteTrack for better re-identification
5. **Alert System**: Automatic alerts when humans detected
6. **Mission Planning**: Integration with QGroundControl or Mission Planner

### Extension Points

- **Tracking**: Extend `tracking/tracker.py` with DeepSORT or custom algorithms
- **Telemetry**: Add more MAVLink messages in `telemetry/mavlink_interface.py`
- **Evaluation**: Add custom metrics in `evaluation/metrics.py`
- **Visualization**: Extend `utils/visualization.py` for map overlays

## 📝 Notes

### Dataset Requirements

- **Minimum**: ~100 images per class (for fine-tuning)
- **Recommended**: 1000+ images per class (for training from scratch)
- **Format**: YOLO format (see "Adding a Dataset" section)
- **Augmentation**: Built into training pipeline

### Performance Tips

1. **Training**: Use larger image size (1280) for small objects, but requires more GPU memory
2. **Inference**: Lower confidence threshold (0.15-0.2) for higher recall in SAR scenarios
3. **Tracking**: Adjust `iou_threshold` in tracker for better association
4. **Jetson**: Use TensorRT for 2-3x speedup

### Troubleshooting

- **CUDA out of memory**: Reduce batch size or image size
- **MAVLink connection fails**: Check serial port permissions (Linux: `sudo usermod -a -G dialout $USER`)
- **Low detection rate**: Lower confidence threshold, check dataset quality
- **Tracking IDs jump**: Adjust `iou_threshold` in tracker or use DeepSORT

## 📄 License

This codebase is provided as-is for SAR applications. Modify and extend as needed.

## 🤝 Contributing

This is a foundational codebase. Contributions welcome for:
- Additional tracking algorithms
- Multi-modal fusion
- Performance optimizations
- Documentation improvements

## 📧 Support

For issues or questions:
1. Check this README
2. Review code comments
3. Consult Ultralytics YOLO documentation: https://docs.ultralytics.com
4. Check MAVLink documentation: https://mavlink.io

---

**Built for Search & Rescue Operations** 🚁👥

