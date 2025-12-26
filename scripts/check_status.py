"""
Check dataset status and paths
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_utils import count_dataset_statistics, validate_dataset

print("="*60)
print("DATASET STATUS CHECK")
print("="*60)

# Check statistics
stats = count_dataset_statistics('configs/data.yaml')
print(f"\nDataset Statistics:")
print(f"  Train:  {stats['train_images']:>5} images, {stats['train_labels']:>5} labels")
print(f"  Val:    {stats['val_images']:>5} images, {stats['val_labels']:>5} labels")

# Check paths from config
import yaml
with open('configs/data.yaml', 'r') as f:
    config = yaml.safe_load(f)

base = Path("datasets/Manipal-UAV Person Detection Dataset")
paths = {
    'Train images': Path(config['train']),
    'Val images': Path(config['val']),
    'Test images': Path(config.get('test', '')),
    'Train labels': base / "train" / "labels" / "labels",
    'Val labels': base / "validation" / "labels" / "labels",
    'Test labels': base / "test" / "labels" / "labels",
}

print(f"\nPath Status:")
for name, path in paths.items():
    exists = path.exists()
    count = 0
    if exists:
        if 'images' in name.lower():
            count = len(list(path.glob('*.jpg')) + list(path.glob('*.png')))
        else:
            count = len(list(path.glob('*.txt')))
    status = "[OK]" if exists else "[MISSING]"
    print(f"  {status} {name:20} {str(path):60} ({count} files)")

# Validate
print(f"\nValidation:")
valid, errors = validate_dataset('configs/data.yaml')
print(f"  Valid: {valid}")
if not valid:
    print(f"  Errors:")
    for error in errors:
        print(f"    - {error}")

print("\n" + "="*60)

