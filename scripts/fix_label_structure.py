"""
Fix label structure to match YOLO expectations
YOLO expects labels at the same level as images directory
"""

import shutil
from pathlib import Path


def fix_labels(base_dir, split_name):
    """
    Fix label structure for a dataset split.
    
    YOLO expects:
    - Images: train/images/
    - Labels: train/labels/ (same level, not nested)
    
    But we have:
    - Images: train/images/
    - Labels: train/labels/labels/ (nested)
    """
    base = Path(base_dir)
    
    # Source labels (nested)
    source_labels = base / split_name / "labels" / "labels"
    
    # Target labels (YOLO expects here)
    target_labels = base / split_name / "labels"
    
    if not source_labels.exists():
        print(f"Warning: Source labels not found: {source_labels}")
        return False
    
    # Check if target already has labels directly
    existing_labels = list(target_labels.glob("*.txt"))
    if existing_labels and target_labels.name == "labels":
        # Check if they're the same
        source_count = len(list(source_labels.glob("*.txt")))
        if len(existing_labels) == source_count:
            print(f"[OK] {split_name}: Labels already in correct location")
            return True
    
    # Create backup of original labels directory if it has files
    if existing_labels:
        backup_dir = base / split_name / "labels_backup"
        if not backup_dir.exists():
            backup_dir.mkdir()
            for label_file in existing_labels:
                shutil.copy2(label_file, backup_dir / label_file.name)
            print(f"  Backed up existing labels to {backup_dir}")
    
    # Copy labels from nested structure to correct location
    print(f"Fixing {split_name} labels...")
    print(f"  Source: {source_labels}")
    print(f"  Target: {target_labels}")
    
    # Ensure target directory exists
    target_labels.mkdir(parents=True, exist_ok=True)
    
    # Copy all label files
    label_files = list(source_labels.glob("*.txt"))
    for label_file in label_files:
        target_file = target_labels / label_file.name
        if not target_file.exists():
            shutil.copy2(label_file, target_file)
    
    print(f"  [OK] Copied {len(label_files)} label files")
    return True


def main():
    """Fix label structure for all splits."""
    base_dir = Path("datasets/Manipal-UAV Person Detection Dataset")
    
    print("="*60)
    print("Fixing Label Structure for YOLO")
    print("="*60)
    print()
    
    # Fix train labels
    if (base_dir / "train" / "labels" / "labels").exists():
        fix_labels(base_dir, "train")
    
    # Fix validation labels
    if (base_dir / "validation" / "labels" / "labels").exists():
        fix_labels(base_dir, "validation")
    
    # Fix test labels
    if (base_dir / "test" / "labels" / "labels").exists():
        fix_labels(base_dir, "test")
    
    print()
    print("="*60)
    print("Label structure fixed!")
    print("="*60)
    print("\nYou can now train the model.")


if __name__ == '__main__':
    main()

