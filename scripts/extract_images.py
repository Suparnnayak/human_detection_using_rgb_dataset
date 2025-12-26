"""
Extract images from tar.xz archives for Manipal-UAV dataset
"""

import tarfile
import os
from pathlib import Path


def extract_images(tar_path, output_dir):
    """
    Extract images from tar.xz archive.
    The tar contains an 'images/' directory, so we extract to parent and move.
    
    Args:
        tar_path: Path to .tar.xz file
        output_dir: Directory to extract images to (final location)
    """
    tar_path = Path(tar_path)
    output_dir = Path(output_dir)
    temp_dir = output_dir.parent / "temp_extract"
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting {tar_path.name}...")
    
    try:
        # Extract to temp directory
        with tarfile.open(tar_path, 'r:xz') as tar:
            tar.extractall(temp_dir)
        
        # Move images from temp/images/ to output_dir
        extracted_images = temp_dir / "images"
        if extracted_images.exists():
            import shutil
            # Move all files from extracted_images to output_dir
            for item in extracted_images.iterdir():
                if item.is_file():
                    shutil.move(str(item), str(output_dir / item.name))
                elif item.is_dir():
                    shutil.move(str(item), str(output_dir / item.name))
            
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"✓ Extracted {len(list(output_dir.glob('*')))} items to {output_dir}")
        else:
            # If no images/ subdirectory, move everything directly
            for item in temp_dir.iterdir():
                if item.name != "images.tar.xz":
                    import shutil
                    if item.is_file():
                        shutil.move(str(item), str(output_dir / item.name))
                    elif item.is_dir():
                        shutil.move(str(item), str(output_dir / item.name))
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"✓ Extracted to {output_dir}")
        
        return True
    except Exception as e:
        print(f"✗ Error extracting {tar_path}: {e}")
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False


def main():
    """Extract all image archives in the dataset."""
    base_dir = Path("datasets/Manipal-UAV Person Detection Dataset")
    
    # Extract train images
    train_tar = base_dir / "train" / "images.tar.xz"
    train_output = base_dir / "train" / "images"
    if train_tar.exists():
        extract_images(train_tar, train_output)
    else:
        print(f"Warning: {train_tar} not found")
    
    # Extract validation images
    val_tar = base_dir / "validation" / "images.tar.xz"
    val_output = base_dir / "validation" / "images"
    if val_tar.exists():
        extract_images(val_tar, val_output)
    else:
        print(f"Warning: {val_tar} not found")
    
    # Extract test images
    test_tar = base_dir / "test" / "images.tar.xz"
    test_output = base_dir / "test" / "images"
    if test_tar.exists():
        extract_images(test_tar, test_output)
    else:
        print(f"Warning: {test_tar} not found")
    
    print("\n" + "="*50)
    print("Image extraction complete!")
    print("="*50)


if __name__ == '__main__':
    main()

