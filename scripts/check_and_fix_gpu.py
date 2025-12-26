"""
Check GPU availability and provide installation instructions
"""

import sys

print("="*60)
print("GPU Setup Check")
print("="*60)
print(f"Python: {sys.executable}")
print()

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print("\n[OK] GPU is ready for training!")
    else:
        print("\n[WARNING] GPU not available!")
        if '+cpu' in torch.__version__:
            print("Reason: CPU-only PyTorch installed")
            print("\nTo fix:")
            print("1. Activate venv: .\\venv\\Scripts\\Activate.ps1")
            print("2. Install CUDA PyTorch:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        else:
            print("Reason: CUDA not detected by PyTorch")
            print("Check NVIDIA drivers and CUDA installation")
        
        print("\nYou can still train on CPU (slower):")
        print("  python scripts/train.py ... --device cpu")
        
except ImportError:
    print("[ERROR] PyTorch not installed!")
    print("Install with: pip install torch torchvision torchaudio")

print("\n" + "="*60)

