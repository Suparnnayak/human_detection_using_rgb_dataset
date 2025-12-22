"""
Evaluation module for SAR detection system
Provides metrics and evaluation utilities
"""

from .metrics import compute_map, compute_recall, evaluate_video

__all__ = ['compute_map', 'compute_recall', 'evaluate_video']

