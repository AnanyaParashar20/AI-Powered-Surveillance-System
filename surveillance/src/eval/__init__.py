# FILE: src/eval/__init__.py

"""
Evaluation utilities for surveillance system.
"""

from .avenue_eval import evaluate_dataset, evaluate_video, calculate_frame_accuracy

__all__ = ['evaluate_dataset', 'evaluate_video', 'calculate_frame_accuracy']
