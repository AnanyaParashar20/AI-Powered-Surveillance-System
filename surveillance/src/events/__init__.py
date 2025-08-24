# FILE: src/events/__init__.py

"""
Event detection modules for surveillance system.
"""

from .loitering import LoiteringDetector
from .abandonment import AbandonmentDetector
from .unusual import UnusualMovementDetector

__all__ = ['LoiteringDetector', 'AbandonmentDetector', 'UnusualMovementDetector']
