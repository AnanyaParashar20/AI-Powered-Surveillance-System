# FILE: src/utils/timers.py

"""
Timing utilities for performance monitoring.
"""

import time
from typing import Dict, Optional
from collections import deque


class FPSCounter:
    """Frames per second counter with moving average."""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize FPS counter.
        
        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.last_time = None
    
    def update(self) -> float:
        """
        Update FPS counter with current frame time.
        
        Returns:
            Current FPS estimate
        """
        current_time = time.time()
        
        if self.last_time is not None:
            frame_duration = current_time - self.last_time
            self.frame_times.append(frame_duration)
        
        self.last_time = current_time
        
        return self.get_fps()
    
    def get_fps(self) -> float:
        """
        Get current FPS estimate.
        
        Returns:
            FPS value
        """
        if not self.frame_times:
            return 0.0
        
        avg_duration = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_duration if avg_duration > 0 else 0.0
    
    def reset(self):
        """Reset FPS counter."""
        self.frame_times.clear()
        self.last_time = None


class PerformanceTimer:
    """Context manager for timing code sections."""
    
    def __init__(self, name: str = "Timer"):
        """
        Initialize performance timer.
        
        Args:
            name: Name for this timer
        """
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and calculate duration."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
    
    def get_duration(self) -> Optional[float]:
        """Get measured duration in seconds."""
        return self.duration
    
    def get_duration_ms(self) -> Optional[float]:
        """Get measured duration in milliseconds."""
        return self.duration * 1000 if self.duration is not None else None
    
    def __str__(self) -> str:
        """String representation of timer result."""
        if self.duration is not None:
            return f"{self.name}: {self.duration:.4f}s ({self.duration*1000:.2f}ms)"
        else:
            return f"{self.name}: Not measured"


class ProfilerManager:
    """Manages multiple timers for profiling different parts of pipeline."""
    
    def __init__(self):
        """Initialize profiler manager."""
        self.timers = {}
        self.cumulative_times = {}
        self.call_counts = {}
    
    def start_timer(self, name: str):
        """Start a named timer."""
        if name not in self.timers:
            self.cumulative_times[name] = 0.0
            self.call_counts[name] = 0
        
        self.timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """
        End a named timer and return duration.
        
        Args:
            name: Timer name
            
        Returns:
            Duration in seconds
        """
        if name not in self.timers:
            return 0.0
        
        duration = time.time() - self.timers[name]
        self.cumulative_times[name] += duration
        self.call_counts[name] += 1
        
        del self.timers[name]
        return duration
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get timing statistics.
        
        Returns:
            Dictionary with timing stats per timer name
        """
        stats = {}
        
        for name in self.cumulative_times:
            total_time = self.cumulative_times[name]
            count = self.call_counts[name]
            avg_time = total_time / count if count > 0 else 0.0
            
            stats[name] = {
                'total_time': total_time,
                'avg_time': avg_time,
                'call_count': count,
                'total_time_ms': total_time * 1000,
                'avg_time_ms': avg_time * 1000
            }
        
        return stats
    
    def print_stats(self):
        """Print timing statistics."""
        stats = self.get_stats()
        
        print("\n=== Performance Statistics ===")
        print(f"{'Timer Name':<20} {'Calls':<8} {'Total (ms)':<12} {'Avg (ms)':<12}")
        print("-" * 55)
        
        for name, data in stats.items():
            print(f"{name:<20} {data['call_count']:<8} "
                  f"{data['total_time_ms']:<12.2f} {data['avg_time_ms']:<12.2f}")
    
    def reset(self):
        """Reset all timers and statistics."""
        self.timers.clear()
        self.cumulative_times.clear()
        self.call_counts.clear()
    
    def timer(self, name: str):
        """
        Context manager for timing a code block.
        
        Args:
            name: Timer name
            
        Returns:
            Context manager that times the block
        """
        return _ProfilerTimer(self, name)


class _ProfilerTimer:
    """Context manager helper for ProfilerManager."""
    
    def __init__(self, profiler: ProfilerManager, name: str):
        self.profiler = profiler
        self.name = name
    
    def __enter__(self):
        self.profiler.start_timer(self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.end_timer(self.name)
