# FILE: test_installation.py

"""
Simple test script to verify surveillance system installation.
"""

import sys
import os
sys.path.insert(0, 'src')

def test_imports():
    """Test basic imports."""
    print("Testing imports...")
    
    try:
        import yaml
        print("✓ PyYAML")
    except ImportError:
        print("✗ PyYAML - install with: pip install pyyaml")
    
    try:
        import cv2
        print("✓ OpenCV")
    except ImportError:
        print("✗ OpenCV - install with: pip install opencv-python")
    
    try:
        import numpy as np
        print("✓ NumPy")
    except ImportError:
        print("✗ NumPy - install with: pip install numpy")
    
    try:
        import pandas as pd
        print("✓ Pandas")
    except ImportError:
        print("✗ Pandas - install with: pip install pandas")
    
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics YOLO")
    except ImportError:
        print("! Ultralytics YOLO - install with: pip install ultralytics (will use HOG fallback)")
    
    try:
        import streamlit as st
        print("✓ Streamlit")
    except ImportError:
        print("✗ Streamlit - install with: pip install streamlit")

def test_modules():
    """Test surveillance modules."""
    print("\nTesting surveillance modules...")
    
    try:
        from src.ingest import VideoReader
        print("✓ Video ingestion module")
    except ImportError as e:
        print(f"✗ Video ingestion module: {e}")
    
    try:
        from src.detect import DetectorManager
        print("✓ Detection module")
    except ImportError as e:
        print(f"✗ Detection module: {e}")
    
    try:
        from src.track import MultiTracker
        print("✓ Tracking module")
    except ImportError as e:
        print(f"✗ Tracking module: {e}")
    
    try:
        from src.events import LoiteringDetector, AbandonmentDetector, UnusualMovementDetector
        print("✓ Event detection modules")
    except ImportError as e:
        print(f"✗ Event detection modules: {e}")

def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✓ Configuration file loaded successfully")
        print(f"  - Target FPS: {config['video']['target_fps']}")
        print(f"  - Detector: {config['model']['detector']}")
        print(f"  - Loitering threshold: {config['events']['loitering_seconds']}s")
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")

def test_data_structure():
    """Test data directory structure."""
    print("\nTesting data structure...")
    
    directories = [
        'data/videos',
        'data/output',
        'data/avenue/test_videos',
        'data/avenue/raw_mat/testing_vol',
        'data/avenue/gt',
        'data/avenue/preds'
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"✓ {directory}")
        else:
            print(f"! {directory} - will be created when needed")

def main():
    """Run all tests."""
    print("Surveillance System Installation Test")
    print("=" * 50)
    
    test_imports()
    test_modules()
    test_config()
    test_data_structure()
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("\nNext steps:")
    print("1. Install missing dependencies from requirements.txt")
    print("2. Test with a sample video:")
    print("   python -m src.pipeline --video data/avenue/test_videos/01.avi --save_dir data/output")
    print("3. Launch dashboard:")
    print("   streamlit run src/dashboard_app.py")

if __name__ == "__main__":
    main()
