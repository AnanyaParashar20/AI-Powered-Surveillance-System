# FILE: USAGE_EXAMPLES.md

# Surveillance System Usage Examples

This document provides practical examples of using the surveillance anomaly detection system.

## Quick Start

### 1. Install Dependencies
```bash
cd surveillance
pip install -r requirements.txt
```

### 2. Process a Single Video
```bash
python -m src.pipeline --video data/avenue/test_videos/01.avi --save_dir data/output
```

### 3. Launch Dashboard
```bash
streamlit run src/dashboard_app.py
```

## Avenue Dataset Workflow

### Step 1: Convert Ground Truth
Convert Avenue .mat files to JSON format:
```bash
python scripts/convert_avenue_mat.py
```

Options:
- `--raw_mat_dir`: Directory with .mat files (default: `data/avenue/raw_mat/testing_vol`)
- `--output_dir`: Output directory for JSON files (default: `data/avenue/gt`)
- `--area_thr`: Minimum area threshold (default: 150)
- `--transpose`: Transpose dimensions if needed

### Step 2: Batch Process Videos
Process all Avenue test videos:
```bash
python scripts/run_avenue_batch.py
```

Options:
- `--video_dir`: Video directory (default: `data/avenue/test_videos`)
- `--output_dir`: Predictions output (default: `data/avenue/preds`)
- `--workers`: Parallel workers (default: 1)

### Step 3: Evaluate Results
Compare predictions against ground truth:
```bash
python -m src.eval.avenue_eval --gt_dir data/avenue/gt --pred_dir data/avenue/preds
```

## Configuration Examples

### Basic Configuration
Edit `config.yaml` for your needs:

```yaml
video:
  target_fps: 15        # Process at 15 FPS
  max_side: 960        # Resize to max 960px side
  frame_skip: 0        # Skip frames at start

model:
  detector: yolo       # Use YOLOv8 detection
  conf_thres: 0.35     # Confidence threshold

events:
  loitering_seconds: 20    # 20s loitering threshold
  unusual:
    z_thresh: 3.0         # Z-score threshold for unusual movement
```

### ROI Configuration
Define region of interest:

```yaml
roi:
  use_full_frame: false
  polygon: [[100, 100], [500, 100], [500, 400], [100, 400]]  # Rectangle
```

## Advanced Usage

### Custom Event Detection
Run with custom event thresholds:

```bash
python -m src.pipeline \
  --video input.mp4 \
  --save_dir output \
  --config custom_config.yaml
```

### Batch Processing with Parallel Workers
Process multiple videos faster:

```bash
python scripts/run_avenue_batch.py --workers 4
```

### Save Processed Video
Save annotated video output:

```bash
python -m src.pipeline \
  --video input.mp4 \
  --save_dir output \
  --save_video
```

### Display Real-time Processing
Show video during processing:

```bash
python -m src.pipeline \
  --video input.mp4 \
  --save_dir output \
  --show
```

## Output Files

After processing, you'll find:

```
data/output/
├── events_video_timestamp.csv    # Alert log
└── frames/                       # Alert screenshots
    ├── video_frame000123_loitering_track1.jpg
    └── video_frame000456_abandonment_track3.jpg
```

### CSV Format
The events CSV contains:
- `frame_idx`: Frame number
- `time_sec`: Timestamp in seconds  
- `type`: Event type (loitering, abandonment, unusual)
- `track_id`: Associated track ID
- `score`: Alert score (0-100)
- `bbox`: Bounding box coordinates
- `contributors`: Additional event data
- `image_path`: Path to alert screenshot

## Dashboard Features

The Streamlit dashboard provides:

1. **Alert Summary**: Total counts and statistics
2. **Timeline View**: Events over time
3. **Filtering**: By event type, score range, time range
4. **Image Gallery**: Alert screenshots with metadata
5. **Export**: Download filtered results as CSV

Access at: http://localhost:8501

## Troubleshooting

### Common Issues

**YOLOv8 not working:**
- Fallback to HOG detector automatically
- Set `detector: hog` in config.yaml

**Low detection accuracy:**
- Adjust `conf_thres` in config
- Try different YOLO model: `yolov8s.pt`, `yolov8m.pt`

**Performance issues:**
- Reduce `target_fps`
- Increase `max_side` to smaller value
- Use fewer parallel workers

**Avenue dataset evaluation:**
- Check .mat file dimensions with `--transpose` flag
- Adjust `--area_thr` for smaller/larger objects
- Verify JSON file format with sample inspection

## Performance Optimization

### For Real-time Processing
```yaml
video:
  target_fps: 10       # Lower FPS
  max_side: 640        # Smaller frames

model:
  detector: hog        # Faster detection
  conf_thres: 0.5      # Higher threshold

events:
  unusual:
    flow_history: 30   # Shorter history
```

### For High Accuracy
```yaml
video:
  target_fps: 25       # Higher FPS
  max_side: 1280       # Larger frames

model:
  detector: yolo
  yolo_weights: yolov8m.pt  # Better model
  conf_thres: 0.25     # Lower threshold

events:
  unusual:
    flow_history: 120  # Longer history
    z_thresh: 2.5      # More sensitive
```
