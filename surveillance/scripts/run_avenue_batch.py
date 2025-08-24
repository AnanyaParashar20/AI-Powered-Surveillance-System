# FILE: scripts/run_avenue_batch.py

"""
Batch processing script for Avenue dataset videos.
"""

import argparse
import os
import sys
import logging
from pathlib import Path
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)


def process_single_video(video_path: str, config_path: str, 
                        output_base_dir: str) -> dict:
    """
    Process a single video file.
    
    Args:
        video_path: Path to video file
        config_path: Path to configuration file
        output_base_dir: Base output directory
        
    Returns:
        Processing result dictionary
    """
    video_name = Path(video_path).stem
    video_output_dir = os.path.join(output_base_dir, video_name)
    
    # Create Avenue prediction ID
    avenue_pred_id = video_name
    
    logger.info(f"Processing {video_name}...")
    
    try:
        # Import and run pipeline
        from src.pipeline import SurveillancePipeline
        
        pipeline = SurveillancePipeline(config_path)
        result = pipeline.process_video(
            video_path=video_path,
            save_dir=video_output_dir,
            show_video=False,
            save_video=False,
            avenue_pred_id=avenue_pred_id
        )
        
        logger.info(f"Completed {video_name}: {result['frames_processed']} frames")
        return {
            'video_name': video_name,
            'status': 'success',
            'result': result
        }
        
    except Exception as e:
        logger.error(f"Failed to process {video_name}: {e}")
        return {
            'video_name': video_name, 
            'status': 'error',
            'error': str(e)
        }


def run_batch_processing(video_dir: str, config_path: str, 
                        output_dir: str, max_workers: int = 1) -> dict:
    """
    Run batch processing on all videos in directory.
    
    Args:
        video_dir: Directory containing video files
        config_path: Path to configuration file
        output_dir: Output directory
        max_workers: Maximum number of parallel workers
        
    Returns:
        Batch processing results
    """
    if not os.path.exists(video_dir):
        raise ValueError(f"Video directory not found: {video_dir}")
    
    # Find all video files
    video_extensions = ['.avi', '.mp4', '.mov', '.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(video_dir).glob(f"*{ext}"))
    
    if not video_files:
        raise ValueError(f"No video files found in {video_dir}")
    
    logger.info(f"Found {len(video_files)} videos to process")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process videos
    results = []
    start_time = time.time()
    
    if max_workers == 1:
        # Sequential processing
        for video_file in video_files:
            result = process_single_video(str(video_file), config_path, output_dir)
            results.append(result)
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_video = {
                executor.submit(process_single_video, str(video_file), config_path, output_dir): str(video_file)
                for video_file in video_files
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_video):
                video_path = future_to_video[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Processing failed for {video_path}: {e}")
                    results.append({
                        'video_name': Path(video_path).stem,
                        'status': 'error',
                        'error': str(e)
                    })
    
    # Calculate summary statistics
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    
    summary = {
        'total_videos': len(video_files),
        'successful': successful,
        'failed': failed,
        'total_time': total_time,
        'avg_time_per_video': total_time / len(video_files),
        'results': results
    }
    
    return summary


def copy_avenue_videos(avenue_base_dir: str, target_dir: str):
    """
    Copy Avenue dataset videos to target directory.
    
    Args:
        avenue_base_dir: Base Avenue dataset directory
        target_dir: Target directory for videos
    """
    testing_videos_dir = os.path.join(avenue_base_dir, 'testing_videos')
    
    if not os.path.exists(testing_videos_dir):
        logger.warning(f"Avenue testing videos not found: {testing_videos_dir}")
        return
    
    os.makedirs(target_dir, exist_ok=True)
    
    # Find all .avi files
    video_files = list(Path(testing_videos_dir).glob("*.avi"))
    
    logger.info(f"Copying {len(video_files)} videos from Avenue dataset...")
    
    for video_file in video_files:
        target_path = os.path.join(target_dir, video_file.name)
        if not os.path.exists(target_path):
            import shutil
            shutil.copy2(str(video_file), target_path)
            logger.debug(f"Copied {video_file.name}")
    
    logger.info(f"Avenue videos copied to: {target_dir}")


def main():
    """Main batch processing script."""
    parser = argparse.ArgumentParser(description='Batch process Avenue dataset videos')
    
    parser.add_argument('--avenue_base_dir',
                       default='Avenue Dataset',
                       help='Base Avenue dataset directory')
    parser.add_argument('--video_dir', 
                       default='data/avenue/test_videos',
                       help='Directory containing videos to process')
    parser.add_argument('--output_dir',
                       default='data/avenue/preds',
                       help='Output directory for predictions') 
    parser.add_argument('--config',
                       default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--workers', type=int, default=1,
                       help='Number of parallel workers')
    parser.add_argument('--copy_videos', action='store_true',
                       help='Copy videos from Avenue dataset first')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Copy videos if requested
        if args.copy_videos:
            copy_avenue_videos(args.avenue_base_dir, args.video_dir)
        
        # Run batch processing
        logger.info("Starting batch processing...")
        summary = run_batch_processing(
            video_dir=args.video_dir,
            config_path=args.config,
            output_dir=args.output_dir,
            max_workers=args.workers
        )
        
        # Print summary
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        print(f"Total videos: {summary['total_videos']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Total time: {summary['total_time']:.1f} seconds")
        print(f"Average time per video: {summary['avg_time_per_video']:.1f} seconds")
        
        if summary['failed'] > 0:
            print(f"\nFailed videos:")
            for result in summary['results']:
                if result['status'] == 'error':
                    print(f"  {result['video_name']}: {result['error']}")
        
        print(f"\nPrediction files saved to: {args.output_dir}")
        print("Run evaluation with:")
        print(f"  python -m src.eval.avenue_eval --gt_dir data/avenue/gt --pred_dir {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
