"""
Validation script to compare bonus feature detections against ground-truth labels.
This helps you see how well your obstruction detection works.
"""

import sys
import numpy as np
from mySolution.MovementPathEstimator import MovementPathEstimator
from bonus_video_labels import ANNOTATIONS

def time_to_frame(time_str, fps=30):
    """Convert MM:SS time string to frame number (assuming 30 fps)."""
    parts = time_str.split(':')
    minutes = int(parts[0])
    seconds = int(parts[1])
    total_seconds = minutes * 60 + seconds
    return int(total_seconds * fps)

def validate_bonus_features(video_number):
    """Validate detected features against ground truth."""
    if video_number not in ANNOTATIONS:
        print(f"No labels for video {video_number}. Skipping...")
        return
    
    print(f"\n{'='*70}")
    print(f"Validation for Video {video_number}")
    print(f"{'='*70}\n")
    
    labels = ANNOTATIONS[video_number]
    estimator = MovementPathEstimator(video_num_to_test=video_number, test_all_videos=False)
    
    # Get detections
    try:
        blockage = estimator.detect_blockage_degree(video_number)
        obstructions = estimator.detect_obstruction_type(video_number)
        visibility = estimator.detect_visibility_score(video_number)
    except Exception as e:
        print(f"ERROR loading detections: {e}")
        return
    
    # Process labels
    print("GROUND TRUTH LABELS:")
    print("-" * 70)
    obstruction_frames = {}
    for label in labels:
        time_str = label['time']
        frame = time_to_frame(time_str)
        
        for key, val in label.items():
            if key == 'time' or key == 'comment':
                continue
            
            print(f"  Frame {frame:5d} (~{time_str}): {key:30s} severity={val}")
            if key not in obstruction_frames:
                obstruction_frames[key] = []
            obstruction_frames[key].append(frame)
    
    # Analyze detections
    print("\n\nDETECTION RESULTS:")
    print("-" * 70)
    
    # Check if high blockage frames align with obstruction labels
    print("\nBlockage Detection vs Obstruction Labels:")
    high_blockage_frames = np.where(blockage > 15)[0]
    labeled_frames = set()
    for frames in obstruction_frames.values():
        labeled_frames.update(frames)
    
    if len(labeled_frames) > 0:
        # Calculate overlap
        overlap = len(set(high_blockage_frames) & labeled_frames)
        precision = overlap / max(1, len(high_blockage_frames))
        recall = overlap / len(labeled_frames)
        print(f"  Labeled frames with obstructions: {len(labeled_frames)}")
        print(f"  Detected high-blockage frames (>15%): {len(high_blockage_frames)}")
        print(f"  Overlap: {overlap} frames")
        print(f"  Precision: {precision:.2%}")
        print(f"  Recall: {recall:.2%}")
    
    # Show detected obstruction types
    print("\nDetected Obstruction Types:")
    for obs_type, ranges in obstructions.items():
        if ranges:
            print(f"  {obs_type}: {len(ranges)} region(s)")
    
    # Check visibility on "No Vision" frames
    no_vision_frames = []
    for label in labels:
        if 'No Vision' in str(label):
            frame = time_to_frame(label['time'])
            no_vision_frames.append(frame)
    
    if no_vision_frames:
        print(f"\nVisibility Detection on No Vision Frames:")
        print(f"  Labeled no-vision frames: {no_vision_frames}")
        avg_visibility = np.mean([visibility[f] for f in no_vision_frames if f < len(visibility)])
        print(f"  Average detected visibility: {avg_visibility:.2f}")
        print(f"  Expected: low visibility (<0.3)")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("BONUS FEATURE VALIDATION")
    print("="*70)
    print("\nComparing detected features against ground-truth labels from")
    print("bonus_video_labels.py...\n")
    
    # Validate all bonus videos
    for video_num in sorted(ANNOTATIONS.keys()):
        try:
            validate_bonus_features(video_num)
        except Exception as e:
            print(f"Error validating video {video_num}: {e}\n")
