"""
Test script to validate bonus obstruction detection features.
Run this to see visibility scores, blockage degrees, and detected obstruction types.
"""

import sys
import numpy as np
from mySolution.MovementPathEstimator import MovementPathEstimator

def test_bonus_features(video_number):
    """Test all bonus detection features on a single video."""
    print(f"\n{'='*60}")
    print(f"Testing Bonus Features for Video {video_number}")
    print(f"{'='*60}\n")
    
    estimator = MovementPathEstimator(video_num_to_test=video_number, test_all_videos=False)
    
    # Test visibility score
    print("1. VISIBILITY SCORE")
    print("-" * 60)
    try:
        visibility = estimator.detect_visibility_score(video_number)
        print(f"   Frames analyzed: {len(visibility)}")
        print(f"   Mean visibility: {np.mean(visibility):.2f}")
        print(f"   Min visibility: {np.min(visibility):.2f}")
        print(f"   Max visibility: {np.max(visibility):.2f}")
        poor_visibility_frames = np.where(visibility < 0.5)[0]
        print(f"   Frames with poor visibility (<0.5): {len(poor_visibility_frames)}")
        if len(poor_visibility_frames) > 0:
            print(f"      Ranges: {poor_visibility_frames[:10]}...")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test blockage degree
    print("\n2. BLOCKAGE DEGREE")
    print("-" * 60)
    try:
        blockage = estimator.detect_blockage_degree(video_number)
        print(f"   Frames analyzed: {len(blockage)}")
        print(f"   Mean blockage: {np.mean(blockage):.2f}%")
        print(f"   Min blockage: {np.min(blockage):.2f}%")
        print(f"   Max blockage: {np.max(blockage):.2f}%")
        blocked_frames = np.where(blockage > 15)[0]
        print(f"   Frames with blockage >15%: {len(blocked_frames)}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test obstruction type detection
    print("\n3. OBSTRUCTION TYPES")
    print("-" * 60)
    try:
        obstructions = estimator.detect_obstruction_type(video_number)
        for obs_type, ranges in obstructions.items():
            if ranges:
                print(f"   {obs_type}: {len(ranges)} region(s)")
                for start, end in ranges[:3]:  # Show first 3
                    print(f"      - Frames {start} to {end} ({end-start+1} frames)")
                if len(ranges) > 3:
                    print(f"      ... and {len(ranges)-3} more")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test dirty locations
    print("\n4. DIRTY LOCATIONS (Blockage >15%)")
    print("-" * 60)
    try:
        dirty = estimator.detect_dirty_locations(video_number, blockage_threshold=15.0)
        dirty_ranges = dirty['dirty_frames']
        print(f"   Dirty frame ranges: {len(dirty_ranges)}")
        for start, end in dirty_ranges[:5]:  # Show first 5
            print(f"      - Frames {start} to {end} ({end-start+1} frames)")
        if len(dirty_ranges) > 5:
            print(f"      ... and {len(dirty_ranges)-5} more")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    # Test on a bonus video (12-15) to validate against labels
    print("Testing bonus features on bonus videos (12-15)...")
    
    for video_num in [12, 13, 14, 15]:
        try:
            test_bonus_features(video_num)
        except FileNotFoundError:
            print(f"Video {video_num} frames not found. Skipping...")
        except Exception as e:
            print(f"Error testing video {video_num}: {e}")
