import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def run_pipeline(frames):
    """Main processing pipeline — called from main.py."""
    results = []

    for i in range(len(frames) - 1):
        frame_a = frames[i]
        frame_b = frames[i + 1]
        result = compare_frames(frame_a, frame_b)
        results.append(result)

    return results


def compare_frames(frame_a, frame_b):
    """Compare two frames and return a difference score or mask."""
    gray_a = to_grayscale(frame_a)
    gray_b = to_grayscale(frame_b)

    diff = frame_difference(gray_a, gray_b)
    # score = similarity_score(gray_a, gray_b)

    return diff


def to_grayscale(frame):
    """Convert a BGR frame to grayscale."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def frame_difference(gray_a, gray_b):
    """Compute absolute pixel difference between two grayscale frames."""
    return cv2.absdiff(gray_a, gray_b)


def threshold_mask(diff, threshold=30):
    """Apply a binary threshold to a difference image."""
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return mask


def similarity_score(gray_a, gray_b):
    """Compute structural similarity index (SSIM) between two frames."""
    score, diff = ssim(gray_a, gray_b, full=True)
    return score, diff


def detect_features(frame):
    """Detect ORB keypoints and descriptors in a frame."""
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(frame, None)
    return keypoints, descriptors


def match_features(desc_a, desc_b):
    """Match ORB descriptors between two frames."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc_a, desc_b)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def optical_flow(gray_a, gray_b):
    """Compute dense optical flow between two grayscale frames."""
    flow = cv2.calcOpticalFlowFarneback(
        gray_a, gray_b,
        None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0
    )
    return flow
