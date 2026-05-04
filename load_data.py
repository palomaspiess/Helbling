import cv2
import os
import numpy as np


def extract_frames(video_path, output_dir):
    """Extract all frames from a video file and save as images."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{output_dir}/frame_{i:05d}.png", frame)
        i += 1
    cap.release()
    print(f"Extracted {i} frames to {output_dir}")


def load_frames(frames_dir):
    """Load all frames from a directory into a list of numpy arrays."""
    files = sorted(os.listdir(frames_dir))
    frames = []
    for f in files:
        if f.endswith(".png") or f.endswith(".jpg"):
            img = cv2.imread(os.path.join(frames_dir, f))
            frames.append(img)
    print(f"Loaded {len(frames)} frames")
    return frames


def load_single_frame(path):
    """Load a single image from a file path."""
    return cv2.imread(path)
