# PAMASA — Helbling Hackathon 2025
Members: Sandro Coti, Matteo De Liquori, Paloma Spiess
Image comparison and feature tracking challenge.
 
---
 
## Project Structure
 
```
hackathon/
├── main.py
├── load_data.py
├── processing.py
├── visualize.py
├── requirements.txt
├── .gitignore
└── data/               ← local only, not tracked by Git
    └── frames/
```
 
---
 
## Setup
 
**1. Create and activate the conda environment:**
```bash
conda create -n hackathon python=3.11
conda activate hackathon
```
 
**2. Install dependencies:**
```bash
pip install -r requirements.txt
```
 
**3. Place video data in the `data/` folder** (not tracked by Git — each person manages this locally)
 
**4. Run:**
```bash
python main.py
```
 
---
 
## File Overview
 
### `main.py`
Entry point for the pipeline. Controls the overall flow:
1. Extract frames from video
2. Load frames into memory
3. Run processing pipeline
4. Display results
Update the `VIDEO_PATH` variable at the top to match the actual filename once we have the data.
 
---
 
### `load_data.py`
Handles all data loading and frame extraction from video files.
 
| Function | Description |
|---|---|
| `extract_frames(video_path, output_dir)` | Reads a video file and saves each frame as a `.png` image |
| `load_frames(frames_dir)` | Loads all saved frames from a folder into a list of numpy arrays |
| `load_single_frame(path)` | Loads one image from a file path |
 
> Frames are numpy arrays (standard format for OpenCV and scikit-image). BGR color format by default.
 
---
 
### `processing.py`
Core image processing logic. Contains multiple approaches — we pick whichever fits the actual task tomorrow.
 
| Function | Description |
|---|---|
| `run_pipeline(frames)` | Main pipeline — loops through frame pairs and runs comparison. Edit this to wire together whichever methods we need |
| `compare_frames(frame_a, frame_b)` | Compares two frames — currently uses frame differencing, can be swapped out |
| `to_grayscale(frame)` | Converts a BGR frame to grayscale (needed by most algorithms) |
| `frame_difference(gray_a, gray_b)` | Computes absolute pixel difference between two frames — simplest comparison method |
| `threshold_mask(diff, threshold)` | Applies a binary threshold to a difference image to isolate changed regions |
| `similarity_score(gray_a, gray_b)` | Computes SSIM (Structural Similarity Index) — a perceptual similarity metric between 0 and 1 |
| `detect_features(frame)` | Detects ORB keypoints and descriptors in a frame — used for feature tracking |
| `match_features(desc_a, desc_b)` | Matches ORB descriptors between two frames to find corresponding features |
| `optical_flow(gray_a, gray_b)` | Computes dense optical flow — tracks how pixels move between frames |
 
---
 
### `visualize.py`
Helper functions for displaying images and results. Call these anywhere for quick visual debugging.
 
| Function | Description |
|---|---|
| `show_results(results)` | Displays the first 5 results from the pipeline as a sanity check |
| `show_image(image, title)` | Displays a single image using matplotlib |
| `show_side_by_side(image_a, image_b)` | Displays two images next to each other for comparison |
| `show_diff(diff)` | Displays a difference or mask image with a heatmap colormap |
| `draw_keypoints(frame, keypoints)` | Draws detected ORB keypoints on a frame |
| `draw_matches(frame_a, kp_a, frame_b, kp_b, matches)` | Draws feature matches between two frames |
| `plot_scores(scores)` | Plots a list of per-frame scores as a line graph over time |
 
---
 
## Key Libraries
 
| Library | Used for |
|---|---|
| `opencv-python` (cv2) | Video reading, frame extraction, feature detection, optical flow |
| `scikit-image` (skimage) | SSIM similarity scoring, filters |
| `numpy` | Array manipulation — all images are numpy arrays under the hood |
| `matplotlib` | Plotting and image display |
| `scipy` | Signal processing utilities |
| `scikit-learn` | ML utilities if needed |
| `Pillow` | General image loading/saving |
 
---
 
## Notes
- The `data/` folder is excluded from Git via `.gitignore` — each team member manages their own local copy of the data
- All frames are stored as numpy arrays in BGR format (OpenCV default) — convert to RGB when using matplotlib: `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`
- `processing.py` contains multiple approaches — `run_pipeline()` should be updated once we understand the exact task requirements