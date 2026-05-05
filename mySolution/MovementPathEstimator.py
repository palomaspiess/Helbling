import numpy as np
import os
from PIL import Image
import cv2
from scipy.signal import savgol_filter

from MovementPath import MovementPath


class MovementPathEstimator:
    """
    Hackathon template: estimate the movement path and turning point
    of an object moving through a channel, based on video frames.

    The framework calls `execute_estimations()` which in turn calls
    `calculate_movement_path_and_turning_point()` for each video.
    Your job is to implement that one method.

    Inputs available inside `calculate_movement_path_and_turning_point`:
      - video_number   : int, which video (1-based)
      - channel_length : float, the physical length of the channel [m]
      - path_to_video  : str, folder containing the frame images (0.png, 1.png, ...)

    Outputs to return (as a tuple):
      - movement_path      : np.ndarray shape (N,)  position in [0, channel_length] per frame
      - turning_point      : float                  frame index where the object reverses
      - movement_direction : np.ndarray shape (N,)  +1 forward, -1 backward, 0 stationary per frame
    """

    def __init__(self, video_num_to_test, test_all_videos):
        self.channel_lengths = np.load('channel_lengths.npy')
        self.test_all_videos = test_all_videos
        self.video_num_to_test = video_num_to_test

        self.path_to_videos = 'frame_images/'
        # Always points to whichever folder this file lives in,
        # so the estimator works regardless of the folder name.
        self.current_folder = os.path.dirname(os.path.abspath(__file__)) + os.sep

        self.calculated_movement_paths = {}

    # ------------------------------------------------------------------ #
    #  TODO: implement your solution here                                  #
    # ------------------------------------------------------------------ #

    def calculate_movement_path_and_turning_point(self, video_number, channel_length):
        # --- Load frames and compute signals ---
        path_to_video = self.path_to_videos + str(video_number)
        frame_files = sorted(os.listdir(path_to_video), key=lambda x: int(os.path.splitext(x)[0]))
        num_frames = len(frame_files)

        brightness = []
        contrast = []
        motion = []
        prev = None

        orb = cv2.ORB_create(nfeatures=200)
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        for f in frame_files:
            img = cv2.imread(os.path.join(path_to_video, f), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

            brightness.append(np.mean(img))
            contrast.append(np.std(img))

            if prev is not None:
                kp1, desc1 = orb.detectAndCompute(prev, None)
                kp2, desc2 = orb.detectAndCompute(img, None)

                if desc1 is not None and desc2 is not None and len(desc1) > 5 and len(desc2) > 5:
                    matches = bf.match(desc1, desc2)
                    if len(matches) > 5:
                        # compute actual pixel displacement of matched keypoints
                        pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
                        pts2 = np.array([kp2[m.trainIdx].pt for m in matches])
                        displacements = np.linalg.norm(pts2 - pts1, axis=1)
                        # use median to ignore outlier matches
                        motion.append(np.median(displacements))
                    else:
                        motion.append(0.0)
                else:
                    motion.append(0.0)

            prev = img

        brightness = np.array(brightness)
        contrast = np.array(contrast)
        motion = np.array(motion)

        # --- Smooth signals ---
        brightness_smooth = savgol_filter(brightness, window_length=51, polyorder=2)
        contrast_smooth = savgol_filter(contrast, window_length=51, polyorder=2)
        motion_smooth = savgol_filter(motion, window_length=31, polyorder=2)
        motion_smooth = np.clip(motion_smooth, 0, None)

        # --- Detect tunnel entry ---
        brightness_threshold = np.percentile(brightness_smooth, 40)
        inside_tunnel = brightness_smooth < brightness_threshold
        tunnel_start = 0
        run_len = 0
        for i, val in enumerate(inside_tunnel):
            if val:
                run_len += 1
                if run_len >= 50:
                    tunnel_start = i - run_len + 1
                    break
            else:
                run_len = 0
        
        # --- Detect bad frames (water/obstruction = low contrast) ---
        # use a rolling window to be more aggressive about catching bad frames
        contrast_threshold = np.percentile(contrast_smooth, 25)
        bad_frame = contrast_smooth < contrast_threshold  # length num_frames


        # --- Detect stationary period ---
        # stationary = motion below 20th percentile of tunnel section
        tunnel_motion = motion_smooth[tunnel_start:]
        stationary_threshold = np.percentile(tunnel_motion, 20)
        stationary = motion_smooth < stationary_threshold
        stationary = np.concatenate([stationary, [False]])  # pad to num_frames

        # zero out motion during stationary and before tunnel
        motion_clean = motion_smooth.copy()
        motion_clean[stationary[:len(motion_clean)]] = 0
        motion_clean[:tunnel_start] = 0

        # --- Detect turning point ---
        # look for brightness minimum in middle 80% of tunnel
        margin = (num_frames - tunnel_start) // 10
        search_start = tunnel_start + margin
        search_end = num_frames - margin
        turning_point = float(search_start + np.argmin(brightness_smooth[search_start:search_end]))
        tp = int(turning_point)

        # --- Build position curve ---
        forward_motion = motion_clean[:tp]
        backward_motion = motion_clean[tp:]

        forward_cumsum = np.cumsum(forward_motion)
        backward_cumsum = np.cumsum(backward_motion)

        if forward_cumsum[-1] > 0:
            forward_position = forward_cumsum / forward_cumsum[-1] * channel_length
        else:
            forward_position = np.linspace(0, channel_length, tp)

        if len(backward_cumsum) > 0 and backward_cumsum[-1] > 0:
            backward_position = channel_length - (backward_cumsum / backward_cumsum[-1] * channel_length)
        else:
            backward_position = np.linspace(channel_length, 0, num_frames - tp)

        movement_path = np.concatenate([[0], forward_position, backward_position])
        movement_path = movement_path[:num_frames]

        # --- Smooth final position ---
        movement_path = savgol_filter(movement_path, window_length=31, polyorder=2)
        movement_path = np.clip(movement_path, 0, channel_length)

        # --- Movement direction ---
        movement_direction = np.zeros(num_frames)
        movement_direction[tunnel_start:tp] = 1
        movement_direction[tp:] = -1
        movement_direction[stationary[:num_frames]] = 0

        return movement_path, turning_point, movement_direction

    # ------------------------------------------------------------------ #
    #  Framework boilerplate – you should not need to change this          #
    # ------------------------------------------------------------------ #

    def execute_estimations(self):
        if self.test_all_videos:
            if not os.path.exists(self.path_to_videos):
                raise FileNotFoundError(f"The folder '{self.path_to_videos}' does not exist.")
            for entry in os.listdir(self.path_to_videos):
                if entry.isdigit():
                    self._run_single(int(entry))
        else:
            self._run_single(self.video_num_to_test)

    def _run_single(self, video_number):
        try:
            channel_length = self.channel_lengths[video_number - 1]
        except Exception:
            print("Cannot load channel length, using 100 m")
            channel_length = 100
        movement_path, turning_point, movement_direction = \
            self.calculate_movement_path_and_turning_point(int(video_number), channel_length)
        self.calculated_movement_paths[int(video_number)] = MovementPath(
            int(video_number), movement_path, movement_direction, turning_point
        )
