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
        # --- Load frames as grayscale numpy arrays ---
        path_to_video = self.path_to_videos + str(video_number)
        frame_files = sorted(os.listdir(path_to_video), key=lambda x: int(os.path.splitext(x)[0]))

        frames = []
        for f in frame_files:
            img = cv2.imread(os.path.join(path_to_video, f), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)  # half resolution
            frames.append(img)

        num_frames = len(frames)

        # --- Optical flow: measure motion magnitude between consecutive frames ---
        motion = []
        for i in range(num_frames - 1):
            flow = cv2.calcOpticalFlowFarneback(
                frames[i], frames[i + 1], None,
                pyr_scale=0.5, levels=2, winsize=9,
                iterations=2, poly_n=5, poly_sigma=1.2, flags=0
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion.append(np.mean(magnitude))

        motion = np.array(motion)

        # --- Smooth motion signal ---
        motion_smooth = savgol_filter(motion, window_length=21, polyorder=2)
        motion_smooth = np.clip(motion_smooth, 0, None)  # no negative values

        # --- Detect stationary zones (where motion drops below threshold) ---
        threshold = np.percentile(motion_smooth, 20)  # bottom 20% = likely stationary
        stationary = motion_smooth < threshold
        stationary = np.concatenate([stationary, [False]])


        # --- Find turning point (lowest motion in middle 80% of video) ---
        margin = num_frames // 10
        search_region = motion_smooth[margin:-margin]
        turning_point = float(np.argmin(search_region) + margin)
        tp = int(turning_point)

        # --- Build position curve from cumulative motion ---
        forward_motion = motion_smooth[:tp].copy()
        backward_motion = motion_smooth[tp:].copy()

        # zero out stationary frames so they don't accumulate position
        forward_motion[stationary[:len(forward_motion)]] = 0
        backward_motion[stationary[tp:tp + len(backward_motion)]] = 0

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

        # --- Final smooth of position curve ---
        movement_path = savgol_filter(movement_path, window_length=21, polyorder=2)
        movement_path = np.clip(movement_path, 0, channel_length)

        # --- Movement direction (0 when stationary) ---
        movement_direction = np.concatenate([
            np.ones(tp),
            -np.ones(num_frames - tp)
        ])
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
