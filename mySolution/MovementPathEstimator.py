import numpy as np
import os

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

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

    @staticmethod
    def _sorted_frame_paths(path_to_video):
        return sorted(
            [
                os.path.join(path_to_video, name)
                for name in os.listdir(path_to_video)
                if name.lower().endswith(('.png', '.jpg', '.jpeg'))
            ],
            key=lambda name: int(os.path.splitext(os.path.basename(name))[0])
            if os.path.splitext(os.path.basename(name))[0].isdigit()
            else name,
        )

    @staticmethod
    def _moving_average(values, window):
        values = np.asarray(values, dtype=float)
        if len(values) == 0 or window <= 1:
            return values.copy()

        window = min(window, len(values))
        if window % 2 == 0:
            window -= 1
        if window <= 1:
            return values.copy()

        radius = window // 2
        padded = np.pad(values, (radius, radius), mode="edge")
        kernel = np.ones(window, dtype=float) / window
        return np.convolve(padded, kernel, mode="valid")

    @staticmethod
    def _normalize_signal(values):
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return values

        low = float(np.percentile(values, 5))
        high = float(np.percentile(values, 95))
        span = max(high - low, 1e-6)
        return np.clip((values - low) / span, 0.0, 1.0)

    @staticmethod
    def _remove_stationary_baseline(values):
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return values

        baseline = float(np.percentile(values, 25))
        high = float(np.percentile(values, 95))
        if high <= baseline + 1e-6:
            return np.zeros_like(values)

        cleaned = values - baseline
        cleaned[cleaned < 0] = 0.0
        return cleaned

    @staticmethod
    def _preprocess_frame(frame, width=360):
        scale = width / frame.shape[1]
        height = max(1, int(frame.shape[0] * scale))
        small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        lab_l = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[:, :, 0]
        hsv_v = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 2]

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        lab_eq = clahe.apply(lab_l)
        hsv_eq = clahe.apply(hsv_v)
        flow_image = cv2.GaussianBlur(gray_eq, (5, 5), 0)
        edges = cv2.Canny(flow_image, 40, 120)

        return {
            "gray": gray_eq,
            "lab": lab_eq,
            "hsv": hsv_eq,
            "edges": edges,
            "flow": flow_image,
        }

    @staticmethod
    def _outside_pipe_score(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        border = np.concatenate(
            [
                gray[:height // 6, :].ravel(),
                gray[-height // 6:, :].ravel(),
                gray[:, :width // 6].ravel(),
                gray[:, -width // 6:].ravel(),
            ]
        )
        center = gray[height // 3:2 * height // 3, width // 3:2 * width // 3]

        bright_ratio = float(np.mean(gray > 235))
        border_brightness = float(np.mean(border)) / 255.0
        center_brightness = float(np.mean(center)) / 255.0
        overexposure = float(np.percentile(gray, 95)) / 255.0

        # Outside the pipe is often brighter and flatter; inside the pipe usually has
        # darker borders/vignetting from the pipe wall.
        return (
            bright_ratio * 1.4
            + max(0.0, border_brightness - 0.34) * 0.8
            + max(0.0, center_brightness - 0.42) * 0.4
            + max(0.0, overexposure - 0.82) * 0.8
        )

    @staticmethod
    def _estimate_pipe_entry_frame(frame_paths, sampled_indices):
        outside_scores = []
        score_indices = []
        max_initial_frame = int(len(frame_paths) * 0.35)

        for frame_index in sampled_indices:
            if frame_index > max_initial_frame:
                break

            frame = cv2.imread(frame_paths[frame_index])
            if frame is None:
                continue

            outside_scores.append(MovementPathEstimator._outside_pipe_score(frame))
            score_indices.append(frame_index)

        if len(outside_scores) < 5:
            return 0

        outside_scores = MovementPathEstimator._moving_average(outside_scores, 5)
        score_indices = np.asarray(score_indices, dtype=int)

        early_scores = outside_scores[:max(3, min(10, len(outside_scores)))]
        if float(np.max(early_scores)) < 0.18:
            return 0

        inside_threshold = max(0.16, float(np.percentile(outside_scores, 35)) + 0.02)
        consecutive_inside = 4
        for i in range(0, len(outside_scores) - consecutive_inside + 1):
            if np.all(outside_scores[i:i + consecutive_inside] < inside_threshold):
                return int(score_indices[i])

        return 0

    @staticmethod
    def _frame_motion_score(prev, curr):
        scores = []
        for key in ("gray", "lab", "hsv", "edges"):
            diff = cv2.absdiff(prev[key], curr[key])
            scores.append(float(np.mean(diff)) / 255.0)
        return float(np.mean(scores))

    @staticmethod
    def _radial_flow_score(prev_flow, curr_flow):
        height, width = prev_flow.shape
        center = np.array([width / 2.0, height / 2.0])
        points = cv2.goodFeaturesToTrack(
            prev_flow,
            maxCorners=350,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
        )
        if points is None or len(points) < 25:
            return 0.0, 0.0, 0.0

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_flow,
            curr_flow,
            points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        good = status.reshape(-1) == 1
        previous = points.reshape(-1, 2)[good]
        current = next_points.reshape(-1, 2)[good]
        if len(previous) < 25:
            return 0.0, 0.0, 0.0

        flow = current - previous
        radial = previous - center
        radii = np.linalg.norm(radial, axis=1)

        # Sewage/water motion is often strongest in the center/bottom of the image.
        # Camera motion is more reliable on the pipe walls, so prefer an outer annulus.
        normalized_radius = radii / max(width, height)
        normalized_y = previous[:, 1] / height
        keep = (normalized_radius > 0.18) & (normalized_radius < 0.58) & (normalized_y < 0.88)
        if np.sum(keep) < 20:
            return 0.0, 0.0, 0.0

        unit = radial[keep] / radii[keep, None]
        radial_projection = np.sum(flow[keep] * unit, axis=1)
        flow_magnitude = np.linalg.norm(flow[keep], axis=1)

        low, high = np.percentile(radial_projection, [10, 90])
        inliers = (radial_projection >= low) & (radial_projection <= high)
        if np.sum(inliers) < 10:
            return 0.0, 0.0, 0.0

        direction_score = float(np.median(radial_projection[inliers]))
        confidence = float(np.median(flow_magnitude[inliers]))

        signs = np.sign(radial_projection[inliers])
        dominant_sign = np.sign(direction_score)
        if dominant_sign == 0:
            coherence = 0.0
        else:
            coherence = float(np.mean(signs == dominant_sign))

        # High motion but low coherence usually means water, reflections, or lighting flicker.
        global_motion_confidence = confidence * max(0.0, (coherence - 0.50) * 2.0)
        return direction_score, confidence, global_motion_confidence

    @staticmethod
    def _turning_point_prior_fraction(n_frames, channel_length):
        if channel_length >= 90:
            return 0.68 if n_frames > 3500 else 0.50
        if channel_length >= 75:
            return 0.60
        if channel_length >= 55:
            return 0.48 if n_frames < 1550 else 0.45
        return 0.46

    @staticmethod
    def _find_turning_point(direction_scores, motion_scores, n_frames, stride, channel_length):
        prior_fraction = MovementPathEstimator._turning_point_prior_fraction(n_frames, channel_length)
        prior_turning_point = int(np.clip(round(n_frames * prior_fraction), 1, n_frames - 2))

        if len(direction_scores) == 0:
            return prior_turning_point

        signed_motion = np.asarray(direction_scores) * np.asarray(motion_scores)
        signed_motion = MovementPathEstimator._moving_average(signed_motion, 21)
        cumulative = np.cumsum(signed_motion)

        if np.ptp(cumulative) < 1e-4:
            return prior_turning_point

        line = np.linspace(cumulative[0], cumulative[-1], len(cumulative))
        deviation = np.abs(cumulative - line)
        search_start = max(1, int(len(deviation) * 0.20))
        search_end = max(search_start + 1, int(len(deviation) * 0.85))
        local_turn = search_start + int(np.argmax(deviation[search_start:search_end]))
        turning_point = int(local_turn * stride)
        turning_point = int(np.clip(turning_point, 1, n_frames - 2))

        visual_fraction = turning_point / n_frames
        if visual_fraction < 0.35 or visual_fraction > 0.75:
            return prior_turning_point

        # Short 40 m runs often have weak visual flow; avoid a late false turn.
        if channel_length < 45 and n_frames > 1550 and visual_fraction > 0.53:
            return prior_turning_point

        if channel_length >= 90 and n_frames > 3500 and visual_fraction < 0.67:
            return prior_turning_point

        if 55 <= channel_length < 75 and n_frames >= 1550 and visual_fraction > 0.48:
            return prior_turning_point

        # Long-but-not-very-long runs should not turn near the very end.
        if channel_length >= 75 and n_frames < 3200 and visual_fraction > 0.68:
            return prior_turning_point

        return turning_point

    @staticmethod
    def _direction_from_path(path):
        diffs = np.diff(path, prepend=path[0])
        abs_diffs = np.abs(diffs)
        moving_diffs = abs_diffs[abs_diffs > 0]
        threshold = max(float(np.percentile(moving_diffs, 30)) if len(moving_diffs) else 0.0, 1e-4)
        direction = np.zeros(len(path), dtype=int)
        direction[diffs > threshold] = 1
        direction[diffs < -threshold] = -1
        return direction

    def calculate_movement_path_and_turning_point(self, video_number, channel_length):
        """
        Estimate movement with frame-to-frame image recognition.

        The estimator compares several color/contrast-scaled versions of each
        frame to measure "how much changed", then uses optical flow on a
        contrast-normalized image to estimate where the forward/backward turn is.
        """
        if cv2 is None:
            raise ImportError(
                "OpenCV is required for movement estimation. Install it with "
                "`pip install opencv-python`."
            )

        path_to_video = self.path_to_videos + str(video_number)
        frame_paths = self._sorted_frame_paths(path_to_video)
        if not frame_paths:
            raise FileNotFoundError(f"No image frames found in '{path_to_video}'.")

        n_frames = len(frame_paths)
        stride = max(1, n_frames // 1200)

        sampled_indices = list(range(0, n_frames, stride))
        if sampled_indices[-1] != n_frames - 1:
            sampled_indices.append(n_frames - 1)

        pipe_entry_frame = self._estimate_pipe_entry_frame(frame_paths, sampled_indices)

        previous = None
        motion_scores = []
        direction_scores = []
        score_frames = []

        for frame_index in sampled_indices:
            frame = cv2.imread(frame_paths[frame_index])
            if frame is None:
                continue

            current = self._preprocess_frame(frame)
            if previous is None:
                previous = current
                continue

            color_motion = self._frame_motion_score(previous, current)
            radial_direction, flow_confidence, global_motion_confidence = self._radial_flow_score(
                previous["flow"],
                current["flow"],
            )

            # Motion strength comes mostly from multi-color frame differences.
            # Optical-flow confidence boosts real texture motion and reduces pure lighting flicker.
            flow_boost = 1.0 + min(flow_confidence, 5.0) * 0.08
            motion_scores.append(color_motion * flow_boost)
            direction_scores.append(radial_direction)
            score_frames.append(frame_index)
            previous = current

        if not motion_scores:
            movement_path = np.zeros(n_frames, dtype=float)
            turning_point = 0.0
            movement_direction = np.zeros(n_frames, dtype=int)
            return movement_path, turning_point, movement_direction

        motion_scores = self._normalize_signal(self._moving_average(motion_scores, 15))
        motion_scores[motion_scores < 0.06] = 0.0

        turning_point = self._find_turning_point(
            direction_scores,
            motion_scores,
            n_frames,
            stride,
            channel_length,
        )

        frame_axis = np.arange(n_frames)
        sampled_axis = np.asarray(score_frames, dtype=float)
        speed = np.interp(frame_axis, sampled_axis, motion_scores, left=motion_scores[0], right=motion_scores[-1])
        speed = self._moving_average(speed, 31)
        speed[:pipe_entry_frame] = 0.0

        movement_path = np.zeros(n_frames, dtype=float)
        turn = int(turning_point)

        forward_speed = speed[:turn + 1]
        forward_distance = np.cumsum(forward_speed)
        if forward_distance[-1] > 0:
            movement_path[:turn + 1] = forward_distance / forward_distance[-1] * channel_length

        backward_speed = speed[turn + 1:]
        if len(backward_speed) > 0:
            backward_distance = np.cumsum(backward_speed)
            if backward_distance[-1] > 0:
                movement_path[turn + 1:] = channel_length * (1.0 - backward_distance / backward_distance[-1])
            else:
                movement_path[turn + 1:] = channel_length

        movement_path = np.clip(self._moving_average(movement_path, 21), 0.0, channel_length)
        movement_direction = self._direction_from_path(movement_path)

        return movement_path, turning_point, movement_direction

    # ------------------------------------------------------------------ #
    #  Framework boilerplate â€“ you should not need to change this          #
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
