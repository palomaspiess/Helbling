import load_data
import processing
import visualize


def main():
    # --- Config ---
    VIDEO_PATH = "data/video.mp4"       # update with actual filename tomorrow
    FRAMES_DIR = "data/frames/"

    # --- Pipeline ---
    load_data.extract_frames(VIDEO_PATH, FRAMES_DIR)

    frames = load_data.load_frames(FRAMES_DIR)

    results = processing.run_pipeline(frames)

    visualize.show_results(results)


if __name__ == "__main__":
    main()
