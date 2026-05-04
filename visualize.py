import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_results(results):
    """Display a summary of processing results."""
    # TODO: adapt based on what results actually contain tomorrow
    for i, result in enumerate(results[:5]):   # show first 5 as a sanity check
        show_image(result, title=f"Result frame {i}")


def show_image(image, title="Image", cmap="gray"):
    """Display a single image with matplotlib."""
    plt.figure()
    plt.title(title)
    plt.axis("off")
    if len(image.shape) == 2:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


def show_side_by_side(image_a, image_b, title_a="Frame A", title_b="Frame B"):
    """Display two images side by side for comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB))
    axes[0].set_title(title_a)
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB))
    axes[1].set_title(title_b)
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


def show_diff(diff, title="Difference"):
    """Display a difference/mask image."""
    show_image(diff, title=title, cmap="hot")


def draw_keypoints(frame, keypoints):
    """Draw detected keypoints on a frame and display it."""
    output = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0))
    show_image(output, title="Keypoints")


def draw_matches(frame_a, kp_a, frame_b, kp_b, matches):
    """Draw feature matches between two frames."""
    output = cv2.drawMatches(frame_a, kp_a, frame_b, kp_b, matches[:20], None)
    show_image(output, title="Feature Matches")


def plot_scores(scores, title="Similarity Scores Over Time"):
    """Plot a list of per-frame scores as a line graph."""
    plt.figure(figsize=(10, 4))
    plt.plot(scores)
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.show()
