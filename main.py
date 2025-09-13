# Roll Number: BSAI23038
# Name: Umer Siddiqui
# Assignment: 01
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
def _rgb_to_gray(img: np.ndarray) -> np.ndarray:
    # extract R, G, B
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    # apply formula
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    return gray.astype(np.uint8)
def read_frames(input_folder: str) -> np.ndarray:
    """
    Reads frames without using OpenCV.

    Input:
        input_folder (str): path to the folder containing all frames

    Output:
        np.ndarray: Array of frames with shape
        [F, H, W, C] for color images where
        F: total no of frames, H: Height, W: Width, C: Channels
        or [F, H, W] for grayscale

    """
    frames = []
    file_list = sorted(os.listdir(input_folder))

    for filename in file_list:
        if filename.endswith(".png"):
            path = os.path.join(input_folder, filename)
            img = Image.open(path)
            img = np.array(img)  # now [H, W, 3]
            gray = _rgb_to_gray(img)  # manual conversion
            frames.append(gray)

    frames = np.stack(frames, axis=0)  # [F, H, W]
    return frames



def plot_frames(frames: np.ndarray, num_frames: int, save_name: str) -> None:
    """
    Plots and saves multiple frames in a single image.

    Input:
        frames (list): List of frame arrays
        num_frames (int): Number of frames to display
        save_name (str): Name for the saved image file
    """
     # select frames to plot
    selected_frames = frames[:num_frames]

    # make grid: rows x cols
    cols = 5
    rows = (num_frames + cols - 1) // cols  # ceil division

    plt.figure(figsize=(15, 3 * rows))

    for i, frame in enumerate(selected_frames):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(frame, cmap="gray")
        plt.axis("off")
        plt.title(f"Frame {i}")

    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches="tight")
    plt.close()

def compute_mean_frames(frames:np.ndarray) -> np.ndarray:
    """
    Compute the mean across all frames for each pixel in a 2D array.

    Parameters:
        arr : numpy.ndarray
            Input array of shape [F, H, W]
            F: Total number of frames
            H: Height of an image
            W: Width of an image

    Returns:
        numpy.ndarray : 2D array of shape [H, W] with mean values
    """
    F, H, W = frames.shape
    mean = np.zeros((H, W), dtype=float)

    for i in range(H):
        for j in range(W):
            sum_val = 0
            for f in range(F):
                sum_val += frames[f, i, j]  
            mean[i, j] = sum_val / F

    return mean.astype(np.uint8) 

# Example Usage:
input_folder = "input/snowFall_frames"
frames = read_frames(input_folder)
print(f"Frames shape: {frames.shape}")  # [F, H, W] 
plot_frames(frames, num_frames=10, save_name="report.pdf")
