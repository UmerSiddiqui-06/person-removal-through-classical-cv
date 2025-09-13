# Roll Number: BSAI23038
# Name: Umer Siddiqui
# Assignment: 01
import os
import numpy as np
from PIL import Image

def rgb_to_gray(img: np.ndarray) -> np.ndarray:
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

    # write your code here
    # ...
    """
    frames = []
    file_list = sorted(os.listdir(input_folder))

    for filename in file_list:
        if filename.endswith(".png"):
            path = os.path.join(input_folder, filename)
            img = Image.open(path)
            img = np.array(img)  # now [H, W, 3]
            gray = rgb_to_gray(img)  # manual conversion
            frames.append(gray)

    frames = np.stack(frames, axis=0)  # [F, H, W]
    return frames

# Example Usage:
input_folder = "input/snowFall_frames"
frames = read_frames(input_folder)
print(f"Frames shape: {frames.shape}")  # [F, H, W]