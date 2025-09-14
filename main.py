# Roll Number: BSAI23038
# Name: Umer Siddiqui
# Assignment: 01
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import deque
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


def compute_variance(frames: np.ndarray, mean_frame: np.ndarray) -> np.ndarray:
    """
    Computes variance for each pixel across frames.

    Input:
        frames (np.ndarray): Array of shape [F, H, W]
        mean_frame (np.ndarray): Mean frame of shape [H, W]

    Output:
        np.ndarray: Variance frame of shape [H, W]

    """
    F, H, W = frames.shape
    variance = np.zeros((H, W), dtype=float)

    for i in range(H):
        for j in range(W):
            sum_sq = 0
            for f in range(F):
                diff = frames[f, i, j] - mean_frame[i, j]
                sum_sq += diff ** 2
            variance[i, j] = sum_sq / F

    return variance.astype(np.uint8)

def compute_mask(frame, mean_frame, variance_frame, threshold=5.0):
    """
    Computes motion mask using Mahalanobis distance.
    
    Parameters:
    frame (np.ndarray): Current frame [H, W]
    mean_frame (np.ndarray): Mean background [H, W]
    variance_frame (np.ndarray): Variance background [H, W]
    threshold (float): Distance threshold
    
    Returns:
    np.ndarray: Binary mask where 1 indicates motion
    """
    # to avoid divide by zero
    epsilon = 1e-6

    # absolute difference between frame and background
    diff = np.abs(frame - mean_frame)

    # standard deviation from variance
    std = np.sqrt(variance_frame + epsilon)

    # Mahalanobis distance
    M = diff / std

    # thresholding
    mask = (M > threshold).astype(np.uint8)

    return mask

def create_kernel(kernel_size=3):
    """Create a square kernel for morphological operations"""
    return np.ones((kernel_size, kernel_size), dtype=np.uint8)
def erode(mask, kernel, anchor, iterations=1):
    """
    Performs erosion on binary mask.
    
    Parameters:
    mask (np.ndarray): Binary mask [H, W]
    kernel (np.ndarray): Structuring element
    anchor: a tuple (r, c)
    iterations (int): Number of times to apply operation
    
    Returns:
    np.ndarray: Eroded mask
    """
    H, W = mask.shape
    kH, kW = kernel.shape
    aH, aW = anchor  # anchor row, col

    result = mask.copy()
    for _ in range(iterations):
        padded = np.pad(result, ((aH, kH - aH - 1), (aW, kW - aW - 1)), 
                        mode='constant', constant_values=0)
        new = np.zeros_like(result)

        for i in range(H):
            for j in range(W):
                region = padded[i:i+kH, j:j+kW]
                # all neighbors under kernel must be 1
                if np.all(region[kernel == 1] == 1):
                    new[i, j] = 1
        result = new
    return result
def dilate(mask, kernel, anchor, iterations=1):
    """
    Performs dilation on binary mask.
    
    Parameters:
    mask (np.ndarray): Binary mask [H, W]
    kernel (np.ndarray): Structuring element
    anchor: a tuple (r, c)
    iterations (int): Number of times to apply operation
    
    Returns:
    np.ndarray: Dilated mask
    """
    H, W = mask.shape
    kH, kW = kernel.shape
    aH, aW = anchor  # anchor row, col

    result = mask.copy()
    for _ in range(iterations):
        padded = np.pad(result, ((aH, kH - aH - 1), (aW, kW - aW - 1)), 
                        mode='constant', constant_values=0)
        new = np.zeros_like(result)

        for i in range(H):
            for j in range(W):
                region = padded[i:i+kH, j:j+kW]
                # at least one neighbor under kernel must be 1
                if np.any(region[kernel == 1] == 1):
                    new[i, j] = 1
        result = new
    return result
def morphological_operations(mask, kernel_size=3):
    """
    Applies opening (erosion followed by dilation) to remove noise.
    
    Parameters:
    mask (np.ndarray): Binary mask [H, W]
    kernel_size (int): Size of structuring element
    
    Returns:
    np.ndarray: Cleaned mask
    """
    kernel = create_kernel(kernel_size)
    # Apply opening: erosion followed by dilation
    eroded = erode(mask, kernel, (1, 1), iterations=1)
    cleaned = dilate(eroded, kernel, (1, 1), iterations=1)

def find_connected_components(mask, connectivity=8):
    """
    Finds connected components in binary mask using BFS.
    
    Parameters:
    mask (np.ndarray): Binary mask [H, W]
    connectivity (int): 4 or 8 connectivity
    
    Returns:
    tuple: (num_components, labeled_mask, component_info)
    """
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    labeled_mask = np.zeros_like(mask, dtype=np.int32)

    # neighbor directions
    if connectivity == 4:
        directions = [(-1,0),(1,0),(0,-1),(0,1)]
    else:  # 8-connectivity
        directions = [(-1,0),(1,0),(0,-1),(0,1),
                      (-1,-1),(-1,1),(1,-1),(1,1)]

    component_info = []
    comp_id = 0

    for i in range(H):
        for j in range(W):
            if mask[i, j] == 1 and not visited[i, j]:
                comp_id += 1
                q = deque([(i,j)])
                visited[i, j] = True
                labeled_mask[i, j] = comp_id

                # stats
                pixels = []
                min_r, max_r = i, i
                min_c, max_c = j, j

                while q:
                    r, c = q.popleft()
                    pixels.append((r,c))

                    # update bbox
                    min_r, max_r = min(min_r,r), max(max_r,r)
                    min_c, max_c = min(min_c,c), max(max_c,c)

                    for dr, dc in directions:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if mask[nr, nc] == 1 and not visited[nr, nc]:
                                visited[nr, nc] = True
                                labeled_mask[nr, nc] = comp_id
                                q.append((nr, nc))

                # compute stats
                area = len(pixels)
                centroid_r = sum([p[0] for p in pixels]) / area
                centroid_c = sum([p[1] for p in pixels]) / area

                component_info.append({
                    "id": comp_id,
                    "area": area,
                    "centroid": (centroid_r, centroid_c),
                    "bbox": (min_r, min_c, max_r, max_c)
                })

    return comp_id, labeled_mask, component_info

# Example Usage:
input_folder = "input/snowFall_frames"
frames = read_frames(input_folder)
print(f"Frames shape: {frames.shape}")  # [F, H, W] 
plot_frames(frames, num_frames=10, save_name="report.pdf")
