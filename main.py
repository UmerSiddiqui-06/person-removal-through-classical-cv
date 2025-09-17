# Roll Number: BSAI23038
# Name: Umer Siddiqui
# Assignment: 01
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from collections import deque
import argparse
import zlib, struct
import random
from concurrent.futures import ProcessPoolExecutor

def _rgb_to_gray(img: np.ndarray) -> np.ndarray:
    # extract R, G, B
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    # apply formula
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    return gray.astype(np.float32)
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
    mean = np.zeros((H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            sum_val = 0.0
            for f in range(F):
                sum_val += float(frames[f, i, j])  
            mean[i, j] = sum_val / F


    return mean


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
    variance = np.zeros((H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            sum_sq = 0.0
            for f in range(F):
                diff = float(frames[f, i, j]) - float(mean_frame[i, j])
                sum_sq += diff ** 2

            variance[i, j] = sum_sq / F

    return variance

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
    frame_f = frame.astype(np.float32)
    # absolute difference between frame and background
    diff = np.abs(frame_f- mean_frame)

    # standard deviation from variance
    std = np.sqrt(variance_frame + epsilon)

    # Mahalanobis distance
    M = diff / std

    # thresholding
    mask = (M > threshold).astype(np.uint8)

    return mask

def create_kernel(kernel_size=5):
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
def morphological_operations(mask, kernel_size=5):
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
    return cleaned

def write_png(image: np.ndarray, filename: str) -> None:
    """
    Save a grayscale or RGB NumPy image to a PNG file using zlib.
    
    Args:
        image (np.ndarray): Image array (H x W) for grayscale or (H x W x 3) for RGB.
        filename (str): Path to save the .png file.
    """
    # Ensure uint8
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    height, width = image.shape[:2]
    depth = 8  # bits per channel
    
    if image.ndim == 2:
        color_type = 0  # grayscale
    elif image.ndim == 3 and image.shape[2] == 3:
        color_type = 2  # RGB
    else:
        raise ValueError("Only grayscale or RGB images supported.")

    # PNG header
    png_signature = b"\x89PNG\r\n\x1a\n"

    def chunk(tag, data):
        return (struct.pack("!I", len(data)) +
                tag +
                data +
                struct.pack("!I", zlib.crc32(tag + data) & 0xffffffff))

    # IHDR chunk
    ihdr = struct.pack("!2I5B", width, height, depth, color_type, 0, 0, 0)

    # Add filter byte (0) at start of each scanline
    if image.ndim == 2:
        raw_data = b"".join(b"\x00" + image[y, :].tobytes() for y in range(height))
    else:  # RGB
        raw_data = b"".join(b"\x00" + image[y, :, :].tobytes() for y in range(height))

    # Compress with zlib
    compressed = zlib.compress(raw_data, level=9)

    # Build PNG
    with open(filename, "wb") as f:
        f.write(png_signature)
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", compressed))
        f.write(chunk(b"IEND", b""))

def scale_to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Normalize a NumPy array to 0–255 and convert to uint8.

    Parameters:
        arr (np.ndarray): Input array (any numeric dtype).

    Returns:
        np.ndarray: Array scaled to range [0, 255] as uint8.
    """
    arr = np.asarray(arr, dtype=np.float32)
    min_val, max_val = arr.min(), arr.max()

    # Handle nearly constant arrays
    if max_val - min_val < 1e-8:
        return np.clip(arr, 0, 255).astype(np.uint8)

    # Normalize to [0, 255]
    scaled = (arr - min_val) * 255.0 / (max_val - min_val)
    return scaled.clip(0, 255).astype(np.uint8)

def find_connected_components(
    mask: np.ndarray,
    connectivity: int = 8,
    min_area: int =40,
    max_area: int = 30000,
):
    H, W = mask.shape
    visited = np.zeros((H, W), dtype=bool)
    labeled_mask = np.zeros((H, W), dtype=np.int32)
    binary_clean = np.zeros((H, W), dtype=np.uint8)  # <--- NEW

    if connectivity == 4:
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
    else:
        directions = [(1,0), (-1,0), (0,1), (0,-1),
                      (1,1), (-1,-1), (1,-1), (-1,1)]

    component_info = []
    label = 0

    for i in range(H):
        for j in range(W):
            if mask[i, j] == 1 and not visited[i, j]:
                label += 1
                q = deque()
                q.append((i, j))
                visited[i, j] = True

                pixels = []
                min_r, min_c = i, j
                max_r, max_c = i, j
                sum_r, sum_c = 0, 0

                while q:
                    r, c = q.popleft()
                    labeled_mask[r, c] = label
                    pixels.append((r, c))
                    sum_r += r
                    sum_c += c
                    min_r, min_c = min(min_r, r), min(min_c, c)
                    max_r, max_c = max(max_r, r), max(max_c, c)

                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if mask[nr, nc] == 1 and not visited[nr, nc]:
                                visited[nr, nc] = True
                                q.append((nr, nc))

                area = len(pixels)
                centroid = (sum_r / area, sum_c / area)
                bbox_w = max_c - min_c + 1
                bbox_h = max_r - min_r + 1

                # Keep only valid blobs
                if min_area <= area <= max_area:
                    component_info.append({
                        "label": label,
                        "area": area,
                        "centroid": centroid,
                        "bbox": (min_r, min_c, max_r, max_c)
                    })
                    for r, c in pixels:
                        binary_clean[r, c] = 255  # <--- mark as white
                else:
                    for r, c in pixels:
                        labeled_mask[r, c] = 0

    return len(component_info), labeled_mask, binary_clean, component_info




def visualize_components(labeled_mask: np.ndarray, output_file: str):
    """
    Save a visualization of connected components with random colors.
    Returns the colored frame as np.ndarray.
    """
    H, W = labeled_mask.shape
    colored = np.zeros((H, W, 3), dtype=np.uint8)

    unique_labels = np.unique(labeled_mask)
    for label in unique_labels:
        if label == 0:  # background
            continue
        color = [random.randint(50, 255) for _ in range(3)]
        colored[labeled_mask == label] = color

    write_png(colored, output_file)
    return colored   # <--- important



def make_video(frames, output_file, fps=20):
    """
    Save a list of frames (grayscale or RGB) as a video.

    Args:
        frames (list of np.ndarray): Each frame is [H, W] (uint8) or [H, W, 3] (uint8)
        output_file (str): Path to save video
        fps (int): Frames per second
    """
    h, w = frames[0].shape[:2]

    # Check if input is grayscale or color
    is_color = (frames[0].ndim == 3 and frames[0].shape[2] == 3)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_file, fourcc, fps, (w, h), isColor=is_color)

    for f in frames:
        if is_color:
            writer.write(f)  # RGB already
        else:
            writer.write(f)  # keep grayscale
    writer.release()



def process_single_frame(idx, frame, mean_frame, var_frame, args, masks_dir, comps_dir):
    # 1. Compute mask + clean
    mask = compute_mask(frame, mean_frame, var_frame, threshold=1)
    mask = morphological_operations(mask, kernel_size=3)
    mask_out = (mask * 255).astype(np.uint8)

    # 2. Save raw mask
    mask_path = os.path.join(masks_dir, f"mask_{idx:04d}.{args.output_ext}")
    write_png(mask_out, mask_path)

    # --- Create subfolders ---
    comp_masks_dir = os.path.join(comps_dir, "comp_masks")
    comp_labels_dir = os.path.join(comps_dir, "comp_labels")
    os.makedirs(comp_masks_dir, exist_ok=True)
    os.makedirs(comp_labels_dir, exist_ok=True)

    # 3. Connected components
    num, labeled_mask, comp_mask, info = find_connected_components(
        mask, connectivity=8, min_area=60
    )

    # Save binary cleaned components (black/white)
    comp_bw_path = os.path.join(comp_masks_dir, f"comp_mask_{idx:04d}.{args.output_ext}")
    write_png(comp_mask, comp_bw_path)

    # Save visualization (colored blobs)
    comp_vis_path = os.path.join(comp_labels_dir, f"components_{idx:04d}.{args.output_ext}")
    comp_vis = visualize_components(labeled_mask, comp_vis_path)

    return idx, mask_out, comp_mask, comp_vis




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--input_ext", type=str, default="png")
    parser.add_argument("--output_ext", type=str, default="png")
    parser.add_argument("--video_format", type=str, default="mp4")
    args = parser.parse_args()

    # --- Get dataset name (last part of input path) ---
    dataset_name = os.path.basename(os.path.normpath(args.input_folder))
    base_out = os.path.join(args.output_folder, dataset_name)
    os.makedirs(base_out, exist_ok=True)

    # --- Subfolders ---
    masks_dir = os.path.join(base_out, "masks")
    comp_masks_dir = os.path.join(base_out, "components", "comp_masks")
    comp_labels_dir = os.path.join(base_out, "components", "comp_labels")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(comp_masks_dir, exist_ok=True)
    os.makedirs(comp_labels_dir, exist_ok=True)

    # 1. Read frames
    frames = read_frames(args.input_folder)

    # 2. Select number of frames for background model
    if "person1" in args.input_folder.lower():
        t = 70
    elif "person3" in args.input_folder.lower():
        t = 60
    else:
        t = len(frames)

    # 3. Compute mean and variance
    mean_frame = compute_mean_frames(frames[:t])
    var_frame = compute_variance(frames[:t], mean_frame)

    write_png(scale_to_uint8(mean_frame), os.path.join(base_out, "mean.png"))
    write_png(scale_to_uint8(var_frame), os.path.join(base_out, "variance.png"))

    # 4. Parallel frame processing
    results = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_single_frame,
                idx, frame, mean_frame, var_frame,
                args, masks_dir, comp_masks_dir, comp_labels_dir
            )
            for idx, frame in enumerate(frames)
        ]
        for f in futures:
            results.append(f.result())

    results.sort(key=lambda x: x[0])
    masks       = [r[1] for r in results]  # raw motion masks
    comp_masks  = [r[2] for r in results]  # binary component masks
    comp_labels = [r[3] for r in results]  # color label visualizations

    # 5. Save videos
    make_video(masks, os.path.join(base_out, f"masks.{args.video_format}"))
    make_video(comp_masks, os.path.join(base_out, "components", f"comp_masks.{args.video_format}"))
    make_video(comp_labels, os.path.join(base_out, "components", f"comp_labels.{args.video_format}"))

if __name__ == "__main__":
    main()
