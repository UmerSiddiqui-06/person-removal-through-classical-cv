Author: Umer Siddiqui
Roll Number: BSAI23038
Assignment: 01

----------------------------------------
Description
----------------------------------------
This script performs **motion detection and background subtraction** on a sequence of images (video frames). 
It detects moving objects (like a person) and can produce:
1. Clean binary motion masks.
2. Connected component masks (individual moving blobs).
3. Color-labeled visualization of components.
4. Alpha-blended frames to gradually remove moving objects.
5. Video outputs for masks, components, and blended frames.

----------------------------------------
Setup
----------------------------------------
1. Create a Python virtual environment:
   Windows:
       python -m venv venv
       venv\Scripts\activate
   Linux/Mac:
       python3 -m venv venv
       source venv/bin/activate


2. Install required libraries:
   pip install -r requirements.txt

----------------------------------------
Usage
----------------------------------------
Run the script from terminal:

python changeDetection.py --input_folder <input_folder> --output_folder <output_folder> [options]

Required arguments:
--input_folder   Folder containing input PNG frames.
--output_folder  Folder to save results (masks, videos, etc.).

Optional arguments:
--input_ext      Input image extension (default: png)  [not used for now]
--output_ext     Output image extension (default: png)
--video_format   Video format for outputs (default: mp4)

Example:
python changeDetection.py --input_folder data/person1 --output_folder results

----------------------------------------
Outputs
----------------------------------------
Inside the output folder, the script creates:
- mean.png          : Average background frame
- variance.png      : Pixel-wise variance of background
- morphological_masks/ : Clean motion masks
- components/comp_masks/  : Binary connected component masks
- components/comp_labels/ : Color-labeled visualization of components
- alpha_blend/      : Frames with gradual removal of moving objects
- Video files (.mp4) corresponding to above masks and components

----------------------------------------
Project Flow
----------------------------------------
1. **Read frames**: Load all PNG images from the input folder.
2. **Compute background**: Calculate mean and variance frames from initial frames.
3. **Motion detection**: Compute motion masks using Mahalanobis distance.
4. **Clean masks**: Apply morphological operations (erosion + dilation) to remove noise.
5. **Connected components**: Identify individual moving objects, save binary masks and color visualizations.
6. **Alpha blending**: Gradually remove moving objects from frames using the masks.
7. **Video generation**: Save masks, components, and blended frames as MP4 videos.
