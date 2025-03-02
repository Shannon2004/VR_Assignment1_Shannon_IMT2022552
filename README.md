# VR_Assignment1_Shannon_IMT2022552

# Computer Vision Image Processing

A comprehensive computer vision project implementing coin detection, segmentation, and image stitching for panorama creation.

## Project Overview

This repository contains two main components:

1. **Coin Detection and Segmentation** - Implements various edge detection methods to identify, count, and extract individual coins from images.
2. **Image Stitching** - Creates panoramic images by stitching multiple overlapping images together using feature-based techniques.

## Repository Structure

```
ASSIGNMENT1/
├── Images_Q1/                 # Input images for coin detection
│   ├── Image1.jpeg
│   ├── Image2.jpeg
│   └── Image3.jpeg
├── Outputs_Q1/                # Edge detection outputs
│   ├── Canny_Edges.jpg
│   ├── Detected_Coins.jpg
│   ├── Laplacian_Edges.jpg
│   └── Sobel_Edges.jpg
├── Output_Q1_p2/              # Coin segmentation outputs
│   ├── Coins/                 # Individual extracted coins
│   │   ├── coin_1.png
│   │   ├── coin_2.png
│   │   └── ...
│   └── segmented_coins.png    # Visualization of all segmented coins
├── Stiching1/                 # Input images for panorama stitching
│   ├── Block1.jpg
│   └── Block2.jpg
├── Stiching2/                 # Additional stitching inputs
│   ├── row-1-column-1.jpg
│   ├── row-1-column-2.jpg
│   └── ...
├── Stiching3/                 # More stitching inputs
├── Stitching_outputs/         # Panorama creation outputs
│   ├── final_panorama.jpg
│   ├── image_*_with_keypoints.jpg
│   └── matches_*_to_*.jpg
├── coins1.py                  # Coin detection and segmentation code
└── Stiching.py                # Image stitching and panorama creation code
```

## Dependencies

The project requires the following dependencies:

- Python 3.9+
- NumPy 2.0.2+
- OpenCV 4.11.0+
- Matplotlib 3.9.4+

You can check your current versions with:

```bash
python3 --version
python3
>>> import numpy
>>> print(numpy.__version__)
>>> import cv2
>>> print(cv2.__version__)
>>> import matplotlib
>>> print(matplotlib.__version__)
>>> exit()
```

## Installation

1. Clone this repository:

```bash
git clone https://github.com/Shannon2004/VR_Assignment1_Shannon_IMT2022552.git
cd VR_Assignment1_Shannon_IMT2022552/Assignment1
```

2. Install the required dependencies:

```bash
pip install numpy opencv-python matplotlib
```

## Usage

### Coin Detection and Segmentation

To detect, segment, and count coins in an image:

```bash
python3 coins1.py
```

This will:
1. Process the input image from `Images_Q1/Image2.jpeg`
2. Apply various edge detection algorithms (Canny, Sobel, Laplacian)
3. Segment individual coins and save them in the `Output_Q1_p2/Coins` directory
4. Save the segmented coins visualization to `Output_Q1_p2/segmented_coins.png`
5. Count the number of coins and display the result

### Image Stitching

To create a panorama by stitching multiple images:

```bash
python3 Stiching.py
```

This will:
1. Load images from the `Stiching1` directory
2. Extract and match features between images
3. Compute homographies and stitch images together
4. Save the final panorama and intermediate results in the `Stitching_outputs` directory

## Methods

### Coin Detection Methods

1. **Preprocessing**:
   - Grayscale conversion
   - Gaussian blur for noise reduction

2. **Edge Detection**:
   - **Canny**: Two-threshold approach for robust edge detection
   - **Sobel**: Gradient-based detection in x and y directions
   - **Laplacian**: Second-order derivative for edge enhancement

3. **Coin Segmentation**:
   - Adaptive thresholding
   - Morphological operations
   - Contour detection and filtering based on area and circularity
   - Circular masking for precise extraction

4. **Coin Counting**:
   - Contour-based counting with area filtering

### Image Stitching Methods

1. **Feature Extraction**:
   - SIFT (Scale-Invariant Feature Transform) for keypoint detection and description

2. **Feature Matching**:
   - Brute-Force matcher with L2 norm
   - Ratio test for filtering good matches

3. **Homography Estimation**:
   - RANSAC algorithm for robust estimation

4. **Image Warping and Blending**:
   - Perspective transformation
   - Distance-based weighted blending for seamless transitions
   - Black border removal

## Expected Outputs

### Coin Detection Expected Outputs

The coin detection and segmentation process will generate the following outputs:

1. **Edge Detection Results** in the `Outputs_Q1` directory:
   - Canny edge detection result
   - Sobel edge detection result
   - Laplacian edge detection result
   - Image with detected coin contours

2. **Segmentation Results** in the `Output_Q1_p2` directory:
   - `segmented_coins.png` - A visualization of all detected coins
   - `Coins/` directory containing individually extracted coins as separate images

3. **Console Output**:
   - Number of coins detected in the image
   - Processing messages

### Image Stitching Expected Outputs

The image stitching process will generate the following outputs in the `Stitching_outputs` directory:

1. **Feature Visualization**:
   - Images showing detected keypoints for each input image

2. **Matching Visualization**:
   - Images showing the matches between consecutive image pairs

3. **Final Panorama**:
   - `final_panorama.jpg` - The complete stitched panoramic image

## Observations and Tips

### Coin Detection

1. **Edge Detection Performance**:
   - Canny edge detector generally provides the most well-defined edges for coin detection
   - Laplacian detector is more sensitive to noise but captures fine details
   - Sobel operator performs well in detecting directional gradients

2. **Segmentation Considerations**:
   - The algorithm works best with clear, well-lit images with good contrast
   - Overlapping coins may be difficult to separate
   - The area filter (5000-35000 pixels) might need adjustment for different image resolutions
   - Circularity threshold (0.7) helps filter out non-coin objects

### Image Stitching

1. **For Best Results**:
   - Use images with significant overlap (30-50%)
   - Maintain consistent lighting between images
   - Avoid extreme perspective changes between consecutive images

2. **Performance Considerations**:
   - Higher resolution images require more processing time
   - The minimum threshold for good matches (10) may need adjustment for different image sets
   - For complex panoramas, consider using images in a specific sequence
