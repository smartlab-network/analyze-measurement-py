import cv2
import numpy as np
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
import time


def define_roi(image):
    """
    Define a Region of Interest (ROI) within the image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image as a numpy array (BGR format).
    Returns
    -------
    numpy.ndarray
        Cropped ROI sub-image.
    """
    height, width = image.shape[:2]
    # ROI placement based on known pole specifications
    roi_x1, roi_y1 = int(width * 0.2), int(height * 0.2)
    roi_x2, roi_y2 = int(width * 0.8), int(height * 0.8)
    return image[roi_y1:roi_y2, roi_x1:roi_x2]


def detect_poles(image):
    """
    Detect poles in an image using thresholding and contour analysis.

    Parameters
    ----------
    image : numpy.ndarray
        Input image (BGR format).
    Returns
    -------
    list of tuple
        List of (x, y) coordinates of detected pole centroids.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Extract ROI
    roi = define_roi(gray)

    # Apply Otsu's thresholding for automatic threshold selection
    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find external contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by minimum area and compute centroids
    poles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10:  # Adjust based on expected pole size
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                x = int(moments["m10"] / moments["m00"])
                y = int(moments["m01"] / moments["m00"])
                # Offset coordinates back to full image frame
                x += int(0.2 * image.shape[1])
                y += int(0.2 * image.shape[0])
                poles.append((x, y))

    return poles


def calculate_distances(poles):
    """
    Compute pairwise Euclidean distances between consecutive poles.

    Parameters
    ----------
    poles : list of tuple
        List of (x, y) coordinates of detected poles.
    Returns
    -------
    list of float
        Distances between consecutive pole pairs.
    """
    distances = []
    for i in range(len(poles) - 1):
        x1, y1 = poles[i]
        x2, y2 = poles[i + 1]
        dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        distances.append(dist)
    return distances


def process_image(image_path):
    """
    Process a single image: detect poles and compute distances.

    Parameters
    ----------
    image_path : str
        Path to the input image file.
    Returns
    -------
    list of float
        Distances between consecutive poles for this image.
    """
    img = cv2.imread(image_path)
    if img is None:
        return []

    poles = detect_poles(img)

    if len(poles) > 1:
        return calculate_distances(poles)
    return []


def parallel_process_images(image_dir, num_workers=4):
    """
    Process all images in a directory in parallel using a thread pool.

    Parameters
    ----------
    image_dir : str
        Directory containing .bmp images.
    num_workers : int, optional
        Number of parallel worker threads (default is 4).
    Returns
    -------
    list of float
        Flattened list of all pole distances from all images.
    """
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".bmp")]
    distances_list = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_image, os.path.join(image_dir, fname))
            for fname in image_files
        ]
        for future in futures:
            distances_list.extend(future.result())
    return distances_list


def save_to_csv(distances, output_file):
    """
    Save a list of distances to a CSV file.

    Parameters
    ----------
    distances : list of float
        Distance values to save.
    output_file : str
        Path to the output CSV file.
    """
    arr = np.array(distances, dtype=np.float64)
    df = pd.DataFrame({"distance": arr})
    df.to_csv(output_file, index=False)


def main():
    """
    Main entry point: orchestrate the full pole distance analysis pipeline.
    """
    IMAGE_DIR = "test_data"
    OUTPUT_FILE = "analyze_measurement/pole_distances.csv"

    if not os.path.exists(IMAGE_DIR):
        raise FileNotFoundError(f"Directory not found: {IMAGE_DIR}")

    print("Starting image processing...")
    start_time = time.time()

    distances = parallel_process_images(IMAGE_DIR, num_workers=4)
    save_to_csv(distances, OUTPUT_FILE)

    elapsed = time.time() - start_time
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"Processing time: {elapsed:.2f} seconds")
    print(f"Total distance measurements: {len(distances)}")

if __name__ == "__main__":
    main()