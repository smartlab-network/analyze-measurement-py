import cv2
import numpy as np
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from pathlib import Path


def define_roi(image: np.ndarray) -> np.ndarray:
    """
    ROI: middle 60% of the image (pre-computed indices for speed).
    """
    h, w = image.shape[:2]
    return image[h // 5:4 * h // 5, w // 5:4 * w // 5]


def detect_poles(image: np.ndarray, area_min: int = 10, area_max: int = 500) -> np.ndarray:
    """
    Detect poles in a grayscale image and return sorted x-coordinates.

    Parameters
    ----------
    image : np.ndarray
        Grayscale input image.
    area_min, area_max : int
        Valid contour area range.

    Returns
    -------
    np.ndarray
        Sorted x-coordinates of detected poles. Empty if < 2 poles.
    """
    roi = define_roi(image)

    # Otsu thresholding - faster than adaptive for well-lit images
    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Single morphological opening to remove noise
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, None, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.array([])

    # Vectorized area computation for all contours
    areas = np.array([cv2.contourArea(c) for c in contours])
    valid_mask = (areas > area_min) & (areas < area_max)

    if not np.any(valid_mask):
        return np.array([])

    # Filter valid contours
    valid_contours = [c for c, m in zip(contours, valid_mask) if m]

    # Compute moments for all valid contours
    moments_list = [cv2.moments(c) for c in valid_contours]
    m00 = np.array([m['m00'] for m in moments_list])
    valid_moments = m00 > 0

    if not np.any(valid_moments):
        return np.array([])

    # Vectorized centroid computation
    x_centers = np.array([m['m10'] / m['m00'] for m, vm in zip(moments_list, valid_moments) if vm])
    y_centers = np.array([m['m01'] / m['m00'] for m, vm in zip(moments_list, valid_moments) if vm])

    # Apply ROI offset to get coordinates in original image
    h_img, w_img = image.shape[:2]
    x_centers += w_img // 5
    y_centers += h_img // 5

    # Sort by x-position (left to right)
    sort_idx = np.argsort(x_centers)
    return x_centers[sort_idx]


def calculate_distances(x_positions: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between consecutive poles.
    Uses numpy vectorization for speed.

    Parameters
    ----------
    x_positions : np.ndarray
        Sorted x-coordinates of detected poles.

    Returns
    -------
    np.ndarray
        Array of distances between consecutive pole pairs.
        Empty if less than 2 poles detected.
    """
    if len(x_positions) < 2:
        return np.array([])

    # Consecutive differences (x only - y varies minimally with good alignment)
    distances = np.diff(x_positions)

    # Optional outlier removal (minimal overhead)
    if len(distances) > 3:
        mean_d = np.mean(distances)
        std_d = np.std(distances)
        distances = distances[np.abs(distances - mean_d) < 2.5 * std_d]

    return distances


def process_image_batch(image_paths: list[str]) -> list[np.ndarray]:
    """
    Process a batch of images in a single process.
    Reduces IPC overhead by processing multiple frames per task.

    Parameters
    ----------
    image_paths : list[str]
        List of paths to BMP files.

    Returns
    -------
    list[np.ndarray]
        List of distance arrays for each image.
    """
    results = []
    for path in image_paths:
        # Read directly as grayscale - saves color conversion step
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            results.append(np.array([]))
            continue

        x_positions = detect_poles(img)
        distances = calculate_distances(x_positions)
        results.append(distances)

    return results


def split_into_batches(files: list[str], num_batches: int) -> list[list[str]]:
    """
    Split file list into roughly equal batches for parallel processing.

    Parameters
    ----------
    files : list[str]
        List of file paths.
    num_batches : int
        Number of batches to create.

    Returns
    -------
    list[list[str]]
        List of file batches.
    """
    batch_size = max(1, len(files) // num_batches)
    return [files[i:i + batch_size] for i in range(0, len(files), batch_size)]


def process_images_parallel(
        image_dir: str,
        num_workers: int = None
) -> tuple[np.ndarray, list[str]]:
    """
    Parallel image processing using ProcessPoolExecutor.
    Uses processes instead of threads for true CPU-bound parallelism.

    Parameters
    ----------
    image_dir : str
        Directory containing .bmp files.
    num_workers : int, optional
        Number of parallel processes. Default: CPU cores - 1.

    Returns
    -------
    tuple of (np.ndarray, list[str])
        2D array: (n_frames, max_pole_pairs), NaN for missing detections
        Sorted list of processed filenames
    """
    image_dir = Path(image_dir)
    image_files = sorted([str(f) for f in image_dir.glob("*.bmp") if f.is_file()])

    if not image_files:
        raise FileNotFoundError(f"No .bmp files found in {image_dir}")

    n_files = len(image_files)
    print(f"Processing {n_files} BMP files from {image_dir}...")

    if num_workers is None:
        num_workers = max(1, os.cpu_count() - 1)  # Reserve one core for OS

    # Split into batches for efficient processing
    batches = split_into_batches(image_files, num_workers)

    results = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_image_batch, batch) for batch in batches]

        for future in as_completed(futures):
            try:
                batch_results = future.result()
                results.extend(batch_results)
            except Exception as e:
                print(f"Batch processing error: {e}")

    # Determine maximum number of pole pairs detected
    max_pairs = max((len(r) for r in results), default=0)

    if max_pairs == 0:
        raise ValueError("No poles detected in any image!")

    # Build 2D array (vectorized)
    result_array = np.full((n_files, max_pairs), np.nan)
    for idx, dist in enumerate(results):
        if len(dist) > 0:
            result_array[idx, :len(dist)] = dist

    elapsed = time.time() - start_time
    print(f"Processed {n_files} frames in {elapsed:.2f}s ({n_files / elapsed:.0f} fps)")
    print(f"Detected {max_pairs} pole pairs")

    return result_array, image_files


def save_to_csv(distances_2d: np.ndarray, image_files: list[str], output_file: str, fps: float = 30.0) -> None:
    """
    Save results as CSV in GUI-compatible format.

    CSV columns: time (seconds), d0, d1, ..., dN (pole pair distances)

    Parameters
    ----------
    distances_2d : np.ndarray
        2D array of distances, shape (n_frames, n_pole_pairs).
    image_files : list[str]
        Sorted list of processed filenames.
    output_file : str
        Path to output CSV file.
    fps : float, optional
        Recording framerate (default 30 fps).
    """
    n_frames = len(distances_2d)
    time_vector = np.arange(n_frames, dtype=np.float64) / fps

    n_distances = distances_2d.shape[1]
    columns = ['time'] + [f'd{i}' for i in range(n_distances)]

    data = np.column_stack([time_vector, distances_2d])
    df = pd.DataFrame(data, columns=columns)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Faster CSV writing without index
    df.to_csv(output_file, index=False, float_format='%.6f')
    print(f"Saved to {output_file}")


def main():
    """
    Optimized main entry point.
    Processes all BMP files in test_data/ and saves results as CSV.
    """
    IMAGE_DIR = r"C:\labhub\Repos\smartlab-network\analyze-measurement-py\test_data/myrPlate_demo_2808_1884.avi"
    OUTPUT_FILE = "analyze_measurement/pole_distances.csv"
    FPS = 60.0

    if not os.path.exists(IMAGE_DIR):
        raise FileNotFoundError(f"Directory not found: {IMAGE_DIR}")

    print("=" * 50)
    print("FOC48 - Pole Distance Analyzer")
    print("=" * 50)

    try:
        distances_2d, image_files = process_images_parallel(IMAGE_DIR)
        save_to_csv(distances_2d, image_files, OUTPUT_FILE, fps=FPS)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())