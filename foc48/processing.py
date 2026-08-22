"""
FOC48 per-frame processing stage.

Everything related to steady-state processing of a live/recorded frame
stream, given a fixed calibration produced by `calibration.py`: locating
each pole within its pre-calibrated ROI, computing intra-pair distances,
batching frames for worker processes, and warming up the process pool.
"""

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor


def _warmup_worker() -> None:
    """
    No-op task submitted once per worker right after pool creation, so
    that interpreter startup and module imports (cv2, numpy) happen
    during a controlled warmup phase instead of during the first real
    frames' processing time.
    """
    import cv2  # noqa: F401
    import numpy as np  # noqa: F401


def warmup_process_pool(executor: ProcessPoolExecutor, num_workers: int) -> float:
    """
    Warm up all worker processes by submitting one no-op task each.

    Parameters
    ----------
    executor : ProcessPoolExecutor
        Pool whose workers should be warmed up.
    num_workers : int
        Number of workers to warm up.

    Returns
    -------
    float
        Warmup time in seconds.
    """
    start = time.perf_counter()
    futures = [executor.submit(_warmup_worker) for _ in range(num_workers)]
    for f in futures:
        f.result()
    return time.perf_counter() - start


def locate_pole_in_roi(
        frame: np.ndarray,
        roi: Tuple[int, int, int, int],
        pix_thresh: float,
        template_local: Tuple[float, float],
        min_blob_area: int = 3
) -> Optional[Tuple[float, float]]:
    """
    Locate a single pole within a pre-calibrated, fixed ROI.

    Parameters
    ----------
    frame : np.ndarray
        Full grayscale frame.
    roi : tuple of int
        (x0, y0, x1, y1) bounding box in original image coordinates.
    pix_thresh : float
        Fixed threshold calibrated for this pole.
    template_local : tuple of float
        Expected (x, y) position of the pole, local to the ROI.
    min_blob_area : int, optional
        Minimum blob area to be considered valid (default 3).

    Returns
    -------
    tuple of float, or None
        (x, y) centroid in original image coordinates, or None if not found.
    """
    x0, y0, x1, y1 = roi
    patch = frame[y0:y1, x0:x1]

    blurred = cv2.GaussianBlur(patch, (3, 3), 0)
    _, mask = cv2.threshold(blurred, pix_thresh, 255, cv2.THRESH_BINARY)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return None

    candidates = [i for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] >= min_blob_area]
    if not candidates:
        return None

    dists = np.linalg.norm(centroids[candidates] - np.array(template_local), axis=1)
    best = candidates[np.argmin(dists)]

    component_mask = (labels == best)
    weights = patch.astype(np.float64) * component_mask
    total = weights.sum()
    if total <= 0:
        cx, cy = centroids[best]
    else:
        ys, xs = np.indices(patch.shape)
        cx, cy = (xs * weights).sum() / total, (ys * weights).sum() / total

    return cx + x0, cy + y0


def process_frame_pairs(
        frame: np.ndarray,
        well_labels: List[str],
        calibration: Dict[str, Dict],
        min_blob_area: int
) -> np.ndarray:
    """
    Compute the intra-pair distance for every well in a single frame.

    Parameters
    ----------
    frame : np.ndarray
        Full grayscale frame.
    well_labels : list of str
        Well labels in output column order.
    calibration : dict of str to dict
        Per-well calibration data.
    min_blob_area : int
        Minimum blob area for a valid pole.

    Returns
    -------
    np.ndarray
        Array of shape (len(well_labels),) with distances in pixels.
        NaN where either pole could not be located.
    """
    distances = np.full(len(well_labels), np.nan)

    for idx, label in enumerate(well_labels):
        c = calibration[label]
        left = locate_pole_in_roi(frame, c['roi_left'], c['pix_thresh_l'], c['template_left_local'], min_blob_area)
        right = locate_pole_in_roi(frame, c['roi_right'], c['pix_thresh_r'], c['template_right_local'], min_blob_area)
        if left is not None and right is not None:
            distances[idx] = np.hypot(right[0] - left[0], right[1] - left[1])

    return distances


def process_frame_batch(
        frames: List[np.ndarray],
        well_labels: List[str],
        calibration: Dict[str, Dict],
        min_blob_area: int
) -> List[np.ndarray]:
    """
    Process a batch of frames within a single worker process.

    Parameters
    ----------
    frames : list of np.ndarray
        Grayscale frames to process.
    well_labels : list of str
        Well labels in output column order.
    calibration : dict of str to dict
        Per-well calibration data.
    min_blob_area : int
        Minimum blob area for a valid pole.

    Returns
    -------
    list of np.ndarray
        Distance array per frame.
    """
    return [process_frame_pairs(f, well_labels, calibration, min_blob_area) for f in frames]


def log_progress(frame_idx: int, total_frames: int, start_time: float) -> None:
    """
    Print a single-line, overwriting progress log to the terminal.

    Parameters
    ----------
    frame_idx : int
        Frames processed so far.
    total_frames : int
        Total expected frames (0 if unknown; omits percentage/ETA).
    start_time : float
        `time.perf_counter()` timestamp when processing started.
    """
    import sys
    elapsed = time.perf_counter() - start_time
    fps = frame_idx / elapsed if elapsed > 0 else 0.0

    if total_frames > 0:
        percent = 100.0 * frame_idx / total_frames
        eta = (total_frames - frame_idx) / fps if fps > 0 else 0.0
        message = f"\r  Progress: {frame_idx}/{total_frames} ({percent:5.1f}%) | {fps:6.1f} fps | ETA: {eta:5.1f}s"
    else:
        message = f"\r  Progress: {frame_idx} frames | {fps:6.1f} fps"

    sys.stdout.write(message.ljust(80))
    sys.stdout.flush()