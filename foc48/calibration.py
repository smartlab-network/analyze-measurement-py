"""
FOC48 calibration stage.

Everything related to the one-time calibration performed on a single
reference frame (typically the first frame of a stream): detecting all
individual poles, pairing them up, assigning stable grid/well labels,
and computing a fixed local ROI and threshold for each pole.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def detect_pole_centroids(
        image: np.ndarray,
        area_min: int = 10,
        area_max: int = 500,
        invert_threshold: bool = False
) -> np.ndarray:
    """
    Detect individual poles across the full image and return their
    centroids. Used only during calibration to discover the grid layout.

    Parameters
    ----------
    image : np.ndarray
        Grayscale input image.
    area_min, area_max : int, optional
        Valid contour area range (default 10, 500).
    invert_threshold : bool, optional
        Use THRESH_BINARY_INV for dark poles on a bright background.
        Default False (bright poles on dark background).

    Returns
    -------
    np.ndarray
        Array of shape (n_poles, 2) with (x, y) centroids. Empty
        (0, 2) if none found.
    """
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    thresh_type = cv2.THRESH_BINARY_INV if invert_threshold else cv2.THRESH_BINARY
    _, thresh = cv2.threshold(blurred, 0, 255, thresh_type + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, None, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.empty((0, 2))

    areas = np.array([cv2.contourArea(c) for c in contours])
    valid_mask = (areas > area_min) & (areas < area_max)
    if not np.any(valid_mask):
        return np.empty((0, 2))

    valid_contours = [c for c, m in zip(contours, valid_mask) if m]
    moments_list = [cv2.moments(c) for c in valid_contours]
    m00 = np.array([m['m00'] for m in moments_list])
    valid_moments = m00 > 0
    if not np.any(valid_moments):
        return np.empty((0, 2))

    x = np.array([m['m10'] / m['m00'] for m, vm in zip(moments_list, valid_moments) if vm])
    y = np.array([m['m01'] / m['m00'] for m, vm in zip(moments_list, valid_moments) if vm])
    return np.column_stack([x, y])


def pair_poles(
        centroids: np.ndarray,
        max_pair_distance: Optional[float] = None
) -> List[Dict]:
    """
    Greedily pair up pole centroids by nearest-neighbor distance.

    Parameters
    ----------
    centroids : np.ndarray
        Array of shape (n_poles, 2) with (x, y) coordinates.
    max_pair_distance : float, optional
        Maximum distance allowed for a valid pair (default: no limit).

    Returns
    -------
    list of dict
        One entry per pair, with keys 'midpoint', 'left', 'right'
        (np.ndarray, shape (2,)) and 'distance' (float).
    """
    n = len(centroids)
    if n < 2:
        return []

    diff = centroids[:, None, :] - centroids[None, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=2))
    iu = np.triu_indices(n, k=1)
    order = np.argsort(dist_matrix[iu])

    matched = np.zeros(n, dtype=bool)
    pairs = []

    for flat_idx in order:
        i, j = iu[0][flat_idx], iu[1][flat_idx]
        d = dist_matrix[i, j]
        if max_pair_distance is not None and d > max_pair_distance:
            break
        if not matched[i] and not matched[j]:
            matched[i] = matched[j] = True
            a, b = centroids[i], centroids[j]
            left, right = (a, b) if a[0] <= b[0] else (b, a)
            pairs.append({'midpoint': (a + b) / 2.0, 'left': left, 'right': right, 'distance': float(d)})

    return pairs


def grid_label(row: int, col: int) -> str:
    """
    Convert 0-based row/column indices to a well label, e.g. (0, 0) -> "A1".

    Parameters
    ----------
    row, col : int
        0-based row and column index.

    Returns
    -------
    str
        Well label.
    """
    return f"{chr(ord('A') + row)}{col + 1}"


def assign_grid_positions(pairs: List[Dict], n_rows: int, n_cols: int) -> Dict[str, Dict]:
    """
    Assign a stable well label to each pole pair via row-major sorting.

    Parameters
    ----------
    pairs : list of dict
        Detected pairs, as returned by `pair_poles`. Must contain exactly
        n_rows * n_cols entries.
    n_rows, n_cols : int
        Grid dimensions.

    Returns
    -------
    dict of str to dict
        Mapping from well label to pair dict.

    Raises
    ------
    ValueError
        If the number of pairs does not match n_rows * n_cols.
    """
    n_expected = n_rows * n_cols
    if len(pairs) != n_expected:
        raise ValueError(f"Expected {n_expected} pole pairs ({n_rows}x{n_cols} grid), found {len(pairs)}.")

    sorted_by_y = [pairs[i] for i in np.argsort([p['midpoint'][1] for p in pairs])]

    labeled = {}
    for row in range(n_rows):
        row_pairs = sorted_by_y[row * n_cols:(row + 1) * n_cols]
        row_sorted = [row_pairs[i] for i in np.argsort([p['midpoint'][0] for p in row_pairs])]
        for col in range(n_cols):
            labeled[grid_label(row, col)] = row_sorted[col]

    return labeled


def calibrate_local_threshold(patch: np.ndarray, fallback_thresh: float) -> float:
    """
    Compute a local Otsu threshold for a small patch, with a fallback
    for degenerate (near-uniform) patches.

    Parameters
    ----------
    patch : np.ndarray
        Grayscale image patch.
    fallback_thresh : float
        Threshold to use if the patch has near-zero variance.

    Returns
    -------
    float
        Calibrated threshold.
    """
    if patch.size == 0 or patch.std() < 1.0:
        return fallback_thresh
    blurred = cv2.GaussianBlur(patch, (3, 3), 0)
    thresh_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(thresh_val)


def compute_pair_margin(
        pair_distance: float,
        margin_factor: float = 0.45,
        min_margin_px: int = 10,
        max_margin_px: int = 100
) -> int:
    """
    Compute a per-pair local ROI margin, guaranteed not to overlap
    between the left and right pole's ROI windows.

    Parameters
    ----------
    pair_distance : float
        Measured intra-pair distance (pixels).
    margin_factor : float, optional
        Fraction of the pair's own distance used as margin (default
        0.45). Must be < 0.5 so that 2 * margin < distance.
    min_margin_px, max_margin_px : int, optional
        Bounds for the margin (default 10, 100).

    Returns
    -------
    int
        Local ROI margin, in pixels.

    Raises
    ------
    ValueError
        If `margin_factor` is not strictly less than 0.5.
    """
    if margin_factor >= 0.5:
        raise ValueError(f"margin_factor must be < 0.5 to avoid ROI overlap, got {margin_factor}")
    return int(np.clip(margin_factor * pair_distance, min_margin_px, max_margin_px))


def resolve_margin_factor(margin_setting: Optional[float]) -> float:
    """
    Resolve the margin factor used for per-pair ROI sizing.

    Parameters
    ----------
    margin_setting : float or None
        None uses the default of 0.45; otherwise a value in (0, 0.5).

    Returns
    -------
    float
        Margin factor, guaranteed < 0.5.

    Raises
    ------
    ValueError
        If `margin_setting` is >= 0.5.
    """
    if margin_setting is None:
        return 0.45
    if margin_setting >= 0.5:
        raise ValueError(f"margin factor must be < 0.5 to avoid ROI overlap, got {margin_setting}")
    return float(margin_setting)


def calibrate_wells(
        first_frame: np.ndarray,
        n_rows: int,
        n_cols: int,
        area_min: int,
        area_max: int,
        invert_threshold: bool,
        max_pair_distance: Optional[float],
        local_margin_px: Optional[float],
        fallback_pix_thresh: float
) -> Dict[str, Dict]:
    """
    Calibrate a fixed local ROI and threshold for each pole of each well.

    Parameters
    ----------
    first_frame : np.ndarray
        Grayscale reference frame.
    n_rows, n_cols : int
        Expected grid dimensions.
    area_min, area_max : int
        Valid contour area range for global pole detection.
    invert_threshold : bool
        Threshold direction for global pole detection.
    max_pair_distance : float, optional
        Maximum distance allowed to form a valid pair.
    local_margin_px : float, optional
        Margin factor, resolved via `resolve_margin_factor`.
    fallback_pix_thresh : float
        Threshold used when local Otsu calibration fails.

    Returns
    -------
    dict of str to dict
        Per-well calibration: 'roi_left', 'roi_right', 'pix_thresh_l',
        'pix_thresh_r', 'template_left', 'template_right',
        'template_left_local', 'template_right_local'.

    Raises
    ------
    ValueError
        If the expected number of poles/pairs is not found.
    """
    n_expected_poles = n_rows * n_cols * 2

    centroids = detect_pole_centroids(first_frame, area_min, area_max, invert_threshold)
    print(f"Calibration: detected {len(centroids)} poles (expected {n_expected_poles})")

    pairs = pair_poles(centroids, max_pair_distance)
    print(f"Calibration: formed {len(pairs)} pairs (expected {n_rows * n_cols})")

    if len(pairs) != n_rows * n_cols:
        raise ValueError(f"Calibration failed: expected {n_rows * n_cols} pairs, found {len(pairs)}.")

    dists = np.array([p['distance'] for p in pairs])
    print(f"Calibration: intra-pair distance - min={dists.min():.1f}, max={dists.max():.1f}, mean={dists.mean():.1f}")

    margin_factor = resolve_margin_factor(local_margin_px)
    print(f"Calibration: margin factor {margin_factor:.2f} "
          f"(>= {(1 - 2 * margin_factor) * 100:.0f}% gap guaranteed between paired ROIs)")

    labeled_pairs = assign_grid_positions(pairs, n_rows, n_cols)
    h_img, w_img = first_frame.shape[:2]

    def local_roi(center: np.ndarray, margin: int) -> Tuple[int, int, int, int]:
        x, y = center
        return (
            max(0, int(round(x - margin))), max(0, int(round(y - margin))),
            min(w_img, int(round(x + margin))), min(h_img, int(round(y + margin)))
        )

    calibration = {}
    margins_used = []

    for label, pair in labeled_pairs.items():
        margin = compute_pair_margin(pair['distance'], margin_factor)
        margins_used.append(margin)

        roi_left = local_roi(pair['left'], margin)
        roi_right = local_roi(pair['right'], margin)

        patch_left = first_frame[roi_left[1]:roi_left[3], roi_left[0]:roi_left[2]]
        patch_right = first_frame[roi_right[1]:roi_right[3], roi_right[0]:roi_right[2]]

        calibration[label] = {
            'roi_left': roi_left,
            'roi_right': roi_right,
            'pix_thresh_l': calibrate_local_threshold(patch_left, fallback_pix_thresh),
            'pix_thresh_r': calibrate_local_threshold(patch_right, fallback_pix_thresh),
            'template_left': tuple(pair['left']),
            'template_right': tuple(pair['right']),
            'template_left_local': (pair['left'][0] - roi_left[0], pair['left'][1] - roi_left[1]),
            'template_right_local': (pair['right'][0] - roi_right[0], pair['right'][1] - roi_right[1]),
        }

    print(f"Calibration: per-pair margins ranged {min(margins_used)}-{max(margins_used)} px")
    return calibration