import numpy as np
from scipy.ndimage import minimum_filter1d, maximum_filter1d


def smooth(values: np.ndarray, buffer: int) -> np.ndarray:
    """
    Moving-average smoothing with edge preservation.

    Parameters
    ----------
    values : np.ndarray, shape (n,)
        Input signal.
    buffer : int
        Kernel width in samples. Forced odd. Values ``< 2`` return
        an unchanged copy.

    Returns
    -------
    result : np.ndarray, shape (n,)
        Smoothed signal with original-value edges.

    Notes
    -----
    .. code-block:: text

        result[i] = mean(values[i - half : i + half + 1])
                    for i in [half, n - half)
        result[i] = values[i]   otherwise
    """
    if buffer < 2:
        return values.copy()
    if buffer % 2 == 0:
        buffer += 1
    half = buffer // 2
    kernel = np.ones(buffer) / buffer
    convolved = np.convolve(values, kernel, mode='valid')
    result = values.copy()
    result[half: len(values) - half] = convolved
    return result


def running_max(values: np.ndarray, window_sec: float, dt: float) -> np.ndarray:
    """
    Diastolic baseline estimated as a slow running maximum (runMax).

    Parameters
    ----------
    values : np.ndarray, shape (n,)
        Smoothed inter-pole distance signal in pixels.
    window_sec : float
        Filter window duration in seconds. Must satisfy
        ``window_sec >> 1 / beat_frequency``.
    dt : float
        Sampling interval in seconds.

    Returns
    -------
    run_max : np.ndarray, shape (n,)
        Diastolic baseline signal in pixels.

    Notes
    -----
    .. code-block:: text

        w_samples = ceil(window_sec / dt)  →  forced odd
        runMax[i] = max(values[i - w//2 : i + w//2 + 1])
    """
    window_samples = max(1, int(window_sec / dt))
    if window_samples % 2 == 0:
        window_samples += 1
    return maximum_filter1d(values, size=window_samples)


def find_peaks(
    values: np.ndarray, min_frq: float, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect systolic peaks (local minima) with guaranteed minimum spacing.

    Parameters
    ----------
    values : np.ndarray, shape (n,)
        Smoothed signal in pixels.
    min_frq : float
        Maximum expected beat frequency in Hz.
        ``min_spacing = floor(1 / min_frq / dt)`` samples.
    dt : float
        Sampling interval in seconds.

    Returns
    -------
    peak_idx : np.ndarray, shape (k,)
        Sample indices of accepted peaks.
    peak_val : np.ndarray, shape (k,)
        Signal values at accepted peaks in pixels.

    Notes
    -----
    .. code-block:: text

        min_spacing [samples] = floor(1 / min_frq / dt)
        kernel_size           = 2 * min_spacing + 1

    Algorithm: candidate detection via ``minimum_filter1d``, then
    greedy deduplication enforcing ``min_spacing`` between peaks;
    within the exclusion zone the deeper peak wins.
    """
    min_spacing = int((1.0 / min_frq) / dt)
    kernel_size = 2 * min_spacing + 1

    window_min = minimum_filter1d(values, size=kernel_size)
    candidates = np.where(values == window_min)[0]

    if len(candidates) == 0:
        return np.array([], dtype=int), np.array([])

    peaks = [candidates[0]]
    for idx in candidates[1:]:
        if idx - peaks[-1] >= min_spacing:
            peaks.append(idx)
        elif values[idx] < values[peaks[-1]]:
            peaks[-1] = idx

    peak_idx = np.array(peaks)
    return peak_idx, values[peak_idx]


def compute_metrics(
    time: np.ndarray,
    values: np.ndarray,
    min_frq: float,
    smooth_buffer: int,
    run_max_window_sec: float = 2.0,
) -> dict:
    """
    Compute all contractility metrics for a single well.

    Pipeline: smooth → running_max → find_peaks →
    peakHeight → contraction% → freq.

    Parameters
    ----------
    time : np.ndarray, shape (n,)
        Time vector in seconds.
    values : np.ndarray, shape (n,)
        Raw inter-pole distance signal in pixels.
    min_frq : float
        Maximum beat frequency in Hz.
    smooth_buffer : int
        Moving-average kernel width.
    run_max_window_sec : float, optional
        Diastolic baseline window in seconds. Default ``2.0``.

    Returns
    -------
    metrics : dict
        ``smoothed``         : np.ndarray, shape (n,) — smoothed signal [px]
        ``run_max``          : np.ndarray, shape (n,) — diastolic baseline [px]
        ``peak_idx``         : np.ndarray, shape (k,) — peak sample indices
        ``peak_val``         : np.ndarray, shape (k,) — peak values [px]
        ``peak_height``      : np.ndarray, shape (k,) — peakHeight = runMax - neigMin [px]
        ``contraction``      : np.ndarray, shape (k,) — contraction per peak [%]
        ``mean_contraction`` : float                  — mean contraction [%]
        ``freq``             : float                  — beat frequency [Hz]

    Notes
    -----
    .. code-block:: text

        peakHeight[i]  = runMax[peak_i] - neigMin[i]
        contraction[i] = peakHeight[i] / runMax[peak_i] * 100
        RR[i]          = time[peak_idx[i+1]] - time[peak_idx[i]]
        freq           = 1 / mean(RR)
    """
    dt = float(time[1] - time[0])

    smoothed             = smooth(values, smooth_buffer)
    run_max_arr          = running_max(smoothed, run_max_window_sec, dt)
    peak_idx, peak_val   = find_peaks(smoothed, min_frq, dt)

    run_max_at_peaks = run_max_arr[peak_idx]
    peak_height      = run_max_at_peaks - peak_val

    with np.errstate(invalid='ignore', divide='ignore'):
        contraction = np.where(
            run_max_at_peaks > 0,
            peak_height / run_max_at_peaks * 100,
            0.0,
        )

    mean_contraction = float(np.mean(contraction)) if len(contraction) > 0 else 0.0

    if len(peak_idx) >= 2:
        rr_intervals = np.diff(time[peak_idx])
        freq = float(1.0 / np.mean(rr_intervals))
    else:
        freq = 0.0

    return dict(
        smoothed=smoothed,
        run_max=run_max_arr,
        peak_idx=peak_idx,
        peak_val=peak_val,
        peak_height=peak_height,
        contraction=contraction,
        mean_contraction=mean_contraction,
        freq=freq,
    )