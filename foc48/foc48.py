"""
FOC48 - Pole Distance Analyzer.

Orchestrates the calibration stage (calibration.py) and the per-frame
processing stage (processing.py) into a single, stateful analyzer:
calibrates well positions/thresholds from a reference frame, then
processes a stream of frames (AVI, camera, or rolling buffer) to
measure each well's intra-pair distance over time. Results are exported
as CSV (distances in pixels) and a calibration XML.
"""

import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd
import yaml

from stream_source import StreamSource
from calibration import calibrate_wells, grid_label
from processing import process_frame_batch, warmup_process_pool, log_progress

_MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = _MODULE_DIR / "config.yaml"


def load_config(config_path: str) -> Dict:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_base_dir() -> Path:
    """
    Resolve the directory to use for locating bundled resources (like
    config.yaml), correctly handling both normal script execution and
    execution as a PyInstaller-frozen executable.

    Returns
    -------
    Path
        - When frozen (PyInstaller): the directory containing the .exe
          file itself, so that a config.yaml placed next to the .exe is
          found.
        - When run as a normal script: the directory containing this
          module.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent

_MODULE_DIR = get_base_dir()
DEFAULT_CONFIG_PATH = _MODULE_DIR / "config.yaml"

class FOC48:
    """
    FOC48 Pole Distance Analyzer.

    Calibrates well positions/thresholds from a reference frame, then
    processes a stream of frames to measure each well's intra-pair
    distance over time.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML config file. Defaults to "config.yaml" next to
        this module.

    Attributes
    ----------
    well_labels : list of str
        Well labels in fixed column order, e.g. ["A1", ..., "H6"].
    calibration : dict of str to dict, or None
        Per-well calibration data, set by `calibrate`.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path or str(DEFAULT_CONFIG_PATH))

        a = self.config["analysis"]
        self.n_rows = a["n_rows"]
        self.n_cols = a["n_cols"]
        self.well_labels = [grid_label(r, c) for r in range(self.n_rows) for c in range(self.n_cols)]
        self.area_min = a["area_min"]
        self.area_max = a["area_max"]
        self.invert_threshold = a["invert_threshold"]
        self.max_pair_distance = a["max_pair_distance"]
        self.local_margin_px = a["local_margin_px"]
        self.fallback_pix_thresh = a["fallback_pix_thresh"]
        self.min_blob_area = a["min_blob_area"]
        self.batch_size = a["batch_size"]
        self.max_pending_batches = a["max_pending_batches"]
        self.fps = a["fps"]

        self.calibration: Optional[Dict[str, Dict]] = None
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Signal the analyzer to stop processing as soon as possible."""
        self._stop_event.set()

    def calibrate(self, first_frame: np.ndarray) -> None:
        """
        Calibrate per-well ROIs and thresholds from a reference frame.

        Parameters
        ----------
        first_frame : np.ndarray
            Grayscale reference frame.
        """
        self.calibration = calibrate_wells(
            first_frame, self.n_rows, self.n_cols, self.area_min, self.area_max,
            self.invert_threshold, self.max_pair_distance, self.local_margin_px, self.fallback_pix_thresh
        )

    def calibrate_from_stream(self, stream_iter: Iterator[np.ndarray], output_dir: str, base_name: str) -> np.ndarray:
        """
        Consume the first frame from a stream, persist it as a reference
        BMP, and calibrate from it.

        Split out from `process_stream` so callers (e.g. benchmarks) can
        time calibration independently of steady-state throughput.

        Parameters
        ----------
        stream_iter : Iterator[np.ndarray]
            Iterator over grayscale frames, i.e. `iter(stream_source)`.
        output_dir : str
            Directory in which the first-frame BMP is saved.
        base_name : str
            Base name for the first-frame BMP file.

        Returns
        -------
        np.ndarray
            The first consumed frame. Include it when counting/processing
            the rest of the stream, since it won't be re-yielded.

        Raises
        ------
        ValueError
            If the stream produced no frames.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            first_frame = next(stream_iter)
        except StopIteration:
            raise ValueError("Stream produced no frames.")

        first_frame_path = output_path / f"{base_name}0.bmp"
        cv2.imwrite(str(first_frame_path), first_frame)
        print(f"First frame saved to: {first_frame_path}")

        self.calibrate(first_frame)
        return first_frame

    def process_stream(
            self,
            stream_iter: Iterator[np.ndarray],
            first_frame: np.ndarray,
            stream_source: StreamSource,
            num_workers: Optional[int] = None
    ) -> np.ndarray:
        """
        Process all frames from a stream into a 2D distance array, using
        a warmed-up pool of worker processes.

        Requires `calibrate` (or `calibrate_from_stream`) beforehand.

        Parameters
        ----------
        stream_iter : Iterator[np.ndarray]
            Iterator over the remaining frames.
        first_frame : np.ndarray
            The already-consumed first frame, included in processing.
        stream_source : StreamSource
            Underlying source, used for `total_frames` and closed when done.
        num_workers : int, optional
            Number of worker processes. Defaults to CPU count - 1.

        Returns
        -------
        np.ndarray
            2D array of shape (n_frames, n_wells), distances in pixels,
            NaN where a pole could not be located.

        Raises
        ------
        RuntimeError
            If no calibration is available.

        Notes
        -----
        The backpressure limit targets a fixed number of buffered frames
        (not batches), so it scales correctly regardless of `batch_size`.
        A batch-count-based limit would over-throttle at small batch
        sizes, serializing submissions and killing parallelism.
        """
        if self.calibration is None:
            raise RuntimeError("No calibration available. Call `calibrate` or `calibrate_from_stream` first.")

        num_workers = num_workers or max(1, os.cpu_count() - 1)
        target_buffered_frames = max(200, num_workers * 20)
        max_pending = max(num_workers + 2, target_buffered_frames // max(1, self.batch_size))

        results_by_batch: Dict[int, List[np.ndarray]] = {}
        pending: Dict = {}
        current_batch = [first_frame]
        batch_idx = 0
        frames_done = 0

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            warmup_time = warmup_process_pool(executor, num_workers)
            print(f"Worker pool warmed up ({num_workers} workers) in {warmup_time:.3f}s")

            start_time = time.perf_counter()

            def submit(batch: List[np.ndarray]) -> None:
                nonlocal batch_idx
                future = executor.submit(process_frame_batch, batch, self.well_labels,
                                          self.calibration, self.min_blob_area)
                pending[future] = (batch_idx, len(batch))
                batch_idx += 1

            def drain_one() -> None:
                nonlocal frames_done
                done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    idx, n = pending.pop(future)
                    results_by_batch[idx] = future.result()
                    frames_done += n
                    log_progress(frames_done, getattr(stream_source, "total_frames", 0), start_time)

            for frame in stream_iter:
                if self._stop_event.is_set():
                    break
                current_batch.append(frame)
                if len(current_batch) >= self.batch_size:
                    submit(current_batch)
                    current_batch = []
                    while len(pending) > max_pending:
                        drain_one()

            if current_batch:
                submit(current_batch)
            while pending:
                drain_one()

        print()
        stream_source.close()

        ordered = [results_by_batch[i] for i in sorted(results_by_batch)]
        flat = [row for batch in ordered for row in batch]
        return np.vstack(flat) if flat else np.empty((0, len(self.well_labels)))

    def run(
            self,
            stream_source: StreamSource,
            output_dir: str,
            base_name: str,
            num_workers: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Run the full pipeline: calibrate, process, save CSV and XML.

        Convenience wrapper around `calibrate_from_stream` and
        `process_stream`. For benchmarking, call those directly instead
        to time each stage independently.

        Parameters
        ----------
        stream_source : StreamSource
            Source of grayscale frames.
        output_dir : str
            Directory for CSV, calibration XML, and first-frame BMP.
        base_name : str
            Base name for all output files.
        num_workers : int, optional
            Number of worker processes. Defaults to CPU count - 1.

        Returns
        -------
        pd.DataFrame
            The resulting distance data (also saved as CSV).

        Notes
        -----
        Output naming: "<base_name>_output.csv", "<base_name>_calibration.xml",
        "<base_name>0.bmp".
        """
        stream_iter = iter(stream_source)
        first_frame = self.calibrate_from_stream(stream_iter, output_dir, base_name)
        distances_2d = self.process_stream(stream_iter, first_frame, stream_source, num_workers)

        n_frames = len(distances_2d)
        n_missing = np.isnan(distances_2d).sum(axis=0)
        missing_wells = [(l, int(n)) for l, n in zip(self.well_labels, n_missing) if n > 0]
        if missing_wells:
            print(f"Warning: missing detections in {len(missing_wells)} well(s):")
            for label, n in sorted(missing_wells, key=lambda x: -x[1])[:10]:
                print(f"  {label}: {n} / {n_frames} frames ({100.0 * n / n_frames:.1f}%)")

        output_path = Path(output_dir)
        df = self.save_csv(distances_2d, str(output_path / f"{base_name}_output.csv"))
        self.save_calibration_xml(str(output_path / f"{base_name}_calibration.xml"), source_name=base_name)
        return df

    def save_csv(self, distances_2d: np.ndarray, output_path: str) -> pd.DataFrame:
        """
        Save distance results (pixels) to CSV.

        Parameters
        ----------
        distances_2d : np.ndarray
            2D array of shape (n_frames, n_wells).
        output_path : str
            Output CSV path.

        Returns
        -------
        pd.DataFrame
            The saved DataFrame.
        """
        time_vector = np.arange(len(distances_2d), dtype=np.float64) / self.fps
        df = pd.DataFrame(np.column_stack([time_vector, distances_2d]), columns=['time'] + self.well_labels)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, float_format='%.6f')
        print(f"CSV saved: {output_path}")
        return df

    def save_calibration_xml(self, output_path: str, source_name: str = "") -> None:
        """
        Save per-well calibration data as XML.

        Parameters
        ----------
        output_path : str
            Output XML path.
        source_name : str, optional
            Source name recorded for traceability.

        Raises
        ------
        RuntimeError
            If no calibration is available.
        """
        if self.calibration is None:
            raise RuntimeError("No calibration available. Call `calibrate` or `run` first.")

        root = ET.Element("calibration")
        meta = ET.SubElement(root, "meta")
        ET.SubElement(meta, "source").text = source_name

        for label in self.well_labels:
            c = self.calibration[label]
            well = ET.SubElement(root, label)
            ET.SubElement(well, "pixThreshL").text = f"{c['pix_thresh_l']:.3f}"
            ET.SubElement(well, "pixThreshR").text = f"{c['pix_thresh_r']:.3f}"
            ET.SubElement(well, "roiLeft").text = str(c['roi_left'])
            ET.SubElement(well, "roiRight").text = str(c['roi_right'])
            ET.SubElement(well, "templateLeft").text = str(c['template_left'])
            ET.SubElement(well, "templateRight").text = str(c['template_right'])

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
        print(f"Calibration XML saved: {output_path}")
