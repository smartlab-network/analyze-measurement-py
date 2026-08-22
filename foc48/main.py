"""
Main entry point for the FOC48 Pole Distance Analyzer.

Supports three input modes:
- avi:    Read frames sequentially from a recorded AVI video file.
- camera: Read frames live from a connected camera (default mode).
- buffer: Read frames from a simulated rolling buffer, fed from
          pre-extracted BMP files (see extract_frames.py). Useful for
          testing the streaming pipeline without a live camera.

Output files (CSV, calibration XML, first-frame BMP) are named based on
the input source, so that results from different videos/sessions don't
overwrite each other.
"""

import argparse
from pathlib import Path
from queue import Queue

import cv2

from foc48.foc48 import FOC48
from stream_source import AVIStreamSource, CameraStreamSource, RollingBufferSource, StreamSource


def build_stream_source(args: argparse.Namespace) -> StreamSource:
    """
    Construct the appropriate StreamSource based on CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    StreamSource
        The constructed stream source, ready to be iterated over.

    Raises
    ------
    ValueError
        If `args.mode` is not one of "avi", "camera", "buffer".
    FileNotFoundError
        If the required input file(s) cannot be found.
    """
    if args.mode == "avi":
        return AVIStreamSource(args.input, args.start_frame, args.end_frame)

    if args.mode == "camera":
        return CameraStreamSource(
            camera_id=args.camera_id,
            fps=int(args.fps) if args.fps else 60
        )

    if args.mode == "buffer":
        bmp_dir = Path(args.input) if args.input not in (None, "", "camera") else Path(
            "data/test_bmps")
        bmp_files = sorted(bmp_dir.glob("*.bmp"))
        if not bmp_files:
            raise FileNotFoundError(f"No .bmp files found in: {bmp_dir.resolve()}")

        frame_queue: Queue = Queue()
        n_loaded = 0
        for bmp_path in bmp_files[:args.buffer_size]:
            img = cv2.imread(str(bmp_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            frame_queue.put(img)
            n_loaded += 1

        return RollingBufferSource(frame_queue, expected_frames=n_loaded)

    raise ValueError(f"Unknown mode: {args.mode}")


def resolve_base_name(args: argparse.Namespace) -> str:
    """
    Derive the output base name from the input source.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    str
        Base name used for all output files (CSV, calibration XML,
        first-frame BMP).

    Notes
    -----
    Naming convention:
    - avi mode:    the video file's stem (e.g. "myrPlate_demo" from
                   "myrPlate_demo.avi")
    - camera mode: "live_stream"
    - buffer mode: "buffer_test"
    """
    if args.mode == "avi":
        return Path(args.input).stem
    if args.mode == "camera":
        return "live_stream"
    return "buffer_test"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="FOC48 - Pole Distance Analyzer"
    )
    parser.add_argument(
        "input",
        help="Path to AVI file (avi mode) or BMP directory (buffer mode). "
             "Ignored in camera mode (pass any placeholder, e.g. 'camera')."
    )
    parser.add_argument(
        "--mode", choices=["avi", "camera", "buffer"], default="camera",
        help="Stream mode (default: camera)"
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to config YAML file (default: config.yaml next to foc48.py)"
    )
    parser.add_argument(
        "--output-dir", default="analyze_measurement",
        help="Output directory for CSV, calibration XML, and first-frame BMP "
             "(default: analyze_measurement)"
    )
    parser.add_argument(
        "--start-frame", type=int, default=0,
        help="Start frame index for avi mode (default: 0)"
    )
    parser.add_argument(
        "--end-frame", type=int, default=None,
        help="End frame index (exclusive) for avi mode (default: read until end)"
    )
    parser.add_argument(
        "--camera-id", type=int, default=0,
        help="Camera device ID for camera mode (default: 0)"
    )
    parser.add_argument(
        "--buffer-size", type=int, default=1000,
        help="Maximum number of BMP frames to load for buffer mode (default: 1000)"
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Override the frame rate used for the CSV time column and, "
             "in camera mode, the requested capture rate"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel worker processes (default: CPU count - 1)"
    )
    return parser.parse_args()


def main() -> int:
    """
    Entry point.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    args = parse_args()

    print("=" * 60)
    print("FOC48 - Pole Distance Analyzer")
    print("=" * 60)
    print(f"Mode: {args.mode}")

    stream_source = build_stream_source(args)
    base_name = resolve_base_name(args)

    analyzer = FOC48(config_path=args.config)
    if args.fps is not None:
        analyzer.fps = args.fps

    analyzer.run(
        stream_source,
        output_dir=args.output_dir,
        base_name=base_name,
        num_workers=args.workers
    )

    return 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    exit(main())