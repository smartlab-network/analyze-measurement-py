"""
Main entry point for the FOC48 Pole Distance Analyzer.

Supports four input modes:
- avi:     Read frames sequentially from a recorded AVI video file.
- camera:  Read frames live from a generic OpenCV-compatible camera.
- basler:  Read frames live from a Basler USB3 camera via Pylon (pypylon).
- buffer:  Read frames from a simulated rolling buffer, fed from
           pre-extracted BMP files (see extract_frames.py).

Output files (CSV, calibration XML, first-frame BMP) are named based on
the input source.
"""

import argparse
from pathlib import Path
from queue import Queue

import cv2

from foc48.foc48 import FOC48

from stream_source import (
    AVIStreamSource, CameraStreamSource, BaslerStreamSource,
    RollingBufferSource, StreamSource
)


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
        The constructed, ready-to-iterate stream source.

    Raises
    ------
    ValueError
        If `args.mode` is not one of "avi", "camera", "basler", "buffer".
    FileNotFoundError
        If the required input file(s) cannot be found.
    """
    if args.mode == "avi":
        if not args.input:
            raise ValueError("avi mode requires an input video path.")
        return AVIStreamSource(args.input, args.start_frame, args.end_frame)

    if args.mode == "basler":
        return BaslerStreamSource(
            serial_number=args.serial,
            fps=args.fps or 60.0,
            exposure_time_us=args.exposure_us,
            duration_seconds=args.duration
        )

    if args.mode == "buffer":
        bmp_dir = Path(args.input) if args.input not in (None, "", "camera") else Path("data/test_bmps")
        bmp_files = sorted(bmp_dir.glob("*.bmp"))
        if not bmp_files:
            raise FileNotFoundError(f"No .bmp files found in: {bmp_dir.resolve()}")

        frame_queue: Queue = Queue()
        n_loaded = 0
        for bmp_path in bmp_files[:args.buffer_size]:
            img = cv2.imread(str(bmp_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
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
        Base name for all output files.
    """
    if args.mode == "avi":
        return Path(args.input).stem
    return {"camera": "live_stream", "basler": "basler_live", "buffer": "buffer_test"}[args.mode]


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="FOC48 - Pole Distance Analyzer")
    parser.add_argument(
        "input", nargs="?", default=None,
        help="AVI file path (avi mode) or BMP directory (buffer mode); "
             "not needed in basler mode"
    )
    parser.add_argument("--mode", choices=["avi", "camera", "basler", "buffer"], default="camera")
    parser.add_argument("--config", default=None, help="Path to config YAML (default: config.yaml next to foc48.py)")
    parser.add_argument("--output-dir", default="analyze_measurement")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=1000)
    parser.add_argument("--fps", type=float, default=None,
                         help="Override CSV time-column fps and, in camera/basler mode, capture rate")
    parser.add_argument("--workers", type=int, default=None, help="Worker processes (default: CPU count - 1)")

    parser.add_argument("--serial", default=None, help="Basler camera serial number (basler mode)")
    parser.add_argument("--exposure-us", type=float, default=None, help="Exposure time in microseconds")
    parser.add_argument("--duration", type=float, default=60.0, help="Recording duration in seconds (basler mode)")

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

    analyzer.run(stream_source, output_dir=args.output_dir, base_name=base_name, num_workers=args.workers)
    return 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    exit(main())