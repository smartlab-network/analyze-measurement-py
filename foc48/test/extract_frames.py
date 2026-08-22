"""
Utility script to extract frames from an AVI file as numbered grayscale
BMP images, for use with the simulated rolling buffer test.
"""

import argparse
from pathlib import Path

import cv2


def extract_frames(video_path: str, output_dir: str, start_frame: int = 0, end_frame: int = None) -> int:
    """
    Extract frames from a video file and save them as numbered grayscale
    BMP images.

    Parameters
    ----------
    video_path : str
        Path to the input AVI video file.
    output_dir : str
        Directory in which to save the extracted BMP frames. Created if
        it does not already exist.
    start_frame : int, optional
        Frame index to start extraction from (default 0).
    end_frame : int, optional
        Frame index (exclusive) to stop extraction at. If None (default),
        extracts until the end of the video.

    Returns
    -------
    int
        Number of frames actually extracted.

    Raises
    ------
    FileNotFoundError
        If the video file does not exist.
    IOError
        If the video file could not be opened by OpenCV.
    """
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_file.resolve()}")

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_file}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    n_extracted = 0

    while True:
        if end_frame is not None and frame_idx >= end_frame:
            break

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        # 6-digit zero-padded filename ensures correct lexicographic sort
        out_path = out_dir / f"frame_{frame_idx:06d}.bmp"
        cv2.imwrite(str(out_path), gray)

        frame_idx += 1
        n_extracted += 1

        if n_extracted % 100 == 0:
            print(f"Extracted {n_extracted} frames...")

    cap.release()
    print(f"Done. Extracted {n_extracted} frames to: {out_dir.resolve()}")
    return n_extracted


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Extract frames from an AVI file as numbered grayscale BMP images."
    )
    parser.add_argument("video", help="Path to the input AVI video file")
    parser.add_argument("-o", "--output-dir", default="data/test_bmps",
                         help="Output directory for extracted BMP frames "
                              "(default: data/test_bmps)")
    parser.add_argument("--start-frame", type=int, default=0,
                         help="Frame index to start extraction from (default: 0)")
    parser.add_argument("--end-frame", type=int, default=None,
                         help="Frame index (exclusive) to stop extraction at "
                              "(default: extract until end of video)")
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
    extract_frames(args.video, args.output_dir, args.start_frame, args.end_frame)
    return 0


if __name__ == "__main__":
    exit(main())