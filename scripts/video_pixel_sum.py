import subprocess
import shutil
import cv2
import numpy as np
import tempfile
import os
import sys


def find_7z_executable() -> str:
    """
    Locates the 7z executable on the system.
    Checks PATH first, then common Windows install locations.
    """
    # Check if 7z is available in PATH
    for name in ("7z", "7z.exe"):
        path = shutil.which(name)
        if path:
            return path

    # Common Windows installation paths
    common_paths = [
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
    ]
    for path in common_paths:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "7z executable not found. Please install 7-Zip "
        "(https://www.7-zip.org/) and ensure it is in your PATH, "
        "or edit find_7z_executable() with the correct path."
    )


def extract_avi_from_7z(archive_path: str, extract_dir: str) -> str:
    """
    Extracts the archive using the system 7z tool and returns the
    path to the extracted .avi file.
    """
    seven_zip = find_7z_executable()

    # "x" = extract with full paths, "-y" = assume yes to prompts,
    # "-o<dir>" = output directory (no space after -o!)
    cmd = [seven_zip, "x", archive_path, f"-o{extract_dir}", "-y"]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"7z extraction failed (exit code {result.returncode}):\n"
            f"{result.stderr}"
        )

    for root, _, files in os.walk(extract_dir):
        for file in files:
            if file.lower().endswith('.avi'):
                return os.path.join(root, file)

    raise FileNotFoundError("No .avi file found inside the archive.")


def compute_frame_differences(avi_path: str) -> list[int]:
    """
    Reads an 8-bit monochrome video frame by frame and computes,
    for each frame (from the second onward), the sum of absolute
    pixel differences relative to the previous frame.
    """
    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {avi_path}")

    differences = []
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = frame.astype(np.uint8)

        if prev_frame is not None:
            diff = np.abs(frame.astype(np.int16) - prev_frame.astype(np.int16))
            pixel_sum = int(np.sum(diff))
            differences.append(pixel_sum)

        prev_frame = frame

    cap.release()
    return differences


def main():
    if len(sys.argv) < 2:
        print("Usage: python frame_diff.py <path_to_file.avi.7z>")
        sys.exit(1)

    archive_path = sys.argv[1]
    if not os.path.isfile(archive_path):
        print(f"File not found: {archive_path}")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        print("Extracting archive using system 7z ...")
        avi_path = extract_avi_from_7z(archive_path, tmp_dir)
        print(f"Extracted video file: {avi_path}")

        print("Computing pixel differences between consecutive frames ...")
        diffs = compute_frame_differences(avi_path)

    print(f"\nNumber of evaluated frame transitions: {len(diffs)}\n")
    for i, d in enumerate(diffs, start=1):
        print(f"Frame {i:5d}: {d}")

    output_file = "frame_differences.csv"
    with open(output_file, "w") as f:
        f.write("frame_index,pixel_difference_sum\n")
        for i, d in enumerate(diffs, start=1):
            f.write(f"{i},{d}\n")
    print(f"\nResults also saved to '{output_file}'.")


if __name__ == "__main__":
    main()