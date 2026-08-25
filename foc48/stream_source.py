"""
StreamSource abstraction for FOC48.

Unified interface for different frame sources: AVI video files, live
generic cameras (OpenCV), live Basler cameras (Pylon/USB3 Vision), and
an in-memory rolling buffer (for testing without a live camera).
"""

import time
from abc import ABC, abstractmethod
from queue import Queue, Empty
from typing import Iterator, Optional

import cv2
import numpy as np

try:
    from pypylon import pylon
except ImportError:
    pylon = None


class StreamSource(ABC):
    """
    Abstract base class for all frame sources.

    Attributes
    ----------
    total_frames : int
        Expected total frame count, or 0 if unknown/unbounded (e.g. a
        live camera). Used only for progress reporting.
    """

    total_frames: int = 0

    @abstractmethod
    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield grayscale frames (H, W), uint8."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release held resources. Safe to call multiple times."""
        raise NotImplementedError


class AVIStreamSource(StreamSource):
    """
    Frame source reading sequentially from an AVI file.

    Parameters
    ----------
    video_path : str
        Path to the AVI file.
    start_frame : int, optional
        Frame index to start from (default 0).
    end_frame : int, optional
        Frame index (exclusive) to stop at (default: end of file).

    Raises
    ------
    IOError
        If the video cannot be opened.
    """

    def __init__(self, video_path: str, start_frame: int = 0, end_frame: Optional[int] = None):
        self.video_path = video_path
        self.start_frame = start_frame
        self.end_frame = end_frame

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = (min(total, end_frame) if end_frame is not None else total) - start_frame
        self.total_frames = max(0, total)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield grayscale frames from `start_frame` to `end_frame`."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        frame_idx = self.start_frame

        while self.end_frame is None or frame_idx < self.end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            frame_idx += 1

    def close(self) -> None:
        """Release the underlying `cv2.VideoCapture`."""
        self.cap.release()


class CameraStreamSource(StreamSource):
    """
    Frame source reading live frames from a generic OpenCV-compatible
    camera. Unbounded (`total_frames = 0`); yields until externally
    stopped.

    Parameters
    ----------
    camera_id : int, optional
        OpenCV camera index (default 0).
    fps : int, optional
        Requested capture rate (default 60); actual rate depends on
        hardware/driver.
    width, height : int, optional
        Requested frame size (default 2820, 1912).

    Raises
    ------
    IOError
        If the camera cannot be opened.
    """

    def __init__(self, camera_id: int = 0, fps: int = 60, width: int = 2820, height: int = 1912):
        self.camera_id = camera_id
        self.total_frames = 0

        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise IOError(f"Could not open camera device: {camera_id}")

        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Yield grayscale frames indefinitely; never raises StopIteration
        on its own.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    def close(self) -> None:
        """Release the underlying `cv2.VideoCapture`."""
        self.cap.release()


class BaslerStreamSource(StreamSource):
    """
    Frame source reading live frames from a Basler camera via Pylon
    (USB3 Vision). Frame acquisition, DMA transfer from the camera
    interface into host memory, and buffering are handled internally by
    the Pylon runtime; this class retrieves already-buffered frames from
    Pylon's acquisition queue.

    Parameters
    ----------
    serial_number : str, optional
        Serial number of the camera to open. If None (default), opens
        the first camera found by the transport layer factory.
    fps : float, optional
        Requested acquisition frame rate (default 60.0).
    exposure_time_us : float, optional
        Exposure time in microseconds. If None (default), uses the
        camera's current/default setting.
    num_buffers : int, optional
        Depth of Pylon's internal acquisition buffer queue (default 10).
    duration_seconds : float, optional
        If given, the stream automatically stops this many seconds after
        the first frame arrives (default None: runs until externally
        stopped via `close()`).
    timeout_ms : int, optional
        Timeout for each frame retrieval attempt, in milliseconds
        (default 1000).

    Raises
    ------
    ImportError
        If `pypylon` is not installed.
    RuntimeError
        If no camera is found, or the specified serial number is not
        connected.
    """

    def __init__(
            self,
            serial_number: Optional[str] = None,
            fps: float = 60.0,
            exposure_time_us: Optional[float] = None,
            num_buffers: int = 10,
            duration_seconds: Optional[float] = None,
            timeout_ms: int = 1000
    ):
        if pylon is None:
            raise ImportError(
                "pypylon is not installed. Run: pip install pypylon\n"
                "The Basler Pylon Runtime must also be installed on this machine."
            )

        self.total_frames = 0
        self.duration_seconds = duration_seconds
        self.timeout_ms = timeout_ms
        self._closed = False

        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        if not devices:
            raise RuntimeError("No Basler camera found.")

        if serial_number is not None:
            device = next((d for d in devices if d.GetSerialNumber() == serial_number), None)
            if device is None:
                raise RuntimeError(f"Camera with serial number {serial_number} not found.")
        else:
            device = devices[0]

        self.camera = pylon.InstantCamera(tl_factory.CreateDevice(device))
        self.camera.Open()

        print(f"Connected to: {self.camera.GetDeviceInfo().GetModelName()} "
              f"(S/N: {self.camera.GetDeviceInfo().GetSerialNumber()})")

        # Mono8 gives a direct (H, W) uint8 array via GrabResult.Array,
        # matching our grayscale pipeline exactly - no color conversion needed.
        self.camera.PixelFormat.SetValue("Mono8")

        if exposure_time_us is not None:
            self.camera.ExposureTime.SetValue(exposure_time_us)

        self.camera.AcquisitionFrameRateEnable.SetValue(True)
        self.camera.AcquisitionFrameRate.SetValue(fps)
        self.camera.MaxNumBuffer = num_buffers

        actual_fps = self.camera.ResultingFrameRate.GetValue()
        print(f"Requested {fps} fps, camera reports achievable rate: {actual_fps:.1f} fps")

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Iterate over grayscale frames from Pylon's acquisition queue, in
        strict acquisition order.

        Yields
        ------
        np.ndarray
            Grayscale frame (H, W), copied out of Pylon's buffer before
            the buffer is released back to the acquisition queue.

        Notes
        -----
        Uses `GrabStrategy_OneByOne` so every acquired frame is yielded
        (rather than `GrabStrategy_LatestImageOnly`, which would silently
        drop frames if the consumer falls behind) - required here since
        a contraction time series needs every frame, not just the latest.

        Each buffer is explicitly copied before `grabResult.Release()`,
        since Pylon reuses that memory for the next incoming frame.
        """
        self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
        start_time = None

        while self.camera.IsGrabbing() and not self._closed:
            grab_result = self.camera.RetrieveResult(self.timeout_ms, pylon.TimeoutHandling_ThrowException)

            if not grab_result.GrabSucceeded():
                print(f"Warning: grab failed - {grab_result.ErrorDescription}")
                grab_result.Release()
                continue

            frame = grab_result.Array.copy()
            grab_result.Release()

            if start_time is None:
                start_time = time.perf_counter()

            yield frame

            if self.duration_seconds is not None and time.perf_counter() - start_time >= self.duration_seconds:
                break

        self.camera.StopGrabbing()

    def close(self) -> None:
        """
        Stop acquisition and release the camera handle. Safe to call
        multiple times.
        """
        self._closed = True
        if self.camera.IsGrabbing():
            self.camera.StopGrabbing()
        if self.camera.IsOpen():
            self.camera.Close()


class RollingBufferSource(StreamSource):
    """
    Frame source reading from an in-memory queue, simulating a rolling
    buffer fed by an external producer.

    Parameters
    ----------
    frame_queue : queue.Queue
        Queue of `np.ndarray` frames. A `None` item signals end-of-stream.
    expected_frames : int, optional
        Stop after this many frames, even if the queue isn't exhausted
        (default: rely on the `None` sentinel or `close()`).
    poll_timeout : float, optional
        Queue read timeout in seconds before re-checking stop conditions
        (default 0.1).
    """

    def __init__(
            self,
            frame_queue: "Queue[Optional[np.ndarray]]",
            expected_frames: Optional[int] = None,
            poll_timeout: float = 0.1
    ):
        self.frame_queue = frame_queue
        self.total_frames = expected_frames or 0
        self._expected_frames = expected_frames
        self._poll_timeout = poll_timeout
        self._closed = False

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Yield frames until `close()` is called, a `None` sentinel is
        read, or `expected_frames` is reached.
        """
        n_yielded = 0
        while not self._closed:
            if self._expected_frames is not None and n_yielded >= self._expected_frames:
                break
            try:
                frame = self.frame_queue.get(timeout=self._poll_timeout)
            except Empty:
                continue
            if frame is None:
                break
            yield frame
            n_yielded += 1

    def close(self) -> None:
        """Signal the source to stop yielding frames."""
        self._closed = True