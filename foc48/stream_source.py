"""
StreamSource abstraction for FOC48.

Provides a unified interface for different frame sources: AVI video files,
live camera input, and an in-memory rolling buffer (used for testing the
streaming pipeline without a live camera).
"""

from abc import ABC, abstractmethod
from queue import Queue, Empty
from typing import Iterator, Optional

import cv2
import numpy as np


class StreamSource(ABC):
    """
    Abstract base class for all frame sources.

    Subclasses must implement `__iter__` (yielding grayscale frames) and
    `close` (releasing any held resources).

    Attributes
    ----------
    total_frames : int
        Total number of frames the source is expected to yield, or 0 if
        unknown/unbounded (e.g. a live camera stream). Used only for
        progress reporting.
    """

    total_frames: int = 0

    @abstractmethod
    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Iterate over grayscale frames.

        Yields
        ------
        np.ndarray
            Grayscale frame (H, W) in uint8 format.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """
        Release any resources held by the stream source (file handles,
        camera device, etc.). Safe to call multiple times.
        """
        raise NotImplementedError


class AVIStreamSource(StreamSource):
    """
    Frame source reading sequentially from an AVI video file.

    Parameters
    ----------
    video_path : str
        Path to the AVI video file.
    start_frame : int, optional
        Frame index to start reading from (default 0).
    end_frame : int, optional
        Frame index (exclusive) to stop reading at. If None (default),
        reads until the end of the file.

    Raises
    ------
    IOError
        If the video file cannot be opened.
    """

    def __init__(self, video_path: str, start_frame: int = 0, end_frame: Optional[int] = None):
        self.video_path = video_path
        self.start_frame = start_frame
        self.end_frame = end_frame

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if end_frame is not None:
            total = min(total, end_frame) - start_frame
        else:
            total = total - start_frame
        self.total_frames = max(0, total)

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Iterate over grayscale frames from `start_frame` to `end_frame`.

        Yields
        ------
        np.ndarray
            Grayscale frame (H, W) in uint8 format.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        frame_idx = self.start_frame

        while True:
            if self.end_frame is not None and frame_idx >= self.end_frame:
                break

            ret, frame = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            yield gray
            frame_idx += 1

    def close(self) -> None:
        """
        Release the underlying `cv2.VideoCapture`.
        """
        self.cap.release()


class CameraStreamSource(StreamSource):
    """
    Frame source reading live frames from a camera device.

    The stream is unbounded (`total_frames = 0`); it yields frames until
    externally stopped (e.g. via `FOC48.stop()`).

    Parameters
    ----------
    camera_id : int, optional
        OpenCV camera device index (default 0).
    fps : int, optional
        Requested capture frame rate (default 60). Actual achievable rate
        depends on the camera hardware and driver.
    width : int, optional
        Requested frame width in pixels (default 2820).
    height : int, optional
        Requested frame height in pixels (default 1912).

    Raises
    ------
    IOError
        If the camera device cannot be opened.
    """

    def __init__(
            self,
            camera_id: int = 0,
            fps: int = 60,
            width: int = 2820,
            height: int = 1912
    ):
        self.camera_id = camera_id
        self.total_frames = 0  # unbounded / unknown

        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise IOError(f"Could not open camera device: {camera_id}")

        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def __iter__(self) -> Iterator[np.ndarray]:
        """
        Iterate indefinitely over grayscale frames from the camera.

        Yields
        ------
        np.ndarray
            Grayscale frame (H, W) in uint8 format.

        Notes
        -----
        This generator never raises `StopIteration` on its own; the
        caller is responsible for breaking out of the loop (e.g. via an
        external stop signal).
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            yield gray

    def close(self) -> None:
        """
        Release the underlying `cv2.VideoCapture`.
        """
        self.cap.release()


class RollingBufferSource(StreamSource):
    """
    Frame source reading from an in-memory queue, simulating a rolling
    buffer fed by an external producer (e.g. a microscope acquisition
    thread). Used to test the streaming pipeline without a live camera.

    Parameters
    ----------
    frame_queue : queue.Queue
        Queue from which frames are consumed. Frames should be
        `np.ndarray` grayscale images. A `None` item in the queue is
        treated as an end-of-stream sentinel.
    expected_frames : int, optional
        If given, the source stops after yielding this many frames, even
        if the queue is not exhausted (default None, i.e. rely on the
        `None` sentinel or `close()` instead).
    poll_timeout : float, optional
        Timeout (seconds) for each queue read attempt before checking
        whether the source should stop (default 0.1).
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
        Iterate over frames from the queue until a stop condition is met.

        Yields
        ------
        np.ndarray
            Grayscale frame (H, W) in uint8 format.

        Notes
        -----
        Stops when:
        - `close()` has been called, or
        - a `None` sentinel is read from the queue, or
        - `expected_frames` frames have been yielded (if set).
        """
        n_yielded = 0

        while not self._closed:
            if self._expected_frames is not None and n_yielded >= self._expected_frames:
                break

            try:
                frame = self.frame_queue.get(timeout=self._poll_timeout)
            except Empty:
                continue

            if frame is None:  # end-of-stream sentinel
                break

            yield frame
            n_yielded += 1

    def close(self) -> None:
        """
        Signal the source to stop yielding frames on the next check.
        """
        self._closed = True