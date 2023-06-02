"""Video buffer thread module."""

from threading import Thread, Event
from collections import deque
from time import sleep
import time
import cv2
import logging

class VideoBufferThread(Thread):
    """Threaded video buffer."""
    def __init__(self, rtsp_url, buffer_length=50):
        """Initialize the VideoBufferThread.

        Args:
            rtsp_url (str): The RTSP URL of the video stream.\n
            fourcc (VideoWriter_fourcc): The fourcc code of the output video file.\n
            buffer_length (int): The number of frames to keep in the buffer.
        """
        super().__init__()
        self.daemon = True
        self.rtsp_url = rtsp_url
        self.buffer_length = buffer_length
        self.buffer = deque(maxlen=buffer_length)
        self._stop_event = Event()
        self.retry = 0

    def run(self):
        """Read frames from the RTSP stream."""
        try:
            cap = self.start_capture()
            if cap is not None:
                while not self.stopped():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    self.buffer.append(frame)
                    if len(self.buffer) == self.buffer_length:
                        self.buffer.popleft()
                cap.release()
        except cv2.error:
            logging.error(f"Error while reading the RTSP stream: {self.rtsp_url}")
            self.stop()
            pass

        
    def start_capture(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self.rtsp_url)
        except cv2.error:
            logging.error(f"Error while connecting the RTSP stream: {self.rtsp_url}")
            logging.error(f"Retry {self.retry}/3")
            if self.retry < 3:
                self.retry += 1
                time.sleep(3)
                self.start_capture()
            self.stop()
        finally:
            return cap

    def wait_first_frame(self):
        """Wait for the first frame to be added to the buffer."""
        while len(self.buffer) == 0 or self.stopped():
            sleep(0.01)

    def stopped(self):
        """Check if the thread has been stopped.

        Returns:
            bool: True if the thread has been stopped, False otherwise.
        """
        return self._stop_event.is_set()

    def stop(self):
        """Stop the thread."""
        self._stop_event.set()

    def get_latest_frame(self):
        """Get the latest frame from the buffer.

        Returns:
            np.ndarray: The latest frame.
        """
        return self.buffer[-1].copy()
