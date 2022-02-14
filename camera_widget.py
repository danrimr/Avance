import cv2
import sys
import time
from threading import Thread
import numpy as np
import urllib.request


class CameraWidget:
    """Allows to show the source frames in a different thread."""

    def __init__(self, src=0, dst="test/test.jpg") -> None:
        if isinstance(src, int):
            self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            self.target = self.show_frame
        if isinstance(src, str):
            self.url = src
            self.target = self.show_ip_frame
        self.frame = np.empty(shape=[600, 800, 3])
        self.dst = dst
        self.text = ""

    def show_frame(self) -> None:
        """Shows a frame from a camera source in a cv2 windows widget."""
        while True:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
            time.sleep(0.01)
            cv2.putText(
                self.frame,
                self.text,
                org=(450, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 0, 0),
                thickness=2,
                lineType=2,
            )
            cv2.imshow("Frame", self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.capture.release()
                cv2.destroyAllWindows()
                sys.exit(0)

    def show_ip_frame(self) -> None:
        """Shows a frame from a web source in a cv2 windows widget."""
        while True:
            frame_resp = urllib.request.urlopen(self.url)
            frame_array = np.array(bytearray(frame_resp.read()), dtype=np.uint8)
            self.frame = cv2.imdecode(frame_array, -1)
            cv2.imshow("IWebcam", self.frame)
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                sys.exit(0)

    def save_frame(self) -> None:
        """Saves the current frame in local storage."""
        cv2.imwrite(self.dst, self.frame)

    def start_stream(self) -> None:
        """Starts the frames stream in a different thread."""
        self.widget_thread = Thread(target=self.target)
        self.widget_thread.daemon = True
        self.widget_thread.start()

    def set_text(self, text) -> None:
        """Add in-frame text to the current capture frame."""
        self.text = text
