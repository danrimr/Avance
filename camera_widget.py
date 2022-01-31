import cv2
import sys
import time
from threading import Thread


class CameraWidget:
    def __init__(self, src=0) -> None:
        self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)

    def show_frame(self):
        while True:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
            time.sleep(0.01)
            cv2.imshow("Frame", self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.capture.release()
                cv2.destroyAllWindows()
                sys.exit(0)

    def save_frame(self, path="test/test.jpg"):
        cv2.imwrite(path, self.frame)

    def start_local_stream(self):
        self.widget_thread = Thread(target=self.show_frame)
        self.widget_thread.daemon = True
        self.widget_thread.start()
