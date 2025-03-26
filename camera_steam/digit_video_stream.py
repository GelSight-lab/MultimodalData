import cv2
import numpy as np
from typing import Tuple
import threading
import time

from ..camera_steam.base_video_stream import BaseVideoStream
from digit_interface import Digit

class DigitVideoStream(BaseVideoStream):
    def __init__(self, serial_num: str, resolution: Tuple[int, int] = (640, 480), 
            format="BGR", restart_interval=10, verbose=True):
        super(DigitVideoStream, self).__init__(resolution, format, verbose=verbose)
        self.serial_num = serial_num
        self.last_restarted = time.time()
        self.restart_interval = restart_interval

    def start(self, create_thread=True):
        self.sensor = Digit(self.serial_num)
        self.sensor.connect()
        self.last_restarted = time.time()

        self.streaming = True
        if create_thread:
            threading.Thread(target=self.update, args=()).start()
    
    def stop(self):
        self.sensor.disconnect()
        del self.sensor

    def update(self):
        while True:
            if not self.streaming:
                time.sleep(0.01)
                continue
            if time.time() - self.last_restarted > self.restart_interval:
                self.restart()
                # buffering to wait for AWB / exposure to settle
                cur_t = time.time()
                while time.time() - cur_t < 2:
                    self.sensor.get_frame(transpose=True)
                    time.sleep(0.01)
            try:
                frame = self.sensor.get_frame(transpose=True)
                if self.resolution != (frame.shape[1], frame.shape[0]):
                    frame = cv2.resize(frame, self.resolution)
                if self.format == "RGB":
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame = frame
                self.write_frame(frame)
                self.last_updated = time.time()
            except Exception as e: 
                print(e)
                print("Error reading frame. Trying to ignore...")
                time.sleep(0.1)
                continue
            else:
                time.sleep(0.01)