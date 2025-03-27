import cv2
import numpy as np
from typing import Tuple
import threading
import time
import pyudev

from .base_video_stream import BaseVideoStream
from ..misc.utils import logging

class USBVideoStream(BaseVideoStream):
    def __init__(self, serial: str = "", usb_id: int = 0, resolution: Tuple[int, int] = (640, 480), format="BGR", verbose=True):
        super(USBVideoStream, self).__init__(resolution, format, verbose=verbose)
        self.serial = serial
        self.usb_id = usb_id
        self.fps = 25
    
    def parse_serial(self, serial: str):
        """Parse serial number and find the corresponding usb id"""
        # list all video devices
        context = pyudev.Context()
        devices = context.list_devices(subsystem="video4linux")
        matching_devices = []
        for device in devices:
            if serial in device.get("ID_SERIAL"):
                matching_devices.append(int(device.sys_number))
        assert len(matching_devices) > 0, "No matching device found with serial: {}".format(serial)
        # one camera can have two devices, one for video, one for metadata. Use the first one
        idx = sorted(matching_devices)[0]
        logging("Found matching camera at /dev/video{} for serial {}".format(idx, serial), verbose=self.verbose, style="warning")
        return idx

    def start(self, create_thread=True):
        if len(self.serial) > 0:
            self.usb_id = self.parse_serial(self.serial)
        self.stream = cv2.VideoCapture(self.usb_id)
        # Only works for cameras that support this resolution as one of the native resolutions
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        if not self.stream.isOpened():
            print("Cannot open camera stream at id {}".format(self.usb_id))
            exit()
        self.streaming = True
        if create_thread:
            threading.Thread(target=self.update, args=()).start()

    def stop(self):
        self.streaming = False
        if self.stream is not None:
            self.stream.release()
            del self.stream

    def update(self):
        while True:
            if not self.streaming:
                time.sleep(0.01)
                continue
            
            while time.time() - self.last_updated < 1.0/self.fps:
                time.sleep(0.001)
            try:
                grabbed, frame = self.stream.read()
            except Exception as e: 
                print(e)
                print("Error reading frame. Trying to ignore...")
                time.sleep(0.1)
                continue
            if grabbed:
                if self.resolution != (frame.shape[1], frame.shape[0]):
                    frame = cv2.resize(frame, self.resolution)
                if self.format == "RGB":
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame = frame
                self.write_frame(frame)
                self.last_updated = time.time()
            else:
                time.sleep(0.01)