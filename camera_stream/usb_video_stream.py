import cv2
import numpy as np
from typing import Tuple
import threading
import time
import pyudev

from .base_video_stream import BaseVideoStream
from misc.utils import logging

class USBVideoStream(BaseVideoStream):
    def __init__(self, serial: str = "", usb_id: int = 0, resolution: Tuple[int, int] = (640, 480), format="BGR", verbose=True):
        super(USBVideoStream, self).__init__(resolution, format, verbose=verbose)
        self.serial = serial
        self.usb_id = usb_id
        self.fps = 30
    
    def parse_serial(self, serial: str):
        """Parse serial number and find the corresponding usb id"""
        # list all video devices
        context = pyudev.Context()
        devices = context.list_devices(subsystem="video4linux")
        matching_devices = []
        for device in devices:
            if serial in device.get("ID_SERIAL"):
                matching_devices.append(int(device.sys_number))
        if len(matching_devices) == 0:
            raise RuntimeError("No matching device found with serial: {}".format(serial))
        # one camera can have two devices, one for video, one for metadata. Use the first one
        idx = sorted(matching_devices)[0]
        logging("Found matching camera at /dev/video{} for serial {}".format(idx, serial), verbose=self.verbose, style="warning")
        return idx

    def start(self, create_thread=True):
        if len(self.serial) > 0:
            self.usb_id = self.parse_serial(self.serial)
        # Force the V4L2 backend so CAP_PROP_BUFFERSIZE / FOURCC are honored.
        self.stream = cv2.VideoCapture(self.usb_id, cv2.CAP_V4L2)
        # MJPG: the GelSight Mini's native compressed format. Without this V4L2
        # may negotiate raw YUYV, which saturates USB bandwidth (esp. alongside
        # the RealSense cameras) and forces the driver to queue/delay frames.
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # Only works for cameras that support this resolution as one of the native resolutions
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        # Shallowest possible driver buffer: with read()/grab() being FIFO over
        # the V4L2 queue, a deep buffer means every frame we read is stale by the
        # buffer depth. 1 keeps us on the freshest frame.
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.stream.isOpened():
            print("Cannot open camera stream at id {}".format(self.usb_id))
            exit()
        self.streaming = True
        if create_thread:
            threading.Thread(target=self.update, args=(), daemon=True).start()

    def stop(self):
        self.streaming = False
        if self.stream is not None:
            self.stream.release()
            del self.stream

    def update(self):
        # IMPORTANT: do NOT throttle reads. cv2.VideoCapture.read() returns the
        # OLDEST frame in the V4L2 queue (FIFO). If we read slower than the
        # camera streams, the queue stays full and every frame we hand out is
        # stale by the buffer depth (this was the ~15-frame GelSight latency).
        # Reading continuously (grab as fast as possible, keep only the latest
        # retrieve) drains the backlog so self.frame is always the freshest.
        while self.streaming:
            grabbed, frame = None, None
            try:
                grabbed, frame = self.stream.read()
            except Exception as e:
                print(e)
                print("Error reading frame. Trying to ignore...")
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