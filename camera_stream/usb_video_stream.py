import cv2
import numpy as np
from typing import Tuple
import threading
import time
import pyudev

from .base_video_stream import BaseVideoStream
from misc.utils import logging

class USBVideoStream(BaseVideoStream):
    def __init__(self, serial: str = "", usb_id: int = 0, resolution: Tuple[int, int] = (640, 480), format="BGR", verbose=True, name=""):
        super(USBVideoStream, self).__init__(resolution, format, verbose=verbose, name=name)
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
        # MJPG: the GelSight Mini's only native format (3280x2464 @ ~18.75 fps
        # real, the sensor's hardware ceiling). We decode it ourselves at reduced
        # scale (see update()) — the full-res decode costs ~71 ms/frame and was
        # dragging the effective rate to ~8 fps; reduced decode recovers ~18 fps.
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # Return the raw MJPG buffer instead of auto-decoding, so we can imdecode
        # at reduced resolution ourselves (much faster than full 8 MP decode).
        self.stream.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        # Shallowest possible driver buffer: with grab()/read() being FIFO over
        # the V4L2 queue, a deep buffer means every frame is stale by the buffer
        # depth. 1 keeps us on the freshest frame. (Correctness comes from the
        # per-frame capture timestamp below; BUFFERSIZE=1 additionally avoids
        # recording a needlessly-old frame and keeps live preview fresh.)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.stream.isOpened():
            print("Cannot open camera stream at id {}".format(self.usb_id))
            exit()
        self.streaming = True
        self.frame_ts = None        # capture timestamp of self.frame (epoch s)
        if create_thread:
            threading.Thread(target=self.update, args=(), daemon=True).start()

    def stop(self):
        self.streaming = False
        stream = getattr(self, "stream", None)
        if stream is not None:
            stream.release()
            self.stream = None

    # Reduced-scale JPEG decode: 1/4 of 3280x2464 = 820x616, still larger than
    # the 640x480 target so the subsequent resize never upscales. Cuts decode
    # from ~71 ms to ~negligible, lifting the effective rate to the ~18.75 fps
    # USB/sensor ceiling. Use REDUCED_COLOR_2 for more headroom if you ever need
    # an output larger than ~820x616.
    _DECODE_FLAG = cv2.IMREAD_REDUCED_COLOR_4

    def update(self):
        # Continuous grab -> retrieve raw MJPG -> reduced-scale decode. The
        # capture timestamp is taken right after grab() (the moment the frame is
        # dequeued from the driver); with BUFFERSIZE=1 that is within ~1 frame of
        # true sensor-capture time. Decode latency afterwards does NOT shift the
        # timestamp, so downstream alignment stays correct regardless of how slow
        # the decode is.
        while self.streaming:
            try:
                grabbed = self.stream.grab()
                ts = time.time()                       # capture time (post-grab)
                if not grabbed:
                    time.sleep(0.005); continue
                ok, raw = self.stream.retrieve()
                if not ok or raw is None:
                    time.sleep(0.005); continue
                # raw is the 1-D MJPG byte buffer (CONVERT_RGB=0); decode reduced.
                img = cv2.imdecode(raw.reshape(-1), self._DECODE_FLAG)
                if img is None:
                    continue
            except Exception as e:
                print(e); print("Error reading frame. Trying to ignore...")
                time.sleep(0.01); continue
            if self.resolution != (img.shape[1], img.shape[0]):
                img = cv2.resize(img, self.resolution)
            if self.format == "RGB":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            with self.lock:
                self.frame = img
                self.frame_ts = ts
            self.write_frame(img)
            self.last_updated = ts

    def get_frame_with_timestamp(self, wait=True, max_no_update_time=0.5):
        """Return (frame_copy, capture_timestamp). The timestamp reflects when
        the frame was captured by the sensor (post-grab), not when this method
        is called — so it stays correct even though decoding is slow."""
        frame = self.get_frame(wait=wait, max_no_update_time=max_no_update_time)
        with self.lock:
            ts = self.frame_ts
        return frame, ts