import time
from typing import Tuple
import threading
import cv2
import os

from misc.utils import logging

class BaseVideoStream(object):
    def __init__(self, resolution: Tuple[int, int] = (640, 480), format="BGR", verbose=True, name=""):
        self.stream = None
        self.frame = None
        self.streaming = False
        self.resolution = resolution
        self.format = format
        self.last_updated = time.time()
        self.lock = threading.Lock()
        self.verbose = verbose
        self.name = name                 # human label, e.g. "left"/"right"

        self.recording = False
        self.record_path = None
        self.record_fps = 10
        self.stop_recording_signal = False
        self.recording_frame_count = 0
        self.last_record_frame_t = time.time()

        if not verbose:
            logging("Camera-related warnings will be turned off", True, "warning")

    def _tag(self):
        """Identifier for log messages: '[<name> serial=<serial>]'."""
        parts = []
        if getattr(self, "name", ""):
            parts.append(self.name)
        ser = getattr(self, "serial", "")
        if ser:
            parts.append(f"serial={ser}")
        return f"[{' '.join(parts)}]" if parts else ""

    def start(self, create_thread=True):
        raise NotImplementedError
    
    def stop(self):
        raise NotImplementedError
    
    def prepare_recording(self, path, fps):
        self.recording = False
        self.recording_frame_count = 0
        self.record_fps = fps
        # cannot get two VideoWriter to work at the same time. save as frames instead.
        # self.video = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*fourcc), fps, self.resolution)
        os.makedirs(path, exist_ok=True)
        self.record_path = path

    def start_recording(self):
        # assert self.video is not None, "Please call prepare_recording() first."
        self.recording = True
    
    def pause_recording(self):
        self.recording = False
    
    def resume_recording(self):
        self.start_recording()
    
    def stop_recording(self):
        self.stop_recording_signal = True
    
    def write_frame(self, frame):
        if not self.recording:
            return
        if self.stop_recording_signal:
            # save the video
            # self.video.release()
            # self.video = None
            self.stop_recording_signal = False
            self.recording = False
        else:
            # self.video.write(frame)
            if time.time() - self.last_record_frame_t > 1. / self.record_fps:
                cv2.imwrite(os.path.join(self.record_path, "frame_{:06d}.jpg".format(self.recording_frame_count)), frame)
                self.recording_frame_count += 1
                self.last_record_frame_t = time.time()
    
    def _start_update_thread(self):
        """Run update() on a fresh daemon thread and remember it."""
        self._update_thread = threading.Thread(target=self.update, daemon=True)
        self._update_thread.start()

    def restart(self):
        """Reopen the camera after a stall.

        stop() ends the update thread (it exits when `streaming` goes False),
        so start() must create a new one: an earlier version reopened the
        device with create_thread=False, which left `frame` None forever and
        hung every get_frame() caller after the first restart.
        """
        self.stop()
        old = getattr(self, "_update_thread", None)
        if old is not None and old is not threading.current_thread():
            old.join(timeout=5.0)
        logging(f"Restarting the camera {self._tag()}...", self.verbose, "cyan")
        self.frame = None
        time.sleep(3)
        self.start(create_thread=True)

    def peek_frame_with_timestamp(self):
        """(frame_copy, capture_ts) of the latest frame, or (None, None).

        Never waits and never restarts the camera; the caller decides what to
        do with a stale frame (the recorder keeps ticking and lets a
        supervisor restart the stream off the capture thread).
        """
        with self.lock:
            frame = self.frame
            ts = getattr(self, "frame_ts", None)
            return (frame.copy() if frame is not None else None), ts
    
    def update(self):
        raise NotImplementedError
    
    def get_frame(self, wait=True, max_no_update_time=0.5):
        if wait:
            while self.frame is None:
                time.sleep(0.01)
        error_flag = False
        while time.time() - self.last_updated > max_no_update_time:
            if time.time() - self.last_updated > 4 * max_no_update_time:
                self.restart()
                self.last_updated = time.time()
                print(f"Restarted the camera {self._tag()}.")
            if not error_flag:
                # only print the error msg once
                error_flag = True
                print("Frame is not updated for more than {} second {}. "
                      "Check the camera connection.".format(max_no_update_time, self._tag()))
            time.sleep(0.01)
        while self.frame is None:
            time.sleep(0.01)
        with self.lock:
            frame = self.frame.copy()
        return frame
    
    def __del__(self):
        self.stop()
