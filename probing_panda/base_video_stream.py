import time
from typing import Tuple
import threading
import cv2
import os

from .utils import logging

class BaseVideoStream(object):
    def __init__(self, resolution: Tuple[int, int] = (640, 480), format="BGR", verbose=True):
        self.stream = None
        self.frame = None
        self.streaming = False
        self.resolution = resolution
        self.format = format
        self.last_updated = time.time()
        self.lock = threading.Lock()
        self.verbose = verbose

        self.recording = False
        self.record_path = None
        self.record_fps = 10
        self.stop_recording_signal = False
        self.recording_frame_count = 0
        self.last_record_frame_t = time.time()

        if not verbose:
            logging("Camera-related warnings will be turned off", True, "warning")

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
    
    def restart(self):
        self.stop()
        logging("Restarting the camera...", self.verbose, "cyan")
        self.frame = None
        time.sleep(3)
        # Avoid creating new thread
        self.start(create_thread=False)
    
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
                print("Restarted the camera.")
            if not error_flag:
                # only print the error msg once
                error_flag = True
                print("Frame is not updated for more than {} second. Check the camera connection.".format(max_no_update_time))
            time.sleep(0.01)
        while self.frame is None:
            time.sleep(0.01)
        with self.lock:
            frame = self.frame.copy()
            time.sleep(0.01)
        return frame
    
    def __del__(self):
        self.stop()
