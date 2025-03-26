import urllib.request
import urllib
import cv2
import numpy as np
from threading import Thread
import time 
from typing import Tuple

from ..camera_steam.base_video_stream import BaseVideoStream


class RaspiVideoStream(BaseVideoStream):
    def __init__(self, url: str, resolution: Tuple[int, int] = (640, 480), format="BGR", verbose=True):
        super(RaspiVideoStream, self).__init__(resolution, format, verbose=verbose)
        self.url = "{}:8080/?action=stream".format(url)

    def start(self, create_thread=True):
        self.streaming = True
        self.stream=urllib.request.urlopen(self.url)

        if create_thread:
            Thread(target=self.update, args=()).start()

    def stop(self):
        if self.streaming == True:
            self.stream.close()
        self.streaming = False

    def update(self):
        stream = self.stream
        bytess=b''

        while True:
            if self.streaming == False:
                time.sleep(0.01)
                continue

            bytess+=stream.read(32767)

            a = bytess.find(b'\xff\xd8') # JPEG start
            b = bytess.find(b'\xff\xd9') # JPEG end

            if a!=-1 and b!=-1:
                jpg = bytess[a:b+2] # actual image
                bytess= bytess[b+2:] # other informations

                frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.IMREAD_COLOR)
                if self.format == "RGB":
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame = cv2.resize(frame, self.resolution)
                self.write_frame(frame)
                self.last_updated = time.time()
