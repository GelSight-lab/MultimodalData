from MultimodalData.camera_stream import RaspiVideoStream
import cv2
import numpy as np

def stream():
    stream = RaspiVideoStream("http://tracking-pi.local")
    stream.start()

    while True:
        frame = stream.get_frame()
        print(frame.shape)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
if __name__ == "__main__":
    stream()