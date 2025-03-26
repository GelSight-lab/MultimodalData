from MultimodalData.camera_steam import DispCollection
import cv2
import numpy as np

import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../config", config_name="config")
def stream(cfg: DictConfig):
    disp = DispCollection(cfg)
    disp.connect_to_tactile_sensor()
    disp.connect_to_external_cam()

    print("Make sure left and right camera are in the correct order.")
    while True:
        left_frame, right_frame = disp.get_gs_images()
        cam_frame = disp.get_external_cam_image()
        frame = np.concatenate((left_frame, right_frame, cam_frame), axis=1)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
if __name__ == "__main__":
    stream()