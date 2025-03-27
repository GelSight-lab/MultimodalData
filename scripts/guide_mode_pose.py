from omegaconf import DictConfig, OmegaConf
import hydra

from MultimodalData.camera_stream import DispCollection
from MultimodalData.misc.utils import logging, color_style
import numpy as np
import os
import cv2
import time

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    planner = DispCollection(cfg)
    planner.connect_to_robot()
    logging("Robot connected.", style=1)
    planner.connect_to_gripper()
    logging("Gripper connected.", style=1)


    cur_pose = planner.get_franka_pose()
    goal_pose = cur_pose.copy()
    planner.gripper.goto_gripper(0.5)
    # goal_pose.translation = [0.3677313,  0.02489768, 0.08026227] # switch pick up pose
    # goal_pose.translation = [0.3500231,  0.02719247, 0.07864657] # VGA pick up pose
    goal_pose.translation = [0.36108827, 0.02599755, 0.08388375] # USB pick up pose
    planner.goto_pose(goal_pose)

    planner.close_gripper(cur=60)
    time.sleep(1)

    goal_pose.translation[2] += 0.015
    planner.goto_pose(goal_pose)

    insert_pose = planner.get_franka_pose().copy()
    # insert_pose.translation = [0.44532288, 0.05251785, 0.05586839] # VGA pre-insert pose
    insert_pose.translation = [0.49654114, 0.03233815, 0.05703585] # USB pre-insert pose
    planner.goto_pose(insert_pose)
    time.sleep(2)

    planner.fa.run_guide_mode(duration=20, block=False)
    t = time.time()
    while time.time() - t < 20:
        print(planner.get_franka_pose())

if __name__ == '__main__':
    main()