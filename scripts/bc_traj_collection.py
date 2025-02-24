from omegaconf import DictConfig, OmegaConf
import hydra

from probing_panda import DispCollection
from probing_panda.utils import logging, color_style
import numpy as np
import os
import cv2
import time

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    agent = DispCollection(cfg)

    agent.connect_to_robot()
    logging("Robot connected.", style=1)
    agent.connect_to_gripper()
    logging("Gripper connected.", style=1)
    agent.connect_to_tactile_sensor()
    logging("Sensor connected.", style=1)
    agent.connect_to_external_cam()
    logging("External camera connected.", style=1)

    def close_gripper():
        gripper_cur = int(np.random.sample() * (cfg.grasp.max_cur - cfg.grasp.min_cur) + cfg.grasp.min_cur)
        agent.close_gripper(cur=gripper_cur)
        logging("1. closed gripper with current {}".format(gripper_cur), True, 1)

    if hasattr(cfg.object, "pick_up_pose"):
        cur_pose = agent.get_franka_pose()
        goal_pose = cur_pose.copy()
        agent.gripper.goto_gripper(0.5)
        goal_pose.translation = np.array(cfg.object.pick_up_pose)
        agent.goto_pose(goal_pose)
        close_gripper()
        time.sleep(0.5)
        goal_pose.translation[2] += 0.015
        agent.goto_pose(goal_pose)
    else:
        confirm = input(color_style[0].format("Close gripper? 'n' to quit"))
        if confirm == "n":
            logging("Quitting... swap the camera devices in config file before trying again", style=1)
            exit()
        close_gripper()

    for i in range(cfg.behavior_cloning.n_rollout):
        # collect a rollout
        logging("Rollout {}/{}".format(i+1, cfg.behavior_cloning.n_rollout), style=2)
        
        agent.goto_pregrasp_pose(i == 0, no_randomization=False)

        agent.collect_bc_rollout()



if __name__ == '__main__':
    main()