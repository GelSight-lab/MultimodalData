from probing_panda.bc_policy import BCOnlinePlanner
import hydra
from omegaconf import DictConfig, OmegaConf
from MultimodalData.misc.utils import logging, color_style
import numpy as np
import time
import cv2

bc_online_weights = "checkpoints_supercloud/best_2024-04-25_20_23_03-step6_withstate_withpretrainedtactile_base0.7_abs"
# bc_online_weights = "checkpoints_supercloud/best_2024-04-25_20_22_59-step6_withstate_withscratchtactile_base0.7_abs"
# bc_online_weights = "checkpoints_supercloud/best_2024-04-25_20_22_57-step6_withstate_notactile_base0.7_abs"

def main():
    vis_frame = np.zeros((480 * 2, 640 * 3, 3), dtype=np.uint8)

    with hydra.initialize(version_base=None, config_path="../config"):
        robot_cfg = hydra.compose(config_name="config")

    with hydra.initialize(version_base=None, config_path="../" + bc_online_weights):
        bc_cfg = hydra.compose(config_name="config", overrides=[
            f"bc_online_weights={bc_online_weights}"
        ])
    
    agent = BCOnlinePlanner(robot_cfg=robot_cfg, bc_cfg=bc_cfg)

    confirm = input(color_style[0].format("Close gripper? 'n' to quit"))
    if confirm == "n":
        logging("Quitting... swap the camera devices in config file before trying again", style=1)
        exit()

    # close gripper
    gripper_cur = int(np.random.sample() * (robot_cfg.grasp.max_cur - robot_cfg.grasp.min_cur) + robot_cfg.grasp.min_cur)
    agent.agent.close_gripper(cur=gripper_cur)
    logging("1. closed gripper with current {}".format(gripper_cur), True, 1)

    time.sleep(2) # sleep to let camera AWB/AE settle
    agent.record_base_gs_imgs(vis_frame=vis_frame)

    agent.agent.goto_pregrasp_pose(True, no_randomization=False)
    step_cnt = 0
    while step_cnt < 30:
        step_cnt += 1
        agent.step(move=True, act_scale=1.5, vis_frame=vis_frame)
        print(f"Step {step_cnt}")
        cv2.imshow("frame", vis_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == '__main__':
    main()