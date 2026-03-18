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
            f"bc_online_weights={bc_online_weights}", 
        ])
    bc_cfg = OmegaConf.to_container(bc_cfg, resolve=True)
    bc_cfg["for_object"] = "switch"
    bc_cfg["mean_abs_pose"] = {"switch": [0.45162369, 0.02617293, 0.06508426]}
    planner = BCOnlinePlanner(robot_cfg=robot_cfg, bc_cfg=OmegaConf.create(bc_cfg))

    # goto pick up pose and lift up
    cur_pose = planner.agent.get_franka_pose()
    goal_pose = cur_pose.copy()
    goal_pose.translation = [0.3657313,  0.02489768, 0.08026227]
    goal_pose.translation += np.random.normal(0, 0.001, 3)
    planner.agent.goto_pose(goal_pose)

    planner.agent.close_gripper(cur=60)
    time.sleep(1)
    goal_pose.translation[2] += 0.015
    planner.agent.goto_pose(goal_pose)
    planner.record_base_gs_imgs(vis_frame=vis_frame)

    init_pose = planner.agent.get_franka_pose().copy()
    init_pose.translation = [0.45502912, 0.02904012, 0.0579782]
    planner.agent.object_specific_init_pose = init_pose
    planner.agent.goto_pose(init_pose)
    time.sleep(3)

    step_cnt = 0
    while step_cnt < 30:
        if step_cnt > 7 and step_cnt % 8 == 0:
            cur_pose = planner.agent.get_franka_pose()
            cur_pose.translation[2] += 0.002
            cur_pose.translation[:2] += np.random.normal(0, 0.001, 2)
            planner.agent.goto_pose(cur_pose)
        step_cnt += 1
        planner.step(move=True, act_scale=1.5, vis_frame=vis_frame)
        print(f"Step {step_cnt}")
        cv2.imshow("frame", vis_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    # release and liftup
    # self.write_goal_current(self.gripper_id, int(cur))
    pose = planner.agent.get_franka_pose()
    pose.translation[2] -= 0.004
    planner.agent.goto_pose(pose)
    planner.agent.gripper.goto_gripper(0.2)
    time.sleep(0.5)
    pose = planner.agent.get_franka_pose()
    pose.translation[0] -= 0.05
    planner.agent.goto_pose(pose)
    pose.translation[2] += 0.05
    planner.agent.goto_pose(pose)
    time.sleep(4)



if __name__ == '__main__':
    main()