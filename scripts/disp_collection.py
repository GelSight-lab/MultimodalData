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
    # confirm = input(color_style[0].format("Are you using the correct franka EE config? (y/n): "))
    # if confirm == "n":
    #     exit()
    
    agent = DispCollection(cfg)
    if len(cfg.env.saving_folder_overwrite) > 0:
        save_path = os.path.join(cfg.env.save_path, cfg.env.saving_folder_overwrite)
        finished_processing_path = os.path.join(cfg.env.save_path, "finished_preprocessing", cfg.env.saving_folder_overwrite)
    else:
        _f = lambda x: x.replace("_", "")
        if len(cfg.object_overwrite) > 0:
            obj_name = cfg.object_overwrite
        else:
            obj_name = cfg.object.name
        folder_name = \
            f"{_f(obj_name)}_{_f(cfg.env.left_finger.type)}_{_f(cfg.env.right_finger.type)}_init_pose_{cfg.env.init_pose_cnt}" 
        save_path = os.path.join(cfg.env.save_path, folder_name)
        finished_processing_path = os.path.join(cfg.env.save_path, "finished_preprocessing", folder_name)
    print("Saving to: ", save_path)

    if not cfg.env.dry_run:
        assert not os.path.exists(save_path), "Folder already exists. Please specify a different folder name."
        assert not os.path.exists(finished_processing_path), "Folder already exists in finished preprocessing folder. Please specify a different folder name."
        os.makedirs(save_path, exist_ok=False)
        with open(os.path.join(save_path, "config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)
    # figure out the rollout counting number
    fs = [fn for fn in os.listdir(save_path) if "rollout_" in fn]
    if len(fs) == 0:
        rollout_count = 0
    else:
        rollout_count = max([int(x.split("_")[-1]) for x in fs]) + 1
    logging("Rollout will be counted starting from: {}".format(rollout_count), style=1)

    agent.connect_to_robot()
    logging("Robot connected.", style=1)
    agent.connect_to_gripper()
    logging("Gripper connected.", style=1)
    agent.connect_to_tactile_sensor()
    logging("Sensor connected.", style=1)

    # visualize video streams to confirm camera ordering
    logging("Visualizing video streams to confirm camera ordering", style="green")
    logging("Make sure left and righ cameras are in the correct order. Press q to quit", style="blue")
    while True:
        left_frame, right_frame = agent.get_gs_images()
        frame = np.concatenate((left_frame, right_frame), axis=1)
        cv2.imshow("Please confirm camera ordering", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
    confirm = input(color_style[0].format("Confirm? 'n' to quit"))
    if confirm == "n":
        logging("Quitting... swap the camera devices in config file before trying again", style=1)
        exit()
    
    if cfg.env.record_gs_video:
        agent.prepare_recording_gs_video(save_path)
        logging("Recording GS video to {}".format(save_path), style=1)

    for i in range(cfg.env.n_rollout):
        # collect a rollout
        logging("Rollout {}/{}".format(i+1, cfg.env.n_rollout), style=2)
        if cfg.env.dry_run:
            logging("Dry run, no data will be saved.", style=1)
            rollout_path = ""
        else:
            rollout_path = save_path
            os.makedirs(rollout_path, exist_ok=True)
            logging("Saving rollout data to {}".format(rollout_path), style=1)
        
        agent.goto_pregrasp_pose(use_init_randomization_range = (i == 0))
        # if i == 0:
        #     confirm = input(color_style[0].format("Confirm pregrasp pose? (y/n): "))
        #     if confirm == "n":
        #         exit()
        
        #     # test close gripper
        #     agent.close_gripper(cur=50)
        #     ready = input(color_style[0].format("Done adjusting object pose?: "))
        #     agent.open_gripper()
        #     time.sleep(1)

        agent.collect_one_rollout(vis=True, dump_path=rollout_path, N=cfg.env.epi_per_rollout)
    
    if cfg.env.record_gs_video:
        print("Stopping GS video recording...")
        agent.stop_recording_gs_video()
        logging("Recording GS video to {}".format(save_path), style=1)



if __name__ == '__main__':
    main()