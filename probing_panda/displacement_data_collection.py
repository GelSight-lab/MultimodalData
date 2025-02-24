from frankapy import FrankaArm
from omegaconf import DictConfig, OmegaConf
import hydra
import csv
import time
import cv2
import os
import numpy as np
from probing_panda import DxlGripperInterface
from PIL import Image
from .utils import sample_transformation, rotmat_to_euler, logging, calc_diff_image, color_style

class DispCollection():
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self.fa = None
        self.gripper = None
        self.left_gs = None
        self.right_gs = None

        # The object specific init pose for panda after object specific delta is applied
        # but before the randomization
        self.object_specific_init_pose = None

        self.current_episode = {
            "object": None,
            "init_delta": None,
        }

        # set object-specific randomization overwrites
        if len(self.cfg.object_overwrite) == 0:
            if "randomize_between_rollouts" in self.cfg.object:
                self.cfg.grasp.randomize_between_rollouts = self.cfg.object.randomize_between_rollouts
            if "randomize_init_rollout" in self.cfg.object:
                self.cfg.grasp.randomize_init_rollout = self.cfg.object.randomize_init_rollout
            if "randomize_episode" in self.cfg.object:
                self.cfg.grasp.randomize_episode = self.cfg.object.randomize_episode

    def connect_to_robot(self, home=True):
        self.fa = FrankaArm(with_gripper=False)
        if home:
            self.fa.reset_joints()
    
    def goto_joints(self, joints):
        self.fa.goto_joints(joints, use_impedance=self.cfg.env.use_impedance, 
                            joint_impedances=list(self.cfg.env.joint_impedance),
                            cartesian_impedances=list(self.cfg.env.cart_impedance),
                            ignore_virtual_walls=False)

    def goto_pose(self, pose, duration=3):
        self.fa.goto_pose(pose, use_impedance=self.cfg.env.use_impedance, 
                          joint_impedances=list(self.cfg.env.joint_impedance),
                          cartesian_impedances=list(self.cfg.env.cart_impedance),
                          duration=duration,
                          ignore_virtual_walls=False)
    
    def connect_to_gripper(self):
        self.gripper = DxlGripperInterface(self.cfg.gripper)
        self.gripper.init()
        self.gripper.home_gripper()
        self.gripper.start()
    
    def close_gripper(self, cur):
        self.gripper.close(cur=cur)
    
    def open_gripper(self, *args, **kwargs):
        self.gripper.open(*args, **kwargs)

    def connect_to_tactile_sensor(self):
        print("Starting Tactile Sensor")

        def _connect(cfg):
            if cfg.type == "Wedge":
                from probing_panda import RaspiVideoStream
                return RaspiVideoStream(cfg.url, resolution=cfg.resolution, format=cfg.format, verbose=cfg.verbose)
            elif cfg.type == "Mini":
                from probing_panda import USBVideoStream
                return USBVideoStream(cfg.mini_serial, resolution=cfg.resolution, format=cfg.format, verbose=cfg.verbose)
            elif cfg.type == "Digit":
                from probing_panda import DigitVideoStream
                return DigitVideoStream(cfg.digit_serial, resolution=cfg.resolution, format=cfg.format, verbose=cfg.verbose)
            else:
                raise ValueError("Unknown tactile sensor type")


        self.left_gs = _connect(self.cfg.env.left_finger)
        self.right_gs = _connect(self.cfg.env.right_finger)
        
        self.left_gs.start()
        self.right_gs.start()
    
    def connect_to_external_cam(self):
        print("Starting External Camera")
        from probing_panda import USBVideoStream
        cam_id = self.cfg.behavior_cloning.external_cam.device_id
        resolution = self.cfg.behavior_cloning.external_cam.resolution
        cam_format = self.cfg.behavior_cloning.external_cam.format
        verbose = self.cfg.behavior_cloning.external_cam.verbose
        self.ext_cam = USBVideoStream(usb_id=cam_id, resolution=resolution, format=cam_format, verbose=verbose)

        self.ext_cam.start()
    
    def prepare_recording_gs_video(self, save_path):
        self.left_gs.prepare_recording(
            os.path.join(save_path, f"gsframes_{self.cfg.env.left_finger.type}_left"), self.cfg.env.record_fps)
        self.right_gs.prepare_recording(
            os.path.join(save_path, f"gsframes_{self.cfg.env.left_finger.type}_right"), self.cfg.env.record_fps)
    
    def stop_recording_gs_video(self):
        self.left_gs.stop_recording()
        self.right_gs.stop_recording()
        time.sleep(10)
    
    def get_franka_pose(self, flush_times=3):
        for _ in range(flush_times-1):
            self.fa.get_pose()
        return self.fa.get_pose().copy()
    
    def goto_pregrasp_pose(self, use_init_randomization_range, verbose=True, no_randomization=False):
        if self.object_specific_init_pose is None:
            # 1. goto init_pose without randomization
            self.goto_joints(self.cfg.env.init_joints)
            logging("1. reached init pose", verbose, 1)
            pose = self.get_franka_pose()
            init_pose = pose.copy()
            
            if len(self.cfg.object_overwrite) > 0:
                use_guide_mode = "y"
            else:
                use_guide_mode = input(color_style[0].format(
                    "Use guide mode insead of pre-defined init pose for this object? Enter 'y' to " \
                    "enter guide mode to manually select an init pose, 'n' "\
                    "to use the init pose defind in the config file: "))
            if use_guide_mode == "n":
                # 2.1 apply object specific delta movement without moving down z for now
                if hasattr(self.cfg.object, "grasp_pose_delta"):
                    delta_no_z = np.array(self.cfg.object.grasp_pose_delta)
                    delta_no_z[2] = 0
                    pose.translation += delta_no_z
                    self.goto_pose(pose)
                    
                    # 2.2 move down z
                    delta_z = np.array(self.cfg.object.grasp_pose_delta)
                    delta_z[:2] = 0
                    pose.translation += delta_z
                    self.goto_pose(pose)
                elif hasattr(self.cfg.object, "grasp_pose_abs"):
                    grasp_pose_abs = np.array(self.cfg.object.grasp_pose_abs)
                    if hasattr(self.cfg.object, "pick_up_randomization"):
                        # add pick up pose randomization if specified
                        rand_range = np.array(self.cfg.object.pick_up_randomization)
                        grasp_pose_abs += np.random.uniform(-rand_range, rand_range)
                    pose.translation[:2] = grasp_pose_abs[:2]
                    self.goto_pose(pose)
                    # move down z
                    pose.translation[2] = grasp_pose_abs[2]
                    self.goto_pose(pose)
            else:
                # enter guide mode to manullay select an init pose
                GUIDE_MODE_TIME = 10
                logging(f"Entering guide mode for {GUIDE_MODE_TIME}s to manually select an init pose", verbose, 1)
                self.fa.run_guide_mode(duration=GUIDE_MODE_TIME, block=True)
                done = input(color_style[0].format("Done? (y/n): "))
                while done == "n":
                    self.fa.run_guide_mode(duration=5, block=True)
                    done = input(color_style[0].format("Done? (y/n): "))
                pose = self.get_franka_pose()
                # sometimes we might want to use guide mode to get the init pose, 
                # but still want to apply the y value defind in the config file
                # to make sure the object is centered in the gripper
                if len(self.cfg.object_overwrite) == 0:
                    use_predefined_y = input(color_style[0].format(
                        "Not use pre-defined y value for this object? (y/n): "))
                    if use_predefined_y == "n":
                        pose.translation[1] = init_pose.translation[1] + self.cfg.object.grasp_pose_delta[1]
                self.goto_pose(pose)
                print("delta:", self.get_franka_pose().translation - init_pose.translation)

            print(self.fa.get_joints())
            logging("2. applied object specific init pose", verbose, 1)
            self.object_specific_init_pose = pose.copy()
        else:
            self.goto_pose(self.object_specific_init_pose, duration=2.0)

        # 3. apply randomized grasp pose
        if not no_randomization:
            if use_init_randomization_range:
                randomize_range = self.cfg.grasp.randomize_init_rollout
            else:
                randomize_range = self.cfg.grasp.randomize_between_rollouts
            delta_pregrasp_T = sample_transformation(
                randomize_range.randomize_tra,
                randomize_range.randomize_rot,
                from_frame="franka_tool",
                to_frame="franka_tool_original"
            )
            pose = self.get_franka_pose()
            pose.from_frame = "franka_tool_original"
            pregrasp_pose = pose.dot(delta_pregrasp_T)
            self.goto_pose(pregrasp_pose, duration=2.0)
            self.goto_pose(pregrasp_pose, duration=1.0)
            logging("3. applied randomized pregrasp pose", verbose, 1)
    
    @staticmethod
    def rgb_to_bgr(img):
        return img[:, :, ::-1]
    
    def get_gs_images(self):
        # Investigate!! the original code uses rgb to bgr here
        return [self.left_gs.get_frame(), self.right_gs.get_frame()]
    
    def get_external_cam_image(self):
        return self.ext_cam.get_frame()
    
    def collect_one_rollout(self, N=10, verbose=True, vis=False, dump_path: str = None):
        init_pose = self.get_franka_pose()
        init_joints = self.fa.get_joints()

        init_imgs = self.get_gs_images()
        if len(dump_path) > 0:
            left_im = Image.fromarray(init_imgs[0])
            left_im.save(os.path.join(dump_path, "blank_left.jpg"))
            right_im = Image.fromarray(init_imgs[1])
            right_im.save(os.path.join(dump_path, "blank_right.jpg"))

        # 1. close gripper
        gripper_cur = int(np.random.sample() * (self.cfg.grasp.max_cur - self.cfg.grasp.min_cur) + self.cfg.grasp.min_cur)
        self.close_gripper(cur=gripper_cur)
        logging("1. closed gripper with current {}".format(gripper_cur), verbose, 1)
        time.sleep(0.8)

        # 1.5 start recording if needed
        if self.cfg.env.record_gs_video:
            self.left_gs.start_recording()
            self.right_gs.start_recording()

        # 2. collect N episodes
        for i in range(N):
            # sample a random pose
            delta_T = sample_transformation(
                self.cfg.grasp.randomize_episode.randomize_tra,
                self.cfg.grasp.randomize_episode.randomize_rot,
                from_frame="franka_tool",
                to_frame="franka_tool_original"
            )
            original_pose = init_pose.copy()
            original_pose.from_frame = "franka_tool_original"
            new_pose = original_pose.dot(delta_T)
            self.goto_pose(new_pose, duration=1.0)
            logging("2.{} reached random pose {}, {}".format(
                i, delta_T.translation*1000, rotmat_to_euler(delta_T.rotation)), verbose, 1)
            time.sleep(0.6)
            # From the new pose, to the original pose
            true_delta_T = original_pose.inverse().dot(self.get_franka_pose())

            current_imgs = self.get_gs_images()
            diff_img = np.hstack((calc_diff_image(current_imgs[0], init_imgs[0]), 
                                                   calc_diff_image(current_imgs[1], init_imgs[1])))
            
            fn_idx = str(time.time()).replace(".", "")
            # logging("Goal pose:   {} | {}".format(delta_T.translation * 1000, rotmat_to_euler(delta_T.rotation)), verbose, 1)
            # logging("Actual pose: {} | {}".format(true_delta_T.translation * 1000, rotmat_to_euler(true_delta_T.rotation)), verbose, 1)
            if len(dump_path) > 0:
                left_im = Image.fromarray(current_imgs[0])
                left_im.save(os.path.join(dump_path, f"left_{fn_idx}.jpg"))
                right_im = Image.fromarray(current_imgs[1])
                right_im.save(os.path.join(dump_path, f"right_{fn_idx}.jpg"))

                np.save(os.path.join(dump_path, f"pose_{fn_idx}.npy"), self.get_franka_pose())
            
            if vis:
                cv2.imshow("diff imgs", diff_img)
                cv2.waitKey(1)

        # 2.5 pause recording if needed
        if self.cfg.env.record_gs_video:
            self.left_gs.pause_recording()
            self.right_gs.pause_recording()

        self.open_gripper()

    def collect_bc_rollout(self):
        os.makedirs(self.cfg.behavior_cloning.save_path, exist_ok=True)

        fs = [fn for fn in os.listdir(self.cfg.behavior_cloning.save_path) if "epi_" in fn]
        if len(fs) == 0:
            rollout_count = 0
        else:
            rollout_count = max([int(x.split("_")[-1]) for x in fs]) + 1
        logging("Rollout will be counted starting from: {}".format(rollout_count), style=1)
        save_path = os.path.join(self.cfg.behavior_cloning.save_path, "epi_{}".format(rollout_count))
        os.makedirs(save_path, exist_ok=True)

        guide_mode_max_time = 30
        guide_mode_started = False
        start_guide_mode_after = 1 
        
        start_t = time.time()
        guide_mode_start_time = 0
        last_data_collect_time = 0
        data_collect_interval = 0.1
        frame_cnt = 0
        
        while True:
            # visualization
            left_frame, right_frame = self.get_gs_images()
            external_cam_frame = self.get_external_cam_image()
            frame = np.concatenate((left_frame, right_frame, external_cam_frame), axis=1)
            cur_pose = self.get_franka_pose()
            cv2.imshow("Collecting", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                # wait until the guide mode is finished
                if guide_mode_started:
                    self.fa.stop_skill()
                break

            # start guide mode
            if not guide_mode_started and time.time() - start_t > start_guide_mode_after:
                logging(f"Entering guide mode for {guide_mode_max_time}s to collect data", True, 1)
                self.fa.run_guide_mode(duration=guide_mode_max_time, block=False)
                guide_mode_started = True
                guide_mode_start_time = time.time()
            if guide_mode_started and time.time() - guide_mode_start_time > guide_mode_max_time:
                logging("Guide mode ended", True, 1)
                time.sleep(1)
                break
            
            # collect data
            if guide_mode_started:
                if time.time() - last_data_collect_time > data_collect_interval:
                    epi_path = os.path.join(save_path, f"frame_{frame_cnt}")
                    os.makedirs(epi_path, exist_ok=True)

                    left_im = Image.fromarray(left_frame)
                    right_im = Image.fromarray(right_frame)
                    cam_im = Image.fromarray(external_cam_frame)
                    left_im.save(os.path.join(epi_path, "left.jpg"))
                    right_im.save(os.path.join(epi_path, "right.jpg"))
                    cam_im.save(os.path.join(epi_path, "cam.jpg"))
                    np.save(os.path.join(epi_path, "pose.npy"), cur_pose)
                    frame_cnt += 1

        time.sleep(0.5)