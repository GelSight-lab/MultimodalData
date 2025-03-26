import torch
import torchvision
from torchvision import transforms
import os
import cv2
import hydra
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
from natsort import natsorted
from ..misc.utils import logging
from torch.utils.data import DataLoader
from fota.models import MTHN
from fota.models.nn_utils import makeMLP
from fota.utils import is_main_process
from datetime import datetime
import wandb
from tqdm import tqdm
from torch.utils.data import random_split

class BCDataset:
    def __init__(self, cfg, load=True):
        self.cfg = cfg
        # setup preproc
        self.cam_preproc = torchvision.transforms.Compose([
            transforms.Resize(self.cfg.cam_preprocess.resize_shape, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cfg.cam_preprocess.mean,std=self.cfg.cam_preprocess.std),
        ])
        self.leftgs_preproc = torchvision.transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cfg.leftgs_preprocess.mean,std=self.cfg.leftgs_preprocess.std)
        ])
        self.rightgs_preproc = torchvision.transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cfg.rightgs_preprocess.mean,std=self.cfg.rightgs_preprocess.std)
        ])
        if load:
            self.load_data()
    
    @staticmethod
    def warp_perspective(img, corners, output_sz):
        # warping for the wedge sensor
        if corners is not None:
            TOPLEFT, TOPRIGHT, BOTTOMLEFT, BOTTOMRIGHT = corners
        else:
            TOPLEFT = (80,119)
            TOPRIGHT = (444,152)
            BOTTOMLEFT = (80,449)
            BOTTOMRIGHT = (453,392)
        WARP_W = output_sz[0]
        WARP_H = output_sz[1]
        points1=np.float32([TOPLEFT,TOPRIGHT,BOTTOMLEFT,BOTTOMRIGHT])
        points2=np.float32([[0,0],[WARP_W,0],[0,WARP_H],[WARP_W,WARP_H]])
        matrix=cv2.getPerspectiveTransform(points1,points2)
        result = cv2.warpPerspective(img, matrix, (WARP_W, WARP_H))
        # then horizontal flip
        result = cv2.flip(result, 1)
        return result
    
    def load_gs_image(self, path, side):
        assert "_warped.jpg" not in path
        assert side in ["left", "right"]
        warped_fn = path.replace(".jpg", "_warped.jpg")
        if not os.path.exists(warped_fn):
            img = cv2.imread(path)
            warped_img = self.warp_perspective(img, corners=None, output_sz=(360, 280))
            cv2.imwrite(warped_fn, warped_img)
        with Image.open(warped_fn) as img:
            if side == "left":
                img_proc = self.leftgs_preproc(img)
            else:
                img_proc = self.rightgs_preproc(img)
        return img_proc
    
    def load_data(self):
        all_data = []
        epi_lengths = []
        n_samples = [] # amount of training samples in each episode
        for d in natsorted(os.listdir(self.cfg.data_dir)):
            if not d.startswith('epi_'):
                continue
            episode = natsorted([os.path.join(self.cfg.data_dir, d, f) for f in os.listdir(os.path.join(self.cfg.data_dir, d)) if f.startswith('frame_')])
            # check epi_length
            if len(episode) < self.cfg.acceptable_traj_length_range[1] and len(episode) > self.cfg.acceptable_traj_length_range[0]:
                # load this episode
                epi_data = []
                for i in range(len(episode)):
                    with Image.open(os.path.join(episode[i], 'cam.jpg')) as img:
                        cam_img = self.cam_preproc(img)
                    left_gs = self.load_gs_image(os.path.join(episode[i], 'left.jpg'), "left")
                    right_gs = self.load_gs_image(os.path.join(episode[i], 'right.jpg'), "right")
                    p = np.load(os.path.join(episode[i], 'pose.npy'), allow_pickle=True)
                    pose = p.item().matrix.copy()
                    del p # free up memory
                    epi_data.append({'cam_img': cam_img, 'left_gs': left_gs, 'right_gs': right_gs, 'pose': pose})
                all_data.append(epi_data)
                epi_lengths.append(len(episode))
                n_samples.append(len(episode) - self.cfg.look_forward_step)
                if self.cfg.max_data > 0 and np.sum(epi_lengths) >= self.cfg.max_data:
                    break
                if len(all_data) % 10 == 0:
                    logging(f"\t...loading {len(all_data)} episodes out of {len(os.listdir(self.cfg.data_dir))} episodes", True, "green")
        logging(f"Loaded {len(all_data)} episodes out of {len(os.listdir(self.cfg.data_dir))} episodes. Total frames: {np.sum(epi_lengths)}", True, "green")
        self.all_data = all_data
        self.epi_lengths = epi_lengths
        self.n_samples = n_samples

    def __len__(self):
        return np.sum(self.n_samples)
    
    def __getitem__(self, idx):
        # first find out which episode this idx belongs to
        epi_idx = 0
        while idx >= self.n_samples[epi_idx]:
            idx -= self.n_samples[epi_idx]
            epi_idx += 1
        
        cur_translation_norm = self.all_data[epi_idx][idx]['pose'][:3, 3] - self.cfg.mean_abs_pose[self.cfg.for_object]
        cur_translation = cur_translation_norm * 1000. # mm
        goal_translation_norm = self.all_data[epi_idx][idx + self.cfg.look_forward_step]['pose'][:3, 3] - self.cfg.mean_abs_pose[self.cfg.for_object]
        goal_translation = goal_translation_norm * 1000. # mm
        delta_pose = goal_translation - cur_translation # mm

        data = {
            'cam_img': self.all_data[epi_idx][idx]['cam_img'],
            'cur_left': self.all_data[epi_idx][idx]['left_gs'],
            'cur_right': self.all_data[epi_idx][idx]['right_gs'],
            'base_left': self.all_data[epi_idx][0]['left_gs'],
            'base_right': self.all_data[epi_idx][0]['right_gs'],
            'cur_trans': torch.from_numpy(cur_translation).type(torch.FloatTensor), # mm
            'goal_trans': torch.from_numpy(goal_translation).type(torch.FloatTensor), # mm
            'delta_trans': torch.from_numpy(delta_pose).type(torch.FloatTensor), # mm
        }
        return data

    def get_loader(self):
        train_val_generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(self, (1-self.cfg.val_ratio, self.cfg.val_ratio), generator=train_val_generator)
        
        def _get_dl(dataset):
            return DataLoader(dataset, batch_size=self.cfg.train.batch_size, 
                              num_workers=self.cfg.train.num_workers, 
                              pin_memory=self.cfg.train.pin_memory, shuffle=True, drop_last=True)
        return _get_dl(train_dataset), _get_dl(val_dataset)


class BCPolicy(torch.nn.Module):
    def __init__(self, cfg):
        super(BCPolicy, self).__init__()
        self.cfg = cfg
        self.with_state = cfg.with_state
        self.with_tactile = cfg.with_tactile

    def setup_model(self, load_from=None):
        # setup vision model
        self.cam_model = torchvision.models.resnet18(weights="IMAGENET1K_V1") # (N, 3, 224, 224) -> (N, 512)
    
        # setup tactile model
        if self.with_tactile:
            self.tactile_model = MTHN(self.cfg.fota_cfg)
            self.tactile_model.set_domains(encoder_domain="wedge", decoder_domain="pooling", forward_mode="single_tower")

        # setup state input encoder
        if self.with_state:
            self.state_encoder = makeMLP(
                input_dim=3,
                output_dim=64,
                hidden_dims=[32, 32],
                dropout_p=0.1,
                tanh_end=False,
                ln=False,
            )

        policy_input_dim = 512
        if self.with_tactile:
            policy_input_dim += 4 * self.cfg.fota_cfg.encoder_embed_dim
        if self.with_state:
            policy_input_dim += 64

        # setup policy model
        self.policy_model = makeMLP(
            input_dim=policy_input_dim,
            output_dim=3,
            hidden_dims=[128, 128, 64, 32],
            dropout_p=0.1,
            tanh_end=False,
            ln=False,
        )

        if load_from is not None:
            self.load_components(load_from)
        else:
            if self.with_tactile: # load pretrained weights for the tactile model
                if len(self.cfg.fota_cfg.weights) > 0:
                    self.tactile_model.load_components(self.cfg.fota_cfg.weights)
                else:
                    logging("No weights provided for the tactile model. Using random initialization.", True, "red")
    
    def save_components(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.cam_model.state_dict(), f"{path}/cam_model.pt")
        torch.save(self.policy_model.state_dict(), f"{path}/policy_model.pt")
        if self.with_tactile:
            self.tactile_model.save_components(path)
        if self.with_state:
            torch.save(self.state_encoder.state_dict(), f"{path}/state_encoder.pt")
    
    def load_components(self, path):
        self.cam_model.load_state_dict(torch.load(f"{path}/cam_model.pt"))
        logging(f"Loaded camera model from {path}", True, "green")
        self.policy_model.load_state_dict(torch.load(f"{path}/policy_model.pt"))
        logging(f"Loaded policy model from {path}", True, "green")
        if self.with_tactile:
            self.tactile_model.load_components(path)
            logging(f"Loaded tactile model from {path}", True, "green")
        if self.with_state:
            self.state_encoder.load_state_dict(torch.load(f"{path}/state_encoder.pt"))
            logging(f"Loaded state encoder from {path}", True, "green")

    def _forward_cam_model(self, cam_img):
        cam_out = self.cam_model.conv1(cam_img)
        cam_out = self.cam_model.bn1(cam_out)
        cam_out = self.cam_model.relu(cam_out)
        cam_out = self.cam_model.maxpool(cam_out)

        cam_out = self.cam_model.layer1(cam_out)
        cam_out = self.cam_model.layer2(cam_out)
        cam_out = self.cam_model.layer3(cam_out)
        cam_out = self.cam_model.layer4(cam_out)

        cam_out = self.cam_model.avgpool(cam_out)
        cam_out = torch.flatten(cam_out, 1)
        return cam_out
    
    def _forward_tactile_model(self, gs_img):
        tactile_out = self.tactile_model.single_tower_forward(gs_img)
        return tactile_out
    
    def forward(self, x):
        cam_out = self._forward_cam_model(x['cam_img']) # (N, 512)

        all_modality = [cam_out]

        if self.with_tactile:
            left_cur_out = self._forward_tactile_model(x['cur_left']) # (N, D)
            right_cur_out = self._forward_tactile_model(x['cur_right'])
            left_base_out = self._forward_tactile_model(x['base_left'])
            right_base_out = self._forward_tactile_model(x['base_right'])
            all_modality += [left_cur_out, left_base_out, right_cur_out, right_base_out]

        if self.with_state:
            state_out = self.state_encoder(x['cur_trans'])
            all_modality.append(state_out)

        combined = torch.cat(all_modality, dim=1)
        policy_out = self.policy_model(combined)
        return policy_out

class PolicyTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset = BCDataset(cfg)
        self.train_loader, self.val_loader = self.dataset.get_loader()
        self.model = BCPolicy(cfg)
        self.model.setup_model()
        self.setup_optimizer()

        self.min_avg_val_loss = np.inf

        self.train_iter = iter(self.train_loader)
        self.val_iter = iter(self.val_loader)

        if torch.cuda.is_available():
            self.device = "cuda:0"
        elif torch.backends.mps.is_available():
            # Apple Silicon
            self.device = "mps"
        else:
            self.device = "cpu"
        
        self.run_id = f"{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}"
        if "comment" in self.cfg:
            self.run_id += "-" + self.cfg.comment

        if self.cfg.train.wandb and is_main_process():
            wandb.init(
                project="FoundationTactile-BCPolicy",
                config=OmegaConf.to_container(self.cfg, resolve=True),
                name=self.run_id,
                entity=self.cfg.train.wandb_entity,
                magic=False)
            # define our custom x axis metric
            wandb.define_metric("train/step")
            wandb.define_metric("eval/step")
            # set all other train/ metrics to use this step
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("eval/*", step_metric="eval/step")
    
    def setup_optimizer(self):
        self.optimizer = eval(self.cfg.train.optimizer["_target_"])(
            params=self.model.parameters(), 
            **{k: v for k, v in self.cfg.train.optimizer.items() if k != "_target_"})
        self.scheduler = hydra.utils.instantiate(self.cfg.train.scheduler, optimizer=self.optimizer)
    
    def get_data_batch(self, is_train):
        try:
            data = next(self.train_iter if is_train else self.val_iter)
        except StopIteration:
            if is_train:
                self.train_iter = iter(self.train_loader)
                data = next(self.train_iter)
            else:
                self.val_iter = iter(self.val_loader)
                data = next(self.val_iter)
        return data
    
    def save_model(self, run_id, avg_val_loss):
        if avg_val_loss < self.min_avg_val_loss:
            # save as the best model
            self.min_avg_val_loss = avg_val_loss
            path = f"checkpoints/best_{run_id}"
            logging(f"Saving model to {path} as the best model", True, "green")
        else:
            path = f"checkpoints/{run_id}"
        logging(f"Current avg. test loss {avg_val_loss} v.s. best so far {self.min_avg_val_loss}. "\
                f"Saving model to {path}", True, "green")
        # save the model
        self.model.save_components(path)
        # save the optimizer and scheduler
        opt_type = self.cfg.train.optimizer["_target_"].split(".")[-1]
        torch.save(self.optimizer.state_dict(), f"{path}/optimizer_{opt_type}.pt")
        sch_type = self.cfg.train.scheduler["_target_"].split(".")[-1]
        torch.save(self.scheduler.state_dict(), f"{path}/scheduler_{sch_type}.pt")
        # save the config file
        with open(f"{path}/config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))
    
    def train_test(self):
        self.model.to(self.device)
        cur_step = 0
        while cur_step < self.cfg.train.total_train_steps:
            # run training for test_every steps
            pbar = tqdm(range(self.cfg.train.test_every), position=0, leave=True)
            self.model.train()
            for idx in pbar:
                cur_step += 1
                if cur_step >= self.cfg.train.total_train_steps:
                    break
                data = self.get_data_batch(is_train=True)
                self.optimizer.zero_grad()
                data_gpu = {k: v.to(self.device) for k, v in data.items()}
                pred = self.model(data_gpu)
                Y = data_gpu['goal_trans'] if self.cfg.abs_prediction else data_gpu['delta_trans']
                loss = torch.nn.functional.mse_loss(pred, Y)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                if self.cfg.train.wandb and is_main_process() and cur_step % self.cfg.train.log_freq == 1:
                    log_dict = {
                        f"train/loss": loss.item(),
                        f"train/epoch": cur_step // len(self.train_loader),
                        f"train/step": cur_step,
                        f"train/lr": self.optimizer.param_groups[0]['lr']}
                    wandb.log(log_dict)
                pbar.set_description(
                    f"Train {cur_step}/{self.cfg.train.total_train_steps} steps | loss: {loss.item():.4f}")
            
            # run eval for test_steps
            self.model.eval()
            pbar = tqdm(range(self.cfg.train.test_steps), position=0, leave=True)
            test_losses = []
            for idx in pbar:
                data = self.get_data_batch(is_train=False)
                data_gpu = {k: v.to(self.device) for k, v in data.items()}
                pred = self.model(data_gpu)
                Y = data_gpu['goal_trans'] if self.cfg.abs_prediction else data_gpu['delta_trans']
                loss = torch.nn.functional.mse_loss(pred, Y)
                test_losses.append(loss.item())
                pbar.set_description(
                    f"Eval {idx}/{self.cfg.train.test_steps} steps | loss: {loss.item():.4f}")
            if self.cfg.train.wandb and is_main_process():
                log_dict = {
                    f"eval/avg_loss": np.mean(test_losses),
                    f"eval/epoch": cur_step // len(self.train_loader),
                    f"eval/step": cur_step}
                wandb.log(log_dict)
            
            # save model
            if self.cfg.train.save_model and is_main_process():
                avg_val_loss = np.mean(test_losses)
                self.save_model(self.run_id, avg_val_loss)

class BCOnlinePlanner:
    def __init__(self, robot_cfg, bc_cfg):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bc_cfg = bc_cfg
        self.robot_cfg = robot_cfg
        self.policy = BCPolicy(bc_cfg)
        self.policy.setup_model(load_from=bc_cfg.bc_online_weights)
        self.policy.to(self.device)
        self.policy.eval()

        from .displacement_data_collection import DispCollection
        self.agent = DispCollection(robot_cfg)
        self.dataset = BCDataset(bc_cfg, load=False)

        self.base_left_gs = None
        self.base_right_gs = None

        self.agent.connect_to_tactile_sensor()
        logging("Sensor connected.", style=1)
        self.agent.connect_to_external_cam()
        logging("External camera connected.", style=1)
        self.agent.connect_to_robot()
        logging("Robot connected.", style=1)
        self.agent.connect_to_gripper()
        logging("Gripper connected.", style=1)
    
    def proc_gs_img(self, img, side):
        assert side in ["left", "right"]
        img = self.dataset.warp_perspective(img, corners=None, output_sz=(360, 280))
        img_pil = Image.fromarray(img)
        if side == "left":
            rval = self.dataset.leftgs_preproc(img_pil)
        else:
            rval = self.dataset.rightgs_preproc(img_pil)
        return rval
    
    def record_base_gs_imgs(self, vis_frame=None):
        self.base_left_gs_ori, self.base_right_gs_ori = self.agent.get_gs_images()
        self.base_left_gs = self.proc_gs_img(self.base_left_gs_ori, "left")
        self.base_right_gs = self.proc_gs_img(self.base_right_gs_ori, "right")
        if vis_frame is not None:
            vis_frame[:480, :640] = self.base_left_gs_ori
            vis_frame[480:, :640] = self.base_right_gs_ori

    def get_obs(self, batched, device="cpu", vis_frame=None):
        left_frame_ori, right_frame_ori = self.agent.get_gs_images()
        external_cam_frame = self.agent.get_external_cam_image()
        cur_pose = self.agent.get_franka_pose()
        # apply pre-processing
        # process gs images
        left_frame = self.proc_gs_img(left_frame_ori, "left")
        right_frame = self.proc_gs_img(right_frame_ori, "right")
        # process external cam image
        cam_im = Image.fromarray(external_cam_frame)
        cam_im = self.dataset.cam_preproc(cam_im)

        cur_translation = 1000. * (cur_pose.translation - self.bc_cfg.mean_abs_pose[self.bc_cfg.for_object]) # mm
        cur_translation = torch.from_numpy(cur_translation).type(torch.FloatTensor)

        if vis_frame is not None:
            vis_frame[:480, 640:1280] = left_frame_ori
            vis_frame[480:, 640:1280] = right_frame_ori
            vis_frame[240:720, 1280:] = external_cam_frame

        if batched:
            return {
                'cam_img': cam_im.unsqueeze(0).to(device),
                'cur_left': left_frame.unsqueeze(0).to(device),
                'cur_right': right_frame.unsqueeze(0).to(device),
                'base_left': self.base_left_gs.unsqueeze(0).to(device),
                'base_right': self.base_right_gs.unsqueeze(0).to(device),
                'cur_trans': cur_translation.unsqueeze(0).to(device),
            }
        else:
            return {
                'cam_img': cam_im.to(device),
                'cur_left': left_frame.to(device),
                'cur_right': right_frame.to(device),
                'base_left': self.base_left_gs.to(device),
                'base_right': self.base_right_gs.to(device),
                'cur_trans': cur_translation.to(device),
            }

    def step(self, move: bool, act_scale = 1.0, vis_frame=None):
        assert self.base_left_gs is not None
        assert self.base_right_gs is not None
        obs = self.get_obs(batched=True, device=self.device, vis_frame=vis_frame)
        pred = self.policy(obs).cpu().detach().numpy()
        if self.bc_cfg.abs_prediction:
            goal_trans = pred / 1000. + self.bc_cfg.mean_abs_pose[self.bc_cfg.for_object] # m
            logging(f"Predicted goal pose: {goal_trans}", True, "green")
        else:
            act = pred / 1000. * act_scale # m
            logging(f"Predicted displacement: {act}", True, "green")

        if move:
            # move the robot
            cur_pose = self.agent.get_franka_pose()
            target_pose = cur_pose.copy()
            target_pose.rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            if self.bc_cfg.abs_prediction:
                assert np.linalg.norm(cur_pose.translation - goal_trans) < 0.01, "Large movement. Aborting."
                target_pose.translation = goal_trans
            else:
                target_pose.translation = cur_pose.translation + act
            self.agent.goto_pose(target_pose)
            