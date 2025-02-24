from omegaconf import OmegaConf, DictConfig
import hydra
import os
from probing_panda import DispCollection

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    agent = DispCollection(cfg)
    agent.connect_to_gripper()
    import IPython; IPython.embed()

if __name__ == "__main__":
    main()