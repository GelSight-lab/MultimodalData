from probing_panda.bc_policy import PolicyTrainer
import hydra

@hydra.main(version_base=None, config_path="../config", config_name="bc")
def train_bc(cfg):
    trainer = PolicyTrainer(cfg)
    trainer.setup_optimizer()
    trainer.train_test()
    # network = BCPolicy(cfg)

if __name__ == "__main__":
    train_bc()