import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("Hello from train-boudless-das!")
    print(cfg)


if __name__ == "__main__":
    main()
