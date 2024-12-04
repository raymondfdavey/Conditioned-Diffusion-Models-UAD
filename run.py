# template: https://github.com/ashleve/lightning-hydra-template/blob/main/run.py
import dotenv
import hydra
from omegaconf import DictConfig
import os 
import sys
import socket

sys.setrecursionlimit(2000)
dir_path = os.path.dirname(os.path.realpath(__file__))
dotenv.load_dotenv(dir_path+'/pc_environment.env',override=True)

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    from src.train import train
    from src.utils import utils

    utils.extras(config)
    return train(config)


if __name__ == "__main__":
    main()