# template: https://github.com/ashleve/lightning-hydra-template/blob/main/run.py
import dotenv
import hydra
from omegaconf import DictConfig
import os 
import sys
import socket
print('syspath', sys.path)


#import multiprocessing as mp
sys.setrecursionlimit(2000)
# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dir_path = os.path.dirname(os.path.realpath(__file__))
dotenv.load_dotenv(dir_path+'/pc_environment.env',override=True)
# dotenv.dotenv_values("pc_environment.env")
print(dir_path)

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
#    import torch
#    torch.multiprocessing.set_sharing_strategy('file_system')
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from src.train import train
    from src.utils import utils

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # You can safely get rid of this line if you don't want those
    
    
    # Get the original working directory
    # original_cwd = hydra.utils.get_original_cwd()
    # print(f"Original working directory: {original_cwd}")
    
        # Check if environment variables are loaded correctly
    # print(f"DATA_DIR: {os.environ.get('DATA_DIR')}")
    # print(f"LOG_DIR: {os.environ.get('LOG_DIR')}")

    
    utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # Train model
    current_dir = os.getcwd()

    # print(f"Current working directory: {current_dir}")

    return train(config)


if __name__ == "__main__":
    main()