import dotenv
import hydra
from omegaconf import DictConfig
import os
import sys

# Set up environment and system configurations
sys.setrecursionlimit(2000)

# Load environment variables from .env file
dir_path = os.path.dirname(os.path.realpath(__file__))
dotenv.load_dotenv(os.path.join(dir_path, 'pc_environment.env'), override=True)

# Main entry point that uses your test_config.yaml
@hydra.main(config_path="configs/", config_name="test_config.yaml")
def main(config: DictConfig):
    # Import test function locally to avoid potential import issues
    from src.test import test
    return test(config)

if __name__ == "__main__":
    main()