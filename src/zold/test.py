import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import os
import sys
import torch
import warnings
from pytorch_lightning import LightningDataModule, Trainer
from src.utils import utils
import pickle
import wandb 
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)


os.environ['NUMEXPR_MAX_THREADS'] = '16'
# Suppress specific warning about batch size inference
warnings.filterwarnings("ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*")

# Configure paths to model checkpoints - modify these for your setup
ENCODER_PATH = "path/to/encoder/checkpoint.ckpt"  # Path to pretrained encoder weights if using cDDPM
MODEL_PATH = "path/to/model/checkpoint.ckpt"      # Path to main model checkpoint

# Initialize logger
log = utils.get_logger(__name__)

@hydra.main(config_path="configs/", config_name="config.yaml")
def test(cfg: DictConfig):
    
    # Force evaluation mode in config
    cfg.onlyEval = True 
    
    # Initialize results dictionary and prefix for logging
    results = {}
    
    # Configure model paths and encoder settings
    if not hasattr(cfg.model.cfg, 'encoder_path'):
        cfg.model.cfg.encoder_path = ENCODER_PATH
    if not hasattr(cfg.model.cfg, 'pretrained_encoder'):
        cfg.model.cfg.pretrained_encoder = True

    #! instantiate DDPM_2d and so all the others too
    cfg.model._target_ = f'src.models.{cfg.model._target_}'
    print(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)

    #! load in all the pretrained shit
    print(f"Loading model checkpoint from: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH)
    state_dict = checkpoint['state_dict']
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    # Log any issues with state dict loading
    if len(missing_keys) > 0:
        print(f"Missing keys in state dict: {missing_keys}")
    if len(unexpected_keys) > 0:
        print(f"Unexpected keys in state dict: {unexpected_keys}")
        
    model.eval()  # Set model to evaluation mode

    # Initialize dictionary to store test predictions
    preds_dict = {'test': {}}


    
    # Initialize data module for current dataset
    cfg.datamodule._target_ = f'src.datamodules.IXI'
    print(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup()

    trainer = Trainer(
        accelerator=cfg.trainer.get('accelerator', 'gpu'),
        devices=cfg.trainer.get('devices', 1),
        precision=cfg.trainer.get('precision', 32)
    )

    # Run evaluation - different dataloaders for IXI vs other datasets
    print(f"Running evaluation on IXI")

    trainer.test(model=model, dataloaders=datamodule.test_eval_dataloader())


    # Store and log results
    preds_dict['test']['IXI'] = trainer.lightning_module.eval_dict  # Store raw predictions
    log_dict = utils.summarize(preds_dict['test']['IXI'], 'test')  # Summarize metrics
    trainer.logger.experiment[0].log(log_dict)                   # Log to wandb or other logger

    # Save predictions to pickle file if configured
    if cfg.get('pickle_preds', True):
        output_path = os.path.join(os.getcwd(), f'{prefix}_preds_dict.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(preds_dict, f)
            print(f"Saved predictions to {output_path}")

if __name__ == "__main__":
    test()