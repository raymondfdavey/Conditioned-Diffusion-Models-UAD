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

# Suppress specific warning about batch size inference
warnings.filterwarnings("ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*")

# Configure paths to model checkpoints - modify these for your setup
ENCODER_PATH = "path/to/encoder/checkpoint.ckpt"  # Path to pretrained encoder weights if using cDDPM
MODEL_PATH = "path/to/model/checkpoint.ckpt"      # Path to main model checkpoint

# Initialize logger
log = utils.get_logger(__name__)

@hydra.main(config_path="configs/", config_name="config.yaml")
def test(cfg: DictConfig):
    
    
    """
    Main testing function to evaluate trained models on various datasets.
    Uses hydra for configuration management.
    
    Args:
        cfg (DictConfig): Hydra configuration containing model and testing parameters
    """
    # Force evaluation mode in config
    cfg.onlyEval = True 
    
    # Initialize results dictionary and prefix for logging
    results = {}
    prefix = f'{cfg.get("fold", 0)+1}/'  # Used for logging when running multiple folds

    # Define which datasets to use for each modality (T1/T2)
    # Each modality has specific datasets it should be evaluated on
    sets = {
        't2': ['Datamodules_eval.Brats21', 'Datamodules_eval.MSLUB', 'Datamodules_train.IXI'],  # T2 datasets
        't1': ['Datamodules_eval.ATLAS', 'Datamodules_eval.WMH', 'Datamodules_train.IXI']       # T1 datasets
    }

    # Configure model paths and encoder settings
    if not hasattr(cfg.model.cfg, 'encoder_path'):
        cfg.model.cfg.encoder_path = ENCODER_PATH
    if not hasattr(cfg.model.cfg, 'pretrained_encoder'):
        cfg.model.cfg.pretrained_encoder = True

    # ====================== Model Initialization ======================
    # The model class is specified in the config as e.g. 'DDPM_2D.DDPM_2D' or 'Spark_2D.Spark_2D'
    # We need to prepend 'src.models.' to make it a full import path
    # Example: 'DDPM_2D.DDPM_2D' becomes 'src.models.DDPM_2D.DDPM_2D'
    cfg.model._target_ = f'src.models.{cfg.model._target_}'
    log.info(f"Instantiating model <{cfg.model._target_}>")

    # hydra.utils.instantiate:
    # 1. Takes the _target_ path and imports that class
    # 2. Uses the rest of cfg.model as kwargs for the class constructor
    # 3. In this case, also passes prefix for logging purposes
    # 
    # For example, if using DDPM_2D, this will:
    # - Import the DDPM_2D class from src.models.DDPM_2D
    # - Create an instance with all the config parameters
    # - Initialize with the encoder configuration we set earlier
    model = hydra.utils.instantiate(cfg.model, prefix=prefix)

    # Load model checkpoint
    log.info(f"Loading model checkpoint from: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH)
    
    # Load state dict with potential encoder keys, using strict=False to handle 
    # cases where the checkpoint might have slightly different keys
    state_dict = checkpoint['state_dict']
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    # Log any issues with state dict loading
    if len(missing_keys) > 0:
        log.info(f"Missing keys in state dict: {missing_keys}")
    if len(unexpected_keys) > 0:
        log.info(f"Unexpected keys in state dict: {unexpected_keys}")
        
    model.eval()  # Set model to evaluation mode

    # Initialize dictionary to store test predictions
    preds_dict = {'test': {}}

    # Iterate through each dataset specified in the config
    for set in cfg.datamodule.cfg.testsets:
        # Skip if dataset doesn't match the current modality (T1/T2)
        if not set in sets[cfg.datamodule.cfg.mode]:
            continue

        # Initialize data module for current dataset
        cfg.datamodule._target_ = f'src.datamodules.{set}'
        log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule, fold=cfg.get('fold',0))
        datamodule.setup()

        # Initialize PyTorch Lightning trainer for testing
        trainer = Trainer(
            accelerator=cfg.trainer.get('accelerator', 'gpu'),
            devices=cfg.trainer.get('devices', 1),
            precision=cfg.trainer.get('precision', 32)
        )

        # Run evaluation - different dataloaders for IXI vs other datasets
        log.info(f"Running evaluation on {set}")
        if 'Datamodules_train.IXI' in set:  # IXI is the healthy training dataset used for evaluation
            trainer.test(model=model, dataloaders=datamodule.test_eval_dataloader())
        else:  # For pathological datasets (BraTS21, MSLUB, ATLAS, WMH)
            trainer.test(model=model, dataloaders=datamodule.test_dataloader())

        # Store and log results
        preds_dict['test'][set] = trainer.lightning_module.eval_dict  # Store raw predictions
        log_dict = utils.summarize(preds_dict['test'][set], 'test')  # Summarize metrics
        log_dict = utils.summarize(log_dict, f'{prefix}'+set)        # Add prefix for logging
        trainer.logger.experiment[0].log(log_dict)                   # Log to wandb or other logger

        # Save predictions to pickle file if configured
        if cfg.get('pickle_preds', True):
            output_path = os.path.join(os.getcwd(), f'{prefix}_preds_dict.pkl')
            with open(output_path, 'wb') as f:
                pickle.dump(preds_dict, f)
                log.info(f"Saved predictions to {output_path}")

if __name__ == "__main__":
    test()