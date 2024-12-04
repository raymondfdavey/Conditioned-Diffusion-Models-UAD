from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything
)
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.plugins import DDPPlugin
import hydra
from omegaconf import DictConfig
from typing import Optional
import os
import warnings
import torch
from src.utils import utils
import logging

os.environ['NUMEXPR_MAX_THREADS'] = '16'
warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

@hydra.main(config_path='configs', config_name='config') # Hydra decorator
def train(cfg: DictConfig): 
    seed_everything(cfg.seed, workers=True)
    _, checkpoints = utils.get_checkpoint(cfg, cfg.get('load_checkpoint'))
    model: LightningModule = hydra.utils.instantiate(cfg.model,prefix='1/') 
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, _convert_="partial", plugins=None,logger=False, enable_progress_bar=False)          
    model.load_state_dict(torch.load(checkpoints[f'fold-1'])['state_dict'])
    cfg.datamodule._target_ = 'src.datamodules.Datamodules_train.IXI'
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule, fold=0)
    datamodule.setup()

    model.test = 'consistency'
    model.test = None
    whatami = trainer.test(
        model=model,
        dataloaders=datamodule.test_eval_dataloader(),
        ckpt_path=checkpoints[f"fold-1"]
    )
    print('return from trainer.test = ', whatami)        

