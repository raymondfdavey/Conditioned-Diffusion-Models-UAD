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
full_model_ckpt_path = '/home/rd81/projects/full_logs/logs/runs/DDPM_cond_2D_spark/DDPM_2D_IXI_DDPM_cond_2D_spark__2024-12-17_12-32-37/checkpoints/epoch-709_step-8519_loss-0.00_fold-1.ckpt'
encoder_ckpt_path='/home/rd81/projects/full_logs/logs/runs/MAE_2D/Spark_2D_IXI_MAE_2D__2024-12-17_10-38-34/checkpoints/epoch-859_step-10319_loss-0.00_fold-1.ckpt'

@hydra.main(config_path='configs', config_name='config') # Hydra decorator
def train(cfg: DictConfig): 
    seed_everything(cfg.seed, workers=True)
    
    model: LightningModule = hydra.utils.instantiate(cfg.model,prefix='1/', encoder_ckpt_path=encoder_ckpt_path) 
    
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, _convert_="partial", plugins=None,logger=False, enable_progress_bar=False)       
       
    model.load_state_dict(torch.load(full_model_ckpt_path)['state_dict'])
    cfg.datamodule._target_ = 'src.datamodules.Datamodules_train.IXI'
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule, fold=0)
    datamodule.setup()

    # model.test = ['consistency', 'augment']
    # model.test = ['consistency', 'augment']
    # model.test = ['consistency', 'synthetic_tumour', 'augment']
    # model.test = ['synthetic']
    
    model.test = ['consistency']
    model.test = ['consistency', 'half']
    # model.test = None
    whatami = trainer.test(
        model=model,
        dataloaders=datamodule.test_eval_dataloader(),
        ckpt_path=full_model_ckpt_path
    )
    print('return from trainer.test = ', whatami)        

