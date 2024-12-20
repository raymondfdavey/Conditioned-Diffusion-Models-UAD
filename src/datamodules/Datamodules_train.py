from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
import src.datamodules.create_dataset as create_dataset
from typing import Optional
import pandas as pd
import os

class IXI(LightningDataModule):

    def __init__(self, cfg, fold = None):
        print('IN DATAMODULES_TRAIN.IXI')
        super(IXI, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload',True)
        

        # IXI

        self.cfg.permute = False # no permutation for IXI


        self.imgpath = {}
        # gets the 
        self.csvpath_train = cfg.path.IXI.IDs.train[fold]
        self.csvpath_val = cfg.path.IXI.IDs.val[fold]
        self.csvpath_test = cfg.path.IXI.IDs.test
        self.csv = {}
        states = ['train','val','test']

        self.csv['train'] = pd.read_csv(self.csvpath_train)
        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        
        if cfg.mode == 't2':
            keep_t2 = pd.read_csv(cfg.path.IXI.keep_t2) # only keep t2 images that have a t1 counterpart
        
        # load data paths and indices
        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'IXI'


            self.csv[state]['img_path'] = cfg.path.pathBase + '/Data' + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase + '/Data' + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = None

            if cfg.mode == 't2': 
                self.csv[state] = self.csv[state][self.csv[state].img_name.isin(keep_t2['0'].str.replace('t2','t1'))]
                self.csv[state]['img_path'] = self.csv[state]['img_path'].str.replace('t1','t2')
    def setup(self, stage: Optional[str] = None):
        debug=False
        # called on every GPU
        if not hasattr(self,'train'):
            if debug == True: # for debugging
                self.train = create_dataset.Train(self.csv['train'][0:1],self.cfg) 
                self.val = create_dataset.Train(self.csv['val'][0:1],self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'][0:1],self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:1],self.cfg)
            else: 
                self.train = create_dataset.Train(self.csv['train'],self.cfg) 
                self.val = create_dataset.Train(self.csv['val'],self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'],self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'],self.cfg)
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=True, drop_last=self.cfg.get('droplast',False))

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def val_eval_dataloader(self):
        return DataLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

    def test_eval_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)
      
