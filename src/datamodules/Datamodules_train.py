from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
import src.datamodules.create_dataset as create_dataset
from typing import Optional
import pandas as pd
import os

class IXI(LightningDataModule):

    def __init__(self, cfg, fold = None):
        print('='*10)
        print('INITIALISING IXI DATASET')
        print('='*10)
        super(IXI, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload',True)
        
        self.cfg.permute = False # no permutation for IXI


        self.imgpath = {}
        self.csvpath_test = cfg.path.IXI.IDs.test
        self.csv = {}

        self.csv['test'] = pd.read_csv(self.csvpath_test)
    
        
    
        self.csv['test']['settype'] = 'test'
        self.csv['test']['setname'] = 'IXI'


        self.csv['test']['img_path'] = cfg.path.pathBase + '/Data' + self.csv['test']['img_path']
        self.csv['test']['mask_path'] = cfg.path.pathBase + '/Data' + self.csv['test']['mask_path']
        self.csv['test']['seg_path'] = None

    
    def setup(self, stage: Optional[str] = None):
        debug=True
        # called on every GPU
        if not hasattr(self,'train'):
            if debug == True: # for debugging
                print('='*10)
                print('USING DEBUG MODE WITH ONLY ONE MRI IMAGE')
                print('='*10)
                self.test_eval = create_dataset.Eval(self.csv['test'][5:6],self.cfg)
            else: 
                self.test_eval = create_dataset.Eval(self.csv['test'],self.cfg)
    
    def test_eval_dataloader(self):
        print('='*10)
        print('RETURNING DATALOADER')
        print('='*10)
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)
      
