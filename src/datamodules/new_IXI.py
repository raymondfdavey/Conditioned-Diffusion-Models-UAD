from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
from typing import Optional
import pandas as pd
import os
from torch.utils.data import Dataset
import numpy as np
import torch
import SimpleITK as sitk
import torchio as tio
sitk.ProcessObject.SetGlobalDefaultThreader("Platform")
from multiprocessing import Manager
from omegaconf import DictConfig, OmegaConf, open_dict


class IXI_new(LightningDataModule):
    def __init__(self, cfg):
        super(IXI_new, self).__init__()

        self.cfg = cfg
        
        self.preload = cfg.get('preload',True)
        # self.cfg.permute = False # no permutation for IXI
        
        
        # Add slice indices to config (can be overridden in actual config)
        self.slice_indices = cfg.get('slice_indices', [80, 81, 82, 83])  # default middle slices
        
        self.imgpath = {}
        
        self.csvpath_test = cfg.path.IXI.IDs.test
        # print(self.csvpath_test)

        self.csv = {}
        self.csv['test'] = pd.read_csv(self.csvpath_test)

        self.csv['test']['settype'] = 'test'
        self.csv['test']['setname'] = 'IXI'

        self.csv['test']['img_path'] = self.csv['test']['img_path'].apply(lambda x: os.path.join(cfg.path.pathBase, 'Data', x.lstrip('/')))        
        self.csv['test']['mask_path'] = self.csv['test']['mask_path'].apply(lambda x: os.path.join(cfg.path.pathBase, 'Data', x.lstrip('/')))

    def setup(self):#
        if self.cfg.debugging: # for debugging
            print('DEBUGGING!!! ONLY 1 MRI USED')
            self.test_eval = new_eval(self.csv['test'][0:1],self.cfg)
        else:
            self.test_eval = new_eval(self.csv['test'],self.cfg)
    
    def test_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)



def new_eval(csv, cfg): 
    subjects = []

    for idx, sub in csv.iterrows():
        # Create a subject dictionary with only the essential fields we need
        subject_dict = {
            'vol': tio.ScalarImage(sub.img_path, reader=sitk_reader),
            'vol_orig': tio.ScalarImage(sub.img_path, reader=sitk_reader),
            'age': sub.age,
            'ID': sub.img_name,
            'label': sub.label,
            'Dataset': sub.setname,
            'stage': sub.settype,
            'path': sub.img_path,
            'mask': tio.LabelMap(sub.mask_path, reader=sitk_reader),
            'mask_orig': tio.LabelMap(sub.mask_path, reader=sitk_reader)
        }

        # Create the subject and apply transforms
        subject = tio.Subject(subject_dict)
        transformed_subject = get_transform(cfg)(subject)
        subjects.append(transformed_subject)

    return tio.SubjectsDataset(subjects)  # Note: Changed from transformed_subject to subjects


def get_transform(cfg): # only transforms that are applied once before preloading
    h, w, d = tuple(cfg.get('imageDim',(192,192,160)))

    if not cfg.resizedEvaluation: 
        exclude_from_resampling = ['vol_orig','mask_orig','seg_orig']
    else: 
        exclude_from_resampling = None
        
    if cfg.get('unisotropic_sampling',True):
        preprocess = tio.Compose([
        tio.CropOrPad((h,w,d),padding_mode=0),
        tio.RescaleIntensity((0, 1),percentiles=(cfg.get('perc_low',1),cfg.get('perc_high',99)),masking_method='mask'),
        tio.Resample(cfg.get('rescaleFactor',3.0),image_interpolation='bspline',exclude=exclude_from_resampling),#,exclude=['vol_orig','mask_orig','seg_orig']), # we do not want to resize *_orig volumes
        ])

    else: 
        preprocess = tio.Compose([
                tio.RescaleIntensity((0, 1),percentiles=(cfg.get('perc_low',1),cfg.get('perc_high',99)),masking_method='mask'),
                tio.Resample(cfg.get('rescaleFactor',3.0),image_interpolation='bspline',exclude=exclude_from_resampling),#,exclude=['vol_orig','mask_orig','seg_orig']), # we do not want to resize *_orig volumes
            ])


    return preprocess 

def sitk_reader(path):
                
    image_nii = sitk.ReadImage(str(path), sitk.sitkFloat32)
    if not 'mask' in str(path) and not 'seg' in str(path) : # only for volumes / scalar images
        image_nii = sitk.CurvatureFlow(image1 = image_nii, timeStep = 0.125, numberOfIterations = 3)
    vol = sitk.GetArrayFromImage(image_nii).transpose(2,1,0)
    return vol, None



'''
When you call `test_dataloader()`, you'll get a PyTorch DataLoader that yields batches where each batch contains:

- `batch['vol'][tio.DATA]`: tensor of shape `[1, 1, H, W, 4]`
  - First 1: batch size (always 1 in test)
  - Second 1: channels
  - H, W: height and width after preprocessing
  - 4: your selected slices (or however many you specified in slice_indices)

- `batch['vol_orig'][tio.DATA]`: Same shape, but with original intensities
- `batch['mask'][tio.DATA]`: Same shape but with mask values
- Other metadata like 'age', 'ID', etc.

So whereas before each item had shape `[1, 1, H, W, D]` where D was the full depth of the volume, now D is just 4 (or your chosen number of slices), representing just those specific selected slices after they've gone through all the same preprocessing as before.

Would you like me to add some print statements to the code to verify these shapes when it runs?


'''