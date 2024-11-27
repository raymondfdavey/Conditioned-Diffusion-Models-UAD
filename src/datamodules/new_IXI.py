from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
import src.datamodules.create_dataset as create_dataset
from typing import Optional
import pandas as pd
import os

class IXI_new(LightningDataModule):
    def __init__(self, cfg):
        super(IXI_new, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload',True)
        self.cfg.permute = False # no permutation for IXI

        self.imgpath = {}
        
        self.csvpath_test = cfg.path.IXI.IDs.test

        self.csv = {}
        self.csv['test'] = pd.read_csv(self.csvpath_test)

        self.csv['test']['settype'] = 'test'
        self.csv['test']['setname'] = 'IXI'

        self.csv['test']['img_path'] = self.csv['test']['img_path'].apply(lambda x: os.path.join(cfg.path.pathBase, 'Data', x.lstrip('/')))        
        self.csv['test']['mask_path'] = self.csv['test']['mask_path'].apply(lambda x: os.path.join(cfg.path.pathBase, 'Data', x.lstrip('/')))

    def setup(self,):
        # called on every GPU
        if not hasattr(self,'train'):
            if self.cfg.sample_set: # for debugging
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8],self.cfg)
            else: 
                self.test_eval = create_dataset.Eval(self.csv['test'],self.cfg)
    
    def test_dataloader(self):
        return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)
      
'''
The return of this: self.test_eval = create_dataset.Eval(self.csv['test'],self.cfg)

is a tio.SubjectsDataset which contains a list tio.Subject items which are dictionary like objects. each of which contains:

    vol: A 3D MRI scan stored as a tio.ScalarImage (main image)
    vol_orig: The same image but in original size
    mask: Brain mask as a tio.LabelMap
    Various metadata (age, ID, etc.)

The images are stored as 3D tensors with shape [C, H, W, D]

    C: Channels (usually 1 for MRI) 
    H: Height
    W: Width
    D: Depth (number of slices)

Interacting with the Data:

        # Load a single subject
        dataset = test_eval_dataloader()
        subject = next(iter(dataset))

        # Access the image data
        image_3d = subject['vol'][tio.DATA]  # Gets the 3D tensor
        mask_3d = subject['mask'][tio.DATA]

        # Display a slice (using matplotlib)
        import matplotlib.pyplot as plt

        # Get middle slice (example)
        slice_idx = image_3d.shape[-1] // 2
        slice_2d = image_3d[0, :, :, slice_idx]  # [0] is for channel

        plt.imshow(slice_2d, cmap='gray')
        plt.show()

        # Get metadata
        print(f"Patient ID: {subject['ID']}")
        print(f"Image shape: {image_3d.shape}")

Data Transformations:
    The dataset already includes transformations defined in get_transform(cfg), which typically handles:
        Resampling to isotropic resolution
        Intensity normalization
        Padding/cropping to fixed dimensions
    
Specifically here:

    Eeach scan is preprocessed using get_transform which:

        Crops/pads to [160,192,160]
        Rescales intensities to [0,1]
        Resamples by a factor of 3.0 (except for '_orig' volumes)


Has shape [1, H, W, D] (1 is channel dimension)

'''

'''
But then this itself is stores as a Pytorch DataLoader here:         
return DataLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False)

The DataLoader:

    Creates batches (here size 1) from your dataset
    Handles parallel loading (num_workers)
    Provides iteration over all 100 scans
    Each iteration gives you one scan and its associated data

    So if you have 100 scans, you'll get 100 iterations, each containing one preprocessed scan and its metadata.


    To use it
    
    # Create iterator
    
    dataloader = test_dataloader()

    # Iterate through all scans
    
    for batch in dataloader:
        # batch is a dictionary containing:
        vol = batch['vol'][tio.DATA]        # Shape: [1, 1, 160, 192, 160]
        #      ^ batch    ^ channel  ^ spatial dimensions
        vol_orig = batch['vol_orig'][tio.DATA]  # Original size
        mask = batch['mask'][tio.DATA]
        
        # Get metadata
        patient_id = batch['ID']
        
        # Process one slice
        slice_idx = 80  # middle slice
        current_slice = vol[0, 0, :, :, slice_idx]  # Shape: [160, 192]

    # Or get a specific scan
    single_batch = next(iter(dataloader))
'''

