import torchvision.transforms as transforms
from src.models.modules.cond_DDPM import GaussianDiffusion
from src.models.modules.OpenAI_Unet import UNetModel as OpenAI_UNet
from src.models.modules.DDPM_encoder import get_encoder
import torch
import numpy as np
from pytorch_lightning.core.lightning import LightningModule
import torch.optim as optim
from typing import Any
import torchio as tio
from src.utils.generate_noise import gen_noise
import wandb
from omegaconf import open_dict
from collections import OrderedDict
from torch import nn
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
import wandb 
import monai
from torch.nn import functional as F
from PIL import Image
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig
import gzip
import matplotlib.colors as colors
import scipy
from scipy import ndimage



class DDPM_2D(LightningModule):
    def __init__(self,cfg,prefix=None, encoder_ckpt_path=None):
        super().__init__()
        print('='*10)
        print('INITIALISING DDPM_2D')
        print('='*10)
        self.prefix = prefix
        self.cfg = cfg
        
        #! initialising new encoder and loading in pretrained encoder weights
        with open_dict(self.cfg):
            self.cfg['cond_dim'] = cfg.get('unet_dim',64) * 4
            self.encoder, out_features = get_encoder(cfg)
        state_dict_pretrained = torch.load(encoder_ckpt_path)['state_dict']
        new_statedict = OrderedDict()
        for key in zip(state_dict_pretrained): 
            if 'slice_encoder' in key[0] :
                new_key = 'slice_encoder'+ key[0].split('encoder')[-1]
                new_statedict[new_key] = state_dict_pretrained[key[0]]
            elif 'sparse_encoder' in key[0] :
                if not 'fc.weight' in key[0] and not 'fc.bias' in key[0]: # remove fc layer
                    new_key = 'encoder' + key[0].split('sp_cnn')[-1]
                    new_statedict[new_key] = state_dict_pretrained[key[0]]
            else:
                new_statedict[key[0]] = state_dict_pretrained[key[0]]
        self.encoder.load_state_dict(new_statedict,strict=False)

        #! initialising new unet (weights added in train.py - for some reason)
        model = OpenAI_UNet(
                            image_size =  (int(cfg.imageDim[0] / cfg.rescaleFactor),int(cfg.imageDim[1] / cfg.rescaleFactor)),
                            in_channels = 1,
                            model_channels = cfg.get('unet_dim',64),
                            out_channels = 1,
                            num_res_blocks = cfg.get('num_res_blocks',3),
                            attention_resolutions = tuple(cfg.get('att_res',[3,6,12])), # 32, 16, 8
                            dropout=cfg.get('dropout_unet',0), # default is 0.1
                            channel_mult=cfg.get('dim_mults',[1, 2, 4, 8]),
                            conv_resample=True,
                            dims=2,
                            num_classes=out_features,
                            use_checkpoint=False,
                            use_fp16=True,
                            num_heads=1,
                            num_head_channels=64,
                            num_heads_upsample=-1,
                            use_scale_shift_norm=True,
                            resblock_updown=True,
                            use_new_attention_order=True,
                            use_spatial_transformer=cfg.get('spatial_transformer',False),    
                            transformer_depth=1,                            
                            )
        
        model.convert_to_fp16()

        timesteps = cfg.get('timesteps',1000)
        sampling_timesteps = cfg.get('sampling_timesteps',timesteps)
        self.test_timesteps = cfg.get('test_timesteps',150) 

        #! initialising new diffusion model (which takes unet as a param - for some reason)
        self.diffusion = GaussianDiffusion(
        model,
        image_size = (int(cfg.imageDim[0] / cfg.rescaleFactor),int(cfg.imageDim[1] / cfg.rescaleFactor)), # only important when sampling
        timesteps = timesteps,   # number of steps
        sampling_timesteps = sampling_timesteps,
        objective = cfg.get('objective','pred_x0'), # pred_noise or pred_x0 which is pred image start
        channels = 1,
        loss_type = cfg.get('loss','l1'),    # L1 or L2
        p2_loss_weight_gamma = cfg.get('p2_gamma',0),
        cfg=cfg
        )
        
        self.save_hyperparameters()

    def forward(self, x): # encode input
        if self.cfg.get('condition',True):
            x = self.encoder(x)
        else: 
            x = None
        return x

    def on_test_start(self):
        print('='*10)
        print('STARTING TESTING...')
        print('='*10)
        self.new_size = [190,190,160]
    
    def test_step(self, batch: Any, batch_idx: int):
                
        self.dataset = batch['Dataset']
        input = batch['vol'][tio.DATA]
        data_orig = batch['vol_orig'][tio.DATA]
        data_seg = batch['seg_orig'][tio.DATA] if batch['seg_available'] else torch.zeros_like(data_orig)
        data_mask = batch['mask_orig'][tio.DATA]
        ID = batch['ID']
        self.stage = batch['stage']
        self.cfg['noise_ensemble'] = False
                        
                        
        total_slices = input.size(4)
        percentages = [0.4, 0.6]  # Can adjust these values
        n_slices = len(percentages)
        slice_indices = [int(total_slices * p) for p in percentages]
        input = input[..., slice_indices]
        data_orig = data_orig[..., slice_indices]
        data_seg = data_seg[..., slice_indices]
        data_mask = data_mask[..., slice_indices]
        final_volume = torch.zeros([input.size(2), input.size(3), input.size(4)], device = self.device)
        assert input.shape[0] == 1, "Batch size must be 1"
        input = input.squeeze(0).permute(3,0,1,2) # [B,C,H,W,D] -> [D,C,H,W]
        
        #! TEST
        if 'consistency' in self.test:
            print('in consistency')
            if 'augment' in self.test:
                print('in consistency + augment')
                tumor1_path = "/home/rd81/projects/diffusion-uad/analysis/brats_tumor_images/image_00002/tumor_only_57.png"
                mask1_path = "/home/rd81/projects/diffusion-uad/analysis/brats_tumor_images/image_00002/tumor_mask_57.png"
                tumor2_path = "/home/rd81/projects/diffusion-uad/analysis/brats_tumor_images/image_00002/tumor_only_85.png"
                mask2_path = "/home/rd81/projects/diffusion-uad/analysis/brats_tumor_images/image_00002/tumor_mask_85.png"

                # Load tumor and mask
                tumor_tensors, mask_tensors = load_and_preprocess_tumor(tumor1_path, mask1_path, tumor2_path, mask2_path, scale_factor=1.5, device=self.device, erosion_pixels=2)
                augmented_input, segmentation_masks = augment_with_tumor(input, tumor_tensors, mask_tensors, n_slices)
                # fig = visualize_results(input, augmented_input, tumor_tensors, mask_tensors)
                data = self.run_repeats(augmented_input, data_orig, segmentation_masks, data_mask, ID, n_runs=100, n_slices=n_slices, save_images_to_disk=True, n_to_save = 10)
            else:
                print('consistency NO augment')
                data = self.run_repeats(input, data_orig, None, data_mask, ID, n_runs=100, n_slices=n_slices, save_images_to_disk=True, n_to_save = 10)
            self.save_repeat_run_data(data, ID, n_slices)
        else:
            print('single run')
            data = self.run_normal(input, data_orig, data_seg, data_mask, ID, save_images_to_disk=True)
            
        self.log('test_metric', 100)  # This will now show up in trainer.test() results
    
    def save_repeat_run_data(self, storage, ID, n_slices):
        print('='*10)
        print(f'Saving run data')
        save_path = HydraConfig.get().run.dir
        torch.save(storage, f"{save_path}/runs_data_{ID}_{n_slices}_slices.pt")
        print(f'Run data saved to {f"{save_path}/runs_data_{ID}_{n_slices}_slices.pt"}')
        print('='*10)
        
    def run_normal(self, input, data_orig, data_seg, data_mask, ID, save_images_to_disk=True):
        print('='*10)
        print(f'RUNNING NORMAL TEST: SINGLE RUN')
        print('='*10)
        features = self(input)
        noise = gen_noise(self.cfg, input.shape).to(self.device)
        
        #! MODEL CALL
        loss_diff, reco, unet_details = self.diffusion(input,cond=features,t=self.test_timesteps-1,noise=noise)
        
        final_volume = reco.clone().squeeze()
        final_volume = final_volume.permute(1,2,0) # to HxWxD
        final_volume = final_volume.unsqueeze(0)
        final_volume = final_volume.unsqueeze(0)
        
        key_points = extract_key_points(unet_details)
        # print_key_details(key_points, 1)
        
        # Resize the images if desired
        if not self.cfg.resizedEvaluation: # in case of full resolution evaluation 
            final_volume = F.interpolate(final_volume, size=self.new_size, mode="trilinear",align_corners=True).squeeze() # resize
        else: 
            final_volume = final_volume.squeeze()
        
        # calculate the residual image
        if self.cfg.get('residualmode','l1'): # l1 or l2 residual
            diff_volume = torch.abs((data_orig-final_volume))
        else:
            diff_volume = (data_orig-final_volume)**2

        # move data to CPU
        data_seg = data_seg.cpu() 
        data_mask = data_mask.cpu()
        diff_volume = diff_volume.cpu()
        data_orig = data_orig.cpu()
        final_volume = final_volume.cpu()

        data_mask[data_mask > 0] = 1

        diff_volume = apply_brainmask_volume(diff_volume.cpu(), data_mask.squeeze().cpu())   
        diff_volume = torch.from_numpy(apply_3d_median_filter(diff_volume.numpy().squeeze(),kernelsize=self.cfg.get('kernelsize_median',5))).unsqueeze(0) # bring back to tensor
        
        if save_images_to_disk:
            save_images(self,diff_volume, data_orig, data_seg, data_mask, final_volume, ID)
        

    def run_repeats(self, input, data_orig, segmentation_mask, data_mask, ID, n_runs=500, n_slices=4, save_images_to_disk=True,  n_to_save = 5):
        print('='*10)
        print(f'RUNNING TEST: REPEATS - {n_runs} runs, {n_slices} slices')
        print('='*10)
        
        assert input.shape == (n_slices, 1, 96, 96), f"Expected input shape (4, 1, 96, 96), got {input.shape}"
        storage=setup_storage(n_runs, n_slices)
        storage['input'][0] = input.squeeze(1)
        storage['brain_mask'][0] = data_mask.squeeze().permute(2, 0, 1)
        features = self(input)
        storage['features_raw_embedding'][0] = features
        if segmentation_mask != None:
            storage['segmentation_mask'][0] = segmentation_mask.squeeze(1)  
            
        for run_idx in range(n_runs):
            print(f'Running iteration {run_idx + 1}/{n_runs}')
            
            noise = gen_noise(self.cfg, input.shape).to(self.device)
            storage['noise'][run_idx] = noise.squeeze(1)

            loss_diff, reco, unet_details = self.diffusion(input, cond=features, t=self.test_timesteps-1, noise=noise)
                        
            final_volume = reco.clone().squeeze(1)  # Just remove the channel dimension, keeps slices first!
            storage['reconstructions'][run_idx] = final_volume
            
            final_volume_for_calc = final_volume.permute(1, 2, 0)  # -> [96, 96, 4]
            final_volume_for_calc = final_volume_for_calc.unsqueeze(0).unsqueeze(0)  # -> [1, 1, 96, 96, 4]
            if self.cfg.get('residualmode','l1'):
                diff_volume = torch.abs((data_orig - final_volume_for_calc))
            else:
                diff_volume = (data_orig - final_volume_for_calc)**2
            
            
            storage['differences'][run_idx] = diff_volume.squeeze().permute(2, 0, 1)

            key_points = extract_key_points(unet_details)
            
            
            # Store UNet features
            for key in storage['features'].keys():
                if key in key_points:
                    storage['features'][key][run_idx] = key_points[key]['tensor']
            
            # Store embeddings
    
            if run_idx == 0:            
                storage['full_raw_embedding'][0] = key_points['raw_embedding']['tensor']
                for key in storage['embeddings'].keys():
                    if key in key_points:
                        storage['embeddings'][key][run_idx] = key_points[key]['tensor']
            
            if save_images_to_disk and run_idx < n_to_save:
                # Save images will handle moving to CPU and post-processing
                
                save_images_from_repeat(
                    self, 
                    input,
                    run_idx,
                    noise,
                    diff_volume,
                    segmentation_mask,
                    data_mask,
                    final_volume.permute(1, 2, 0),
                    ID
                )
            
            # Optional: Clear GPU cache after each run if memory is a concern
            torch.cuda.empty_cache()
        return storage
            
    def on_test_end(self):
        print('='*10)
        print('FINISHED TESTING')
        print('='*10)
         
def setup_storage(n_runs, n_slices):

    storage = {
        # Input/Output Image Storage
        'input': torch.zeros(1, n_slices, 96, 96),  # Removed the channel dimension
        'features_raw_embedding': torch.zeros(1, n_slices, 512), 
        'full_raw_embedding': torch.zeros(1, n_slices, 1024),
        'brain_mask': torch.zeros(1, n_slices, 96, 96),     # Brain masks
        'segmentation_mask': torch.zeros(1, n_slices, 96, 96),   # Segmentation masks
        
        'reconstructions': torch.zeros(n_runs, n_slices, 96, 96), # Reconstructed images
        'noise': torch.zeros(n_runs, n_slices, 96, 96),
        'differences': torch.zeros(n_runs, n_slices, 96, 96),     # Difference maps
        
        # # Segmentation and Mask Storage (for completeness)
        
        # UNet Feature Storage
        'features': {
            'down_post_1': torch.zeros(n_runs, n_slices, 128, 96, 96),
            'down_post_4': torch.zeros(n_runs, n_slices, 128, 48, 48),
            'down_post_8': torch.zeros(n_runs, n_slices, 256, 24, 24),
            'middle_post': torch.zeros(n_runs, n_slices, 256, 24, 24),
            'up_post_3': torch.zeros(n_runs, n_slices, 256, 48, 48),
            'up_post_7': torch.zeros(n_runs, n_slices, 256, 96, 96),
            'up_post_final_layer': torch.zeros(n_runs, n_slices, 1, 96, 96)
        },
        
        # Embedding Storage
        'embeddings': {
            'down_post_1_embedding': torch.zeros(1, n_slices, 256),
            'down_post_4_embedding': torch.zeros(1, n_slices, 256),
            'down_post_8_embedding': torch.zeros(1, n_slices, 512),
            'middle_post_embedding': torch.zeros(1, n_slices, 512),
            'up_post_3_embedding': torch.zeros(1, n_slices, 512),
            'up_post_7_embedding': torch.zeros(1, n_slices, 512)
        }
    }
    return storage

def save_images(self, diff_volume, data_orig, data_seg, data_mask, final_volume, ID, diff_volume_KL=None,  flow=None ):
    print('='*10)
    print('SAVING IMAGES - NORMAL RUN')
    print('='*10)
    
    ImagePathList = {
                    'imagesGrid': os.path.join(os.getcwd(),'grid')}
    
    for key in ImagePathList :
        if not os.path.isdir(ImagePathList[key]):
            os.mkdir(ImagePathList[key])

    for j in range(0,diff_volume.squeeze().shape[2],1):
        
        # create a figure of images with 1 row and 4 columns for subplots
        fig, ax = plt.subplots(1,3,figsize=(16,4))
        # change spacing between subplots
        fig.subplots_adjust(wspace=0.0)
        # orig
        ax[0].imshow(data_orig.squeeze()[...,j].rot90(3),'gray')
        # reconstructed
        ax[1].imshow(final_volume[...,j].rot90(3).squeeze(),'gray')
        # difference
        ax[2].imshow(diff_volume.squeeze()[:,...,j].rot90(3),'inferno',norm=colors.Normalize(vmin=0, vmax=diff_volume.max()+.01))
        # mask
        # ax[3].imshow(data_seg.squeeze()[...,j].rot90(3),'gray')
        
        # remove all the ticks (both axes), and tick labels
        for axes in ax:
            axes.set_xticks([])
            axes.set_yticks([])
        # remove the frame of the chart
        for axes in ax:
            axes.spines['top'].set_visible(False)
            axes.spines['right'].set_visible(False)
            axes.spines['bottom'].set_visible(False)
            axes.spines['left'].set_visible(False)
        # remove the white space around the chart
        plt.tight_layout()
        
        plt.savefig(os.path.join(ImagePathList['imagesGrid'], '{}_{}_Grid.png'.format(ID[0],j)),bbox_inches='tight')

        # self.logger.experiment[0].log({'images/{}/{}_Grid.png'.format(self.dataset[0],j) : wandb.Image(plt)})
        plt.clf()
        plt.cla()
        plt.close()
    print('='*10)
    print(f"IMAGES SAVED TO {ImagePathList['imagesGrid']}")
    print('='*10)
          
def save_images_from_repeat(self, input, run_idx, noise, diff_volume, segmentation_mask, data_mask, final_volume, ID):
    print('='*10)
    print(f'Saving iteration {run_idx+1} images...')
    
    # Move tensors to CPU here
    diff_volume = diff_volume.cpu()
    input = input.cpu()
    noise = noise.cpu()
    segmentation_mask = segmentation_mask.cpu() if segmentation_mask is not None else None
    data_mask = data_mask.cpu()
    final_volume = final_volume.cpu()
    
    # Apply brain mask and filtering
    data_mask[data_mask > 0] = 1
    diff_volume = apply_brainmask_volume(diff_volume, data_mask.squeeze())
    diff_volume = torch.from_numpy(
        apply_3d_median_filter(
            diff_volume.numpy().squeeze(),
            kernelsize=self.cfg.get('kernelsize_median', 5)
        )
    ).unsqueeze(0)
    
    ImagePathList = {
        'imagesGrid': os.path.join(os.getcwd(), 'grid')
    }
    
    for key in ImagePathList:
        if not os.path.isdir(ImagePathList[key]):
            os.mkdir(ImagePathList[key])
    
    for j in range(0, diff_volume.squeeze().shape[2], 1):
        # create a figure of images with 1 row and 6 columns (or 5 if no segmentation)
        n_cols = 6 if segmentation_mask is not None else 5
        fig, ax = plt.subplots(1, n_cols, figsize=(20, 4))
        # change spacing between subplots
        fig.subplots_adjust(wspace=0.0)
        
        # orig
        ax[0].imshow(input.squeeze(1).permute(1, 2, 0)[..., j].rot90(3), 'gray')
        ax[0].set_title('Original')
        
        ax[1].imshow(final_volume[..., j].rot90(3).squeeze(), 'gray')
        ax[1].set_title('Reconstructed')
        
        ax[2].imshow(diff_volume.squeeze()[:, ..., j].rot90(3), 'inferno',
                    norm=colors.Normalize(vmin=0, vmax=diff_volume.max()+.01))
        ax[2].set_title('Difference')
        
        ax[3].imshow(noise[j, 0].rot90(3), 'gray')
        ax[3].set_title('Noise')
        
        # mask
        ax[4].imshow(data_mask.squeeze()[..., j].rot90(3), 'gray')
        ax[4].set_title('Mask')
        
        if segmentation_mask is not None:
            ax[5].imshow(segmentation_mask.squeeze(1).permute(1, 2, 0)[..., j].rot90(3), 'gray')
            ax[5].set_title('Segmentation')
        
        # remove all the ticks and frames
        for axes in ax:
            axes.set_xticks([])
            axes.set_yticks([])
            for spine in axes.spines.values():
                spine.set_visible(False)
                
        plt.tight_layout()
        
        if self.cfg.get('save_to_disc', True):
            plt.savefig(
                os.path.join(ImagePathList['imagesGrid'], 
                           f'{ID[0]}_{j}_Grid_RUN_{run_idx}.png'),
                bbox_inches='tight'
            )
            
        plt.clf()
        plt.cla()
        plt.close()
    print(f"Images saved to {ImagePathList['imagesGrid']}")
    print('='*10)

def print_key_details(key_points, indent):
    for key, value in key_points.items():
        print("  " * indent + f"📍 {key}")
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if subkey != 'tensor':
                    print("  " * (indent+1) + f"└─ {subkey}: {subvalue}")
        print()  # Add blank line between sections

def log_images_with_patch(self, diff_volume, data_orig, data_seg, data_mask, final_volume, patched_input, patch_mask, ID, diff_volume_KL=None, flow=None):
    """
    Modified log_images function that includes the patch visualization.
    """
    print('SAVING IMAGES WITH PATCH!')
    diff_volume = diff_volume.cpu()
    data_orig = data_orig.cpu()
    data_seg = data_seg.cpu()
    data_mask = data_mask.cpu()
    final_volume = final_volume.cpu()
    patched_input = patched_input.cpu()
    patch_mask = patch_mask.cpu()
    
    ImagePathList = {
        'imagesGrid': os.path.join(os.getcwd(), 'grid')
    }
    
    for key in ImagePathList:
        if not os.path.isdir(ImagePathList[key]):
            os.mkdir(ImagePathList[key])

    for j in range(0, diff_volume.squeeze().shape[2], 1):
        # Create figure with 5 columns: original, patch mask, patched input, reconstruction, difference
        fig, ax = plt.subplots(1, 5, figsize=(25, 4))
        fig.subplots_adjust(wspace=0.02)
        
        # Original
        ax[0].imshow(data_orig.squeeze()[...,j].rot90(3), 'gray')
        ax[0].set_title('Original')
        
        # Patch Mask
        ax[1].imshow(patch_mask.squeeze()[j].rot90(3), 'gray')
        ax[1].set_title('Patch Mask')
        
        # Patched Input
        ax[2].imshow(patched_input.squeeze()[j].rot90(3), 'gray')
        ax[2].set_title('Patched Input')
        
        # Reconstruction
        ax[3].imshow(final_volume[...,j].rot90(3).squeeze(), 'gray')
        ax[3].set_title('Reconstruction')
        
        # Difference
        im = ax[4].imshow(diff_volume.squeeze()[:,...,j].rot90(3), 'inferno',
                         norm=colors.Normalize(vmin=0, vmax=diff_volume.max()+.01))
        ax[4].set_title('Difference Map')
        
        # Add colorbar to difference map
        plt.colorbar(im, ax=ax[4])
        
        # Remove ticks and frames
        for axes in ax:
            axes.set_xticks([])
            axes.set_yticks([])
            for spine in axes.spines.values():
                spine.set_visible(False)
        
        plt.tight_layout()
        
        if self.cfg.get('save_to_disc', True):
            plt.savefig(os.path.join(ImagePathList['imagesGrid'], 
                                   '{}_{}_{}_Grid.png'.format(ID[0], j, 'patched')),
                       bbox_inches='tight', dpi=300)
        
        self.logger.experiment[0].log({
            'images_patched/{}/{}_Grid.png'.format(self.dataset[0], j): wandb.Image(plt)
        })
        
        plt.clf()
        plt.cla()
        plt.close()

def add_square_patch(image_tensor, patch_size=20, intensity=1.0, position=None, black=False):
    """
    Adds a square patch to the input image tensor.
    Args:
        image_tensor: Shape [D,C,H,W]
        patch_size: Size of the square patch
        intensity: Intensity value of the patch
        position: Tuple of (y,x) coordinates for patch center. If None, places in center
        black: If True, makes the patch black (0.0), if False uses the provided intensity
    Returns:
        patched_image: Modified image tensor
        patch_mask: Binary mask showing patch location
    """
    patched_image = image_tensor.clone()
    
    # Get image dimensions
    D, C, H, W = image_tensor.shape
    
    # Default to center if no position specified
    if position is None:
        y = H // 2
        x = W // 2
    else:
        y, x = position
        
    # Calculate patch boundaries
    half_size = patch_size // 2
    y1 = max(y - half_size, 0)
    y2 = min(y + half_size, H)
    x1 = max(x - half_size, 0)
    x2 = min(x + half_size, W)
    
    # Create patch mask
    patch_mask = torch.zeros_like(image_tensor)
    
    # Determine patch intensity
    patch_value = 0.0 if black else intensity
    
    # Add patch to all slices
    for d in range(D):
        patched_image[d, :, y1:y2, x1:x2] = patch_value
        patch_mask[d, :, y1:y2, x1:x2] = 1
        
    return patched_image, patch_mask
                
def apply_brainmask(x, brainmask, erode , iterations):
    strel = scipy.ndimage.generate_binary_structure(2, 1)
    brainmask = np.expand_dims(brainmask, 2)
    if erode:
        brainmask = scipy.ndimage.morphology.binary_erosion(np.squeeze(brainmask), structure=strel, iterations=iterations)
    return np.multiply(np.squeeze(brainmask), np.squeeze(x))

def apply_brainmask_volume(vol,mask_vol,erode=True, iterations=10) : 
    for s in range(vol.squeeze().shape[2]): 
        slice = vol.squeeze()[:,:,s]
        mask_slice = mask_vol.squeeze()[:,:,s]
        eroded_vol_slice = apply_brainmask(slice, mask_slice, erode = True, iterations=vol.squeeze().shape[1]//25)
        vol.squeeze()[:,:,s] = eroded_vol_slice
    return vol

def apply_3d_median_filter(volume, kernelsize=5):  # kernelsize 5 works quite well
    volume = scipy.ndimage.filters.median_filter(volume, (kernelsize, kernelsize, kernelsize))
    return volume

def normalize(tensor): # THanks DZimmerer
    tens_deta = tensor.detach().cpu()
    tens_deta -= float(np.min(tens_deta.numpy()))
    tens_deta /= float(np.max(tens_deta.numpy()))

    return tens_deta

def extract_key_points(unet_details):
    # Initialize dictionary to store our key monitoring points
    key_monitoring_points = {}
    
    # Define the main points we want to capture
    main_points = [
        'input',                  # Input state
        'raw_embedding',
        'down_post_1',           # Early features
        'down_post_4',           # Mid-down features
        'down_post_8',           # Deep features
        'middle_post',           # Bottleneck
        'up_post_3',            # Mid-up features 
        'up_post_7',            # Late up features
        'up_post_final_layer'    # Final output
    ]
    
    # Define points where we expect associated embeddings
    embedding_associations = {
        'down_post_1': 'projected_embedding_level_0_block_1',
        'down_post_4': 'projected_embedding_level_0_block_None',
        'down_post_8': 'projected_embedding_level_1_block_None',
        'middle_post': 'projected_embedding_level_None_block_None',
        'up_post_3': 'projected_embedding_level_2_block_3',
        'up_post_7': 'projected_embedding_level_1_block_3'
    }
    
    # Extract the main points and their associated embeddings
    for point in main_points:
        if point in unet_details:
            # Store the main point details
            key_monitoring_points[point] = unet_details[point]
            
            # If this point has an associated embedding, store it too
            if point in embedding_associations:
                embedding_key = embedding_associations[point]
                if embedding_key in unet_details:
                    key_monitoring_points[f"{point}_embedding"] = unet_details[embedding_key]
    
    return key_monitoring_points

def resize_and_center(image, mask, scale_factor=1.5, erosion_pixels=2):
    """
    Resize image and mask by scale_factor and erode mask slightly to avoid boundary artifacts
    """
    # Convert both to numpy for manipulation
    img_array = np.array(image)
    mask_array = np.array(mask)
    
    # Get original size
    orig_height, orig_width = img_array.shape
    
    # Calculate new size
    new_width = int(orig_width / scale_factor)
    new_height = int(orig_height / scale_factor)
    
    # Create PIL images for resizing
    img_pil = Image.fromarray(img_array)
    mask_pil = Image.fromarray(mask_array)
    
    # Resize both
    img_resized = img_pil.resize((new_width, new_height), Image.LANCZOS)
    mask_resized = mask_pil.resize((new_width, new_height), Image.NEAREST)
    
    # Convert back to numpy
    img_small = np.array(img_resized)
    mask_small = np.array(mask_resized)
    
    # Erode the mask slightly
    eroded_mask = ndimage.binary_erosion(mask_small > 0, iterations=erosion_pixels)
    mask_small[~eroded_mask] = 0
    
    # Create output arrays
    final_img = np.zeros((orig_height, orig_width), dtype=np.uint8)
    final_mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
    
    # Calculate position to paste (center)
    top = (orig_height - new_height) // 2
    left = (orig_width - new_width) // 2
    
    # Insert resized images into center
    final_img[top:top+new_height, left:left+new_width] = img_small
    final_mask[top:top+new_height, left:left+new_width] = mask_small
    
    return Image.fromarray(final_img), Image.fromarray(final_mask)

def load_and_preprocess_tumor(tumor1_path, mask1_path, tumor2_path, mask2_path, device, scale_factor=1.5, erosion_pixels=2):
    """
    Load and preprocess two different tumors and their masks
    """
    # Load all images and convert to grayscale
    tumor1 = Image.open(tumor1_path).convert('L')
    mask1 = Image.open(mask1_path).convert('L')
    tumor2 = Image.open(tumor2_path).convert('L')
    mask2 = Image.open(mask2_path).convert('L')
    
    # Process each tumor-mask pair
    tumor1, mask1 = resize_and_center(tumor1, mask1, scale_factor, erosion_pixels)
    tumor2, mask2 = resize_and_center(tumor2, mask2, scale_factor, erosion_pixels)
    
    # Rotate all images 90 degrees counterclockwise
    tumor1 = tumor1.rotate(90, expand=True)
    mask1 = mask1.rotate(90, expand=True)
    tumor2 = tumor2.rotate(90, expand=True)
    mask2 = mask2.rotate(90, expand=True)
    
    # Convert to tensors
    transform = transforms.ToTensor()
    tumor1_tensor = transform(tumor1).to(device)
    mask1_tensor = transform(mask1).to(device)
    tumor2_tensor = transform(tumor2).to(device)
    mask2_tensor = transform(mask2).to(device)
    
    # Convert masks to binary (0 or 1)
    mask1_tensor = (mask1_tensor > 0.5).float()
    mask2_tensor = (mask2_tensor > 0.5).float()
    
    # Stack tensors
    tumor_tensors = torch.stack([tumor1_tensor, tumor2_tensor])
    mask_tensors = torch.stack([mask1_tensor, mask2_tensor])
    
    return tumor_tensors, mask_tensors

def augment_with_tumor(input_slices, tumor_tensors, mask_tensors, intensity_factor=1.0):
    """
    Augment brain MRI slices with their corresponding tumors
    """
    augmented_slices = input_slices.clone()
    segmentation_masks = mask_tensors.clone()  # Keep masks in same format

    
    # Apply each tumor to its corresponding slice
    for i in range(2):
        # Get the corresponding tumor and mask for this slice
        tumor = tumor_tensors[i]
        mask = mask_tensors[i]
        slice_img = augmented_slices[i]
        
        # Copy tumor values where mask is 1, keep original values where mask is 0
        augmented_slices[i] = torch.where(
            mask > 0,
            tumor,
            slice_img
        )
    
    return augmented_slices, segmentation_masks

def visualize_results(original_slices, augmented_slices, tumor_tensors, mask_tensors):
    """
    Visualize original and augmented slices along with both tumors and masks
    """
    # Move to CPU
    original_slices = original_slices.detach().cpu()
    augmented_slices = augmented_slices.detach().cpu()
    tumor_tensors = tumor_tensors.detach().cpu()
    mask_tensors = mask_tensors.detach().cpu()
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Get consistent min/max values for visualization
    vmin = original_slices.min()
    vmax = original_slices.max()
    
    # First row: Original slices and tumors
    axes[0,0].imshow(original_slices[0,0].numpy(), cmap='gray', vmin=vmin, vmax=vmax)
    axes[0,0].set_title(f'Original Slice 1\nRange: [{original_slices[0,0].min():.3f}, {original_slices[0,0].max():.3f}]')
    
    axes[0,1].imshow(original_slices[1,0].numpy(), cmap='gray', vmin=vmin, vmax=vmax)
    axes[0,1].set_title(f'Original Slice 2\nRange: [{original_slices[1,0].min():.3f}, {original_slices[1,0].max():.3f}]')
    
    axes[0,2].imshow(tumor_tensors[0,0].numpy(), cmap='gray')
    axes[0,2].set_title(f'Tumor 1\nRange: [{tumor_tensors[0,0].min():.3f}, {tumor_tensors[0,0].max():.3f}]')
    
    axes[0,3].imshow(tumor_tensors[1,0].numpy(), cmap='gray')
    axes[0,3].set_title(f'Tumor 2\nRange: [{tumor_tensors[1,0].min():.3f}, {tumor_tensors[1,0].max():.3f}]')
    
    # Second row: Augmented slices and masks
    axes[1,0].imshow(augmented_slices[0,0].numpy(), cmap='gray', vmin=vmin, vmax=vmax)
    axes[1,0].set_title(f'Augmented Slice 1\nRange: [{augmented_slices[0,0].min():.3f}, {augmented_slices[0,0].max():.3f}]')
    
    axes[1,1].imshow(augmented_slices[1,0].numpy(), cmap='gray', vmin=vmin, vmax=vmax)
    axes[1,1].set_title(f'Augmented Slice 2\nRange: [{augmented_slices[1,0].min():.3f}, {augmented_slices[1,0].max():.3f}]')
    
    axes[1,2].imshow(mask_tensors[0,0].numpy(), cmap='gray')
    axes[1,2].set_title('Tumor Mask 1')
    
    axes[1,3].imshow(mask_tensors[1,0].numpy(), cmap='gray')
    axes[1,3].set_title('Tumor Mask 2')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    return fig

def analyze_image_data(input_slices, tumor_tensor, mask_tensor, name=""):
    """Analyze a tensor's statistics and distribution"""
    print(f"\n{name} Statistics:")
    print(f"Shape: {input_slices.shape}")
    print(f"Min/Max: {input_slices.min():.4f}/{input_slices.max():.4f}")
    print(f"Mean/Std: {input_slices.mean():.4f}/{input_slices.std():.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Original data
    for i in range(2):
        slice_data = input_slices[i,0].detach().cpu().numpy()
        axes[i,0].imshow(slice_data, cmap='gray')
        axes[i,0].set_title(f'Slice {i+1}\nMin: {slice_data.min():.4f}, Max: {slice_data.max():.4f}')
        
        # Histogram of non-zero values (ignore background)
        non_zero = slice_data[slice_data != 0]
        axes[i,1].hist(non_zero.ravel(), bins=50)
        axes[i,1].set_title(f'Slice {i+1} Histogram (non-zero values)')
    
    plt.tight_layout()
    return fig

def analyze_tumor_data(tumor_tensor, mask_tensor):
    """Analyze tumor data specifically"""
    tumor_np = tumor_tensor.squeeze().cpu().numpy()
    mask_np = mask_tensor.squeeze().cpu().numpy()
    
    # Get tumor region only
    tumor_region = tumor_np[mask_np > 0]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Show tumor with its histogram
    axes[0].imshow(tumor_np, cmap='gray')
    axes[0].set_title(f'Tumor\nMin: {tumor_region.min():.4f}, Max: {tumor_region.max():.4f}')
    
    # Histogram of tumor region only
    axes[1].hist(tumor_region.ravel(), bins=50)
    axes[1].set_title('Tumor Histogram (masked region only)')
    
    plt.tight_layout()
    return fig



def analyze_images(tumor_path, mask_path):
    """Analyze the raw image data in detail"""
    # Load images in different ways to compare
    # 1. Direct PIL load
    tumor_pil = Image.open(tumor_path).convert('L')
    mask_pil = Image.open(mask_path).convert('L')
    
    # 2. Convert to numpy arrays
    tumor_np = np.array(tumor_pil)
    mask_np = np.array(mask_pil)
    
    # 3. Use torchvision transform
    transform = transforms.ToTensor()
    tumor_torch = transform(tumor_pil)
    mask_torch = transform(mask_pil)
    
    # Create visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Raw PIL data
    axes[0, 0].imshow(tumor_pil, cmap='gray')
    axes[0, 0].set_title('Tumor (PIL)')
    axes[0, 1].hist(np.array(tumor_pil).ravel(), bins=50)
    axes[0, 1].set_title('Tumor Histogram (PIL)')
    axes[0, 2].imshow(mask_pil, cmap='gray')
    axes[0, 2].set_title('Mask (PIL)')
    
    # Numpy data
    axes[1, 0].imshow(tumor_np, cmap='gray')
    axes[1, 0].set_title(f'Tumor (Numpy)\nMin: {tumor_np.min()}, Max: {tumor_np.max()}')
    axes[1, 1].hist(tumor_np.ravel(), bins=50)
    axes[1, 1].set_title('Tumor Histogram (Numpy)')
    axes[1, 2].imshow(mask_np, cmap='gray')
    axes[1, 2].set_title(f'Mask (Numpy)\nMin: {mask_np.min()}, Max: {mask_np.max()}')
    
    # Torch data
    axes[2, 0].imshow(tumor_torch.squeeze().numpy(), cmap='gray')
    axes[2, 0].set_title(f'Tumor (Torch)\nMin: {tumor_torch.min():.4f}, Max: {tumor_torch.max():.4f}')
    axes[2, 1].hist(tumor_torch.numpy().ravel(), bins=50)
    axes[2, 1].set_title('Tumor Histogram (Torch)')
    axes[2, 2].imshow(mask_torch.squeeze().numpy(), cmap='gray')
    axes[2, 2].set_title(f'Mask (Torch)\nMin: {mask_torch.min():.4f}, Max: {mask_torch.max():.4f}')
    
    # Add masked tumor analysis
    mask_bool = mask_torch.squeeze().bool()
    tumor_masked = tumor_torch.squeeze()[mask_bool]
    
    print("\nTumor Statistics:")
    print(f"PIL Image Mode: {tumor_pil.mode}")
    print(f"PIL Image Size: {tumor_pil.size}")
    print(f"\nNumpy array stats:")
    print(f"Shape: {tumor_np.shape}")
    print(f"Min/Max: {tumor_np.min()}/{tumor_np.max()}")
    print(f"Mean/Std: {tumor_np.mean():.4f}/{tumor_np.std():.4f}")
    
    print(f"\nTorch tensor stats:")
    print(f"Shape: {tumor_torch.shape}")
    print(f"Min/Max: {tumor_torch.min():.4f}/{tumor_torch.max():.4f}")
    print(f"Mean/Std: {tumor_torch.mean():.4f}/{tumor_torch.std():.4f}")
    
    print(f"\nMasked tumor region stats:")
    print(f"Min/Max: {tumor_masked.min():.4f}/{tumor_masked.max():.4f}")
    print(f"Mean/Std: {tumor_masked.mean():.4f}/{tumor_masked.std():.4f}")
    
    plt.tight_layout()
    return fig
