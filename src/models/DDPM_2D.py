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

import matplotlib.colors as colors


class DDPM_2D(LightningModule):
    def __init__(self,cfg,prefix=None):
        super().__init__()
        print('='*10)
        print('INITIALISING DDPM_2D')
        print('='*10)
        self.prefix = prefix
        self.cfg = cfg
        
        with open_dict(self.cfg):
            self.cfg['cond_dim'] = cfg.get('unet_dim',64) * 4
            self.encoder, out_features = get_encoder(cfg)

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
        
        if cfg.get('pretrained_encoder',False): # load pretrained encoder from cfg.modelpath
            assert cfg.get('encoder_path',None) is not None
            state_dict_pretrained = torch.load(cfg.get('encoder_path',None))['state_dict']
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

    def run_1000_repeats(self, input, data_orig, data_seg, data_mask, final_volume, n_runs=1000, n_slices=4):
        #! MORNING RAY!!!
        #! YOU JUST NEED TO CHECK THE DIMENSIONALITY OF YOUR STORAGE THING AND THEN THE DIFFERENT DIMENSIONS OF THE DIFFERENT BITS.
        assert input.shape == (n_slices, 1, 96, 96), f"Expected input shape (4, 1, 96, 96), got {input.shape}"

        storage = setup_storage(n_runs, n_slices)
        for run_idx in range(n_runs):
            storage['input'][run_idx] = input

            features = self(input)

            noise = gen_noise(self.cfg, input.shape).to(self.device)
            _, reco, unet_details = self.diffusion(input, cond=features, 
                                                        t=self.test_timesteps-1, 
                                                        noise=noise)

                    
                    
            final_volume = reco.clone().squeeze()
            final_volume = final_volume.permute(1,2,0) # to HxWxD
            final_volume = final_volume.unsqueeze(0)
            final_volume = final_volume.unsqueeze(0)

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

            
            data_mask[data_mask > 0] = 1
            diff_volume = apply_brainmask_volume(diff_volume.cpu(), data_mask.squeeze().cpu())   
            diff_volume = torch.from_numpy(apply_3d_median_filter(diff_volume.numpy().squeeze(),kernelsize=self.cfg.get('kernelsize_median',5))).unsqueeze(0) # bring 
            
            
            
            key_points = extract_key_points(unet_details)
            for key, tensor in key_points.items():
                if key in storage['features']:
                    storage['features'][key][run_idx] = tensor
                elif key in storage['embeddings']:
                    storage['embeddings'][key][run_idx] = tensor


        return storage

    
        
    def test_step(self, batch: Any, batch_idx: int):        
        self.dataset = batch['Dataset']
        input = batch['vol'][tio.DATA]
        data_orig = batch['vol_orig'][tio.DATA]
        data_seg = batch['seg_orig'][tio.DATA] if batch['seg_available'] else torch.zeros_like(data_orig)
        data_mask = batch['mask_orig'][tio.DATA]
        ID = batch['ID']
        self.stage = batch['stage']
        latentSpace = []
        self.cfg['noise_ensemble'] = False
                        
                        
        total_slices = input.size(4)
        percentages = [0.2, 0.4, 0.6, 0.8]  # Can adjust these values
        slice_indices = [int(total_slices * p) for p in percentages]
        input = input[..., slice_indices]
        data_orig = data_orig[..., slice_indices]
        data_seg = data_seg[..., slice_indices]
        data_mask = data_mask[..., slice_indices]
        final_volume = torch.zeros([input.size(2), input.size(3), input.size(4)], device = self.device)
        assert input.shape[0] == 1, "Batch size must be 1"
        input = input.squeeze(0).permute(3,0,1,2) # [B,C,H,W,D] -> [D,C,H,W]
    
        if self.test == 'consistency':
            self.run_500_repeats(input, data_seg, data_mask, final_volume)
        else:
            print(input.shape)
            features = self(input)
            print(features.shape)
            features_single = features
            latentSpace.append(features_single.mean(0).squeeze().detach().cpu())
            noise = gen_noise(self.cfg, input.shape).to(self.device)
            
            #! MODEL CALL
            loss_diff, reco, unet_details = self.diffusion(input,cond=features,t=self.test_timesteps-1,noise=noise)
            
            # reassamble the reconstruction volume
            final_volume = reco.clone().squeeze()
            final_volume = final_volume.permute(1,2,0) # to HxWxD
            final_volume = final_volume.unsqueeze(0)
            final_volume = final_volume.unsqueeze(0)
            # print('unsqueeze final volume shape (DDPM2d)', final_volume.shape)

            # print(unet_details['raw_embedding'])
            key_points = extract_key_points(unet_details)
            print_key_details(key_points, 1)
            
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
            print(final_volume.shape)
            print()
            data_mask[data_mask > 0] = 1
            diff_volume = apply_brainmask_volume(diff_volume.cpu(), data_mask.squeeze().cpu())   
            diff_volume = torch.from_numpy(apply_3d_median_filter(diff_volume.numpy().squeeze(),kernelsize=self.cfg.get('kernelsize_median',5))).unsqueeze(0) # bring back to tensor
            save_images(self,diff_volume, data_orig, data_seg, data_mask, final_volume, ID)
            self.log('test_metric', 100)  # This will now show up in trainer.test() results

    def on_test_end(self):
        print('='*10)
        print('FINISHED TESTING')
        print('='*10)

def setup_storage(n_runs, n_slices):
    """
    Set up comprehensive storage for diffusion runs including original images.
    The original images will be identical across runs, but storing them helps validate
    the experimental setup and provides a convenient reference point.
    """
    storage = {
        # Input/Output Image Storage
        'input': torch.zeros(n_runs, 4, 1, 96, 96),          # Original input slices
        'reconstructions': torch.zeros(n_runs, n_slices, 96, 96), # Reconstructed images
        'differences': torch.zeros(n_runs, n_slices, 96, 96),     # Difference maps
        'final_volumes': torch.zeros(n_runs, n_slices, 96, 96),   # Final processed volumes
        
        # Segmentation and Mask Storage (for completeness)
        'segmentations': torch.zeros(n_runs, n_slices, 96, 96),   # Segmentation masks
        'brain_masks': torch.zeros(n_runs, n_slices, 96, 96),     # Brain masks
        
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
            'raw_embedding': torch.zeros(n_runs, n_slices, 1024),
            'down_post_1_embedding': torch.zeros(n_runs, n_slices, 256),
            'down_post_4_embedding': torch.zeros(n_runs, n_slices, 256),
            'down_post_8_embedding': torch.zeros(n_runs, n_slices, 512),
            'middle_post_embedding': torch.zeros(n_runs, n_slices, 512),
            'up_post_3_embedding': torch.zeros(n_runs, n_slices, 512),
            'up_post_7_embedding': torch.zeros(n_runs, n_slices, 512)
        }
    }
    return storage


def save_images(self, diff_volume, data_orig, data_seg, data_mask, final_volume, ID, diff_volume_KL=None,  flow=None ):
    print('='*10)
    print('SAVING IMAGES')
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
        
        if self.cfg.get('save_to_disc',True):
            plt.savefig(os.path.join(ImagePathList['imagesGrid'], '{}_{}_Grid.png'.format(ID[0],j)),bbox_inches='tight')
        # self.logger.experiment[0].log({'images/{}/{}_Grid.png'.format(self.dataset[0],j) : wandb.Image(plt)})
        plt.clf()
        plt.cla()
        plt.close()


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