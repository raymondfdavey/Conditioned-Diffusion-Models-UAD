from src.utils.ray_utils import setup_storage, save_images, save_images_from_repeat, apply_brainmask_volume, apply_3d_median_filter, extract_key_points, load_and_preprocess_tumor,  augment_with_tumor
from src.models.modules.cond_DDPM import GaussianDiffusion
from src.models.modules.OpenAI_Unet import UNetModel as OpenAI_UNet
from src.models.modules.DDPM_encoder import get_encoder
import torch
from pytorch_lightning.core.lightning import LightningModule
from typing import Any
import torchio as tio
from src.utils.generate_noise import gen_noise
from omegaconf import open_dict
from collections import OrderedDict
from torch.nn import functional as F
from hydra.core.hydra_config import HydraConfig
import os
import shutil

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
        SCAN_ID = batch['ID']
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
        assert input.shape[0] == 1, "Batch size must be 1"
        input = input.squeeze(0).permute(3,0,1,2) # [B,C,H,W,D] -> [D,C,H,W]
        
        #! TEST
        save_images_to_disk=True
        
        if 'consistency' in self.test:
            n_runs = 100
            n_images_to_save = 5
            if 'augment' in self.test:
                test_type='augmented_repeated'
                print('in consistency + augment')
                
                tumor1_path = "/home/rd81/projects/data-complete/extracted_tumour_images_and_segments/image_00002/tumor_only_57.png"
                mask1_path = "/home/rd81/projects/data-complete/extracted_tumour_images_and_segments/image_00002/tumor_mask_57.png"
                tumor2_path = "/home/rd81/projects/data-complete/extracted_tumour_images_and_segments/image_00002/tumor_only_85.png"
                mask2_path = "/home/rd81/projects/data-complete/extracted_tumour_images_and_segments/image_00002/tumor_mask_85.png"

                # Load tumor and mask
                tumor_tensors, mask_tensors = load_and_preprocess_tumor(tumor1_path, mask1_path, tumor2_path, mask2_path, scale_factor=1.5, device=self.device, erosion_pixels=2)
                augmented_input, segmentation_masks = augment_with_tumor(input, tumor_tensors, mask_tensors, n_slices)
                # fig = visualize_results(input, augmented_input, tumor_tensors, mask_tensors)
                data = self.run_repeats(augmented_input, data_orig, segmentation_masks, data_mask, SCAN_ID, n_runs=n_runs, n_slices=n_slices, save_images_to_disk=save_images_to_disk, n_images_to_save = n_images_to_save)
            else:
                test_type='healthy_repeated'
                print('in consistency + healthy')

                data = self.run_repeats(input, data_orig, None, data_mask, SCAN_ID, n_runs=n_runs, n_slices=n_slices, save_images_to_disk=save_images_to_disk, n_images_to_save = n_images_to_save)
            self.save_repeat_run_data(data, SCAN_ID, test_type, n_runs, n_slices)
        else:            
            data = self.run_normal(input, data_orig, data_seg, data_mask, SCAN_ID, save_images_to_disk=save_images_to_disk)
            
        self.log('test_metric', 100)  # This will now show up in trainer.test() results
    
    def save_repeat_run_data(self, storage, SCAN_ID, test_type, n_runs, n_slices):
        print('='*10)
        print(f'Saving run data')
        save_path = HydraConfig.get().run.dir
        # Ensure the run_data directory exists
        full_path = f"{save_path}/{SCAN_ID[0].removesuffix('.nii.gz')}_{test_type}_{n_runs}_runs_{n_slices}_slices_{self.test_timesteps-1}_timesteps.pt" 
        torch.save(storage, full_path)
        self.for_deletion_dir = save_path
        print(f"Run data saved to {full_path}")
        print('='*10)
        
    def run_normal(self, input, data_orig, data_seg, data_mask, SCAN_ID, save_images_to_disk=True):
        print('='*10)
        print(f'RUNNING NORMAL TEST: SINGLE RUN')
        print('='*10)
        features = self(input)
        noise = gen_noise(self.cfg, input.shape).to(self.device)
        
        #! MODEL CALL
        loss_diff, reco, unet_details, noisy_image = self.diffusion(input,cond=features,t=self.test_timesteps-1,noise=noise)
        
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
            save_images(self,diff_volume, data_orig, data_seg, data_mask, final_volume, SCAN_ID)
        
    def run_repeats(self, input, data_orig, segmentation_mask, data_mask, SCAN_ID, n_runs=500, n_slices=4, save_images_to_disk=True,  n_images_to_save = 5):
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
            # storage['noise'][run_idx] = noise.squeeze(1)

            loss_diff, reco, unet_details, noisy_image = self.diffusion(input, cond=features, t=self.test_timesteps-1, noise=noise)
                        
            final_volume = reco.clone().squeeze(1)  # Just remove the channel dimension, keeps slices first!
            
            storage['noisy_images'][run_idx] = noisy_image.squeeze(1)
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
            
            if save_images_to_disk and run_idx < n_images_to_save:
                # Save images will handle moving to CPU and post-processing
                
                save_images_from_repeat(
                    self, 
                    input,
                    run_idx,
                    noise,
                    noisy_image,
                    diff_volume,
                    segmentation_mask,
                    data_mask,
                    final_volume.permute(1, 2, 0),
                    SCAN_ID
                )
            
            # Optional: Clear GPU cache after each run if memory is a concern
            torch.cuda.empty_cache()
        return storage
    def delete_hydra_and_logs(self):
        # Get the directory to operate on
        target_dir = self.for_deletion_dir
        
        # Delete the `.hydra` directory if it exists
        hydra_dir = os.path.join(target_dir, ".hydra")
        if os.path.exists(hydra_dir) and os.path.isdir(hydra_dir):
            shutil.rmtree(hydra_dir)
            print(f"Deleted directory: {hydra_dir}")
        
        # Delete all `.log` files in the directory
        for filename in os.listdir(target_dir):
            if filename.endswith(".log"):
                file_path = os.path.join(target_dir, filename)
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
                
    def on_test_end(self):
        print('='*10)
        self.delete_hydra_and_logs()
        print('FINISHED TESTING')
        print('='*10)
         