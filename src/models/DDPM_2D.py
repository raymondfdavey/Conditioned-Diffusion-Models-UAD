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
from src.models.LDM.modules.diffusionmodules.util import timestep_embedding
from torch import nn
import torch
from skimage.measure import regionprops, label
from torchvision.transforms import ToTensor, ToPILImage
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.metrics import  confusion_matrix, roc_curve, accuracy_score, precision_recall_fscore_support, auc,precision_recall_curve, average_precision_score
import wandb 
import monai
from torch.nn import functional as F
from PIL import Image

import matplotlib.colors as colors


class DDPM_2D(LightningModule):
    def __init__(self,cfg,prefix=None):
        #! IN DDPM
        print('='*10)
        print('INITIALISING DDPM_2D')
        print('='*10)
        super().__init__()
        
        self.cfg = cfg
        
        #! INITIALISES conditioning net
        if cfg.get('condition',True):
            with open_dict(self.cfg):
                self.cfg['cond_dim'] = cfg.get('unet_dim',64) * 4

            self.encoder, out_features = get_encoder(cfg)
            print('encoder out features:', out_features)
        else: 
            out_features = None
        #! INITIALISED UNET
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

        #! INITIALISES GUASSIAN DIFFUSION WHICH USES UNET
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
            print('Loading pretrained encoder from: ', cfg.encoder_path)
            assert cfg.get('encoder_path',None) is not None
            print('encoder path, ', cfg.get('encoder_path',None))
            
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
        
        self.prefix = prefix
        
        self.save_hyperparameters()

    def forward(self, x): # encode input
        if self.cfg.get('condition',True):
            x = self.encoder(x)
            #! this is c - the 128 dim encoding of the image that is the basis for c` and c` proj
            print("i'm i DDPM2d forward block (whch is basically the encoder)")
            print("Context vector c shape:", x.shape)  # Should be [batch_size, 128]
            # print("Context vector c:", x)
        else: 
            x = None
        return x


    def training_step(self, batch, batch_idx: int):
        print('='*10)
        print('IN DDPM TRAINING STEP')
        print('='*10)
        # process batch
        input = batch['vol'][tio.DATA].squeeze(-1)


        # calc features for guidance     
        features = self(input)

        # generate noise
        if self.cfg.get('noisetype') is not None:
            noise = gen_noise(self.cfg, input.shape).to(self.device)
        else: 
            noise = None
        #! ACTUAL CALLLLLLLLLLLL
        # reconstruct
        loss, reco, unet_details = self.diffusion(input, cond = features, noise = noise)

        self.log(f'{self.prefix}train/Loss', loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=input.shape[0],sync_dist=True)
        return {"loss": loss}
    
    def validation_step(self, batch: Any, batch_idx: int):
        print('='*10)
        print('IN DDPM VALIDATION STEP')
        print('='*10)
        input = batch['vol'][tio.DATA].squeeze(-1) 

        # calc features for guidance
        features = self(input)
        # generate noise
        # generate noise
        if self.cfg.get('noisetype') is not None:
            noise = gen_noise(self.cfg, input.shape).to(self.device)
        else: 
            noise = None
        # reconstruct
        loss, reco, unet_details = self.diffusion(input,cond=features,noise=noise)

        self.log(f'{self.prefix}val/Loss_comb', loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=input.shape[0],sync_dist=True)
        return {"loss": loss}

    def on_test_start(self):
        print('='*10)
        print('IN DDPM ON TEST START DDPM2D')
        print('='*10)
        
        self.eval_dict = get_eval_dictionary()
        self.inds = []
        self.latentSpace_slice = []
        self.new_size = [160,190,160]
        self.diffs_list = []
        self.seg_list = []
        if not hasattr(self,'threshold'):
            self.threshold = {}

    def test_step(self, batch: Any, batch_idx: int):        
        self.dataset = batch['Dataset']
        input = batch['vol'][tio.DATA]
        data_orig = batch['vol_orig'][tio.DATA]
        data_seg = batch['seg_orig'][tio.DATA] if batch['seg_available'] else torch.zeros_like(data_orig)
        data_mask = batch['mask_orig'][tio.DATA]
        ID = batch['ID']
        age = batch['age']
        self.stage = batch['stage']
        label = batch['label']
        AnomalyScoreComb = []
        AnomalyScoreReg = []
        AnomalyScoreReco = []
        latentSpace = []
                        
        num_slices = 2

        if num_slices != input.size(4):
            total_slices = input.size(4)
            # percentages = [0.2, 0.4, 0.6, 0.8]  # Can adjust these values
            percentages = [0.4, 0.6]  # Can adjust these values
            # percentages = [0.5]  # Can adjust these values
            slice_indices = [int(total_slices * p) for p in percentages]
            
            input = input[..., slice_indices]
            data_orig = data_orig[..., slice_indices]
            data_seg = data_seg[..., slice_indices]
            data_mask = data_mask[..., slice_indices]
            ind_offset = slice_indices[0]
        else:
            ind_offset = 0
        

        final_volume = torch.zeros([input.size(2), input.size(3), input.size(4)], device = self.device)


        assert input.shape[0] == 1, "Batch size must be 1"
        patching = True
        if patching:
            input = input.squeeze(0).permute(3,0,1,2)  # [B,C,H,W,D] -> [D,C,H,W]
            
            # Add artificial anomaly
            patched_input, patch_mask = add_square_patch(input, 
                                                    patch_size=20,  # adjust size as needed
                                                    intensity=0.5, black=False)  # adjust intensity as needed
            
            # Generate features from patched input
            features = self(patched_input)
            features_single = features
            if self.cfg.condition:
                latentSpace.append(features_single.mean(0).squeeze().detach().cpu())
            self.cfg['noise_ensemble'] = False

            # Rest of the processing remains the same but uses patched_input
            if self.cfg.get('noise_ensemble', False):
                print('ensemble true')
                timesteps = self.cfg.get('step_ensemble',[250,500,750]) # timesteps to evaluate
                reco_ensemble = torch.zeros_like(patched_input)
                for t in timesteps:
                    # generate noise
                    if self.cfg.get('noisetype') is not None:
                        print('noisetupe true')
                        
                        noise = gen_noise(self.cfg, patched_input.shape).to(self.device)
                    else: 
                        noise = None
                    #! actual model call <- THIS IS WHAT IS BEING RETUNED
                    loss_diff, reco, unet_details = self.diffusion(patched_input,cond=features,t=t-1,noise=noise)
                    reco_ensemble += reco
                    
                reco = reco_ensemble / len(timesteps) # average over timesteps
            else:
                if self.cfg.get('noisetype') is not None:
                    noise = gen_noise(self.cfg, patched_input.shape).to(self.device)
                else:
                    noise = None
                loss_diff, reco, unet_details = self.diffusion(patched_input, cond=features,
                                                            t=self.test_timesteps-1, noise=noise)
        else:
            
            input = input.squeeze(0).permute(3,0,1,2) # [B,C,H,W,D] -> [D,C,H,W]
            features = self(input)
            features_single = features
            if self.cfg.condition:
                latentSpace.append(features_single.mean(0).squeeze().detach().cpu())

            if self.cfg.get('noise_ensemble',False): # evaluate with different noise levels
                print('ensemble true')
                timesteps = self.cfg.get('step_ensemble',[250,500,750]) # timesteps to evaluate
                reco_ensemble = torch.zeros_like(input)
                for t in timesteps:
                    # generate noise
                    if self.cfg.get('noisetype') is not None:
                        print('noisetupe true')
                        
                        noise = gen_noise(self.cfg, input.shape).to(self.device)
                    else: 
                        noise = None
                    #! actual model call <- THIS IS WHAT IS BEING RETUNED
                    loss_diff, reco, unet_details = self.diffusion(input,cond=features,t=t-1,noise=noise)
                    reco_ensemble += reco
                    
                reco = reco_ensemble / len(timesteps) # average over timesteps
            else :
                if self.cfg.get('noisetype') is not None:
                    noise = gen_noise(self.cfg, input.shape).to(self.device)
                else: 
                    noise = None
                loss_diff, reco, unet_details = self.diffusion(input,cond=features,t=self.test_timesteps-1,noise=noise)
        
        # calculate loss and Anomalyscores
        AnomalyScoreComb.append(loss_diff.cpu())
        AnomalyScoreReg.append(loss_diff.cpu())
        AnomalyScoreReco.append(loss_diff.cpu())

        # reassamble the reconstruction volume
        final_volume = reco.clone().squeeze()
        final_volume = final_volume.permute(1,2,0) # to HxWxD
        print('FINAL VOLUME (DDPM) shape', final_volume.shape)
        # print('FINAL VOLUME', final_volume)

    

        # average across slices to get volume-based scores
        self.latentSpace_slice.extend(latentSpace)
        self.eval_dict['latentSpace'].append(torch.mean(torch.stack(latentSpace),0))
        # AnomalyScoreComb_vol = np.mean(AnomalyScoreComb) 
        # AnomalyScoreReg_vol = np.mean(AnomalyScoreReg)
        # AnomalyScoreReco_vol = np.mean(AnomalyScoreReco)

        # self.eval_dict['AnomalyScoreRegPerVol'].append(AnomalyScoreReg_vol)


        # if not self.cfg.get('use_postprocessed_score', True):
            # self.eval_dict['AnomalyScoreRecoPerVol'].append(AnomalyScoreReco_vol)
            # self.eval_dict['AnomalyScoreCombPerVol'].append(AnomalyScoreComb_vol)
            # self.eval_dict['AnomalyScoreCombiPerVol'].append(AnomalyScoreReco_vol * AnomalyScoreReg_vol)
            # self.eval_dict['AnomalyScoreCombPriorPerVol'].append(AnomalyScoreReco_vol + self.cfg.beta * 0)
            # self.eval_dict['AnomalyScoreCombiPriorPerVol'].append(AnomalyScoreReco_vol * 0)

        final_volume = final_volume.unsqueeze(0)
        final_volume = final_volume.unsqueeze(0)
        print('unsqueeze final volume shape (DDPM2d)', final_volume.shape)
        # print('unsqueeze final volume (DDPM 2D)', final_volume)
        # calculate metrics
        print('='*100)
        indent=1
        # for key, value in unet_details.items():
        #     print("  " * indent + f"📍 {key}")
        #     if isinstance(value, dict):
        #         # Skip printing the tensor value, but print its shape if available
        #         for subkey, subvalue in value.items():
        #             if subkey != 'tensor':
        #                 print("  " * (indent + 1) + f"└─ {subkey}: {subvalue}")
        #     print()  # Add blank line between sections
        key_points = extract_key_points(unet_details)
        # Function to print the extracted points with the same formatting
        for key, value in key_points.items():
            print("  " * indent + f"📍 {key}")
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if subkey != 'tensor':
                        print("  " * (indent + 1) + f"└─ {subkey}: {subvalue}")
            print()  # Add blank line between sections
        # print(unet_details.keys())
        print('='*100)

        self.healthy_sets = ['IXI']
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

       # Calculate Reconstruction errors with respect to anomal/normal regions
        l1err = nn.functional.l1_loss(final_volume.squeeze(),data_orig.squeeze())
        l2err = nn.functional.mse_loss(final_volume.squeeze(),data_orig.squeeze())
        # l1err_anomal = nn.functional.l1_loss(final_volume.squeeze()[data_seg.squeeze() > 0],data_orig[data_seg > 0]) 
        # l1err_healthy = nn.functional.l1_loss(final_volume.squeeze()[data_seg.squeeze() == 0],data_orig[data_seg == 0]) 
        # l2err_anomal = nn.functional.mse_loss(final_volume.squeeze()[data_seg.squeeze() > 0],data_orig[data_seg > 0]) 
        # l2err_healthy = nn.functional.mse_loss(final_volume.squeeze()[data_seg.squeeze() == 0],data_orig[data_seg == 0])

        # store in eval dict
        self.eval_dict['l1recoErrorAll'].append(l1err.item())
        # self.eval_dict['l1recoErrorUnhealthy'].append(l1err_anomal.item())
        # self.eval_dict['l1recoErrorHealthy'].append(l1err_healthy.item())
        self.eval_dict['l2recoErrorAll'].append(l2err.item())
        # self.eval_dict['l2recoErrorUnhealthy'].append(l2err_anomal.item())
        # self.eval_dict['l2recoErrorHealthy'].append(l2err_healthy.item())

        # move data to CPU
        # data_seg = data_seg.cpu() 
        data_mask = data_mask.cpu()
        diff_volume = diff_volume.cpu()
        data_orig = data_orig.cpu()
        final_volume = final_volume.cpu()
        # binarize the segmentation
        # data_seg[data_seg > 0] = 1
        data_mask[data_mask > 0] = 1

        # Erode the Brainmask 
        if self.cfg['erodeBrainmask']:
            print("ERODING BRAIN MASK")
            diff_volume = apply_brainmask_volume(diff_volume.cpu(), data_mask.squeeze().cpu())   

        # Filter the DifferenceImage
        if self.cfg['medianFiltering']:
            print("MEDIAN FILTERING")
            diff_volume = torch.from_numpy(apply_3d_median_filter(diff_volume.numpy().squeeze(),kernelsize=self.cfg.get('kernelsize_median',5))).unsqueeze(0) # bring back to tensor

        # save image grid
        if self.cfg['saveOutputImages'] :
            print("SAVING OUTPUT - ORIGINAL, FINAL, MASK ETC")
            if patching:
                log_images_with_patch(self, diff_volume, data_orig, data_seg, data_mask, 
                         final_volume, patched_input, patch_mask, ID)
            else:
                log_images(self,diff_volume, data_orig, data_seg, data_mask, final_volume, ID)
            
        ### Compute Metrics per Volume / Step ###
        # if self.cfg.evalSeg and self.dataset[0] not in self.healthy_sets: # only compute metrics if segmentation is available

            # Pixel-Wise Segmentation Error Metrics based on Differenceimage
            # AUC, _fpr, _tpr, _threshs = compute_roc(diff_volume.squeeze().flatten(), np.array(data_seg.squeeze().flatten()).astype(bool))
            # AUPRC, _precisions, _recalls, _threshs = compute_prc(diff_volume.squeeze().flatten(),np.array(data_seg.squeeze().flatten()).astype(bool))

            # gready search for threshold
            # bestDice, bestThresh = find_best_val(np.array(diff_volume.squeeze()).flatten(),  # threshold search with a subset of EvaluationSet
                                                # np.array(data_seg.squeeze()).flatten().astype(bool), 
                                                # val_range=(0, np.max(np.array(diff_volume))),
                                                # max_steps=10, 
                                                # step=0, 
                                                # max_val=0, 
                                                # max_point=0)

            # if 'test' in self.stage:
            #     bestThresh = self.threshold['total']

            # if self.cfg["threshold"] == 'auto':
            #     diffs_thresholded = diff_volume > bestThresh
            # else: # never used
            #     diffs_thresholded = diff_volume > self.cfg["threshold"]    
            
            # # Connected Components
            # if not 'node' in self.dataset[0].lower(): # no 3D data
            #     diffs_thresholded = filter_3d_connected_components(np.squeeze(diffs_thresholded)) # this is only done for patient-wise evaluation atm
            
            # # Calculate Dice Score with thresholded volumes
            # diceScore = dice(np.array(diffs_thresholded.squeeze()),np.array(data_seg.squeeze().flatten()).astype(bool))
            
            # # Other Metrics
            # TP, FP, TN, FN = confusion_matrix(np.array(diffs_thresholded.squeeze().flatten()), np.array(data_seg.squeeze().flatten()).astype(bool),labels=[0, 1]).ravel()
            # TPR = tpr(np.array(diffs_thresholded.squeeze()), np.array(data_seg.squeeze().flatten()).astype(bool))
            # FPR = fpr(np.array(diffs_thresholded.squeeze()), np.array(data_seg.squeeze().flatten()).astype(bool))
            # self.eval_dict['lesionSizePerVol'].append(np.count_nonzero(np.array(data_seg.squeeze().flatten()).astype(bool)))
            # self.eval_dict['DiceScorePerVol'].append(diceScore)
            # self.eval_dict['BestDicePerVol'].append(bestDice)
            # self.eval_dict['BestThresholdPerVol'].append(bestThresh)
            # self.eval_dict['AUCPerVol'].append(AUC)
            # self.eval_dict['AUPRCPerVol'].append(AUPRC)
            # self.eval_dict['TPPerVol'].append(TP)
            # self.eval_dict['FPPerVol'].append(FP)
            # self.eval_dict['TNPerVol'].append(TN)
            # self.eval_dict['FNPerVol'].append(FN) 
            # self.eval_dict['TPRPerVol'].append(TPR)
            # self.eval_dict['FPRPerVol'].append(FPR)
            # self.eval_dict['IDs'].append(ID[0])

            # PrecRecF1PerVol = precision_recall_fscore_support(np.array(data_seg.squeeze().flatten()).astype(bool),np.array(diffs_thresholded.squeeze()).flatten(),labels=[0,1])
            # self.eval_dict['AccuracyPerVol'].append(accuracy_score(np.array(data_seg.squeeze().flatten()).astype(bool),np.array(diffs_thresholded.squeeze()).flatten()))
            # self.eval_dict['PrecisionPerVol'].append(PrecRecF1PerVol[0][1])
            # self.eval_dict['RecallPerVol'].append(PrecRecF1PerVol[1][1])
            # self.eval_dict['SpecificityPerVol'].append(TN / (TN+FP+0.0000001))

            # # other metrics from monai:
            # if len(data_seg.shape) == 4:
            #     data_seg = data_seg.unsqueeze(0)
            # Haus = monai.metrics.compute_hausdorff_distance(diffs_thresholded.unsqueeze(0).unsqueeze(0),data_seg, include_background=False, distance_metric='euclidean', percentile=None, directed=False)
            # self.eval_dict['HausPerVol'].append(Haus.item())

            # # compute slice-wise metrics
            # for slice in range(data_seg.squeeze().shape[0]): 
            #         if np.array(data_seg.squeeze()[slice].flatten()).astype(bool).any():
            #             self.eval_dict['DiceScorePerSlice'].append(dice(np.array(diff_volume.squeeze()[slice] > bestThresh),np.array(data_seg.squeeze()[slice].flatten()).astype(bool)))
            #             PrecRecF1PerSlice = precision_recall_fscore_support(np.array(data_seg.squeeze()[slice].flatten()).astype(bool),np.array(diff_volume.squeeze()[slice] > bestThresh).flatten(),warn_for=tuple(),labels=[0,1])
            #             self.eval_dict['PrecisionPerSlice'].append(PrecRecF1PerSlice[0][1])
            #             self.eval_dict['RecallPerSlice'].append(PrecRecF1PerSlice[1][1])
            #             self.eval_dict['lesionSizePerSlice'].append(np.count_nonzero(np.array(data_seg.squeeze()[slice].flatten()).astype(bool)))

        # if 'val' in self.stage  :
        #     if batch_idx == 0:
        #         self.diffs_list = np.array(diff_volume.squeeze().flatten())
        #         self.seg_list = np.array(data_seg.squeeze().flatten()).astype(np.int8)
        #     else: 
        #         self.diffs_list = np.append(self.diffs_list,np.array(diff_volume.squeeze().flatten()),axis=0)
        #         self.seg_list = np.append(self.seg_list,np.array(data_seg.squeeze().flatten()),axis=0).astype(np.int8)

        # Reconstruction based Anomaly score for Slice-Wise evaluation  
        # if self.cfg.get('use_postprocessed_score', True):
        #     AnomalyScoreReco_vol = diff_volume.squeeze()[data_mask.squeeze()>0].mean() # for sample-wise detection 

        # AnomalyScoreReco = [] # Reconstruction based Anomaly score
        # if len(diff_volume.squeeze().shape) !=2:
        #     for slice in range(diff_volume.squeeze().shape[0]): 
        #         score = diff_volume.squeeze()[slice][data_mask.squeeze()[slice]>0].mean()
        #         if score.isnan() : # if no brain exists in that slice
        #             AnomalyScoreReco.append(0.0) 
        #         else: 
        #             AnomalyScoreReco.append(score) 

            # create slice-wise labels 
            # data_seg_downsampled = np.array(data_seg.squeeze())
            # label = [] # store labels here
            # for slice in range(data_seg_downsampled.shape[0]) :  #iterate through volume
                # if np.array(data_seg_downsampled[slice]).astype(bool).any(): # if there is an anomaly segmentation
                    # label.append(1) # label = 1
                # else :
                    # label.append(0) # label = 0 if there is no Anomaly in the slice
                    
        #     if self.dataset[0] not in self.healthy_sets:
        #         AUC, _fpr, _tpr, _threshs = compute_roc(np.array(AnomalyScoreReco),np.array(label))
        #         AUPRC, _precisions, _recalls, _threshs = compute_prc(np.array(AnomalyScoreReco),np.array(label))
        #         self.eval_dict['AUCAnomalyRecoPerSlice'].append(AUC)
        #         self.eval_dict['AUPRCAnomalyRecoPerSlice'].append(AUPRC)
        #         self.eval_dict['labelPerSlice'].extend(label)
        #         # store Slice-wise Anomalyscore (reconstruction based)
        #         self.eval_dict['AnomalyScoreRecoPerSlice'].extend(AnomalyScoreReco)

        # # sample-Wise Anomalyscores
        # if self.cfg.get('use_postprocessed_score', True):
        #     self.eval_dict['AnomalyScoreRecoPerVol'].append(AnomalyScoreReco_vol)
        #     self.eval_dict['AnomalyScoreCombPerVol'].append(AnomalyScoreReco_vol)
        #     self.eval_dict['AnomalyScoreCombiPerVol'].append(AnomalyScoreReco_vol )
        #     self.eval_dict['AnomalyScoreCombPriorPerVol'].append(AnomalyScoreReco_vol)
        #     self.eval_dict['AnomalyScoreCombiPriorPerVol'].append(AnomalyScoreReco_vol )
        

        # self.eval_dict['labelPerVol'].append(label_vol.item())
            #! NEW TESTS BABY
            
            # # Visualization: Compare input and output for a specific slice
            # slice_idx = 50  # Choose the slice you want to visualize (can be dynamic)
            # visualize_comparison(input, final_volume, slice_idx=slice_idx)

            # # Optional: Visualize the difference
            # visualize_difference(input, final_volume, slice_idx=slice_idx)

            # # Optional: Visualize a range of slices
            # visualize_multiple_slices(input, final_volume, slice_range=range(40, 60))
    
    def on_test_end(self) :
        # calculate metrics
        _test_end(self) # everything that is independent of the model choice 


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.cfg.lr)
    
    def update_prefix(self, prefix):
        self.prefix = prefix 
        
#! SAVES IMAGES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def log_images(self, diff_volume, data_orig, data_seg, data_mask, final_volume, ID, diff_volume_KL=None,  flow=None ):
    print('SAVING IMAGES!')
    
    ImagePathList = {
                    'imagesGrid': os.path.join(os.getcwd(),'grid')}
    
    for key in ImagePathList :
        if not os.path.isdir(ImagePathList[key]):
            os.mkdir(ImagePathList[key])

    for j in range(0,diff_volume.squeeze().shape[2],1) : #! the 1 here is how many to save

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
        self.logger.experiment[0].log({'images/{}/{}_Grid.png'.format(self.dataset[0],j) : wandb.Image(plt)})
        plt.clf()
        plt.cla()
        plt.close()

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

def _test_end(self) :
        print('IN TEST END')
        pass
#     # average over all test samples
#         self.eval_dict['l1recoErrorAllMean'] = np.nanmean(self.eval_dict['l1recoErrorAll'])
#         self.eval_dict['l1recoErrorAllStd'] = np.nanstd(self.eval_dict['l1recoErrorAll'])
#         self.eval_dict['l2recoErrorAllMean'] = np.nanmean(self.eval_dict['l2recoErrorAll'])
#         self.eval_dict['l2recoErrorAllStd'] = np.nanstd(self.eval_dict['l2recoErrorAll'])

#         self.eval_dict['l1recoErrorHealthyMean'] = np.nanmean(self.eval_dict['l1recoErrorHealthy'])
#         self.eval_dict['l1recoErrorHealthyStd'] = np.nanstd(self.eval_dict['l1recoErrorHealthy'])
#         self.eval_dict['l1recoErrorUnhealthyMean'] = np.nanmean(self.eval_dict['l1recoErrorUnhealthy'])
#         self.eval_dict['l1recoErrorUnhealthyStd'] = np.nanstd(self.eval_dict['l1recoErrorUnhealthy'])

#         self.eval_dict['l2recoErrorHealthyMean'] = np.nanmean(self.eval_dict['l2recoErrorHealthy'])
#         self.eval_dict['l2recoErrorHealthyStd'] = np.nanstd(self.eval_dict['l2recoErrorHealthy'])
#         self.eval_dict['l2recoErrorUnhealthyMean'] = np.nanmean(self.eval_dict['l2recoErrorUnhealthy'])
#         self.eval_dict['l2recoErrorUnhealthyStd'] = np.nanstd(self.eval_dict['l2recoErrorUnhealthy'])

#         self.eval_dict['AUPRCPerVolMean'] = np.nanmean(self.eval_dict['AUPRCPerVol'])
#         self.eval_dict['AUPRCPerVolStd'] = np.nanstd(self.eval_dict['AUPRCPerVol'])
#         self.eval_dict['AUCPerVolMean'] = np.nanmean(self.eval_dict['AUCPerVol'])
#         self.eval_dict['AUCPerVolStd'] = np.nanstd(self.eval_dict['AUCPerVol'])

#         self.eval_dict['DicePerVolMean'] = np.nanmean(self.eval_dict['DiceScorePerVol'])
#         self.eval_dict['DicePerVolStd'] = np.nanstd(self.eval_dict['DiceScorePerVol'])
#         self.eval_dict['BestDicePerVolMean'] = np.mean(self.eval_dict['BestDicePerVol'])
#         self.eval_dict['BestDicePerVolStd'] = np.std(self.eval_dict['BestDicePerVol'])
#         self.eval_dict['BestThresholdPerVolMean'] = np.mean(self.eval_dict['BestThresholdPerVol'])
#         self.eval_dict['BestThresholdPerVolStd'] = np.std(self.eval_dict['BestThresholdPerVol'])


#         self.eval_dict['TPPerVolMean'] = np.nanmean(self.eval_dict['TPPerVol'])
#         self.eval_dict['TPPerVolStd'] = np.nanstd(self.eval_dict['TPPerVol'])
#         self.eval_dict['FPPerVolMean'] = np.nanmean(self.eval_dict['FPPerVol'])
#         self.eval_dict['FPPerVolStd'] = np.nanstd(self.eval_dict['FPPerVol'])
#         self.eval_dict['TNPerVolMean'] = np.nanmean(self.eval_dict['TNPerVol'])
#         self.eval_dict['TNPerVolStd'] = np.nanstd(self.eval_dict['TNPerVol'])
#         self.eval_dict['FNPerVolMean'] = np.nanmean(self.eval_dict['FNPerVol'])
#         self.eval_dict['FNPerVolStd'] = np.nanstd(self.eval_dict['FNPerVol'])
#         self.eval_dict['TPRPerVolMean'] = np.nanmean(self.eval_dict['TPRPerVol'])
#         self.eval_dict['TPRPerVolStd'] = np.nanstd(self.eval_dict['TPRPerVol'])
#         self.eval_dict['FPRPerVolMean'] = np.nanmean(self.eval_dict['FPRPerVol'])
#         self.eval_dict['FPRPerVolStd'] = np.nanstd(self.eval_dict['FPRPerVol'])
#         self.eval_dict['HausPerVolMean'] = np.nanmean(np.array(self.eval_dict['HausPerVol'])[np.isfinite(self.eval_dict['HausPerVol'])])
#         self.eval_dict['HausPerVolStd'] = np.nanstd(np.array(self.eval_dict['HausPerVol'])[np.isfinite(self.eval_dict['HausPerVol'])])
        


#         self.eval_dict['PrecisionPerVolMean'] = np.mean(self.eval_dict['PrecisionPerVol'])
#         self.eval_dict['PrecisionPerVolStd'] =np.std(self.eval_dict['PrecisionPerVol'])
#         self.eval_dict['RecallPerVolMean'] = np.mean(self.eval_dict['RecallPerVol'])
#         self.eval_dict['RecallPerVolStd'] = np.std(self.eval_dict['RecallPerVol'])
#         self.eval_dict['PrecisionPerSliceMean'] = np.mean(self.eval_dict['PrecisionPerSlice'])
#         self.eval_dict['PrecisionPerSliceStd'] = np.std(self.eval_dict['PrecisionPerSlice'])
#         self.eval_dict['RecallPerSliceMean'] = np.mean(self.eval_dict['RecallPerSlice'])
#         self.eval_dict['RecallPerSliceStd'] = np.std(self.eval_dict['RecallPerSlice'])
#         self.eval_dict['AccuracyPerVolMean'] = np.mean(self.eval_dict['AccuracyPerVol'])
#         self.eval_dict['AccuracyPerVolStd'] = np.std(self.eval_dict['AccuracyPerVol'])
#         self.eval_dict['SpecificityPerVolMean'] = np.mean(self.eval_dict['SpecificityPerVol'])
#         self.eval_dict['SpecificityPerVolStd'] = np.std(self.eval_dict['SpecificityPerVol'])


#         if 'test' in self.stage :
#             del self.threshold
                
#         if 'val' in self.stage: 
#             if self.dataset[0] not in self.healthy_sets:
#                 bestdiceScore, bestThresh = find_best_val((self.diffs_list).flatten(), (self.seg_list).flatten().astype(bool), 
#                                         val_range=(0, np.max((self.diffs_list))), 
#                                         max_steps=10, 
#                                         step=0, 
#                                         max_val=0, 
#                                         max_point=0)

#                 self.threshold['total'] = bestThresh 
#                 if self.cfg.get('KLDBackprop',False): 
#                     bestdiceScoreKLComb, bestThreshKLComb = find_best_val((self.diffs_listKLComb).flatten(), (self.seg_list).flatten().astype(bool), 
#                         val_range=(0, np.max((self.diffs_listKLComb))), 
#                         max_steps=10, 
#                         step=0, 
#                         max_val=0, 
#                         max_point=0)

#                     self.threshold['totalKLComb'] = bestThreshKLComb 
#                     bestdiceScoreKL, bestThreshKL = find_best_val((self.diffs_listKL).flatten(), (self.seg_list).flatten().astype(bool), 
#                         val_range=(0, np.max((self.diffs_listKL))), 
#                         max_steps=10, 
#                         step=0, 
#                         max_val=0, 
#                         max_point=0)

#                     self.threshold['totalKL'] = bestThreshKL 
#             else: # define thresholds based on the healthy validation set
#                 _, fpr_healthy, _, threshs = compute_roc((self.diffs_list).flatten(), np.zeros_like(self.diffs_list).flatten().astype(int))
#                 self.threshholds_healthy= {
#                         'thresh_1p' : threshs[np.argmax(fpr_healthy>0.01)], # 1%
#                         'thresh_5p' : threshs[np.argmax(fpr_healthy>0.05)], # 5%
#                         'thresh_10p' : threshs[np.argmax(fpr_healthy>0.10)]} # 10%}
#                 self.eval_dict['t_1p'] = self.threshholds_healthy['thresh_1p']
#                 self.eval_dict['t_5p'] = self.threshholds_healthy['thresh_5p']
#                 self.eval_dict['t_10p'] = self.threshholds_healthy['thresh_10p']
                
def calc_thresh(dataset):
    data = dataset['Datamodules_train.Chexpert']
    _, fpr_healthy_comb, _, threshs_healthy_comb = compute_roc(np.array(data['AnomalyScoreCombPerVol']),np.array(data['labelPerVol'])) 
    _, fpr_healthy_combPrior, _, threshs_healthy_combPrior = compute_roc(np.array(data['AnomalyScoreCombPriorPerVol']),np.array(data['labelPerVol']))
    _, fpr_healthy_reg, _, threshs_healthy_reg = compute_roc(np.array(data['AnomalyScoreRegPerVol']),np.array(data['labelPerVol']))
    _, fpr_healthy_reco, _, threshs_healthy_reco = compute_roc(np.array(data['AnomalyScoreRecoPerVol']),np.array(data['labelPerVol']))
    _, fpr_healthy_prior_kld, _, threshs_healthy_prior_kld = compute_roc(np.array(data['KLD_to_learned_prior']),np.array(data['labelPerVol']))
    threshholds_healthy= {
                'thresh_1p_comb' : threshs_healthy_comb[np.argmax(fpr_healthy_comb>0.01)], 
                'thresh_1p_combPrior' : threshs_healthy_combPrior[np.argmax(fpr_healthy_combPrior>0.01)], 
                'thresh_1p_reg' : threshs_healthy_reg[np.argmax(fpr_healthy_reg>0.01)], 
                'thresh_1p_reco' : threshs_healthy_reco[np.argmax(fpr_healthy_reco>0.01)], 
                'thresh_1p_prior_kld' : threshs_healthy_prior_kld[np.argmax(fpr_healthy_prior_kld>0.01)], 
                'thresh_5p_comb' : threshs_healthy_comb[np.argmax(fpr_healthy_comb>0.05)], 
                'thresh_5p_combPrior' : threshs_healthy_combPrior[np.argmax(fpr_healthy_combPrior>0.05)], 
                'thresh_5p_reg' : threshs_healthy_reg[np.argmax(fpr_healthy_reg>0.05)], 
                'thresh_5p_reco' : threshs_healthy_reco[np.argmax(fpr_healthy_reco>0.05)], 
                'thresh_5p_prior_kld' : threshs_healthy_prior_kld[np.argmax(fpr_healthy_prior_kld>0.05)], 
                'thresh_10p_comb' : threshs_healthy_comb[np.argmax(fpr_healthy_comb>0.1)], 
                'thresh_10p_combPrior' : threshs_healthy_combPrior[np.argmax(fpr_healthy_combPrior>0.1)],
                'thresh_10p_reg' : threshs_healthy_reg[np.argmax(fpr_healthy_reg>0.1)], 
                'thresh_10p_reco' : threshs_healthy_reco[np.argmax(fpr_healthy_reco>0.1)],
                'thresh_10p_prior_kld' : threshs_healthy_prior_kld[np.argmax(fpr_healthy_prior_kld>0.1)], } 
    return threshholds_healthy

def get_eval_dictionary():
    _eval = {
        'IDs': [],
        'x': [],
        'reconstructions': [],
        'diffs': [],
        'diffs_volume': [],
        'Segmentation': [],
        'reconstructionTimes': [],
        'latentSpace': [],
        'Age': [],
        'AgeGroup': [],
        'l1reconstructionErrors': [],
        'l1recoErrorAll': [],
        'l1recoErrorUnhealthy': [],
        'l1recoErrorHealthy': [],
        'l2recoErrorAll': [],
        'l2recoErrorUnhealthy': [],
        'l2recoErrorHealthy': [],
        'l1reconstructionErrorMean': 0.0,
        'l1reconstructionErrorStd': 0.0,
        'l2reconstructionErrors': [],
        'l2reconstructionErrorMean': 0.0,
        'l2reconstructionErrorStd': 0.0,
        'HausPerVol': [],
        'TPPerVol': [],
        'FPPerVol': [],
        'FNPerVol': [],
        'TNPerVol': [],
        'TPRPerVol': [],
        'FPRPerVol': [],
        'TPTotal': [],
        'FPTotal': [],
        'FNTotal': [],
        'TNTotal': [],
        'TPRTotal': [],
        'FPRTotal': [],

        'PrecisionPerVol': [],
        'RecallPerVol': [],
        'PrecisionPerSlice': [],
        'RecallPerSlice': [],
        'lesionSizePerSlice': [],
        'lesionSizePerVol': [],
        'Dice': [],
        'DiceScorePerSlice': [],
        'DiceScorePerVol': [],
        'BestDicePerVol': [],
        'BestThresholdPerVol': [],
        'AUCPerVol': [],
        'AUPRCPerVol': [],
        'SpecificityPerVol': [],
        'AccuracyPerVol': [],
        'TPgradELBO': [],
        'FPgradELBO': [],
        'FNgradELBO': [],
        'TNgradELBO': [],
        'TPRgradELBO': [],
        'FPRgradELBO': [],
        'DicegradELBO': [],
        'DiceScorePerVolgradELBO': [],
        'BestDicePerVolgradELBO': [],
        'BestThresholdPerVolgradELBO': [],
        'AUCPerVolgradELBO': [],
        'AUPRCPerVolgradELBO': [],
        'KLD_to_learned_prior':[],

        'AUCAnomalyCombPerSlice': [], # PerVol!!! + Confusionmatrix.
        'AUPRCAnomalyCombPerSlice': [],
        'AnomalyScoreCombPerSlice': [],


        'AUCAnomalyKLDPerSlice': [],
        'AUPRCAnomalyKLDPerSlice': [],
        'AnomalyScoreKLDPerSlice': [],


        'AUCAnomalyRecoPerSlice': [],
        'AUPRCAnomalyRecoPerSlice': [],
        'AnomalyScoreRecoPerSlice': [],
        'AnomalyScoreRecoBinPerSlice': [],
        'AnomalyScoreAgePerSlice': [],
        'AUCAnomalyAgePerSlice': [],
        'AUPRCAnomalyAgePerSlice': [],

        'labelPerSlice' : [],
        'labelPerVol' : [],
        'AnomalyScoreCombPerVol' : [],
        'AnomalyScoreCombiPerVol' : [],
        'AnomalyScoreCombMeanPerVol' : [],
        'AnomalyScoreRegPerVol' : [],
        'AnomalyScoreRegMeanPerVol' : [],
        'AnomalyScoreRecoPerVol' : [],
        'AnomalyScoreCombPriorPerVol': [],
        'AnomalyScoreCombiPriorPerVol': [],
        'AnomalyScoreAgePerVol' : [],
        'AnomalyScoreRecoMeanPerVol' : [],
        'DiceScoreKLPerVol': [],
        'DiceScoreKLCombPerVol': [],
        'BestDiceKLCombPerVol': [],
        'BestDiceKLPerVol': [],
        'AUCKLCombPerVol': [],
        'AUPRCKLCombPerVol': [],
        'AUCKLPerVol': [],
        'AUPRCKLPerVol': [],
        'TPKLCombPerVol': [],
        'FPKLCombPerVol': [],
        'TNKLCombPerVol': [],
        'FNKLCombPerVol': [],
        'TPRKLCombPerVol': [],
        'FPRKLCombPerVol': [],
        'TPKLPerVol': [],
        'FPKLPerVol': [],
        'TNKLPerVol': [],
        'FNKLPerVol': [],
        'TPRKLPerVol': [],
        'FPRKLPerVol': [],



    }
    return _eval

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
def apply_2d_median_filter(volume, kernelsize=5):  # kernelsize 5 works quite well
    img = scipy.ndimage.filters.median_filter(volume, (kernelsize, kernelsize))
    return img
def squash_intensities(img):
    # logistic function intended to squash reconstruction errors from [0;0.2] to [0;1] (just an example)
    k = 100
    offset = 0.5
    return 2.0 * ((1.0 / (1.0 + np.exp(-k * img))) - offset)
def apply_colormap(img, colormap_handle):
    img = img - img.min()
    if img.max() != 0:
        img = img / img.max()
    img = Image.fromarray(np.uint8(colormap_handle(img) * 255))
    return img

def add_colorbar(img):
    for i in range(img.squeeze().shape[0]):
        img[i, -1] = float(i) / img.squeeze().shape[0]

    return img

def filter_3d_connected_components(volume):
    sz = None
    if volume.ndim > 3:
        sz = volume.shape
        volume = np.reshape(volume, [sz[0] * sz[1], sz[2], sz[3]])

    cc_volume = label(volume, connectivity=3)
    props = regionprops(cc_volume)
    for prop in props:
        if prop['filled_area'] <= 7:
            volume[cc_volume == prop['label']] = 0

    if sz is not None:
        volume = np.reshape(volume, [sz[0], sz[1], sz[2], sz[3]])
    return volume

# From Zimmerer iterative algorithm for threshold search
def find_best_val(x, y, val_range=(0, 1), max_steps=4, step=0, max_val=0, max_point=0):  #x: Image , y: Label
    if step == max_steps:
        return max_val, max_point

    if val_range[0] == val_range[1]:
        val_range = (val_range[0], 1)

    bottom = val_range[0]
    top = val_range[1]
    center = bottom + (top - bottom) * 0.5

    q_bottom = bottom + (top - bottom) * 0.25
    q_top = bottom + (top - bottom) * 0.75
    val_bottom = dice(x > q_bottom, y)
    #print(str(np.mean(x>q_bottom)) + str(np.mean(y)))
    val_top = dice(x > q_top, y)
    #print(str(np.mean(x>q_top)) + str(np.mean(y)))
    #val_bottom = val_fn(x, y, q_bottom) # val_fn is the dice calculation dice(p, g)
    #val_top = val_fn(x, y, q_top)

    if val_bottom >= val_top:
        if val_bottom >= max_val:
            max_val = val_bottom
            max_point = q_bottom
        return find_best_val(x, y, val_range=(bottom, center), step=step + 1, max_steps=max_steps,
                             max_val=max_val, max_point=max_point)
    else:
        if val_top >= max_val:
            max_val = val_top
            max_point = q_top
        return find_best_val(x, y, val_range=(center, top), step=step + 1, max_steps=max_steps,
                             max_val=max_val,max_point=max_point)
def dice(P, G):
    psum = np.sum(P.flatten())
    gsum = np.sum(G.flatten())
    pgsum = np.sum(np.multiply(P.flatten(), G.flatten()))
    score = (2 * pgsum) / (psum + gsum)
    return score
    
def compute_roc(predictions, labels):
    _fpr, _tpr, _ = roc_curve(labels.astype(int), predictions,pos_label=1)
    roc_auc = auc(_fpr, _tpr)
    return roc_auc, _fpr, _tpr, _


def compute_prc(predictions, labels):
    precisions, recalls, thresholds = precision_recall_curve(labels.astype(int), predictions)
    auprc = average_precision_score(labels.astype(int), predictions)
    return auprc, precisions, recalls, thresholds   

# Dice Score 
def xfrange(start, stop, step):
    i = 0
    while start + i * step < stop:
        yield start + i * step
        i += 1

def tpr(P, G):
    tp = np.sum(np.multiply(P.flatten(), G.flatten()))
    fn = np.sum(np.multiply(np.invert(P.flatten()), G.flatten()))
    return tp / (tp + fn)


def fpr(P, G):
    tp = np.sum(np.multiply(P.flatten(), G.flatten()))
    fp = np.sum(np.multiply(P.flatten(), np.invert(G.flatten())))
    return fp / (fp + tp)


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