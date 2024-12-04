from sklearn.metrics import  confusion_matrix, roc_curve, accuracy_score, precision_recall_fscore_support, auc,precision_recall_curve, average_precision_score
from src.models.LDM.modules.diffusionmodules.util import timestep_embedding
from src.models.modules.OpenAI_Unet import UNetModel as OpenAI_UNet
from pytorch_lightning.core.lightning import LightningModule
from src.models.modules.cond_DDPM import GaussianDiffusion
from src.models.modules.DDPM_encoder import get_encoder
import torch
import numpy as np
from typing import Any
import torchio as tio
from src.utils.generate_noise import gen_noise
import wandb
from omegaconf import open_dict
from collections import OrderedDict
from torch import nn
from skimage.measure import regionprops, label
import os
import matplotlib.pyplot as plt
import scipy
from torch.nn import functional as F
from PIL import Image
import matplotlib.colors as colors


class DDPM_2D_ray(LightningModule):
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

    def on_test_start(self):
        print('='*10)
        print('IN  DDPM RAY ON TEST START DDPM2D')
        print('='*10)
        
        self.eval_dict = get_eval_dictionary()
        self.inds = []
        self.latentSpace_slice = []
        self.new_size = [160,190,160]
        self.diffs_list = []
        self.seg_list = []
        if not hasattr(self,'threshold'):
            self.threshold = {}

    def test_step(self, batch: Any, batch_idx: int, num_slices):
        print('='*10)
        print('IN DDPM RAY ON TEST STEP')
        print('='*10)
        
        self.dataset = batch['Dataset']
        input = batch['vol'][tio.DATA]
        data_orig = batch['vol_orig'][tio.DATA]
        # data_seg = batch['seg_orig'][tio.DATA] if batch['seg_available'] else torch.zeros_like(data_orig)
        data_mask = batch['mask_orig'][tio.DATA]
        ID = batch['ID']
        age = batch['age']
        self.stage = batch['stage']
        label = batch['label']
        AnomalyScoreComb = []
        AnomalyScoreReg = []
        AnomalyScoreReco = []
        latentSpace = []
        
        
        print(num_slices)
        if num_slices != input.size(4):
            start_slice = int((input.size(4) - num_slices) / 2)
            input = input[...,start_slice:start_slice+num_slices]
            data_orig = data_orig[...,start_slice:start_slice+num_slices] 
            data_mask = data_mask[...,start_slice:start_slice+num_slices]
            ind_offset = start_slice
        else: 
            ind_offset = 0 

        final_volume = torch.zeros([input.size(2), input.size(3), input.size(4)], device = self.device)


        # reorder depth to batch dimension
        assert input.shape[0] == 1, "Batch size must be 1"
        input = input.squeeze(0).permute(3,0,1,2) # [B,C,H,W,D] -> [D,C,H,W]
        #!
        # calc features for guidance
        print('THE WEIRD CONDITION BIT CALLED "features" in DDPM_2D')
        features = self(input)
        print('shape features: ', features.shape)
        features_single = features
        #! CONDITION BIT!!!
        if self.cfg.condition:
            print('additng features to latern space')
            print('latent space:', latentSpace)
            latentSpace.append(features_single.mean(0).squeeze().detach().cpu())
        else: 
            latentSpace.append(torch.tensor([0],dtype=float).repeat(input.shape[0]))

        if self.cfg.get('noise_ensemble',False): # evaluate with different noise levels
            timesteps = self.cfg.get('step_ensemble',[250,500,750]) # timesteps to evaluate
            reco_ensemble = torch.zeros_like(input)
            for t in timesteps:
                # generate noise
                if self.cfg.get('noisetype') is not None:
                    noise = gen_noise(self.cfg, input.shape).to(self.device)
                else: 
                    noise = None
                #! actual model call
                loss_diff, reco, unet_details = self.diffusion(input,cond=features,t=t-1,noise=noise)
                reco_ensemble += reco
                
            reco = reco_ensemble / len(timesteps) # average over timesteps
            print("reconstruction shape", reco.shape)
            print("reconstruction", reco)
        else :
            if self.cfg.get('noisetype') is not None:
                noise = gen_noise(self.cfg, input.shape).to(self.device)
            else: 
                noise = None
            #! actual model call
            loss_diff, reco, unet_details = self.diffusion(input,cond=features,t=self.test_timesteps-1,noise=noise)
            print("reconstruction shape", reco.shape)
            print("reconstruction", reco)
    #     # calculate loss and Anomalyscores
    #     AnomalyScoreComb.append(loss_diff.cpu())
    #     AnomalyScoreReg.append(loss_diff.cpu())
    #     AnomalyScoreReco.append(loss_diff.cpu())

        # reassamble the reconstruction volume
        final_volume = reco.clone().squeeze()
        final_volume = final_volume.permute(1,2,0) # to HxWxD
        print('FINAL VOLUME (DDPM) shape', final_volume.shape)
        print('FINAL VOLUME', final_volume)

       

    #     # average across slices to get volume-based scores
    #     self.latentSpace_slice.extend(latentSpace)
    #     self.eval_dict['latentSpace'].append(torch.mean(torch.stack(latentSpace),0))
    #     AnomalyScoreComb_vol = np.mean(AnomalyScoreComb) 
    #     AnomalyScoreReg_vol = np.mean(AnomalyScoreReg)
    #     AnomalyScoreReco_vol = np.mean(AnomalyScoreReco)

    #     self.eval_dict['AnomalyScoreRegPerVol'].append(AnomalyScoreReg_vol)


    #     if not self.cfg.get('use_postprocessed_score', True):
    #         self.eval_dict['AnomalyScoreRecoPerVol'].append(AnomalyScoreReco_vol)
    #         self.eval_dict['AnomalyScoreCombPerVol'].append(AnomalyScoreComb_vol)
    #         self.eval_dict['AnomalyScoreCombiPerVol'].append(AnomalyScoreReco_vol * AnomalyScoreReg_vol)
    #         self.eval_dict['AnomalyScoreCombPriorPerVol'].append(AnomalyScoreReco_vol + self.cfg.beta * 0)
    #         self.eval_dict['AnomalyScoreCombiPriorPerVol'].append(AnomalyScoreReco_vol * 0)

        final_volume = final_volume.unsqueeze(0)
        final_volume = final_volume.unsqueeze(0)
        print('unsqueeze final volume shape (DDPM2d)', final_volume.shape)
        print('unsqueeze final volume (DDPM 2D)', final_volume)
        # calculate metrics

        # _test_step(self, final_volume, data_orig, data_mask, batch_idx, ID, label) # everything that is independent of the model choice

    #     #! NEW TESTS BABY
        
    #     # # Visualization: Compare input and output for a specific slice
    #     # slice_idx = 50  # Choose the slice you want to visualize (can be dynamic)
    #     # visualize_comparison(input, final_volume, slice_idx=slice_idx)

    #     # # Optional: Visualize the difference
    #     # visualize_difference(input, final_volume, slice_idx=slice_idx)

    #     # # Optional: Visualize a range of slices
    #     # visualize_multiple_slices(input, final_volume, slice_range=range(40, 60))
    
    # def on_test_end(self) :
    #     # calculate metrics
    #     _test_end(self) # everything that is independent of the model choice 


    # def configure_optimizers(self):
    #     return optim.Adam(self.parameters(), lr=self.cfg.lr)
    
    # def update_prefix(self, prefix):
    #     self.prefix = prefix 
    

def _test_step(self, final_volume, data_orig,data_mask, batch_idx, ID, label_vol) :
        print('='*10)
        print('IN _TEST_STEP')
        print('='*10)
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


        # store in eval dict
        self.eval_dict['l1recoErrorAll'].append(l1err.item())


        # move data to CPU
        data_mask = data_mask.cpu()
        diff_volume = diff_volume.cpu()
        data_orig = data_orig.cpu()
        final_volume = final_volume.cpu()
        # binarize the segmentation
        data_mask[data_mask > 0] = 1

        # Erode the Brainmask 
        if self.cfg['erodeBrainmask']:
            print("ERODING BRAIN MASK")
            diff_volume = apply_brainmask_volume(diff_volume.cpu(), data_mask.squeeze().cpu())   

        # Filter the DifferenceImage
        if self.cfg['medianFiltering']:
            print("MEDIAN FILTERING")
            diff_volume = torch.from_numpy(apply_3d_median_filter(diff_volume.numpy().squeeze(),kernelsize=self.cfg.get('kernelsize_median',5))).unsqueeze(0) # bring back to tensor

        #! save image grid

        print("SAVING OUTPUT - ORIGINAL, FINAL, MASK ETC")
        log_images(self,diff_volume, data_orig, data_mask, final_volume, ID)
            
        # ### Compute Metrics per Volume / Step ###
        # if self.cfg.evalSeg and self.dataset[0] not in self.healthy_sets: # only compute metrics if segmentation is available


        #     if 'test' in self.stage:
        #         bestThresh = self.threshold['total']

        #     if self.cfg["threshold"] == 'auto':
        #         diffs_thresholded = diff_volume > bestThresh
        #     else: # never used
        #         diffs_thresholded = diff_volume > self.cfg["threshold"]    
            
        #     # Connected Components
        #     if not 'node' in self.dataset[0].lower(): # no 3D data
        #         diffs_thresholded = filter_3d_connected_components(np.squeeze(diffs_thresholded)) # this is only done for patient-wise evaluation atm
            
            

        #     self.eval_dict['BestThresholdPerVol'].append(bestThresh)
        #     self.eval_dict['AUCPerVol'].append(AUC)
        #     self.eval_dict['AUPRCPerVol'].append(AUPRC)
        #     self.eval_dict['IDs'].append(ID[0])


        # if 'val' in self.stage  :
        #     if batch_idx == 0:
        #         self.diffs_list = np.array(diff_volume.squeeze().flatten())
        #     else: 
        #         self.diffs_list = np.append(self.diffs_list,np.array(diff_volume.squeeze().flatten()),axis=0)

        # # Reconstruction based Anomaly score for Slice-Wise evaluation  
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


#! SAVES IMAGES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def log_images(self, diff_volume, data_orig,data_mask, final_volume, ID, diff_volume_KL=None,  flow=None ):
    print('SAVING IMAGES!')
    
    ImagePathList = {
                    'imagesGrid': os.path.join(os.getcwd(),'RAY_TEST')}
    for key in ImagePathList :
        if not os.path.isdir(ImagePathList[key]):
            os.mkdir(ImagePathList[key])

    for j in range(0,diff_volume.squeeze().shape[2],10) : 

        # create a figure of images with 1 row and 4 columns for subplots
        fig, ax = plt.subplots(1,4,figsize=(16,4))
        # change spacing between subplots
        fig.subplots_adjust(wspace=0.0)
        # orig
        ax[0].imshow(data_orig.squeeze()[...,j].rot90(3),'gray')
        # reconstructed
        ax[1].imshow(final_volume[...,j].rot90(3).squeeze(),'gray')
        # difference
        ax[2].imshow(diff_volume.squeeze()[:,...,j].rot90(3),'inferno',norm=colors.Normalize(vmin=0, vmax=diff_volume.max()+.01))
        # mask
        
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




def _test_end(self) :
        print('IN TEST END')

    # average over all test samples
        self.eval_dict['l1recoErrorAllMean'] = np.nanmean(self.eval_dict['l1recoErrorAll'])
        self.eval_dict['l1recoErrorAllStd'] = np.nanstd(self.eval_dict['l1recoErrorAll'])
        self.eval_dict['l2recoErrorAllMean'] = np.nanmean(self.eval_dict['l2recoErrorAll'])
        self.eval_dict['l2recoErrorAllStd'] = np.nanstd(self.eval_dict['l2recoErrorAll'])

        self.eval_dict['l1recoErrorHealthyMean'] = np.nanmean(self.eval_dict['l1recoErrorHealthy'])
        self.eval_dict['l1recoErrorHealthyStd'] = np.nanstd(self.eval_dict['l1recoErrorHealthy'])
        self.eval_dict['l1recoErrorUnhealthyMean'] = np.nanmean(self.eval_dict['l1recoErrorUnhealthy'])
        self.eval_dict['l1recoErrorUnhealthyStd'] = np.nanstd(self.eval_dict['l1recoErrorUnhealthy'])

        self.eval_dict['l2recoErrorHealthyMean'] = np.nanmean(self.eval_dict['l2recoErrorHealthy'])
        self.eval_dict['l2recoErrorHealthyStd'] = np.nanstd(self.eval_dict['l2recoErrorHealthy'])
        self.eval_dict['l2recoErrorUnhealthyMean'] = np.nanmean(self.eval_dict['l2recoErrorUnhealthy'])
        self.eval_dict['l2recoErrorUnhealthyStd'] = np.nanstd(self.eval_dict['l2recoErrorUnhealthy'])

        self.eval_dict['AUPRCPerVolMean'] = np.nanmean(self.eval_dict['AUPRCPerVol'])
        self.eval_dict['AUPRCPerVolStd'] = np.nanstd(self.eval_dict['AUPRCPerVol'])
        self.eval_dict['AUCPerVolMean'] = np.nanmean(self.eval_dict['AUCPerVol'])
        self.eval_dict['AUCPerVolStd'] = np.nanstd(self.eval_dict['AUCPerVol'])

        self.eval_dict['DicePerVolMean'] = np.nanmean(self.eval_dict['DiceScorePerVol'])
        self.eval_dict['DicePerVolStd'] = np.nanstd(self.eval_dict['DiceScorePerVol'])
        self.eval_dict['BestDicePerVolMean'] = np.mean(self.eval_dict['BestDicePerVol'])
        self.eval_dict['BestDicePerVolStd'] = np.std(self.eval_dict['BestDicePerVol'])
        self.eval_dict['BestThresholdPerVolMean'] = np.mean(self.eval_dict['BestThresholdPerVol'])
        self.eval_dict['BestThresholdPerVolStd'] = np.std(self.eval_dict['BestThresholdPerVol'])


        self.eval_dict['TPPerVolMean'] = np.nanmean(self.eval_dict['TPPerVol'])
        self.eval_dict['TPPerVolStd'] = np.nanstd(self.eval_dict['TPPerVol'])
        self.eval_dict['FPPerVolMean'] = np.nanmean(self.eval_dict['FPPerVol'])
        self.eval_dict['FPPerVolStd'] = np.nanstd(self.eval_dict['FPPerVol'])
        self.eval_dict['TNPerVolMean'] = np.nanmean(self.eval_dict['TNPerVol'])
        self.eval_dict['TNPerVolStd'] = np.nanstd(self.eval_dict['TNPerVol'])
        self.eval_dict['FNPerVolMean'] = np.nanmean(self.eval_dict['FNPerVol'])
        self.eval_dict['FNPerVolStd'] = np.nanstd(self.eval_dict['FNPerVol'])
        self.eval_dict['TPRPerVolMean'] = np.nanmean(self.eval_dict['TPRPerVol'])
        self.eval_dict['TPRPerVolStd'] = np.nanstd(self.eval_dict['TPRPerVol'])
        self.eval_dict['FPRPerVolMean'] = np.nanmean(self.eval_dict['FPRPerVol'])
        self.eval_dict['FPRPerVolStd'] = np.nanstd(self.eval_dict['FPRPerVol'])
        self.eval_dict['HausPerVolMean'] = np.nanmean(np.array(self.eval_dict['HausPerVol'])[np.isfinite(self.eval_dict['HausPerVol'])])
        self.eval_dict['HausPerVolStd'] = np.nanstd(np.array(self.eval_dict['HausPerVol'])[np.isfinite(self.eval_dict['HausPerVol'])])
        


        self.eval_dict['PrecisionPerVolMean'] = np.mean(self.eval_dict['PrecisionPerVol'])
        self.eval_dict['PrecisionPerVolStd'] =np.std(self.eval_dict['PrecisionPerVol'])
        self.eval_dict['RecallPerVolMean'] = np.mean(self.eval_dict['RecallPerVol'])
        self.eval_dict['RecallPerVolStd'] = np.std(self.eval_dict['RecallPerVol'])
        self.eval_dict['PrecisionPerSliceMean'] = np.mean(self.eval_dict['PrecisionPerSlice'])
        self.eval_dict['PrecisionPerSliceStd'] = np.std(self.eval_dict['PrecisionPerSlice'])
        self.eval_dict['RecallPerSliceMean'] = np.mean(self.eval_dict['RecallPerSlice'])
        self.eval_dict['RecallPerSliceStd'] = np.std(self.eval_dict['RecallPerSlice'])
        self.eval_dict['AccuracyPerVolMean'] = np.mean(self.eval_dict['AccuracyPerVol'])
        self.eval_dict['AccuracyPerVolStd'] = np.std(self.eval_dict['AccuracyPerVol'])
        self.eval_dict['SpecificityPerVolMean'] = np.mean(self.eval_dict['SpecificityPerVol'])
        self.eval_dict['SpecificityPerVolStd'] = np.std(self.eval_dict['SpecificityPerVol'])


        if 'test' in self.stage :
            del self.threshold
                
        if 'val' in self.stage: 
            if self.dataset[0] not in self.healthy_sets:
                bestdiceScore, bestThresh = find_best_val((self.diffs_list).flatten(), (self.seg_list).flatten().astype(bool), 
                                        val_range=(0, np.max((self.diffs_list))), 
                                        max_steps=10, 
                                        step=0, 
                                        max_val=0, 
                                        max_point=0)

                self.threshold['total'] = bestThresh 
                if self.cfg.get('KLDBackprop',False): 
                    bestdiceScoreKLComb, bestThreshKLComb = find_best_val((self.diffs_listKLComb).flatten(), (self.seg_list).flatten().astype(bool), 
                        val_range=(0, np.max((self.diffs_listKLComb))), 
                        max_steps=10, 
                        step=0, 
                        max_val=0, 
                        max_point=0)

                    self.threshold['totalKLComb'] = bestThreshKLComb 
                    bestdiceScoreKL, bestThreshKL = find_best_val((self.diffs_listKL).flatten(), (self.seg_list).flatten().astype(bool), 
                        val_range=(0, np.max((self.diffs_listKL))), 
                        max_steps=10, 
                        step=0, 
                        max_val=0, 
                        max_point=0)

                    self.threshold['totalKL'] = bestThreshKL 
            else: # define thresholds based on the healthy validation set
                _, fpr_healthy, _, threshs = compute_roc((self.diffs_list).flatten(), np.zeros_like(self.diffs_list).flatten().astype(int))
                self.threshholds_healthy= {
                        'thresh_1p' : threshs[np.argmax(fpr_healthy>0.01)], # 1%
                        'thresh_5p' : threshs[np.argmax(fpr_healthy>0.05)], # 5%
                        'thresh_10p' : threshs[np.argmax(fpr_healthy>0.10)]} # 10%}
                self.eval_dict['t_1p'] = self.threshholds_healthy['thresh_1p']
                self.eval_dict['t_5p'] = self.threshholds_healthy['thresh_5p']
                self.eval_dict['t_10p'] = self.threshholds_healthy['thresh_10p']
                
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
