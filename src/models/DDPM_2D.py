from src.models.modules.cond_DDPM import GaussianDiffusion
from src.models.modules.OpenAI_Unet import UNetModel as OpenAI_UNet
from src.models.modules.DDPM_encoder import get_encoder
import torch
from src.utils.utils_eval import _test_step, _test_end, get_eval_dictionary
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

class DDPM_2D(LightningModule):
    def __init__(self,cfg,prefix=None):
        #! IN DDPM
        # print('='*10)
        # print('INITIALISING DDPM_2D')
        # print('='*10)
        super().__init__()
        
        self.cfg = cfg
        
        #! INITIALISES conditioning net
        if cfg.get('condition',True):
            with open_dict(self.cfg):
                self.cfg['cond_dim'] = cfg.get('unet_dim',128)

            self.encoder, out_features = get_encoder(cfg)
            # print('encoder out features:', out_features)
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
            # print('Loading pretrained encoder from: ', cfg.encoder_path)
            assert cfg.get('encoder_path',None) is not None
            # print('encoder path, ', cfg.get('encoder_path',None))
            
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
            # print("i'm i DDPM2d forward block (whch is basically the encoder)")
            # print("Context vector c shape:", x.shape)  # Should be [batch_size, 128]
            # # print("Context vector c:", x)
        else: 
            x = None
        return x


    def training_step(self, batch, batch_idx: int):
        # print('='*10)
        # print('IN DDPM TRAINING STEP')
        # print('='*10)
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
        loss, reco = self.diffusion(input, cond = features, noise = noise)

        self.log(f'{self.prefix}train/Loss', loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=input.shape[0],sync_dist=True)
        return {"loss": loss}
    
    def validation_step(self, batch: Any, batch_idx: int):
        # print('='*10)
        # print('IN DDPM VALIDATION STEP')
        # print('='*10)
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
        loss, reco = self.diffusion(input,cond=features,noise=noise)

        self.log(f'{self.prefix}val/Loss_comb', loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=input.shape[0],sync_dist=True)
        return {"loss": loss}

    def on_test_start(self):
        # print('='*10)
        # print('IN DDPM ON TEST START DDPM2D')
        # print('='*10)
        
        self.eval_dict = get_eval_dictionary()
        self.inds = []
        self.latentSpace_slice = []
        self.new_size = [160,190,160]
        self.diffs_list = []
        self.seg_list = []
        if not hasattr(self,'threshold'):
            self.threshold = {}

    def test_step(self, batch: Any, batch_idx: int):
        # print('='*10)
        # print('IN DDPM ON TEST STEP')
        # print('='*10)
        
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
        # print('8'*1000)
        # print(input.size())
        # print(input.size(4))
        
        self.cfg['num_eval_slices']=4
        if self.cfg.get('num_eval_slices', input.size(4)) != input.size(4):
            num_slices = self.cfg.get('num_eval_slices', input.size(4))  # number of center slices to evaluate. If not set, the whole Volume is evaluated
            start_slice = int((input.size(4) - num_slices) / 2)
            input = input[...,start_slice:start_slice+num_slices]
            data_orig = data_orig[...,start_slice:start_slice+num_slices] 
            data_seg = data_seg[...,start_slice:start_slice+num_slices]
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
        # print('THE WEIRD CONDITION BIT CALLED "features" in DDPM_2D')
        features = self(input)
        # print('shape features: ', features.shape)
        features_single = features
        #! CONDITION BIT!!!
        if self.cfg.condition:
            # print('additng features to latern space')
            # print('latent space:', latentSpace)
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
                loss_diff, reco = self.diffusion(input,cond=features,t=t-1,noise=noise)
                reco_ensemble += reco
                
            reco = reco_ensemble / len(timesteps) # average over timesteps
            # print("reconstruction shape", reco.shape)
            # print("reconstruction", reco)
        else :
            if self.cfg.get('noisetype') is not None:
                noise = gen_noise(self.cfg, input.shape).to(self.device)
            else: 
                noise = None
            #! actual model call
            loss_diff, reco = self.diffusion(input,cond=features,t=self.test_timesteps-1,noise=noise)
            # print("reconstruction shape", reco.shape)
            # print("reconstruction", reco)
        # calculate loss and Anomalyscores
        AnomalyScoreComb.append(loss_diff.cpu())
        AnomalyScoreReg.append(loss_diff.cpu())
        AnomalyScoreReco.append(loss_diff.cpu())

        # reassamble the reconstruction volume
        final_volume = reco.clone().squeeze()
        final_volume = final_volume.permute(1,2,0) # to HxWxD
        # print('FINAL VOLUME (DDPM) shape', final_volume.shape)
        # print('FINAL VOLUME', final_volume)

       

        # average across slices to get volume-based scores
        self.latentSpace_slice.extend(latentSpace)
        self.eval_dict['latentSpace'].append(torch.mean(torch.stack(latentSpace),0))
        AnomalyScoreComb_vol = np.mean(AnomalyScoreComb) 
        AnomalyScoreReg_vol = np.mean(AnomalyScoreReg)
        AnomalyScoreReco_vol = np.mean(AnomalyScoreReco)

        self.eval_dict['AnomalyScoreRegPerVol'].append(AnomalyScoreReg_vol)


        if not self.cfg.get('use_postprocessed_score', True):
            self.eval_dict['AnomalyScoreRecoPerVol'].append(AnomalyScoreReco_vol)
            self.eval_dict['AnomalyScoreCombPerVol'].append(AnomalyScoreComb_vol)
            self.eval_dict['AnomalyScoreCombiPerVol'].append(AnomalyScoreReco_vol * AnomalyScoreReg_vol)
            self.eval_dict['AnomalyScoreCombPriorPerVol'].append(AnomalyScoreReco_vol + self.cfg.beta * 0)
            self.eval_dict['AnomalyScoreCombiPriorPerVol'].append(AnomalyScoreReco_vol * 0)

        final_volume = final_volume.unsqueeze(0)
        final_volume = final_volume.unsqueeze(0)
        # print('unsqueeze final volume shape (DDPM2d)', final_volume.shape)
        # print('unsqueeze final volume (DDPM 2D)', final_volume)
        # calculate metrics

        _test_step(self, final_volume, data_orig, data_seg, data_mask, batch_idx, ID, label) # everything that is independent of the model choice

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