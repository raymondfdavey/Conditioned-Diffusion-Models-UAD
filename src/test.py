import torch
import hydra
import torchio as tio
import os
import pickle
import numpy as np
from src.datamodules.new_IXI import IXI_new
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf, open_dict
from src.utils.generate_noise import gen_noise
from src.utils.utils_eval import apply_3d_median_filter
from torch.nn import functional as F




def visualize_slices(input_tensor, title=""):
    """
    Simple function to visualize all slices in a tensor.
    Expects tensor shape: [B, C, H, W, D] where D is number of slices.
    """
    
    # Get the data from the first batch and channel
    data = input_tensor[0, 0].cpu().numpy()
    num_slices = data.shape[-1]
    
    # Create a figure with a subplot for each slice
    fig, axes = plt.subplots(1, num_slices, figsize=(4*num_slices, 4))
    if num_slices == 1:  # Handle case of single slice
        axes = [axes]
        
    # Plot each slice
    for i in range(num_slices):
        axes[i].imshow(data[..., i], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Slice {i}')
    
    plt.suptitle(f'{title}\nShape: {input_tensor.shape}')
    plt.tight_layout()
    plt.show()

def setup_model(cfg, model_path: str, encoder_path: str = None):
    """Set up the model with proper configuration and checkpoint loading."""
    if encoder_path:
        cfg.model.cfg.encoder_path = encoder_path
        cfg.model.cfg.pretrained_encoder = True
        
    cfg.model._target_ = f'{cfg.model._target_}'
    model = hydra.utils.instantiate(cfg.model)
    
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['state_dict']
    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict, strict=False
    )
    
    if missing_keys or unexpected_keys:
        print("State dict loading issues:")
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        
    model.eval()
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        
    return model

def setup_datamodule(cfg):
    """Initialize the IXI datamodule for slice-based testing.
    
    The config structure has the datamodule configuration nested under cfg.datamodule.cfg,
    so we pass that specific part to the IXI_new class.
    """
    datamodule = IXI_new(cfg.datamodule.cfg)
    datamodule.setup()
    return datamodule

def run_inference(model, batch, cfg):
    """Run model inference on a single batch."""
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Move data to device and extract the 4 slices
        input_data = batch['vol'][tio.DATA].to(device)  # Shape: [B, 1, H, W, 4]
        features = model.forward(input_data)
        
        # Handle noise generation
        noise = None
        if cfg.get('noisetype') is not None:
            noise = gen_noise(cfg, input_data.shape).to(device)
            
        # Get reconstruction
        loss_diff, reconstruction, unet_details = model.diffusion(
            input_data, 
            cond=features,
            t=model.test_timesteps-1,
            noise=noise
        )
        
        return reconstruction, loss_diff, unet_details

def evaluate_batch(model, batch, cfg):
    """Evaluate a single batch and compute metrics."""
    reconstruction, loss_diff, unet_details = run_inference(model, batch, cfg)
    device = next(model.parameters()).device
    
    # Calculate metrics
    metrics = compute_metrics(
        input_image=batch['vol'][tio.DATA].to(device),
        reconstruction=reconstruction,
        original=batch['vol_orig'][tio.DATA].to(device),
        mask=batch['mask_orig'][tio.DATA].to(device),
        loss=loss_diff
    )
    
    return {
        'reconstruction': reconstruction.cpu(),
        'metrics': metrics,
        'unet_details': unet_details,
        'metadata': {
            'ID': batch['ID'],
            'age': batch['age'],
            'Dataset': batch['Dataset'],
            'stage': batch['stage']
        }
    }

def test_dataset(model, datamodule, cfg):
    """Run testing on the entire dataset."""
    model.eval()
    batch_results = {}
    
    print("Starting testing...")
    test_loader = datamodule.test_dataloader()
    
    for batch_idx, batch in enumerate(test_loader):
        batch_output = evaluate_batch(model, batch, cfg)
        batch_results[f'batch_{batch_idx}'] = batch_output
        
        if batch_idx % cfg.testing.log_every_n_steps == 0:
            print(f"Processed batch {batch_idx}/{len(test_loader)}")
            print(f"Current sample ID: {batch_output['metadata']['ID']}")
            print("Current metrics:", {k: f"{v:.4f}" for k, v in batch_output['metrics'].items()})
    
    # Calculate average metrics
    all_metrics = {}
    for batch_output in batch_results.values():
        for metric_name, value in batch_output['metrics'].items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            all_metrics[metric_name].append(value)
    
    avg_metrics = {
        metric_name: torch.tensor(values).mean().item()
        for metric_name, values in all_metrics.items()
    }
    
    batch_results['average_metrics'] = avg_metrics
    
    # Print summary
    print("\nTest Results:")
    for metric_name, value in avg_metrics.items():
        print(f"{metric_name}: {value:.4f}")
        
    return batch_results

def save_results(results, output_dir: str = None):
    """Save test results and predictions."""
    if output_dir is None:
        output_dir = os.getcwd()
        
    output_path = os.path.join(output_dir, 'test_results.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump({'test': {'IXI': results}}, f)
    print(f"Saved test results to {output_path}")


@hydra.main(config_path='configs', config_name='test_config.yaml')
def test(cfg: DictConfig):
    """Main testing function."""
    print("Setting up testing environment...")
    
    # Set up model and datamodule
    model_path = cfg.get('load_checkpoint', None)
    model = setup_model(cfg, model_path)
    datamodule = setup_datamodule(cfg)
    
    # Get the test dataloader
    test_loader = datamodule.test_dataloader()
    
    model.on_test_start()
    
    num_slices = 4

    for batch_idx, batch in enumerate(test_loader):
        model.test_step(batch, batch_idx, num_slices)
    
    
    
    
    

    
    
    
    # print("Starting testing...")
    # results = []
    
        
    #     # Process each batch from the test loader
    # for batch_idx, batch in enumerate(test_loader):
    #     print(f"Processing batch {batch_idx}...")
        
    #     # Get and prepare input data
    #     input = batch['vol'][tio.DATA]
    #     data_orig = batch['vol_orig'][tio.DATA]
    #     data_mask = batch['mask_orig'][tio.DATA]
        
    #     print(f"Selecting slices at indices: {slice_indices}")
    #     input = input[..., slice_indices]
    #     data_orig = data_orig[..., slice_indices]
    #     data_mask = data_mask[..., slice_indices]
        
    #     print("\nAfter slice selection:")
    #     print(f"Selected slices shape: {input.shape}")

    #     with torch.no_grad():
    #         if torch.cuda.is_available():
    #             input = input.cuda()
            
    #         # Transform volume to slices
    #         assert input.shape[0] == 1, "Batch size must be 1"
    #         input = input.squeeze(0).permute(3,0,1,2)  # [1,1,96,96,4] -> [4,1,96,96]
    #         print("After transformation:", input.shape)
            
    #         # Process slices
    #         all_reconstructions = []
    #         for slice_idx in range(input.shape[0]):
    #             current_slice = input[slice_idx:slice_idx+1]
    #             print(f"\nProcessing slice {slice_idx}")
    #             print("Single slice shape:", current_slice.shape)
                
    #             features = model(current_slice)
    #             print("Features shape:", features.shape)
                
    #             if cfg.get('noise_ensemble', False):
    #                 timesteps = cfg.get('step_ensemble', [250,500,750])
    #                 slice_reco_ensemble = torch.zeros_like(current_slice)
                    
    #                 for t in timesteps:
    #                     noise = None
    #                     if cfg.get('noisetype') is not None:
    #                         noise = gen_noise(cfg, current_slice.shape).to(current_slice.device)
                        
    #                     loss_diff, reco, unet_details = model.diffusion(
    #                         current_slice,
    #                         cond=features,
    #                         t=t-1,
    #                         noise=noise
    #                     )
    #                     slice_reco_ensemble += reco
                    
    #                 slice_reconstruction = slice_reco_ensemble / len(timesteps)
    #             else:
    #                 noise = None
    #                 if cfg.get('noisetype') is not None:
    #                     noise = gen_noise(cfg, current_slice.shape).to(current_slice.device)
                    
    #                 loss_diff, slice_reconstruction, unet_details = model.diffusion(
    #                     current_slice,
    #                     cond=features,
    #                     t=model.test_timesteps-1,
    #                     noise=noise
    #                 )
                
    #             # Remove the batch dimension before adding to list
    #             all_reconstructions.append(slice_reconstruction.squeeze(0))
            
    #         # Stack and reshape reconstructions correctly
    #         reconstruction = torch.stack(all_reconstructions, dim=0)  # [4,1,96,96]
    #         print("Initial stacked shape:", reconstruction.shape)
            
    #         # Now prepare it for visualization:
    #         # 1. Add channel dimension in the right place
    #         reconstruction = reconstruction.unsqueeze(0)  # [1,4,1,96,96]
    #         # 2. Rearrange to [B,C,H,W,D] format
    #         reconstruction_viz = reconstruction.permute(0,2,3,4,1)  # [1,1,96,96,4]
    #         print("Visualization shape:", reconstruction_viz.shape)

    #         # Visualize using the provided method
    #         print("\nVisualing reconstructed slices:")
    #         visualize_slices(reconstruction_viz.cpu(), title="Model Output")

    #         # You can also visualize the original for comparison
    #         print("\nVisualing original slices:")
    #         visualize_slices(data_orig.cpu(), title="Original Input")
    #     #     # Combine all reconstructions back into a volume
    #     # reconstruction = torch.stack(all_reconstructions, dim=-1)
    #     # print("\nFinal reconstruction shape:", reconstruction.shape)
    #         # # Apply post-processing
    #         # if not cfg.get('resizedEvaluation', False):
    #         #     reconstruction = F.interpolate(
    #         #         reconstruction.unsqueeze(0), 
    #         #         size=cfg.get('new_size', [160,190,160]), 
    #         #         mode="trilinear",
    #         #         align_corners=True
    #         #     ).squeeze()
    #         # else:
    #         #     reconstruction = reconstruction.squeeze()

    #         # # Apply median filtering if configured
    #         # if cfg.get('medianFiltering', False):
    #         #     reconstruction = torch.from_numpy(
    #         #         apply_3d_median_filter(
    #         #             reconstruction.cpu().numpy(),
    #         #             kernelsize=cfg.get('kernelsize_median', 5)
    #         #         )
    #         #     ).to(reconstruction.device)

    #         # # For visualization, transform back to original format
    #         # reconstruction_viz = reconstruction.unsqueeze(0).permute(0,2,3,4,1)
    #         # visualize_slices(reconstruction_viz.cpu(), title="Model Output")
            


    #     # # Run inference
    #     # with torch.no_grad():
    #     #     if torch.cuda.is_available():
    #     #         input = input.cuda()
            
    #     #     # Get features
    #     #     features = model(input)
            
    #     #     # Now we handle both noise approaches
    #     #     if cfg.get('noise_ensemble', False):
    #     #         print("Using noise ensemble approach...")
    #     #         # Get timesteps from config, with a default if not specified
    #     #         timesteps = cfg.get('step_ensemble', [250, 500, 750])
    #     #         reco_ensemble = torch.zeros_like(input)
                
    #     #         # Loop through different timesteps
    #     #         for t in timesteps:
    #     #             noise = None
    #     #             if cfg.get('noisetype') is not None:
    #     #                 noise = gen_noise(cfg, input.shape).to(input.device)
                    
    #     #             loss_diff, reco, unet_details = model.diffusion(
    #     #                 input,
    #     #                 cond=features,
    #     #                 t=t-1,
    #     #                 noise=noise
    #     #             )
    #     #             reco_ensemble += reco
                    
    #     #         # Average the reconstructions
    #     #         reconstruction = reco_ensemble / len(timesteps)
                
    #     #     else:
    #     #         print("Using single noise approach...")
    #     #         # Single noise generation
    #     #         noise = None
    #     #         if cfg.get('noisetype') is not None:
    #     #             noise = gen_noise(cfg, input.shape).to(input.device)
                    
    #     #         loss_diff, reconstruction, unet_details = model.diffusion(
    #     #             input,
    #     #             cond=features,
    #     #             t=model.test_timesteps-1,
    #     #             noise=noise
    #     #         )

    #     #     print("Reconstruction shape:", reconstruction.shape)
    #     #     #! reverse earlier wierdness
    #     #     # Earlier we transformed the input from [1,1,96,96,4] -> [4,1,96,96]
    #     #     # Now our reconstruction is in shape [4,1,96,96]
    #     #     # We need to transform it back to the original format for visualization

    #     #     # First, let's understand what happened:
    #     #     print("\nModel reconstruction:")
    #     #     print("Current reconstruction shape:", reconstruction.shape)  # [4,1,96,96]

    #     #     # To visualize properly, we need to:
    #     #     # 1. Add batch dimension
    #     #     # 2. Move the slice dimension back to the end
    #     #     reconstruction_viz = reconstruction.unsqueeze(0)  # Add batch: [4,1,96,96] -> [1,4,1,96,96]
    #     #     reconstruction_viz = reconstruction_viz.permute(0,2,3,4,1)  # Reorder: [1,4,1,96,96] -> [1,1,96,96,4]

    #     #     # Now we can visualize
    #     #     print("Visualization shape:", reconstruction_viz.shape)
    #     #     visualize_slices(reconstruction_viz.cpu(), title="Model Output")
            
        
    #     # Store results as before
    #     results.append({
    #         'ID': batch['ID'],
    #         'input': input,
    #         'reconstruction': reconstruction,
    #         'original': data_orig,
    #         'mask': data_mask
    #     })
    
    
    # print("Testing completed!")
    # return results