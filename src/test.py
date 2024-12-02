import torch
import hydra
import torchio as tio
import os
import pickle
import numpy as np
from src.datamodules.new_IXI import IXI_new
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf, open_dict

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

# def visualize_slice_selection(input_tensor, orig_tensor, mask_tensor, slice_indices=None, stage_name=""):
#     """
#     Visualizes the selected slices from the MRI volume at different stages of processing.
    
#     Args:
#         input_tensor: The input tensor (processed) [B, C, H, W, D]
#         orig_tensor: The original tensor [B, C, H, W, D]
#         mask_tensor: The mask tensor [B, C, H, W, D]
#         slice_indices: List of specific slice indices to visualize
#         stage_name: Name of the processing stage (for the plot title)
#     """
#     import matplotlib.pyplot as plt
#     import torch
    
#     # Convert tensors to CPU and get numpy arrays
#     input_np = input_tensor[0, 0].cpu().numpy()  # Remove batch and channel dims
#     orig_np = orig_tensor[0, 0].cpu().numpy()
#     mask_np = mask_tensor[0, 0].cpu().numpy()
    
#     # If no specific indices provided, use all available slices
#     if slice_indices is None:
#         slice_indices = list(range(input_np.shape[-1]))
    
#     n_slices = len(slice_indices)
    
#     # Create a figure with three rows (input, original, and mask)
#     fig, axes = plt.subplots(3, n_slices, figsize=(4*n_slices, 12))
    
#     # Add a title for the entire figure
#     fig.suptitle(f'Slice Visualization - {stage_name}\nSlice Indices: {slice_indices}', fontsize=16, y=1.02)
    
#     # Row titles
#     row_titles = ['Processed Input', 'Original', 'Mask Overlay']
    
#     for row_idx, (data, title) in enumerate([(input_np, 'Processed Input'),
#                                            (orig_np, 'Original'),
#                                            (mask_np, 'Mask')]):
#         # Add row title
#         fig.text(0.02, 0.75 - row_idx*0.3, title, rotation=90, fontsize=12)
        
#         for col_idx, slice_idx in enumerate(slice_indices):
#             ax = axes[row_idx, col_idx]
            
#             if row_idx < 2:  # For input and original data
#                 ax.imshow(data[..., slice_idx], cmap='gray')
#             else:  # For mask overlay
#                 ax.imshow(orig_np[..., slice_idx], cmap='gray')
#                 ax.imshow(data[..., slice_idx], cmap='Reds', alpha=0.3)
            
#             ax.axis('off')
#             ax.set_title(f'Slice {slice_idx}')
    
#     plt.tight_layout()
#     plt.show()
    
#     # Print tensor information
#     print(f"\n=== Tensor Information at {stage_name} ===")
#     print(f"Input tensor shape: {input_tensor.shape}")
#     print(f"Original tensor shape: {orig_tensor.shape}")
#     print(f"Mask tensor shape: {mask_tensor.shape}")
#     print(f"Value ranges:")
#     print(f"  Input: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
#     print(f"  Original: [{orig_tensor.min():.3f}, {orig_tensor.max():.3f}]")
#     print(f"  Mask: [{mask_tensor.min():.3f}, {mask_tensor.max():.3f}]")
    
def debug_batch_visualization(batch, cfg=None, num_eval_slices=None):
    """
    Comprehensive visualization function to debug and understand the data at each stage.
    
    Args:
        batch: The batch from your dataloader
        cfg: Optional configuration object
        num_eval_slices: Optional number of slices to evaluate
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torchio as tio
    
    # Extract data from batch
    input_vol = batch['vol'][tio.DATA]
    orig_vol = batch['vol_orig'][tio.DATA]
    mask_vol = batch['mask_orig'][tio.DATA]
    
    # Get basic information
    print("\n=== Batch Information ===")
    print(f"Subject ID: {batch['ID']}")
    print(f"Age: {batch['age']}")
    print(f"Label: {batch['label']}")
    print(f"Dataset: {batch['Dataset']}")
    print(f"Stage: {batch['stage']}")
    
    print("\n=== Volume Shapes ===")
    print(f"Input volume shape: {input_vol.shape}")
    print(f"Original volume shape: {orig_vol.shape}")
    print(f"Mask shape: {mask_vol.shape}")
    
    print("\n=== Value Ranges ===")
    print(f"Input volume range: [{input_vol.min():.3f}, {input_vol.max():.3f}]")
    print(f"Original volume range: [{orig_vol.min():.3f}, {orig_vol.max():.3f}]")
    print(f"Mask range: [{mask_vol.min():.3f}, {mask_vol.max():.3f}]")
    
    # If we're selecting specific slices
    if num_eval_slices and num_eval_slices != input_vol.size(4):
        start_slice = int((input_vol.size(4) - num_eval_slices) / 2)
        print(f"\n=== Slice Selection ===")
        print(f"Selecting {num_eval_slices} slices starting from index {start_slice}")
        
        input_vol = input_vol[..., start_slice:start_slice+num_eval_slices]
        orig_vol = orig_vol[..., start_slice:start_slice+num_eval_slices]
        mask_vol = mask_vol[..., start_slice:start_slice+num_eval_slices]
        
        print(f"New shapes after slice selection:")
        print(f"Input volume: {input_vol.shape}")
        print(f"Original volume: {orig_vol.shape}")
        print(f"Mask: {mask_vol.shape}")
    
    # Visualization function for a single volume
    def plot_volume(vol, title, max_slices=8):
        n_slices = min(vol.shape[-1], max_slices)
        if n_slices < vol.shape[-1]:
            slice_indices = np.linspace(0, vol.shape[-1]-1, n_slices, dtype=int)
        else:
            slice_indices = range(n_slices)
            
        fig, axes = plt.subplots(1, n_slices, figsize=(3*n_slices, 3))
        if n_slices == 1:
            axes = [axes]
            
        for i, idx in enumerate(slice_indices):
            axes[i].imshow(vol[0, 0, :, :, idx].cpu(), cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f'Slice {idx}')
        fig.suptitle(title)
        plt.show()
    
    print("\n=== Visualizing Volumes ===")
    plot_volume(input_vol, "Input Volume (Processed)")
    plot_volume(orig_vol, "Original Volume")
    plot_volume(mask_vol, "Mask")
    
    # Combined visualization with mask overlay
    middle_slice = input_vol.shape[-1] // 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Show processed input
    axes[0].imshow(input_vol[0, 0, :, :, middle_slice].cpu(), cmap='gray')
    axes[0].set_title('Processed Input')
    axes[0].axis('off')
    
    # Show original
    axes[1].imshow(orig_vol[0, 0, :, :, middle_slice].cpu(), cmap='gray')
    axes[1].set_title('Original')
    axes[1].axis('off')
    
    # Show mask overlay on original
    orig_slice = orig_vol[0, 0, :, :, middle_slice].cpu()
    mask_slice = mask_vol[0, 0, :, :, middle_slice].cpu()
    
    # Create the overlay using a proper colormap
    axes[2].imshow(orig_slice, cmap='gray')
    # Use 'Reds' colormap instead of 'red' for the mask overlay
    axes[2].imshow(mask_slice, cmap='Reds', alpha=0.3)
    axes[2].set_title('Original + Mask Overlay')
    axes[2].axis('off')
    
    plt.suptitle(f'Middle Slice Comparison (Slice {middle_slice})')
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
    
    # Get first batch for debugging
    # first_batch = next(iter(test_loader))
    
    # print("\nDebugging first batch:")
    # # Show what the data looks like before slice selection
    # debug_batch_visualization(first_batch)
    
    # # Show what the data will look like with slice selection
    # print("\nDebugging with slice selection:")
    # debug_batch_visualization(first_batch, num_eval_slices=4)
    
    
    
    
    print("Starting testing...")
    results = []
    
    
    for batch_idx, batch in enumerate(test_loader):
        print(f"Processing batch {batch_idx}...")
        
        # Get input data
        input = batch['vol'][tio.DATA]
        data_orig = batch['vol_orig'][tio.DATA]
        data_mask = batch['mask_orig'][tio.DATA]
        slice_indices = cfg.datamodule.cfg.get('slice_indices', [28, 35, 42, 48])

        # Visualize before slice selection
        print("\nBefore slice selection:")
        # visualize_slices(input, title="Input Volume Before Selection")
        
        print('OINPUT SIZE 4', input.size(4))
        print(f"Selecting slices at indices: {slice_indices}")

        # Select the specified slices using indexing
        input = input[..., slice_indices]  # This will pick exactly the slices we want
        data_orig = data_orig[..., slice_indices]
        data_mask = data_mask[..., slice_indices]

        # Visualize after slice selection
        print("\nAfter slice selection:")
        print(f"Selected slices shape: {input.shape}")
        visualize_slices(input, title=f"Selected Slices (indices {slice_indices})")
            
    #     # Run inference
    #     with torch.no_grad():
    #         if torch.cuda.is_available():
    #             input = input.cuda()
    #         reconstruction = model(input)
            
    #         # Visualize reconstruction if you want
    #         print("\nModel reconstruction:")
    #         visualize_slices(reconstruction.cpu(), title="Model Output")
        
    #     # Store results as before
    #     results.append({
    #         'ID': batch['ID'],
    #         'input': input,
    #         'reconstruction': reconstruction,
    #         'original': data_orig,
    #         'mask': data_mask
    #     })
        
    #     if cfg.datamodule.cfg.debugging and batch_idx >= 2:
    #         print("Debug mode: stopping after 2 batches")
    #         break
    
    # print("Testing completed!")
    # return results