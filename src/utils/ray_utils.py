import torchvision.transforms as transforms
import torch
import numpy as np
import wandb
import os
import matplotlib.pyplot as plt
import scipy
from PIL import Image
import matplotlib.colors as colors
import scipy
from scipy import ndimage
import torchio as tio


def add_processed_features(storage):
    """
    Add processed features to the existing storage dictionary
    """
    # Process features using both approaches
    features_approach1 = approach1_feature_statistics(storage)
    features_approach2 = approach2_feature_statistics(storage)
    
    # Add processed features to storage
    storage['features_processed'] = {
        'approach1': features_approach1,
        'approach2': features_approach2
    }
    
    return storage

def approach1_feature_statistics(storage):
    """
    Approach 1: Calculate mean and variance across runs for each position in the feature maps.
    For each slice and each layer, produces two volumes of the same dimension as the original feature map.
    
    Args:
        storage (dict): The storage dictionary containing feature maps
        
    Returns:
        dict: Dictionary containing mean and variance volumes for each layer and slice
    """
    n_slices = storage['features']['down_post_1'].shape[1]
    results = {}
    
    # Process each layer in the features
    for layer_name, feature_tensor in storage['features'].items():
        results[layer_name] = {
            'mean': [],
            'variance': []
        }
        
        # Process each slice separately
        for slice_idx in range(n_slices):
            # Get all runs for this slice
            slice_data = feature_tensor[:, slice_idx, ...]  # Shape: [n_runs, channels, height, width]
            
            # Calculate mean and variance across runs
            # Mean shape will be [channels, height, width]
            mean_volume = torch.mean(slice_data, dim=0)
            variance_volume = torch.var(slice_data, dim=0)
            
            results[layer_name]['mean'].append(mean_volume)
            results[layer_name]['variance'].append(variance_volume)
            
        # Stack slices together
        results[layer_name]['mean'] = torch.stack(results[layer_name]['mean'])
        results[layer_name]['variance'] = torch.stack(results[layer_name]['variance'])
    
    return results

def approach2_feature_statistics(storage):
    """
    Approach 2: For each run, calculate mean and variance across the feature volume,
    then average these statistics across runs.
    
    Args:
        storage (dict): The storage dictionary containing feature maps
        
    Returns:
        dict: Dictionary containing averaged mean and variance maps for each layer and slice
    """
    n_slices = storage['features']['down_post_1'].shape[1]
    results = {}
    
    # Process each layer in the features
    for layer_name, feature_tensor in storage['features'].items():
        results[layer_name] = {
            'mean_map': [],
            'variance_map': []
        }
        
        # Process each slice separately
        for slice_idx in range(n_slices):
            slice_data = feature_tensor[:, slice_idx, ...]  # Shape: [n_runs, channels, height, width]
            
            # For each run, calculate mean and variance across channels
            run_means = torch.mean(slice_data, dim=1)  # Shape: [n_runs, height, width]
            run_variances = torch.var(slice_data, dim=1)  # Shape: [n_runs, height, width]
            
            # Average the means and variances across runs
            final_mean_map = torch.mean(run_means, dim=0)  # Shape: [height, width]
            final_variance_map = torch.mean(run_variances, dim=0)  # Shape: [height, width]
            
            results[layer_name]['mean_map'].append(final_mean_map)
            results[layer_name]['variance_map'].append(final_variance_map)
        
        # Stack slices together
        results[layer_name]['mean_map'] = torch.stack(results[layer_name]['mean_map'])
        results[layer_name]['variance_map'] = torch.stack(results[layer_name]['variance_map'])
    
    return results










def generate_and_save_synthetic_anomaly(save_dir):
    """
    Creates synthetic patterns and saves them as PNG files that can be used
    in place of tumor images
    """
    import torch
    import os
    from PIL import Image
    import numpy as np
    
    def create_synthetic_pattern(size=96, pattern_type='circles'):
        pattern = torch.zeros((size, size))
        mask = torch.zeros((size, size))
        
        center = size // 2
        
        if pattern_type == 'circles':
            # Create concentric circles with different intensities
            for r in [30, 20, 10]:
                for i in range(size):
                    for j in range(size):
                        dist = ((i - center) ** 2 + (j - center) ** 2) ** 0.5
                        if abs(dist - r) < 1:
                            pattern[i, j] = 0.8
                            mask[i, j] = 1
        
            # Add a central square
            square_size = 15
            start = center - square_size // 2
            end = start + square_size
            pattern[start:end, start:end] = 1.0
            mask[start:end, start:end] = 1
            
        elif pattern_type == 'grid':
            # Create a grid pattern
            for i in range(0, size, 10):
                pattern[i:i+2, :] = 0.8
                pattern[:, i:i+2] = 0.8
                mask[i:i+2, :] = 1
                mask[:, i:i+2] = 1
                
        # Add the diagonal lines
        for i in range(size):
            if abs(i - center) < 30:  # Only add diagonals near center
                pattern[i, i] = 0.6
                pattern[i, size-1-i] = 0.6
                mask[i, i] = 1
                mask[i, size-1-i] = 1
        
        return pattern, mask
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate and save multiple pattern types
    patterns = ['circles', 'grid']
    saved_files = []
    
    for idx, pattern_type in enumerate(patterns):
        pattern, pattern_mask = create_synthetic_pattern(96, pattern_type)
        
        # Convert to numpy arrays and scale to 0-255
        pattern_np = (pattern.numpy() * 255).astype(np.uint8)
        mask_np = (pattern_mask.numpy() * 255).astype(np.uint8)
        
        # Create PIL images
        pattern_img = Image.fromarray(pattern_np)
        mask_img = Image.fromarray(mask_np)
        
        # Save files
        pattern_path = os.path.join(save_dir, f'synthetic_tumor_{idx}.png')
        mask_path = os.path.join(save_dir, f'synthetic_mask_{idx}.png')
        
        pattern_img.save(pattern_path)
        mask_img.save(mask_path)
        
        saved_files.append((pattern_path, mask_path))
        
    return saved_files


def create_synthetic_batch(device):
    """
    Creates a synthetic batch matching torchio format for testing
    Returns dict with same structure as regular batch
    """
    
    # Create synthetic pattern
    def create_pattern(size=96):
        pattern = torch.zeros((size, size))
        
        # Create concentric circles
        center = size // 2
        for r in [30, 20, 10]:
            for i in range(size):
                for j in range(size):
                    dist = ((i - center) ** 2 + (j - center) ** 2) ** 0.5
                    if abs(dist - r) < 1:
                        pattern[i, j] = 0.8
        
        # Add some diagonal lines
        for i in range(size):
            pattern[i, i] = 0.6
            pattern[i, size-1-i] = 0.6
            
        # Add a central square
        square_size = 15
        start = center - square_size // 2
        end = start + square_size
        pattern[start:end, start:end] = 1.0
        
        return pattern
    
    # Create multiple slices
    n_slices = 4
    pattern_volume = torch.zeros((1, 1, 96, 96, n_slices))
    mask_volume = torch.zeros((1, 1, 96, 96, n_slices))
    
    # Fill each slice with slightly different patterns
    for i in range(n_slices):
        pattern = create_pattern()
        pattern_volume[0, 0, :, :, i] = pattern
        
        # Create mask (1 where pattern > 0)
        mask_volume[0, 0, :, :, i] = (pattern > 0).float()
    
    # Create batch dictionary matching your expected format
    batch = {
        'Dataset': ['synthetic'],
        'vol': {tio.DATA: pattern_volume.to(device)},
        'vol_orig': {tio.DATA: pattern_volume.to(device)},
        'seg_orig': {tio.DATA: mask_volume.to(device)},
        'mask_orig': {tio.DATA: mask_volume.to(device)},
        'ID': ['synthetic_test'],
        'seg_available': True,
        'stage': 'test'  # Added missing stage field
    }
    
    return batch

def setup_storage(n_runs, n_slices):

    storage = {
        'input': torch.zeros(1, n_slices, 96, 96),  # Removed the channel dimension
        'features_raw_embedding': torch.zeros(1, n_slices, 512), 
        'full_raw_embedding': torch.zeros(1, n_slices, 1024),
        'brain_mask': torch.zeros(1, n_slices, 96, 96),     # Brain masks
        'segmentation_mask': torch.zeros(1, n_slices, 96, 96),   # Segmentation masks
        
        'noisy_images': torch.zeros(n_runs, n_slices, 96, 96), # Reconstructed images
        'reconstructions': torch.zeros(n_runs, n_slices, 96, 96), # Reconstructed images
        'differences': torch.zeros(n_runs, n_slices, 96, 96),     # Difference maps
        
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
   
def save_images_from_repeat(self, input, run_idx, noise, noisy_image, diff_volume, segmentation_mask, data_mask, final_volume, SCAN_ID):
    print('='*10)
    print(f'Saving iteration {run_idx+1} images...')
    
    # Move tensors to CPU here
    diff_volume = diff_volume.cpu()
    input = input.cpu()
    noise = noise.cpu()
    noisy_image = noisy_image.cpu()
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
        'imagesGrid': os.getcwd()
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
        
        ax[1].imshow(noisy_image[j, 0].rot90(3), 'gray')
        ax[1].set_title('Noisy_image')
        
        ax[2].imshow(final_volume[..., j].rot90(3).squeeze(), 'gray')
        ax[2].set_title('Reconstructed')
        
        ax[3].imshow(diff_volume.squeeze()[:, ..., j].rot90(3), 'inferno',
                    norm=colors.Normalize(vmin=0, vmax=diff_volume.max()+.01))
        ax[3].set_title('Difference')
                
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
                           f'{SCAN_ID[0]}_RUN_{run_idx}_SLICE_{j}_Grid.png'),
                bbox_inches='tight'
            )
            
        plt.clf()
        plt.cla()
        plt.close()
    print(f"Images saved to {ImagePathList['imagesGrid']}")
    print('='*10)

def save_images(self, diff_volume, data_orig, data_seg, data_mask, final_volume, SCAN_ID, diff_volume_KL=None,  flow=None ):
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
        
        plt.savefig(os.path.join(ImagePathList['imagesGrid'], '{}_{}_Grid.png'.format(SCAN_ID[0],j)),bbox_inches='tight')

        # self.logger.experiment[0].log({'images/{}/{}_Grid.png'.format(self.dataset[0],j) : wandb.Image(plt)})
        plt.clf()
        plt.cla()
        plt.close()
    print('='*10)
    print(f"IMAGES SAVED TO {ImagePathList['imagesGrid']}")
    print('='*10)
       
def print_key_details(key_points, indent):
    for key, value in key_points.items():
        print("  " * indent + f"📍 {key}")
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if subkey != 'tensor':
                    print("  " * (indent+1) + f"└─ {subkey}: {subvalue}")
        print()  # Add blank line between sections

def log_images_with_patch(self, diff_volume, data_orig, data_seg, data_mask, final_volume, patched_input, patch_mask, SCAN_ID, diff_volume_KL=None, flow=None):
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
                                   '{}_{}_{}_Grid.png'.format(SCAN_ID[0], j, 'patched')),
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

def load_and_preprocess_tumor(tumor1_path, mask1_path, tumor2_path, mask2_path, device, scale_factor=1.5, erosion_pixels=2, rotation=90):
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
    tumor1 = tumor1.rotate(rotation, expand=True)
    mask1 = mask1.rotate(rotation, expand=True)
    tumor2 = tumor2.rotate(rotation, expand=True)
    mask2 = mask2.rotate(rotation, expand=True)
    
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
