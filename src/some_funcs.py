import matplotlib.pyplot as plt

def visualize_comparison(input, final_volume, slice_idx=50):
    """
    Visualizes a slice comparison between the input and the output (final_volume).
    
    Args:
    - input: Original input data, shape [D, C, H, W]
    - final_volume: Reconstructed output, shape [D, C, H, W]
    - slice_idx: The index of the slice to display (z-axis)
    """
    
    # Select slice along depth (z-axis)
    input_slice = input[slice_idx].cpu().detach().numpy()
    output_slice = final_volume[slice_idx].cpu().detach().numpy()
    
    # Plot the input and output slices side by side
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(input_slice[0], cmap='gray')  # Channel 0 of input (you might want to adjust for multi-channel data)
    ax[0].set_title("Input Slice")
    ax[1].imshow(output_slice[0], cmap='gray')  # Channel 0 of output
    ax[1].set_title("Reconstructed Output Slice")
    
    for a in ax:
        a.axis('off')
    
    plt.show()

# Call the function with a specific slice index (e.g., 50)
visualize_comparison(input, final_volume, slice_idx=50)

import numpy as np

def visualize_difference(input, final_volume, slice_idx=50):
    """
    Visualizes the difference between input and output (final_volume).
    
    Args:
    - input: Original input data, shape [D, C, H, W]
    - final_volume: Reconstructed output, shape [D, C, H, W]
    - slice_idx: The index of the slice to display (z-axis)
    """
    
    # Select slice along depth (z-axis)
    input_slice = input[slice_idx].cpu().detach().numpy()
    output_slice = final_volume[slice_idx].cpu().detach().numpy()
    
    # Compute difference (absolute difference between input and output)
    diff_slice = np.abs(input_slice - output_slice)
    
    # Plot the difference slice
    plt.figure(figsize=(6, 6))
    plt.imshow(diff_slice[0], cmap='hot')  # Show the difference in heatmap
    plt.title(f"Difference between Input and Output (Slice {slice_idx})")
    plt.axis('off')
    plt.show()

# Call the function to visualize the difference for a particular slice
visualize_difference(input, final_volume, slice_idx=50)


def visualize_multiple_slices(input, final_volume, slice_range=range(40, 60)):
    """
    Visualizes a range of slices from input and output volumes.
    
    Args:
    - input: Original input data, shape [D, C, H, W]
    - final_volume: Reconstructed output, shape [D, C, H, W]
    - slice_range: The range of slices to display
    """
    
    n_slices = len(slice_range)
    fig, axes = plt.subplots(n_slices, 2, figsize=(10, 5 * n_slices))
    
    for idx, slice_idx in enumerate(slice_range):
        input_slice = input[slice_idx].cpu().detach().numpy()
        output_slice = final_volume[slice_idx].cpu().detach().numpy()
        
        # Display input and output side-by-side
        axes[idx, 0].imshow(input_slice[0], cmap='gray')
        axes[idx, 0].set_title(f"Input Slice {slice_idx}")
        axes[idx, 1].imshow(output_slice[0], cmap='gray')
        axes[idx, 1].set_title(f"Output Slice {slice_idx}")
        
        for ax in axes[idx]:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Call the function with a range of slices (e.g., 40 to 60)
visualize_multiple_slices(input, final_volume, slice_range=range(40, 60))


import pyvista as pv

def visualize_3d(input, final_volume, slice_idx=50):
    """
    Visualizes the 3D volumes (input vs output) using PyVista.
    """
    input_slice = input[slice_idx].cpu().detach().numpy()
    output_slice = final_volume[slice_idx].cpu().detach().numpy()

    # Create 3D grid for input and output
    grid_input = pv.StructuredGrid(*np.meshgrid(np.arange(input_slice.shape[0]),
                                               np.arange(input_slice.shape[1]),
                                               np.arange(input_slice.shape[2])))
    grid_output = pv.StructuredGrid(*np.meshgrid(np.arange(output_slice.shape[0]),
                                                np.arange(output_slice.shape[1]),
                                                np.arange(output_slice.shape[2])))

    # Add the data as point arrays
    grid_input.point_arrays["input"] = input_slice.flatten()
    grid_output.point_arrays["output"] = output_slice.flatten()

    # Plot the volumes
    plotter = pv.Plotter()
    plotter.add_mesh(grid_input, scalars="input", cmap="coolwarm", opacity=0.5)
    plotter.add_mesh(grid_output, scalars="output", cmap="coolwarm", opacity=0.5)
    plotter.show()

# Call the function for 3D visualization
visualize_3d(input, final_volume, slice_idx=50)

