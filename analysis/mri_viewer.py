import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from ipywidgets import interactive, IntSlider, Dropdown, VBox, HBox
import os
from glob import glob
from IPython.display import display, clear_output

def get_valid_files(directory):
    """Get all valid .nii.gz files, skipping hidden files"""
    all_files = glob(os.path.join(directory, "*.nii.gz"))
    valid_files = [f for f in all_files if not os.path.basename(f).startswith('._')]
    return sorted(valid_files)

class MRIViewer:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.t1_files = get_valid_files(os.path.join(base_dir, "t1"))
        self.seg_files = get_valid_files(os.path.join(base_dir, "seg"))
        
        if not self.t1_files:
            raise ValueError("No valid T1 files found")
            
        # Load first image to get dimensions
        first_img = nib.load(self.t1_files[0]).get_fdata()
        self.max_layer = first_img.shape[2] - 1
        
        # Create the widgets
        self.file_slider = IntSlider(
            min=0,
            max=len(self.t1_files)-1,
            step=1,
            description='File:',
            continuous_update=False  # Only update when slider is released
        )
        
        self.layer_slider = IntSlider(
            min=0,
            max=self.max_layer,
            step=1,
            value=self.max_layer//2,
            description='Layer:',
            continuous_update=False
        )
        
        self.view_dropdown = Dropdown(
            options=['T1', 'Segmentation'],
            description='Type:',
            value='T1'
        )
        
    def plot_scan(self, file_idx, layer, view_type):
        """Function to plot the scan"""
        try:
            if view_type == 'T1':
                file_path = self.t1_files[file_idx]
                img = nib.load(file_path).get_fdata()
                title = f'T1 Image - {os.path.basename(file_path)} - Layer {layer}'
                if len(img.shape) == 4:
                    display_data = img[:, :, layer, 0]
                else:
                    display_data = img[:, :, layer]
                cmap = 'gray'
            else:
                file_path = self.seg_files[file_idx]
                img = nib.load(file_path).get_fdata()
                display_data = img[:, :, layer]
                title = f'Segmentation - {os.path.basename(file_path)} - Layer {layer}'
                cmap = 'viridis'
            
            # Clear the current figure
            plt.clf()
            
            # Create new figure
            plt.figure(figsize=(10, 5))
            plt.imshow(display_data, cmap=cmap)
            plt.title(title)
            plt.axis('off')
            plt.show()
            
            # Print info below the plot
            print(f"Displaying {os.path.basename(file_path)}")
            print(f"Image shape: {img.shape}")
            
        except Exception as e:
            print(f"Error plotting scan: {e}")
    
    def create_viewer(self):
        """Create the interactive viewer"""
        # Create the interactive wrapper around plot_scan
        interactive_plot = interactive(
            self.plot_scan,
            file_idx=self.file_slider,
            layer=self.layer_slider,
            view_type=self.view_dropdown
        )
        
        # Arrange widgets horizontally
        widgets_box = HBox([self.file_slider, self.layer_slider, self.view_dropdown])
        
        # Stack widgets and output vertically
        viewer_box = VBox([widgets_box, interactive_plot.children[-1]])
        
        return viewer_box

def interactive_viewer(base_dir):
    """Main function to launch the viewer"""
    try:
        print(f"Loading viewer for: {base_dir}")
        viewer = MRIViewer(base_dir)
        viewer_box = viewer.create_viewer()
        display(viewer_box)
        # Trigger initial plot
        viewer.plot_scan(0, viewer.max_layer//2, 'T1')
    except Exception as e:
        print(f"Error creating viewer: {e}")
        raise