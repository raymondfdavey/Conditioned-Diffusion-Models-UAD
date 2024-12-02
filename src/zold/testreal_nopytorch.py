import torch
from torch.utils.data import DataLoader
import hydra
import torchio as tio
import os
import pickle

class DiffusionModelTester:
    """
    A dedicated testing framework for diffusion models using pure PyTorch
    """
    def __init__(self, cfg: DictConfig, model_path: str, encoder_path: str = None):
        """
        Initialize the tester with configuration and model paths.
        
        Args:
            cfg: Configuration dictionary
            model_path: Path to the main model checkpoint
            encoder_path: Path to the encoder checkpoint (for cDDPM)
        """
        self.cfg = cfg
        self.model = self._setup_model(model_path, encoder_path)
        self.eval_dict = {}
        
    def _setup_model(self, model_path: str, encoder_path: str = None):
        """Set up the model with proper configuration and checkpoint loading."""
        # Configure encoder settings if needed
        if encoder_path:
            self.cfg.model.cfg.encoder_path = encoder_path
            self.cfg.model.cfg.pretrained_encoder = True
            
        # Instantiate model
        self.cfg.model._target_ = f'src.models.{self.cfg.model._target_}'
        model = hydra.utils.instantiate(self.cfg.model)
        
        # Load checkpoint
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['state_dict']
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict, strict=False
        )
        
        # Log loading issues
        if missing_keys or unexpected_keys:
            print("State dict loading issues:")
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")
            
        model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            
        return model

    def run_inference(self, batch):
        """Run model inference on a single batch."""
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            # Move data to device
            input_data = batch['vol'][tio.DATA].to(device)
            features = self.model.forward(input_data)
            
            # Handle noise generation
            noise = None
            if self.cfg.get('noisetype') is not None:
                noise = self.gen_noise(input_data.shape).to(device)
                
            # Get reconstruction
            loss_diff, reconstruction, unet_details = self.model.diffusion(
                input_data, 
                cond=features,
                t=self.model.test_timesteps-1,
                noise=noise
            )
            
            return reconstruction, loss_diff, unet_details

    def evaluate_batch(self, batch, batch_idx):
        """Evaluate a single batch and compute metrics."""
        reconstruction, loss_diff, unet_details = self.run_inference(batch)
        device = next(self.model.parameters()).device
        
        # Calculate metrics
        metrics = self.compute_metrics(
            input_image=batch['vol'][tio.DATA].to(device),
            reconstruction=reconstruction,
            original=batch['vol_orig'][tio.DATA].to(device),
            mask=batch['mask_orig'][tio.DATA].to(device),
            loss=loss_diff
        )
        
        # Store results
        self.eval_dict[f'batch_{batch_idx}'] = {
            'reconstruction': reconstruction.cpu(),
            'metrics': metrics,
            'unet_details': unet_details
        }
        
        return metrics

    def test_dataset(self, dataloader):
        """Run testing on the entire dataset."""
        self.model.eval()
        
        print("Starting testing...")
        for batch_idx, batch in enumerate(dataloader):
            metrics = self.evaluate_batch(batch, batch_idx)
            
            if batch_idx % self.cfg.testing.log_every_n_steps == 0:
                print(f"Processed batch {batch_idx}/{len(dataloader)}")
                print("Current metrics:", {k: f"{v:.4f}" for k, v in metrics.items()})
        
        # Calculate and store average metrics
        all_metrics = {}
        for batch_results in self.eval_dict.values():
            if isinstance(batch_results, dict) and 'metrics' in batch_results:
                for metric_name, value in batch_results['metrics'].items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(value)
        
        avg_metrics = {
            metric_name: torch.tensor(values).mean().item()
            for metric_name, values in all_metrics.items()
        }
        
        self.eval_dict['average_metrics'] = avg_metrics
        
        # Print summary
        print("\nTest Results:")
        for metric_name, value in avg_metrics.items():
            print(f"{metric_name}: {value:.4f}")

    def save_results(self, output_dir: str = None):
        """Save test results and predictions."""
        if output_dir is None:
            output_dir = os.getcwd()
            
        output_path = os.path.join(output_dir, 'test_results.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump({'test': {'IXI': self.eval_dict}}, f)
        print(f"Saved test results to {output_path}")

def test_model(cfg: DictConfig, model_path: str, encoder_path: str = None, output_dir: str = None):
    """Main function to run testing."""
    # Initialize tester
    tester = DiffusionModelTester(cfg, model_path, encoder_path)
    
    # Setup dataloader
    dataset = hydra.utils.instantiate(cfg.datamodule)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.testing.batch_size,
        num_workers=cfg.testing.num_workers,
        shuffle=False
    )
    
    # Run testing
    tester.test_dataset(dataloader)
    
    # Save results
    if output_dir:
        tester.save_results(output_dir)
    
    return tester.eval_dict

#! functional

import torch
from torch.utils.data import DataLoader
import hydra
import torchio as tio
import os
import pickle

def setup_model(cfg, model_path: str, encoder_path: str = None):
    """Set up the model with proper configuration and checkpoint loading."""
    # Configure encoder settings if needed
    if encoder_path:
        cfg.model.cfg.encoder_path = encoder_path
        cfg.model.cfg.pretrained_encoder = True
        
    # Instantiate model
    cfg.model._target_ = f'src.models.{cfg.model._target_}'
    model = hydra.utils.instantiate(cfg.model)
    
    # Load checkpoint
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['state_dict']
    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict, strict=False
    )
    
    # Log loading issues
    if missing_keys or unexpected_keys:
        print("State dict loading issues:")
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        
    model.eval()
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        
    return model

def setup_dataloader(cfg):
    """Initialize the dataset and create a dataloader."""
    dataset = hydra.utils.instantiate(cfg.datamodule)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.testing.batch_size,
        num_workers=cfg.testing.num_workers,
        shuffle=False
    )
    return dataloader

def run_inference(model, batch, cfg):
    """Run model inference on a single batch."""
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Move data to device
        input_data = batch['vol'][tio.DATA].to(device)
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
        'unet_details': unet_details
    }

def test_dataset(model, dataloader, cfg):
    """Run testing on the entire dataset."""
    model.eval()
    batch_results = {}
    
    print("Starting testing...")
    for batch_idx, batch in enumerate(dataloader):
        batch_output = evaluate_batch(model, batch, cfg)
        batch_results[f'batch_{batch_idx}'] = batch_output
        
        if batch_idx % cfg.testing.log_every_n_steps == 0:
            print(f"Processed batch {batch_idx}/{len(dataloader)}")
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

def test_diffusion_model(cfg: DictConfig, model_path: str, encoder_path: str = None, output_dir: str = None):
    """Main function to run the entire testing pipeline."""
    # Setup
    model = setup_model(cfg, model_path, encoder_path)
    dataloader = setup_dataloader(cfg)
    
    # Run testing
    results = test_dataset(model, dataloader, cfg)
    
    # Save results if output directory is provided
    if output_dir:
        save_results(results, output_dir)
    
    return results

# Example usage:
if __name__ == "__main__":
    # Assuming cfg is loaded via hydra
    results = test_diffusion_model(
        cfg=cfg,
        model_path="path/to/model.ckpt",
        encoder_path="path/to/encoder.ckpt",
        output_dir="path/to/output"
    )