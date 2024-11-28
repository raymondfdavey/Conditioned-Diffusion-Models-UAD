import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import os
import sys
import torch
import warnings
from pytorch_lightning import LightningDataModule, Trainer
from src.utils import utils
import pickle
import wandb 

# Configure paths to model checkpoints - modify these for your setup
ENCODER_PATH = "path/to/encoder/checkpoint.ckpt"  # Path to pretrained encoder weights if using cDDPM
MODEL_PATH = "path/to/model/checkpoint.ckpt"      # Path to main model checkpoint

@hydra.main(config_path="configs/", config_name="config.yaml")
def test(cfg: DictConfig):
    """Main testing function using the DiffusionModelTester."""
    cfg.onlyEval = True
    
    # Initialize tester
    tester = DiffusionModelTester(
        cfg=cfg,
        model_path=MODEL_PATH,
        encoder_path=ENCODER_PATH
    )
    
    # Set up data module
    tester.setup_datamodule('IXI')
    
    # Run tests
    results = tester.test_dataset()
    
    # Save results
    tester.save_results()
    
if __name__ == "__main__":
    test()
    

class DiffusionModelTester:
    """
    A dedicated testing framework for diffusion models that separates testing logic 
    from model implementation while maintaining compatibility with the original setup.
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
        self.setup_model(model_path, encoder_path)
        self.results = {'test': {}}
        
    def setup_model(self, model_path: str, encoder_path: str = None):
        """Set up the model with proper configuration and checkpoint loading."""
        # Configure encoder settings if needed
        if encoder_path:
            self.cfg.model.cfg.encoder_path = encoder_path
            self.cfg.model.cfg.pretrained_encoder = True
            
        # Instantiate model
        self.cfg.model._target_ = f'src.models.{self.cfg.model._target_}'
        self.model = hydra.utils.instantiate(self.cfg.model)
        
        # Load checkpoint
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['state_dict']
        missing_keys, unexpected_keys = self.model.load_state_dict(
            state_dict, strict=False
        )
        
        # Log loading issues
        if missing_keys or unexpected_keys:
            print("State dict loading issues:")
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")
            
        self.model.eval()
        
    def setup_datamodule(self, dataset_name: str):
        """Initialize and set up the data module for testing."""
        self.cfg.datamodule._target_ = f'src.datamodules.{dataset_name}'
        self.datamodule = hydra.utils.instantiate(self.cfg.datamodule)
        self.datamodule.setup()
        
    def run_inference(self, batch):
        """
        Run model inference on a single batch, separate from PyTorch Lightning framework.
        """
        with torch.no_grad():
            # Extract the core model operations
            input_data = batch['vol'][tio.DATA]
            features = self.model.forward(input_data)
            
            # Handle noise generation
            noise = None
            if self.cfg.get('noisetype') is not None:
                noise = gen_noise(self.cfg, input_data.shape)
                
            # Get reconstruction
            loss_diff, reconstruction, unet_details = self.model.diffusion(
                input_data, 
                cond=features,
                t=self.model.test_timesteps-1,
                noise=noise
            )
            
            return reconstruction, loss_diff, unet_details
            
    def evaluate_batch(self, batch):
        """
        Evaluate a single batch and compute metrics.
        """
        reconstruction, loss_diff, unet_details = self.run_inference(batch)
        
        # Calculate metrics
        metrics = self.compute_metrics(
            input_image=batch['vol'][tio.DATA],
            reconstruction=reconstruction,
            original=batch['vol_orig'][tio.DATA],
            mask=batch['mask_orig'][tio.DATA],
            loss=loss_diff
        )
        
        return {
            'reconstruction': reconstruction,
            'metrics': metrics,
            'unet_details': unet_details
        }
        
    def test_dataset(self, trainer: Trainer = None):
        """
        Run comprehensive testing on the dataset.
        """
        if trainer is None:
            trainer = Trainer(
                accelerator=self.cfg.trainer.get('accelerator', 'gpu'),
                devices=self.cfg.trainer.get('devices', 1),
                precision=self.cfg.trainer.get('precision', 32)
            )
            
        # Run evaluation
        trainer.test(
            model=self.model, 
            dataloaders=self.datamodule.test_eval_dataloader()
        )
        
        # Store results
        self.results['test']['IXI'] = trainer.lightning_module.eval_dict
        
        return self.results
        
    def save_results(self, output_dir: str = None):
        """Save test results and predictions."""
        if output_dir is None:
            output_dir = os.getcwd()
            
        output_path = os.path.join(output_dir, 'test_results.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Saved test results to {output_path}")