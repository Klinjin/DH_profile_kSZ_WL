import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import lightning.pytorch as pl
from lightning.pytorch import seed_everything, Trainer


@dataclass
class ModelConfig:
    """Configuration for the power spectrum regression model."""
    n_halo_features: int = 6000  # 300 halos * 20 radial bins
    n_power_bins: int = 100  # k-bins in power spectrum
    n_cosmo_params: int = 0  # Set to >0 if using cosmological parameters
    hidden_dims: List[int] = None
    dropout_rate: float = 0.2  # Increased for regularization
    use_residual: bool = True
    use_batch_norm: bool = True  # Add batch normalization
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256, 128]  # Simpler architecture


class PowerSpectrumRegressor(nn.Module):
    """
    Neural network for predicting power spectrum suppression from halo observables.
    Supports optional conditioning on cosmological/astrophysical parameters.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Determine input dimension
        input_dim = config.n_halo_features
        if config.n_cosmo_params > 0:
            input_dim += config.n_cosmo_params
        
        # Build encoder layers with residual connections
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(config.hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            else:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout_rate))
            prev_dim = hidden_dim
        
        self.encoder = nn.ModuleList(layers)
        
        # Residual projection layers (for skip connections)
        self.residual_projections = nn.ModuleList()
        if config.use_residual:
            proj_input_dim = input_dim
            for hidden_dim in config.hidden_dims:
                self.residual_projections.append(
                    nn.Linear(proj_input_dim, hidden_dim)
                )
                proj_input_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(config.hidden_dims[-1], config.n_power_bins)
        
    def forward(self, halo_features: torch.Tensor, 
                cosmo_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            halo_features: (batch_size, n_halo_features)
            cosmo_params: Optional (batch_size, n_cosmo_params)
            
        Returns:
            power_spectrum: (batch_size, n_power_bins)
        """
        # Concatenate cosmological parameters if provided
        if cosmo_params is not None:
            x = torch.cat([halo_features, cosmo_params], dim=-1)
        else:
            x = halo_features
        
        # Forward through encoder with residual connections
        residual = x
        for i in range(0, len(self.encoder), 4):  # Process in blocks of 4 (Linear, LN, ReLU, Dropout)
            # Apply block
            for j in range(4):
                x = self.encoder[i + j](x)
            
            # Add residual connection
            if self.config.use_residual and len(self.residual_projections) > i // 4:
                residual = self.residual_projections[i // 4](residual)
                x = x + residual
                residual = x
        
        # Output layer
        power_spectrum = self.output_layer(x)
        
        return power_spectrum


class EnsembleRegressor:
    """
    Ensemble of power spectrum regressors for uncertainty quantification.
    """
    
    def __init__(self, config: ModelConfig, n_models: int = 5, device: str = 'cuda'):
        self.config = config
        self.n_models = n_models
        self.device = device
        
        # Create ensemble members
        self.models = [
            PowerSpectrumRegressor(config).to(device) 
            for _ in range(n_models)
        ]
        
        # Separate optimizers for each model
        self.optimizers = None
        
    def initialize_optimizers(self, lr: float = 1e-3, weight_decay: float = 1e-5):
        """Initialize optimizers for all ensemble members."""
        self.optimizers = [
            torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            for model in self.models
        ]
        
    def train_step(self, halo_features: torch.Tensor, 
                   targets: torch.Tensor,
                   cosmo_params: Optional[torch.Tensor] = None,
                   loss_fn: nn.Module = None) -> float:
        """
        Single training step for all ensemble members.
        
        Args:
            halo_features: (batch_size, n_halo_features)
            targets: (batch_size, n_power_bins)
            cosmo_params: Optional (batch_size, n_cosmo_params)
            loss_fn: Loss function (defaults to MSE)
            
        Returns:
            Average loss across ensemble
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        
        total_loss = 0.0
        
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(halo_features, cosmo_params)
            loss = loss_fn(predictions, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / self.n_models
    
    def predict(self, halo_features: torch.Tensor,
                cosmo_params: Optional[torch.Tensor] = None,
                return_std: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict with uncertainty quantification.
        
        Args:
            halo_features: (batch_size, n_halo_features)
            cosmo_params: Optional (batch_size, n_cosmo_params)
            return_std: Whether to return standard deviation
            
        Returns:
            mean_prediction: (batch_size, n_power_bins)
            std_prediction: Optional (batch_size, n_power_bins)
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(halo_features, cosmo_params)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # (n_models, batch_size, n_power_bins)
        
        mean_pred = predictions.mean(dim=0)
        
        if return_std:
            std_pred = predictions.std(dim=0)
            return mean_pred, std_pred
        
        return mean_pred, None
    
    def save(self, path: str):
        """Save all ensemble models."""
        torch.save({
            'config': self.config,
            'n_models': self.n_models,
            'model_states': [model.state_dict() for model in self.models],
            'optimizer_states': [opt.state_dict() for opt in self.optimizers] if self.optimizers else None
        }, path)
    
    def load(self, path: str):
        """Load all ensemble models."""
        checkpoint = torch.load(path, map_location=self.device)
        
        for model, state_dict in zip(self.models, checkpoint['model_states']):
            model.load_state_dict(state_dict)
        
        if checkpoint['optimizer_states'] and self.optimizers:
            for opt, state_dict in zip(self.optimizers, checkpoint['optimizer_states']):
                opt.load_state_dict(state_dict)


class LightningEnsembleRegressor(pl.LightningModule):
    """
    PyTorch Lightning wrapper for EnsembleRegressor for simplified training.
    """
    
    def __init__(self, config: ModelConfig, n_models: int = 5, 
                 lr: float = 1e-3, weight_decay: float = 1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.n_models = n_models
        self.lr = lr
        self.weight_decay = weight_decay
        
        # Create ensemble members
        self.models = nn.ModuleList([
            PowerSpectrumRegressor(config) 
            for _ in range(n_models)
        ])
        
        self.loss_fn = nn.MSELoss()
        
    def forward(self, halo_features: torch.Tensor, 
                cosmo_params: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with uncertainty quantification."""
        predictions = []
        for model in self.models:
            pred = model(halo_features, cosmo_params)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred
    
    def training_step(self, batch, batch_idx):
        halo_features, targets, cosmo_params = batch
        
        # Train each ensemble member
        total_loss = 0.0
        for model in self.models:
            predictions = model(halo_features, cosmo_params)
            loss = self.loss_fn(predictions, targets)
            total_loss += loss
        
        avg_loss = total_loss / self.n_models
        self.log('train_loss', avg_loss, prog_bar=True)
        
        return avg_loss
    
    def validation_step(self, batch, batch_idx):
        halo_features, targets, cosmo_params = batch
        
        # Get predictions with uncertainty
        mean_pred, std_pred = self(halo_features, cosmo_params)
        
        # Calculate loss
        loss = self.loss_fn(mean_pred, targets)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_uncertainty', std_pred.mean(), prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step for evaluation on test set."""
        halo_features, targets, cosmo_params = batch
        
        # Get predictions with uncertainty
        mean_pred, std_pred = self(halo_features, cosmo_params)
        
        # Calculate loss
        loss = self.loss_fn(mean_pred, targets)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_uncertainty', std_pred.mean(), prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        # Combine all parameters from ensemble into one optimizer
        all_params = []
        for model in self.models:
            all_params.extend(model.parameters())
        
        optimizer = torch.optim.AdamW(all_params, lr=self.lr, weight_decay=self.weight_decay)
        
        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


# Example usage and training loop
if __name__ == "__main__":
    import sys
    import os
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    from lightning.pytorch.loggers import TensorBoardLogger
    
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data.camels_dataloader import get_train_val_test_dataloaders
    
    # Set seed for reproducibility
    seed_everything(42, workers=True)
    
    # Data paths
    data_dir = "/pscratch/sd/l/lindajin/DH_profile_kSZ_WL/data"
    
    # Toggle whether to use cosmological parameters
    use_cosmo_params = True
    
    # Determine n_cosmo_params if using them
    n_cosmo_params = 0
    if use_cosmo_params:
        cosmo_params_path = os.path.join(data_dir, "camels_params_matrix.npy")
        cosmo_params_full = np.load(cosmo_params_path)
        n_cosmo_params = cosmo_params_full.shape[1]
        print(f"Loaded cosmo params with shape: {cosmo_params_full.shape}")
    
    # Configuration with real data dimensions
    config = ModelConfig(
        n_halo_features=13500,  # 15 * 900 from dataloader
        n_power_bins=255,  # Power spectrum bins
        n_cosmo_params=n_cosmo_params,  # From camels_params_matrix.npy or 0
        hidden_dims=[256, 256, 128],  # Simpler architecture
        dropout_rate=0.2,  # Increased regularization
        use_residual=True,
        use_batch_norm=True
    )
    
    # Create train/val/test dataloaders
    batch_size = 16
    train_dataloader, val_dataloader, test_dataloader = get_train_val_test_dataloaders(
        data_dir, 
        batch_size=batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        include_cosmo_params=use_cosmo_params
    )
    
    # Get actual dimensions from dataset
    dataset = train_dataloader.dataset.dataset  # Access underlying dataset from Subset
    actual_n_halo_features = 15 * dataset.n_features_per_sample
    actual_n_power_bins = dataset.ptot_data.shape[1]
    
    print(f"\nActual data dimensions detected:")
    print(f"  n_halo_features: {actual_n_halo_features}")
    print(f"  n_power_bins: {actual_n_power_bins}")
    
    # Update config with actual dimensions
    config.n_halo_features = actual_n_halo_features
    config.n_power_bins = actual_n_power_bins
    
    # Create Lightning model
    print("\n=== Training with PyTorch Lightning ===")
    model = LightningEnsembleRegressor(
        config=config,
        n_models=5,
        lr=1e-3,
        weight_decay=1e-5
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='ensemble-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=True
    )
    
    # Setup logger
    logger = TensorBoardLogger('lightning_logs', name='power_spectrum_emulator')
    
    # Create Trainer
    trainer = Trainer(
        max_epochs=50,  # Increased for better convergence
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=5,
        enable_progress_bar=True,
        # gradient_clip_val=1.0,  # Add gradient clipping
        accumulate_grad_batches=2  # Accumulate gradients for larger effective batch size
    )
    
    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)
    print("Training complete!")
    
    # Test the model
    print("\n=== Testing on test set ===")
    test_results = trainer.test(model, test_dataloader)
    
    # Test predictions with uncertainty
    print("\n=== Testing prediction with uncertainty ===")
    model.eval()
    test_batch = next(iter(test_dataloader))
    test_halo_features, test_targets, test_cosmo_params = test_batch
    test_halo_features = test_halo_features.to(model.device)
    if test_cosmo_params is not None:
        test_cosmo_params = test_cosmo_params.to(model.device)
    
    mean_pred, std_pred = model(test_halo_features, test_cosmo_params)
    print(f"Prediction shape: {mean_pred.shape}")
    print(f"Uncertainty shape: {std_pred.shape}")
    print(f"Mean uncertainty: {std_pred.mean().item():.4f}")
    
    # Compute percentage error with uncertainty
    test_targets = test_targets.to(model.device)
    abs_error = torch.abs(mean_pred - test_targets)
    percentage_error = (abs_error / (torch.abs(test_targets) + 1e-10)) * 100
    percentage_uncertainty = (std_pred / (torch.abs(test_targets) + 1e-10)) * 100
    
    mean_pct_error = percentage_error.mean().item()
    mean_pct_uncertainty = percentage_uncertainty.mean().item()
    print(f"Test accuracy: {mean_pct_error:.2f}% ± {mean_pct_uncertainty:.2f}%")
    print(f"Median accuracy: {percentage_error.median().item():.2f}% ± {percentage_uncertainty.median().item():.2f}%")
    
    # Example without cosmo params
    print("\n=== Training without cosmo params ===")
    config_no_cosmo = ModelConfig(
        n_halo_features=actual_n_halo_features,
        n_power_bins=actual_n_power_bins,
        n_cosmo_params=0,  # No cosmo params
        hidden_dims=[256, 256, 128],
        dropout_rate=0.2,
        use_batch_norm=True
    )
    
    # Create dataloaders without cosmo params
    train_dataloader_no_cosmo, val_dataloader_no_cosmo, test_dataloader_no_cosmo = get_train_val_test_dataloaders(
        data_dir,
        batch_size=batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        include_cosmo_params=False
    )
    
    model_no_cosmo = LightningEnsembleRegressor(
        config=config_no_cosmo,
        n_models=5,
        lr=1e-3,
        weight_decay=1e-5
    )
    
    trainer_no_cosmo = Trainer(
        max_epochs=50,
        accelerator='auto',
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=5,
        # gradient_clip_val=1.0,
        accumulate_grad_batches=2
    )
    
    trainer_no_cosmo.fit(model_no_cosmo, train_dataloader_no_cosmo, val_dataloader_no_cosmo)
    print("Training without params complete!")

    print("\n=== Testing on test set ===")
    test_results = trainer_no_cosmo.test(model_no_cosmo, test_dataloader_no_cosmo)
    
    # Test predictions with uncertainty
    print("\n=== Testing prediction with uncertainty ===")
    model_no_cosmo.eval()
    test_batch = next(iter(test_dataloader_no_cosmo))
    test_halo_features, test_targets, test_cosmo_params = test_batch
    test_halo_features = test_halo_features.to(model_no_cosmo.device)
    # test_cosmo_params should be None for this case
    
    mean_pred, std_pred = model_no_cosmo(test_halo_features, None)
    print(f"Prediction shape: {mean_pred.shape}")
    print(f"Uncertainty shape: {std_pred.shape}")
    print(f"Mean uncertainty: {std_pred.mean().item():.4f}")
    
    # Compute percentage error with uncertainty (no cosmo)
    test_targets_no_cosmo = test_targets.to(model_no_cosmo.device)
    abs_error_no_cosmo = torch.abs(mean_pred - test_targets_no_cosmo)
    percentage_error_no_cosmo = (abs_error_no_cosmo / (torch.abs(test_targets_no_cosmo) + 1e-10)) * 100
    percentage_uncertainty_no_cosmo = (std_pred / (torch.abs(test_targets_no_cosmo) + 1e-10)) * 100
    
    mean_pct_error_no_cosmo = percentage_error_no_cosmo.mean().item()
    mean_pct_uncertainty_no_cosmo = percentage_uncertainty_no_cosmo.mean().item()
    print(f"Test accuracy (no cosmo): {mean_pct_error_no_cosmo:.2f}% ± {mean_pct_uncertainty_no_cosmo:.2f}%")
    print(f"Median accuracy (no cosmo): {percentage_error_no_cosmo.median().item():.2f}% ± {percentage_uncertainty_no_cosmo.median().item():.2f}%")
