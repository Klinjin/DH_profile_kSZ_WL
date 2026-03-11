import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
import lightning.pytorch as pl
from lightning.pytorch import seed_everything, Trainer


@dataclass
class ModelConfig:
    """Configuration for the power spectrum regression model."""
    n_radial_bins: int = 15  # Number of radial bins
    n_spatial_positions: int = 900  # Number of spatial positions per sample
    n_power_bins: int = 255  # k-bins in power spectrum
    n_cosmo_params: int = 0  # Set to >0 if using cosmological parameters
    hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 128, 256])
    dropout_rate: float = 0.2  # Regularization
    use_residual: bool = True
    use_batch_norm: bool = True  # Add batch normalization
    
    @property
    def input_dim(self):
        """Calculate total input dimension."""
        base_dim = self.n_radial_bins * self.n_spatial_positions
        return base_dim + self.n_cosmo_params


class PowerSpectrumRegressor(nn.Module):
    """
    Neural network for predicting power spectrum suppression from kSZ/DeltaSigma observables.
    Supports optional conditioning on cosmological/astrophysical parameters.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Build encoder layers with residual connections
        layers = []
        prev_dim = config.input_dim
        
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
            proj_input_dim = config.input_dim
            for hidden_dim in config.hidden_dims:
                self.residual_projections.append(
                    nn.Linear(proj_input_dim, hidden_dim)
                )
                proj_input_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(config.hidden_dims[-1], config.n_power_bins)
        
    def forward(self, ksz_spatial: torch.Tensor, 
                cosmo_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            ksz_spatial: (batch_size, n_radial_bins * n_spatial_positions)
            cosmo_params: Optional (batch_size, n_cosmo_params)
            
        Returns:
            power_spectrum: (batch_size, n_power_bins)
        """
        # Concatenate cosmological parameters if provided
        if cosmo_params is not None:
            x = torch.cat([ksz_spatial, cosmo_params], dim=-1)
        else:
            x = ksz_spatial
        
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
        
    def train_step(self, ksz_spatial: torch.Tensor, 
                   targets: torch.Tensor,
                   cosmo_params: Optional[torch.Tensor] = None,
                   loss_fn: nn.Module = None) -> float:
        """
        Single training step for all ensemble members.
        
        Args:
            ksz_spatial: (batch_size, n_radial_bins * n_spatial_positions)
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
            predictions = model(ksz_spatial, cosmo_params)
            loss = loss_fn(predictions, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / self.n_models
    
    def predict(self, ksz_spatial: torch.Tensor,
                cosmo_params: Optional[torch.Tensor] = None,
                return_std: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict with uncertainty quantification.
        
        Args:
            ksz_spatial: (batch_size, n_radial_bins * n_spatial_positions)
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
                pred = model(ksz_spatial, cosmo_params)
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
                 lr: float = 1e-3, weight_decay: float = 1e-5,
                 l1_lambda: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.n_models = n_models
        self.lr = lr
        self.weight_decay = weight_decay
        self.l1_lambda = l1_lambda
        
        # Create ensemble members
        self.models = nn.ModuleList([
            PowerSpectrumRegressor(config) 
            for _ in range(n_models)
        ])
        
        self.loss_fn = nn.MSELoss()
        
    def forward(self, ksz_spatial: torch.Tensor, 
                cosmo_params: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with uncertainty quantification."""
        predictions = []
        for model in self.models:
            pred = model(ksz_spatial, cosmo_params)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred
    
    def training_step(self, batch, batch_idx):
        ksz_spatial, targets, cosmo_params = batch
        
        # Train each ensemble member
        total_loss = 0.0
        for model in self.models:
            predictions = model(ksz_spatial, cosmo_params)
            mse_loss = self.loss_fn(predictions, targets)
            # L1 regularization: penalise sum of absolute weight values
            l1_penalty = sum(p.abs().sum() for p in model.parameters())
            loss = mse_loss + self.l1_lambda * l1_penalty
            total_loss += loss
        
        avg_loss = total_loss / self.n_models
        self.log('train_loss', avg_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_l1_penalty', self.l1_lambda * l1_penalty, prog_bar=False, on_step=True, on_epoch=True)
        
        return avg_loss
    
    def validation_step(self, batch, batch_idx):
        ksz_spatial, targets, cosmo_params = batch
        
        # Get predictions with uncertainty
        mean_pred, std_pred = self(ksz_spatial, cosmo_params)
        
        # Calculate loss
        loss = self.loss_fn(mean_pred, targets)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_uncertainty', std_pred.mean(), prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step for evaluation on test set."""
        ksz_spatial, targets, cosmo_params = batch
        
        # Get predictions with uncertainty
        mean_pred, std_pred = self(ksz_spatial, cosmo_params)
        
        # Calculate loss
        loss = self.loss_fn(mean_pred, targets)
        
        # Calculate percentage errors
        abs_error = torch.abs(mean_pred - targets)
        percentage_error = (abs_error / (torch.abs(targets) + 1e-10)) * 100
        percentage_uncertainty = (std_pred / (torch.abs(targets) + 1e-10)) * 100
        
        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_uncertainty', std_pred.mean(), prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_mean_percent_error', percentage_error.mean(), prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_median_percent_error', percentage_error.median(), prog_bar=True, on_step=False, on_epoch=True)
        
        return {
            'test_loss': loss,
            'mean_pred': mean_pred,
            'std_pred': std_pred,
            'targets': targets,
            'percentage_error': percentage_error
        }
    
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
    import argparse
    import matplotlib.pyplot as plt
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
    from lightning.pytorch.loggers import TensorBoardLogger
    
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.camels_dataloader import get_train_val_test_dataloaders, KSZDeltaSigmaDataset
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train neural network for power spectrum emulation')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum number of training epochs')
    parser.add_argument('--run_name', type=str, default='pk_nn_run',
                       help='Name for this run (used in checkpoint and figure filenames)')
    parser.add_argument('--use_cosmo_params', action='store_true',
                       help='Include cosmological parameters as conditioning')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--n_samples', type=int, default=5,
                       help='Number of ensemble members')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--mean_profile', action='store_true',
                       help='Use mean profile for kSZ/DeltaSigma ratios')
    args = parser.parse_args()
    
    # Set seed for reproducibility
    seed_everything(42, workers=True)
    
    # Options
    max_epochs = args.max_epochs
    run_name = args.run_name
    use_cosmo_params = args.use_cosmo_params
    batch_size = args.batch_size
    n_models = args.n_samples
    lr = args.lr
    mean_profile = args.mean_profile
    data_dir = "/pscratch/sd/l/lindajin/DH_profile_kSZ_WL/data"
    
    print(f"\n{'='*60}")
    print(f"  RUN NAME: {run_name}")
    print(f"  Use cosmological parameters: {use_cosmo_params}")
    print(f"  Batch size: {batch_size}")
    print(f"  Ensemble members: {n_models}")
    print(f"  Learning rate: {lr}")
    print(f"  Use mean profile: {mean_profile}")
    print(f"{'='*60}\n")
    
    # Load a temporary dataset to get data dimensions
    temp_dataset = KSZDeltaSigmaDataset(data_dir, include_cosmo_params=use_cosmo_params, mean_prof=mean_profile)
    
    # Extract dimensions from data
    n_radial_bins = temp_dataset.n_radial_bins
    n_spatial_positions = temp_dataset.n_spatial_positions
    n_power_bins = temp_dataset.n_power_bins
    k_bins = temp_dataset.k_bins
    r_bins = temp_dataset.r_bins
    
    # Determine n_cosmo_params from data if using them
    if use_cosmo_params:
        n_cosmo_params = temp_dataset.cosmo_params.shape[1]
        print(f"Detected {n_cosmo_params} cosmological parameters from data")
    else:
        n_cosmo_params = 0
        print("Training without cosmological parameters")
    
    print(f"\n=== Data Dimensions ===")
    print(f"  n_radial_bins: {n_radial_bins}")
    print(f"  n_spatial_positions: {n_spatial_positions}")
    print(f"  n_power_bins: {n_power_bins}")
    print(f"  n_cosmo_params: {n_cosmo_params}")
    print(f"  Total input dim: {temp_dataset.input_dim}")
    print(f"========================\n")
    
    # Configuration with real data dimensions
    config = ModelConfig(
        n_radial_bins=n_radial_bins,
        n_spatial_positions=n_spatial_positions,
        n_power_bins=n_power_bins,
        n_cosmo_params=n_cosmo_params,
        hidden_dims=[512, 512, 256, 128],
        dropout_rate=0.2,
        use_residual=True,
        use_batch_norm=True
    )
    
    del temp_dataset  # Free memory
    
    # Create train/val/test dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_train_val_test_dataloaders(
        data_dir, 
        batch_size=batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        include_cosmo_params=use_cosmo_params
    )
    
    # Create Lightning model
    print("\n=== Training with PyTorch Lightning ===")
    model = LightningEnsembleRegressor(
        config=config,
        n_models=n_models,
        lr=lr,
        weight_decay=1e-5
    )
    
    print(f"\n=== Model Configuration ===")
    print(f"  Input dimension: {config.input_dim}")
    print(f"  Hidden dims: {config.hidden_dims}")
    print(f"  Dropout rate: {config.dropout_rate}")
    print(f"  Use residual: {config.use_residual}")
    print(f"  Use batch norm: {config.use_batch_norm}")
    print(f"=========================\n")
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
            m.bias.data.fill_(0.01)
    
    model.apply(init_weights)
    
    # Create run-specific directories
    checkpoint_dir = os.path.join('checkpoints', run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'{run_name}-epoch={{epoch:02d}}-val_loss={{val_loss:.4f}}',
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
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # Setup logger
    logger = TensorBoardLogger('lightning_logs', name=run_name)
    
    # Create Trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=5,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
        val_check_interval=1.0,
    )
    
    # Train the model
    print("\n=== Starting Training ===")
    trainer.fit(model, train_dataloader, val_dataloader)
    print("Training complete!")
    
    # Test the model
    print("\n=== Testing on test set ===")
    test_results = trainer.test(model, test_dataloader)
    
    # Evaluation on test set
    print("\n=== Evaluating on test set ===")
    single_batch = next(iter(test_dataloader))
    ksz_spatial, ptot_data, cosmo_params = single_batch
    
    # Ensure data is on correct device
    device = model.device
    ksz_spatial = ksz_spatial.to(device)
    ptot_data = ptot_data.to(device)
    if cosmo_params is not None:
        cosmo_params = cosmo_params.to(device)
    
    # Get predictions with uncertainty
    model.eval()
    with torch.no_grad():
        mean_pred, std_pred = model(ksz_spatial, cosmo_params)
    
    print(f"\nPrediction shape: {mean_pred.shape}")
    print(f"Uncertainty shape: {std_pred.shape}")
    print(f"Mean uncertainty: {std_pred.mean().item():.4f}")
    
    # Compute percentage error with uncertainty
    abs_error = torch.abs(mean_pred - ptot_data)
    percentage_error = (abs_error / (torch.abs(ptot_data) + 1e-10)) * 100
    percentage_uncertainty = (std_pred / (torch.abs(ptot_data) + 1e-10)) * 100
    
    mean_pct_error = percentage_error.mean().item()
    median_pct_error = percentage_error.median().item()
    mean_pct_uncertainty = percentage_uncertainty.mean().item()
    
    cosmo_status = "with cosmo" if use_cosmo_params else "without cosmo"
    print(f"\nTest accuracy ({cosmo_status}):")
    print(f"  Mean error: {mean_pct_error:.2f}% ± {mean_pct_uncertainty:.2f}%")
    print(f"  Median error: {median_pct_error:.2f}%")
    
    # ========================
    # Visualization: Error Percentiles
    # ========================
    
    print("\n=== Creating error percentile visualizations ===")
    
    # Calculate percentiles
    percentile_50 = np.percentile(percentage_error.cpu().numpy(), 50, axis=0)  # Median
    percentile_68 = np.percentile(percentage_error.cpu().numpy(), 84, axis=0)  # ~1 sigma
    percentile_99 = np.percentile(percentage_error.cpu().numpy(), 99, axis=0)  # ~2.5 sigma
    
    # Figure 1: Error percentiles
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top panel: Percentile bands
    ax1.fill_between(k_bins, percentile_50, percentile_99, 
                     alpha=0.2, color='red', label='99%')
    ax1.fill_between(k_bins, percentile_50, percentile_68, 
                     alpha=0.3, color='blue', label='68%')
    ax1.plot(k_bins, percentile_50, 'k-', linewidth=2, label='Median (50th)')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('k [h/Mpc]', fontsize=12)
    ax1.set_ylabel('Percentage Error (%)', fontsize=12)
    ax1.set_title(f'Test Error Percentiles ({cosmo_status})' + "\n" + f"Median error: {median_pct_error:.2f}%", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Box plot for selected k-bins
    selected_bins = np.linspace(0, n_power_bins-1, 10, dtype=int)
    box_data = [percentage_error[:, i].cpu().numpy() for i in selected_bins]
    positions = selected_bins + 1
    
    bp = ax2.boxplot(box_data, positions=positions, widths=3, 
                     patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.6)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('k [h/Mpc]', fontsize=12)
    ax2.set_ylabel('Percentage Error (%)', fontsize=12)
    ax2.set_title('Error Distribution at Selected k-bins', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_dir = checkpoint_dir
    fig_path = os.path.join(fig_dir, f'{run_name}_test_error_percentiles.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved error percentile plot to {fig_path}")
    plt.close()
    
    # ========================
    # Visualization: Power Spectrum Reconstruction
    # ========================
    
    print("\n=== Creating power spectrum reconstruction visualizations ===")
    
    # Select first 4 examples from test batch for visualization
    n_examples = min(4, ptot_data.shape[0])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx in range(n_examples):
        ax = axes[idx]
        
        # Ground truth
        truth = ptot_data[idx].cpu().numpy()
        
        # Prediction with uncertainty
        pred = mean_pred[idx].cpu().numpy()
        std = std_pred[idx].cpu().numpy()
        
        # Plot prediction with uncertainty band
        ax.plot(k_bins, pred, 'b-', linewidth=2.5, 
               label='Mean Prediction', zorder=10)
        ax.fill_between(k_bins, pred - std, pred + std, 
                       alpha=0.3, color='cornflowerblue', label='±1σ uncertainty')
        
        # Plot ground truth
        ax.plot(k_bins, truth, 'r-', linewidth=2, 
               label='Ground Truth', zorder=11)
        
        # Calculate error for this example
        example_error = 100 * np.abs(pred - truth) / truth
        mean_abs_error = np.mean(example_error)
        
        ax.set_xlabel('k [h/Mpc]', fontsize=11)
        ax.set_ylabel('Power Spectrum', fontsize=11)
        ax.set_title(f'Example {idx+1}  |Mean error|: {mean_abs_error:.2f}%', 
                    fontsize=12)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Power Spectrum Reconstruction ({cosmo_status})', 
                fontsize=15, y=1.00)
    plt.tight_layout()
    
    fig_path = os.path.join(fig_dir, f'{run_name}_power_spectrum_reconstruction.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved power spectrum reconstruction plot to {fig_path}")
    plt.close()
    
    print(f"\n{'='*60}")
    print(f"  All outputs saved with run name: {run_name}")
    print(f"  Checkpoints: {checkpoint_dir}/")
    print(f"  Figures: {fig_dir}/")
    print(f"  TensorBoard logs: lightning_logs/{run_name}/")
    print(f"{'='*60}\n")
