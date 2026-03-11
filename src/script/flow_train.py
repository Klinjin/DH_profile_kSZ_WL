import os
import sys
sys.path.append("/pscratch/sd/l/lindajin/DH_profile_kSZ_WL")
from src.models.flow_nn import *
from src.data.camels_dataloader import get_train_val_test_dataloaders
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import argparse

from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

# Example usage
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train flow matching model for power spectrum emulation')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum number of training epochs')
    parser.add_argument('--run_name', type=str, default='flow_run',
                       help='Name for this run (used in checkpoint and figure filenames)')
    parser.add_argument('--use_cosmo_params', action='store_true',
                       help='Include cosmological parameters as conditioning')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--n_samples', type=int, default=10,
                       help='Number of samples to generate per input during evaluation')
    parser.add_argument('--mean_profile', action='store_true',
                       help='Use mean profile for kSZ/DeltaSigma ratios')
    args = parser.parse_args()
    
    # Options
    max_epochs = args.max_epochs
    run_name = args.run_name
    use_cosmo_params = args.use_cosmo_params
    batch_size = args.batch_size
    n_samples = args.n_samples
    mean_profile = args.mean_profile
    data_dir = "/pscratch/sd/l/lindajin/DH_profile_kSZ_WL/data"
    
    print(f"\n{'='*60}")
    print(f"  RUN NAME: {run_name}")
    print(f"  Use cosmological parameters: {use_cosmo_params}")
    print(f"  Batch size: {batch_size}")
    print(f"  Evaluation samples: {n_samples}")
    print(f"{'='*60}\n")
    
    # Load a temporary dataset to get data dimensions
    from src.data.camels_dataloader import KSZDeltaSigmaDataset
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
    
    # Configuration with dynamically determined dimensions
    config = FlowConfig(
        n_radial_bins=n_radial_bins,
        n_spatial_positions=n_spatial_positions,
        n_power_bins=n_power_bins,
        n_cosmo_params=n_cosmo_params,
        encoder_latent_dim= 64, #512, 
        flow_hidden_dims=[100, 100, 256], #[1024, 1024, 512, 512, 256],
        time_embed_dim=128,
        context_embed_dim=256,
        dropout_rate=0.1,
        sigma=0.02,  # Deterministic flow
        reverse=False,
        smoothness_weight = 0.5  # Weight for smoothness loss (tune between 0.01 - 0.5)
    )
    
    del temp_dataset  # Free memory
    

    fm = PkFlow(config)
    
    print(f"\n=== Model Configuration ===")
    print(f"  Smoothness weight: {config.smoothness_weight}")
    print(f"  Encoder latent dim: {config.encoder_latent_dim}")
    print(f"  Flow hidden dims: {config.flow_hidden_dims}")
    print(f"  Sigma (stochasticity): {config.sigma}")
    print(f"=========================\n")
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
            m.bias.data.fill_(0.01)
    
    fm.apply(init_weights)
    fm.train()

    # Load data
    train_dataloader, val_dataloader, test_dataloader = get_train_val_test_dataloaders(
        data_dir, 
        batch_size=batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        include_cosmo_params=use_cosmo_params
    )
    
    # Create run-specific directories
    checkpoint_dir = os.path.join('checkpoints', run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'{run_name}-epoch={{epoch:02d}}-val_loss={{val_loss:.4f}}',
        monitor='val_loss',
        mode='min',
        save_top_k=3
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger('lightning_logs', name=run_name)

    trainer = Trainer(
        max_epochs=max_epochs, 
        gradient_clip_val=1.0,
        logger=logger,
        log_every_n_steps=5,
        accumulate_grad_batches=2,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator='cpu',
        devices=1,
        val_check_interval=1.0,
        profiler=None,
    )

    # Start training
    trainer.fit(
        model=fm,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # Evaluation on test set
    print("\n=== Evaluating on test set ===")
    single_batch = next(iter(test_dataloader))
    ksz_features, ptot_data, cosmo_params = single_batch
    
    # Ensure data is on CPU (model is on CPU)
    ksz_features = ksz_features.cpu()
    ptot_data = ptot_data.cpu()
    if cosmo_params is not None:
        cosmo_params = cosmo_params.cpu()
    
    # Generate samples with better integration settings
    samples, latent = fm.sample(
        ksz_spatial=ksz_features,
        cosmo_params=cosmo_params,
        n_samples=n_samples,
        solver='midpoint',  # Use midpoint for better accuracy (or 'dopri5' if torchdiffeq available)
        n_steps=500  # Increased from 100 for smoother trajectories
    )  # samples: (batch_size * 10, 118), latent: (batch_size, 64)

    print(f"\nGenerated {samples.shape[0]} samples")
    print(f"Sample shape: {samples.shape}")
    print(f"Sample mean across ensemble: {samples.mean(dim=0)[:5]}")
    print(f"Sample std across ensemble: {samples.std(dim=0)[:5]}")
    
    # Calculate percentage errors

    batch_size_actual = ptot_data.shape[0]
    samples_reshaped = samples.view(batch_size_actual, n_samples, -1)  # (batch, 10, 118)
    
    # Expand ptot_data to (batch_size, 1, 118) for broadcasting
    ptot_expanded = ptot_data.unsqueeze(1)  # (batch, 1, 118)
    
    # Calculate percentage error: 100 * (predicted - target) / target
    percent_errors = 100 * (samples_reshaped - ptot_expanded) / ptot_expanded  # (batch, 10, 118)
    
    # Statistics across samples and batch
    mean_percent_error = percent_errors.mean(dim=(0, 1))  # (118,) - mean across batch and samples
    std_percent_error = percent_errors.std(dim=(0, 1))    # (118,) - std across batch and samples
    
    # Compute median properly: flatten to (batch*n_samples, 118) then take median
    percent_errors_flat = percent_errors.view(-1, 118)  # (batch*10, 118)
    median_percent_error = percent_errors_flat.median(dim=0).values  # (118,)
    
    # Overall statistics
    overall_mean_error = mean_percent_error.mean().item()
    overall_std_error = std_percent_error.mean().item()
    overall_median_error = median_percent_error.mean().item()
    
    cosmo_status = "with cosmo" if use_cosmo_params else "without cosmo"
    print(f"\nPrediction accuracy ({cosmo_status}):")
    print(f"  Mean error: {overall_mean_error:.2f}% +/- {overall_std_error:.2f}%")
    print(f"  Median error: {overall_median_error:.2f}%")
    print(f"\nFirst 5 k-bins:")
    print(f"  Mean error: {[f'{x:.2f}' for x in mean_percent_error[:5].tolist()]}")
    print(f"  Median error: {[f'{x:.2f}' for x in median_percent_error[:5].tolist()]}")
    print(f"  Std: {[f'{x:.2f}' for x in std_percent_error[:5].tolist()]}")
    
    # ========================
    # Visualization: Error Percentiles
    # ========================
    
    print("\n=== Creating error percentile visualizations ===")
    
    # Calculate percentiles across all samples and batch
    # percent_errors_flat is (batch*n_samples, 118)
    percentile_50 = np.percentile(percent_errors_flat.cpu().numpy(), 50, axis=0)  # Median
    percentile_68 = np.percentile(percent_errors_flat.cpu().numpy(), 84, axis=0)  # ~1 sigma
    percentile_99 = np.percentile(percent_errors_flat.cpu().numpy(), 99, axis=0)  # ~2.5 sigma
    
    # Figure 1: Error percentiles
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top panel: Percentile bands
    ax1.fill_between(k_bins, percentile_50, percentile_99, 
                     alpha=0.2, color='red', label='99%')
    ax1.fill_between(k_bins,  percentile_50, percentile_68, 
                     alpha=0.3, color='blue', label='68%')
    ax1.plot(k_bins, percentile_50, 'k-', linewidth=2, label='Median (50th)')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('k-bin index', fontsize=12)
    ax1.set_ylabel('Percentage Error (%)', fontsize=12)
    ax1.set_title(f'Test Error Percentiles Across Ensemble ({cosmo_status})'+ "\n" +f"Median error: {overall_median_error:.2f}%", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Bottom panel: Box plot for selected k-bins
    selected_bins = np.linspace(0, n_power_bins-1, 10, dtype=int)
    box_data = [percent_errors_flat[:, i].cpu().numpy() for i in selected_bins]
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
    n_examples = min(4, batch_size_actual)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx in range(n_examples):
        ax = axes[idx]
        
        # Ground truth
        truth = ptot_data[idx].cpu().numpy()
        
        # Samples for this example: (n_samples, 118)
        example_samples = samples_reshaped[idx].cpu().numpy()
        
        # Median prediction
        mean_pred = np.mean(example_samples, axis=0)
        
        # Plot all individual samples as faint lines
        for sample_idx in range(n_samples):
            ax.plot(k_bins, example_samples[sample_idx], 
                   color='cornflowerblue', alpha=0.15, linewidth=0.8)
        
        # Plot median prediction as solid line
        ax.plot(k_bins, mean_pred, 'b-', linewidth=2.5, 
               label=f'Mean Prediction (n={n_samples})', zorder=10)
        
        # Plot ground truth
        ax.plot(k_bins, truth, 'r-', linewidth=2, 
               label='Ground Truth', zorder=11)
        
        # Calculate error for this example
        example_error = 100 * (mean_pred - truth) / truth
        mean_abs_error = np.mean(np.abs(example_error))
        
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
    