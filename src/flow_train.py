from models.flow_nn import *
from data.camels_dataloader import get_train_val_test_dataloaders
import os
import torch
import torch.nn as nn

from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

# Example usage
if __name__ == "__main__":
    # Options
    use_cosmo_params = False  # Set to True to include cosmological parameters
    batch_size = 16
    data_dir = "/pscratch/sd/l/lindajin/DH_profile_kSZ_WL/data"
    
    # Determine n_cosmo_params from data if using them
    if use_cosmo_params:
        cosmo_params_path = os.path.join(data_dir, "camels_params_matrix.npy")
        cosmo_params_array = np.load(cosmo_params_path)
        n_cosmo_params = cosmo_params_array.shape[1]
        print(f"Detected {n_cosmo_params} cosmological parameters from data")
    else:
        n_cosmo_params = 0
        print("Training without cosmological parameters")
    
    # Configuration
    config = FlowConfig(
        n_radial_bins=15,
        n_spatial_positions=900,
        n_power_bins=255,
        n_cosmo_params=n_cosmo_params,
        encoder_channels=[32, 64, 128, 256],
        encoder_latent_dim=64,
        flow_hidden_dims=[512, 512, 512, 256]
    )
    
    # Initialize model
    fm = PkFlow(config)
    
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
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='ensemble-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger('lightning_logs', name='power_spectrum_emulator')

    trainer = Trainer(
        max_epochs=50, 
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
    
    # Generate samples
    samples, latent = fm.sample(
        ksz_spatial=ksz_features,
        cosmo_params=cosmo_params,
        n_samples=10,
    )  # samples: (batch_size * 10, 255), latent: (batch_size, 64)

    print(f"\nGenerated {samples.shape[0]} samples")
    print(f"Sample shape: {samples.shape}")
    print(f"Sample mean across ensemble: {samples.mean(dim=0)[:5]}")
    print(f"Sample std across ensemble: {samples.std(dim=0)[:5]}")
    
    # Calculate percentage errors
    n_samples = 10
    batch_size_actual = ptot_data.shape[0]
    samples_reshaped = samples.view(batch_size_actual, n_samples, -1)  # (batch, 10, 255)
    
    # Expand ptot_data to (batch_size, 1, 255) for broadcasting
    ptot_expanded = ptot_data.unsqueeze(1)  # (batch, 1, 255)
    
    # Calculate percentage error: 100 * (predicted - target) / target
    percent_errors = 100 * (samples_reshaped - ptot_expanded) / ptot_expanded  # (batch, 10, 255)
    
    # Statistics across samples and batch
    mean_percent_error = percent_errors.mean(dim=(0, 1))  # (255,) - mean across batch and samples
    std_percent_error = percent_errors.std(dim=(0, 1))    # (255,) - std across batch and samples
    
    # Compute median properly: flatten to (batch*n_samples, 255) then take median
    percent_errors_flat = percent_errors.view(-1, 255)  # (batch*10, 255)
    median_percent_error = percent_errors_flat.median(dim=0).values  # (255,)
    
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
