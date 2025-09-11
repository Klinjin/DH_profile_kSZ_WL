"""
Comprehensive Gaussian Process training module for cosmological analysis.

This module consolidates all GP training functionality including:
- GPTrainer class for complete training/testing workflows
- Standard GP training with hierarchical kernels
- Neural Network + GP hybrid models
- Data preparation and model management utilities

Replaces both the previous gp_trainer.py and trainer.py for better organization.
"""

# Core scientific computing imports
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime
from tqdm import tqdm, trange
import pickle
import json
from typing import Dict, List, Tuple, Optional, Any

# JAX imports
import jax
import jax.numpy as jnp

# Local imports
from src.config.config import GP_TRAINING_DEFAULTS, N_COSMO_PARAMS, TRAINED_MODELS_DIR
from src.data.sim_dataloader import SimulationDataLoader, DataLoaderConfig
from src.models.gp_trainer_one import train_single_gp_model, train_single_gp_model_with_validation

# Import kernel builders
try:
    from src.models.kernels import build_hierarchical_gp, get_kernel_builder, initialize_gp_parameters
except ImportError:
    print("Warning: Could not import kernel builders - some functionality may be limited")

# Setup JAX environment
try:
    from src.utils.environment import setup_jax_environment
    setup_jax_environment()
except ImportError:
    # Fallback if src/ structure not fully implemented yet
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    jax.config.update("jax_enable_x64", False)
    print(f"JAX devices: {jax.devices()}")
    print(f"Using device: {jax.devices()[0]}")


class GPTrainer:
    """
    Gaussian Process Trainer for Cosmological Simulation Profiles.

    This class provides a unified interface for:
    - Loading and splitting simulation data (train/val/test)
    - Training Gaussian Process models with various kernel types (e.g., hierarchical, robust, physics-informed)
    - Evaluating models with comprehensive metrics and error analysis
    - Making predictions with uncertainty quantification
    - Visualizing training progress, predictions, and residuals
    - Saving and loading trained models

    Usage Example:
        ```python
        # Initialize trainer with simulation indices and configuration
        # Initialize trainer with simulation indices
        trainer = GPTrainer(
            sim_indices_total=list(range(100)),  # Use first 100 sims for quick demo
            train_test_val_split=(0.8, 0.2, 0.0),
            filterType='CAP',
            ptype='gas'
        )

        # Train GP models with hierarchical kernel
        training_info = trainer.train(kernel_type='hierarchical', maxiter=500)

        # Evaluate on test set
        metrics = trainer.test(plot=True)
        print(f"Test MAPE: {metrics['mape']:.1f}%")

        # Make predictions on new data
        # X_new should have shape (n_samples, n_features) where n_features = n_cosmo + 1 + n_k
        pred_means, pred_vars = trainer.pred(X_new)  # Returns (n_radius_bins, n_samples)
        ```
    """
    
    def __init__(
        self,
        sim_indices_total: List[int],
        train_test_val_split: Tuple[float, float, float] = (0.7, 0.3, 0.0),
        filterType: str = 'CAP',
        ptype: str = 'gas',
        save_dir: Optional[str] = None,
    ):
        """
        Initialize GPTrainer with simulation data.
        
        Args:
            sim_indices_total: List of simulation indices for all data
            filterType: Type of filter ('CAP', 'cumulative', 'dsigma')
            ptype: Particle type ('gas', 'dm', 'star', 'bh', 'total', 'baryon')
            save_dir: Directory to save results and plots
            train_test_val_split: Tuple of ratios for train/test/validation splits
        """
        self.sim_indices_total = sim_indices_total
        self.filterType = filterType
        self.ptype = ptype
        self.train_ratio, self.test_ratio, self.val_ratio = train_test_val_split
        
        # Setup save directory
        if save_dir is None:
            timestamp = datetime.now().strftime('%m%d%y_%H%M')
            save_dir = f"{TRAINED_MODELS_DIR}/GPTrainer_{timestamp}_{filterType}_{ptype}"
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize data storage
        self.is_data_loaded = False
        self.is_trained = False
        self.is_pretrained_loaded = False
        self.trained_models = None
        self.training_info = {}
        
        print(f"GPTrainer initialized:")
        print(f"  - Total sims: {len(sim_indices_total)}")
        print(f"  - Filter: {filterType}, Particle: {ptype}")
        print(f"  - Save dir: {save_dir}")

        self._load_data()
    
    def _load_data(self):
        """Load and prepare training and test data."""
        if self.is_data_loaded:
            return
            
        print("Loading training data...")
        
        # Create DataLoader configuration
        config = DataLoaderConfig(
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            normalize_features=True,
            log_transform_mass=True,
            shuffle=False, 
            batch_size=128,
            random_seed=42
        )
        
        # Load training data
        dl = SimulationDataLoader(
            sim_indices=self.sim_indices_total,
            config=config,
                filterType=self.filterType,
            ptype=self.ptype
        )
        
        self.train_dl, self.val_dl, self.test_dl = dl.get_dataloaders() # Iterate through batches

        self.X_train, self.y_train = dl.get_split_data('train')
        self.X_val, self.y_val = dl.get_split_data('val')
        self.X_test, self.y_test = dl.get_split_data('test')

        # Store metadata
        self.r_bins = dl.r_bins
        self.k_bins = dl.k_bins
        self.n_radius_bins = len(self.r_bins)
        self.n_features = self.X_train.shape[1]
        
        # Only train linear small bins if using CAP filter for kSZ
        if self.filterType == 'CAP' and self.ptype == 'gas':
            self.n_radius_bins = self.n_radius_bins//2
            self.r_bins = self.r_bins[:self.n_radius_bins]
            self.y_train = self.y_train[:, :self.n_radius_bins]
            self.y_val = self.y_val[:, :self.n_radius_bins] if self.val_ratio > 0 else None
            self.y_test = self.y_test[:, :self.n_radius_bins] if self.test_ratio > 0 else None

        self.is_data_loaded = True
        print(f"Data loaded:")
        print(f"  - Train: X={self.X_train.shape}, y={self.y_train.shape}")
        if self.X_val is not None:
            print(f"  - Val: X={self.X_val.shape}, y={self.y_val.shape}")
        print(f"  - Test: X={self.X_test.shape}, y={self.y_test.shape}")
        print(f"  - Radius bins: {self.n_radius_bins}")

    def train(
        self,
        kernel_type: str = 'hierarchical',
        maxiter: int = 1000,
        lr: float = 3e-4,
        plot: bool = True,
        save: bool = True
    ) -> Dict[str, Any]:
        """
        Train GP models for all radius bins.
        
        Args:
            kernel_type: Type of kernel ('hierarchical', 'robust', 'physics_informed', etc.)
            maxiter: Maximum training iterations
            lr: Learning rate
            plot: Whether to plot training progress
            save: Whether to save trained models
            
        Returns:
            Dictionary with training information and results
        """        
        if self.is_trained or self.is_pretrained_loaded:
            return 
        print(f"\n=== Training GP with {kernel_type} kernel ===")
        start_time = time.time()
        
        # Import kernel builder
        try:
            from src.models.kernels import get_kernel_builder
            build_gp = get_kernel_builder(kernel_type)
        except ImportError:
            print("Warning: Using fallback hierarchical kernel")
            build_gp = build_hierarchical_gp
        
        # Train models for each radius bin
        trained_models = []
        trained_params = []
        training_losses = []
        
        print(f"Training {self.n_radius_bins} GP models...")
        
        for i in trange(self.n_radius_bins, desc="Training GP models"):
            initial_params = initialize_gp_parameters(
                N_COSMO_PARAMS, len(self.k_bins)  # n_cosmo_params, n_k_bins
            )
            
            # Train single model
            lr = GP_TRAINING_DEFAULTS.get('learning_rate', 3e-4)
            gp_model, best_params, losses = train_single_gp_model(
                build_gp, self.X_train, self.y_train[:, i],
                initial_params, maxiter=maxiter, lr=lr
            )
            
            trained_models.append(gp_model)
            trained_params.append(best_params)
            training_losses.append(losses)
        
        train_time = time.time() - start_time
        
        # Store training results
        self.trained_models = trained_models
        self.trained_params = trained_params
        self.is_trained = True
        
        self.training_info = {
            'kernel_type': kernel_type,
            'train_time': train_time,
            'maxiter': maxiter,
            'lr': lr,
            'training_losses': training_losses,
            'n_models': len(trained_models),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"Training completed in {train_time:.1f}s")
        
        # Plot training progress
        if plot:
            self._plot_training_progress()
        
        # Save models
        if save:
            self._save_models()
        
        return self.training_info
    
    def test(
        self, 
        mode: str = 'test',
        plot: bool = True,
        save_plots: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate trained models on test data with comprehensive metrics.
        
        Args:
            plot: Whether to generate evaluation plots
            save_plots: Whether to save plots to disk
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained and not self.is_pretrained_loaded:
            raise ValueError("Model must be trained/loaded before testing. Call train() first.")
        
        print(f"\n=== Testing GP Models ===")
        start_time = time.time()
        
        # Generate predictions on test set
        pred_means = []
        pred_vars = []
        
        if mode == 'train':
            X_eval = self.X_train
            y_eval = self.y_train
            print("Evaluating on training set...")
        elif mode == 'test' and self.X_val is None:
            X_eval = self.X_test
            y_eval = self.y_test
            print("Evaluating on test set...")
        else:
            raise ValueError("Invalid mode or missing validation data.")

        print("Generating predictions...")
        for i, gp_model in enumerate(tqdm(self.trained_models, desc="Predicting")):
            _, cond_gp = gp_model.condition(self.y_train[:, i], X_eval)
            pred_means.append(cond_gp.mean)
            pred_vars.append(cond_gp.variance)
        
        pred_means = np.array(pred_means)  # Shape: (n_radius_bins, n_test_samples)
        pred_vars = np.array(pred_vars)
        
        pred_time = time.time() - start_time
        print(f"Predictions generated in {pred_time:.1f}s")
        
        # Compute metrics
        metrics = self._compute_metrics(pred_means, pred_vars, y_eval.T)

        # Store test results
        self.test_results = {
            'pred_means': pred_means,
            'pred_vars': pred_vars,
            'metrics': metrics,
            'pred_time': pred_time
        }
        
        # Generate plots
        if plot:
            self._plot_test_results(save=save_plots)
        
        print(f"\nTest Results Summary:")
        print(f"  - MAPE: {metrics['mape']:.1f}%")
        print(f"  - MSE: {metrics['mse']:.2e}")
        print(f"  - High-radius MAPE: {metrics['high_radius_mape']:.1f}%")
        
        return metrics
    

    def pred(self, X_new: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            X_new: New input features (shape: [n_samples, n_features])
            
        Returns:
            Tuple of (pred_means, pred_vars) with shapes [n_radius_bins, n_samples]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions. Call train() first.")
        
        pred_means = []
        pred_vars = []
        
        for i, gp_model in enumerate(self.trained_models):
            _, cond_gp = gp_model.condition(self.y_train[:, i], X_new)
            pred_means.append(cond_gp.mean)
            pred_vars.append(cond_gp.variance)
        
        return jnp.array(pred_means), jnp.array(pred_vars)
    
    def _compute_metrics(
        self, 
        pred_means: np.ndarray, 
        pred_vars: np.ndarray, 
        y_true: np.ndarray
    ) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics."""
        # Take mean across test samples for each radius bin
        pred_median = np.mean(pred_means, axis=1)
        true_median = np.median(y_true, axis=1)
        
        # Remove invalid data points
        valid_mask = ~(np.isnan(pred_median) | np.isnan(true_median) | (true_median == 0))
        
        if not np.any(valid_mask):
            return {'error': 'No valid predictions'}
        
        pred_valid = pred_median[valid_mask]
        true_valid = true_median[valid_mask]
        
        # Compute metrics
        mse = np.mean((pred_valid - true_valid)**2)
        mae = np.mean(np.abs(pred_valid - true_valid))
        mape = np.mean(np.abs((pred_valid - true_valid) / true_valid)) * 100
        
        # R-squared
        ss_res = np.sum((true_valid - pred_valid)**2)
        ss_tot = np.sum((true_valid - np.mean(true_valid))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # High radius performance (last 5 bins)
        high_r_mask = valid_mask[-5:]
        if np.any(high_r_mask):
            high_r_mape = np.mean(np.abs((pred_median[-5:][high_r_mask] - true_median[-5:][high_r_mask]) / 
                                       true_median[-5:][high_r_mask])) * 100
        else:
            high_r_mape = np.nan
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'mape': float(mape),
            'r2': float(r2),
            'high_radius_mape': float(high_r_mape),
            'n_valid_bins': int(np.sum(valid_mask))
        }
    
    def _plot_training_progress(self):
        """Plot training loss curves for all models."""
        if not hasattr(self, 'training_info') or 'training_losses' not in self.training_info:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        losses = self.training_info['training_losses']
        
        # Plot individual loss curves (sample a few)
        n_show = min(4, len(losses))
        indices_to_show = np.linspace(0, len(losses)-1, n_show, dtype=int)
        
        for i, idx in enumerate(indices_to_show):
            if i < len(axes):
                axes[i].plot(losses[idx])
                axes[i].set_title(f'Training Loss - Radius Bin {idx}')
                axes[i].set_xlabel('Iteration')
                axes[i].set_ylabel('Negative Log Likelihood')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/training_progress.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Plot summary statistics
        final_losses = [loss[-1] for loss in losses]
        _, _ = plt.subplots(1, 2, figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.r_bins, final_losses, 'bo-')
        plt.xlabel('Radius [Mpc/h]')
        plt.ylabel('Final Training Loss')
        plt.title('Final Training Loss vs Radius')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(final_losses, bins=10, alpha=0.7, edgecolor='black')
        plt.xlabel('Final Training Loss')
        plt.ylabel('Count')
        plt.title('Distribution of Final Losses')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/training_summary.png', dpi=150, bbox_inches='tight')
        plt.show()

    def _plot_test_results(self, mode: str = 'test', save: bool = True):
        """Generate comprehensive test result plots."""
        if not hasattr(self, 'test_results'):
            return
        
        pred_means = self.test_results['pred_means']
        pred_vars = self.test_results['pred_vars']
        y_test_T = self.y_test.T
        
        # Compute statistics for plotting
        pred_median = np.mean(pred_means, axis=1)
        pred_std = np.mean(np.sqrt(pred_vars), axis=1)
        
        true_median = np.median(y_test_T, axis=1)
        true_lower = np.quantile(y_test_T, 0.25, axis=1)
        true_upper = np.quantile(y_test_T, 0.75, axis=1)
        
        # 1. Profile comparison plot
        plt.figure(figsize=(12, 8))
        
        # Plot ground truth
        plt.errorbar(self.r_bins, true_median, 
                    yerr=[true_median - true_lower, true_upper - true_median],
                    fmt='o', capsize=5, capthick=2, linewidth=2, markersize=6, 
                    color='black', label='Ground Truth')
        
        # Plot predictions
        plt.errorbar(self.r_bins, pred_median, yerr=pred_std,
                    fmt='s', capsize=5, capthick=2, linewidth=2, markersize=6,
                    color='red', label='GP Prediction')
        
        plt.fill_between(self.r_bins, pred_median - pred_std, pred_median + pred_std,
                        color='red', alpha=0.2, label='GP 1σ Uncertainty')
        
        plt.yscale('log')
        plt.xlabel('Radius [Mpc/h]', fontsize=14)
        plt.ylabel(f'{self.filterType} {self.ptype} Profile', fontsize=14)
        plt.title(f'GP Prediction vs Ground Truth', fontsize=16)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(f'{self.save_dir}/{mode}_profile_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # 2. Percentage error plot
        plt.figure(figsize=(12, 6))
        
        percent_error = 100 * (pred_median - true_median) / true_median
        error_std = 100 * pred_std / true_median
        
        plt.plot(self.r_bins, percent_error, 'ro-', linewidth=2, markersize=6, 
                label='Percentage Error')
        plt.fill_between(self.r_bins, percent_error - error_std, percent_error + error_std,
                        color='red', alpha=0.2, label='Error Uncertainty')
        
        plt.axhline(0, color='k', linestyle='--', linewidth=1)
        plt.xlabel('Radius [Mpc/h]', fontsize=14)
        plt.ylabel('Percentage Error [%]', fontsize=14)
        plt.title('GP Prediction Percentage Error', fontsize=16)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(f'{self.save_dir}/{mode}_percentage_error.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # 3. Residual analysis
        residuals = pred_median - true_median
        
        _, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Residuals vs radius
        axes[0].plot(self.r_bins, residuals, 'bo-')
        axes[0].axhline(0, color='k', linestyle='--')
        axes[0].set_xlabel('Radius [Mpc/h]')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Radius')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals vs predictions
        axes[1].scatter(pred_median, residuals, alpha=0.6)
        axes[1].axhline(0, color='k', linestyle='--')
        axes[1].set_xlabel('Predicted Value')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residuals vs Predictions')
        axes[1].grid(True, alpha=0.3)
        
        # Residual histogram
        axes[2].hist(residuals, bins=15, alpha=0.7, edgecolor='black')
        axes[2].set_xlabel('Residuals')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Residual Distribution')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save:
            plt.savefig(f'{self.save_dir}/{mode}_residual_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def _save_models(self):
        """Save trained models and parameters."""
        model_path = f'{self.save_dir}/trained_models.pkl'
        params_path = f'{self.save_dir}/trained_params.pkl'
        info_path = f'{self.save_dir}/training_info.json'
        
        # Save models and parameters
        with open(model_path, 'wb') as f:
            pickle.dump(self.trained_models, f)
        
        with open(params_path, 'wb') as f:
            pickle.dump(self.trained_params, f)
        
        # Save training info (JSON serializable parts only)
        training_info_json = {k: v for k, v in self.training_info.items() 
                             if k != 'training_losses'}  # Skip large arrays
        
        with open(info_path, 'w') as f:
            json.dump(training_info_json, f, indent=2)
        
        print(f"Models saved to {self.save_dir}/")
    
    def _load_pretrained(self, pretrained_dir: str):
        """
        Load previously trained models.
        
        Args:
            model_dir: Directory containing saved models
        """
        if self.is_trained or self.is_pretrained_loaded:
            return 
        
        print(f"\n=== Loading GP from {pretrained_dir} ===")        
    
        if not os.path.exists(pretrained_dir):
            raise FileNotFoundError(f"No pretrained model found at {pretrained_dir}")
        
        model_path = f'{pretrained_dir}/trained_models.pkl'
        params_path = f'{pretrained_dir}/trained_params.pkl'
        info_path = f'{pretrained_dir}/training_info.json'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No trained models found at {model_path}")
        
        with open(model_path, 'rb') as f:
            self.trained_models = pickle.load(f)
        
        with open(params_path, 'rb') as f:
            self.trained_params = pickle.load(f)
        
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                self.training_info = json.load(f)
        
        self.is_pretrained_loaded = True
        print(f"Loaded {len(self.trained_models)} trained models")
    
    def tune_hyperparameters(self, 
                           subset_ratio: float = 0.1,
                           lr_candidates: list = [1e-4, 3e-4, 1e-3, 3e-3],
                           kernel_types: list = ['hierarchical', 'robust', 'physics_informed'],
                           max_iter_tune: int = 500,
                           early_stop_patience: int = 50,
                           n_radius_bins_tune: int = 5) -> dict:
        """
        Fast hyperparameter tuning using a subset of data.
        
        Args:
            subset_ratio: Fraction of training data to use (default: 0.1 = 10%)
            lr_candidates: Learning rates to test
            kernel_types: Kernel types to test  
            max_iter_tune: Max iterations for each tuning run
            early_stop_patience: Stop if no improvement for N iterations
            n_radius_bins_tune: Only tune on first N radius bins for speed
            
        Returns:
            dict: Best hyperparameters and validation scores
        """
        print(f"\n🔍 Starting hyperparameter tuning...")
        print(f"  - Subset ratio: {subset_ratio:.1%} ({int(len(self.X_train) * subset_ratio)} samples)")
        print(f"  - Testing {len(lr_candidates)} learning rates × {len(kernel_types)} kernels")
        print(f"  - Max iterations per run: {max_iter_tune}")
        print(f"  - Early stopping patience: {early_stop_patience}")
        print(f"  - Radius bins for tuning: {min(n_radius_bins_tune, self.n_radius_bins)}")
        
        # Create subset indices
        n_subset = int(len(self.X_train) * subset_ratio)
        subset_indices = np.random.choice(len(self.X_train), n_subset, replace=False)
        
        X_subset = self.X_train[subset_indices]
        y_subset = self.y_train[subset_indices]
        
        # Also create validation subset for faster evaluation
        n_val_subset = min(1000, len(self.X_val))  # Use up to 1000 validation samples
        val_indices = np.random.choice(len(self.X_val), n_val_subset, replace=False)
        X_val_subset = self.X_val[val_indices]
        y_val_subset = self.y_val[val_indices]
        
        results = []
        n_bins_tune = min(n_radius_bins_tune, self.n_radius_bins)
        
        from src.models.kernels import get_kernel_builder
        
        for kernel_type in kernel_types:
            for lr in lr_candidates:
                print(f"\n  Testing {kernel_type} kernel with lr={lr:.0e}")
                
                try:
                    # Get kernel builder
                    build_gp = get_kernel_builder(kernel_type)
                    
                    # Train on subset of radius bins for speed
                    bin_val_losses = []
                    train_times = []
                    
                    for i in range(n_bins_tune):
                        start_time = time.time()
                        
                        # Initialize parameters
                        initial_params = initialize_gp_parameters(
                            N_COSMO_PARAMS, len(self.k_bins)
                        )
                        
                        # Train with validation-based early stopping
                        gp_model, _, train_losses, val_losses = train_single_gp_model_with_validation(
                            build_gp, X_subset, y_subset[:, i], X_val_subset, y_val_subset[:, i],
                            initial_params, maxiter=max_iter_tune, lr=lr, patience=early_stop_patience
                        )
                        
                        train_time = time.time() - start_time
                        train_times.append(train_time)
                        
                        # Use final validation loss from training
                        final_val_loss = val_losses[-1] if val_losses else float('inf')
                        bin_val_losses.append(final_val_loss)
                    
                    # Average metrics across radius bins
                    avg_val_loss = np.mean(bin_val_losses)
                    avg_train_time = np.mean(train_times)
                    total_train_time = np.sum(train_times)
                    
                    result = {
                        'kernel_type': kernel_type,
                        'learning_rate': lr,
                        'val_loss': avg_val_loss,
                        'avg_train_time_per_bin': avg_train_time,
                        'total_train_time': total_train_time,
                        'estimated_full_time': avg_train_time * self.n_radius_bins * (1/subset_ratio),
                        'val_losses_per_bin': bin_val_losses
                    }
                    
                    results.append(result)
                    print(f"    Val Loss: {avg_val_loss:.6f}, Time/bin: {avg_train_time:.1f}s")
                    print(f"    Est. full training time: {result['estimated_full_time']/3600:.1f} hours")
                    
                except Exception as e:
                    print(f"    ❌ Failed: {str(e)}")
                    continue
        
        # Find best configuration
        if not results:
            raise RuntimeError("All hyperparameter combinations failed!")
            
        best_result = min(results, key=lambda x: x['val_loss'])
        
        print(f"\n🏆 Best hyperparameters found:")
        print(f"  - Kernel: {best_result['kernel_type']}")
        print(f"  - Learning rate: {best_result['learning_rate']:.0e}")
        print(f"  - Validation loss: {best_result['val_loss']:.6f}")
        print(f"  - Estimated full training time: {best_result['estimated_full_time']/3600:.1f} hours")
        
        # Store results
        tuning_results = {
            'best_config': {
                'kernel_type': best_result['kernel_type'],
                'learning_rate': best_result['learning_rate']
            },
            'best_val_loss': best_result['val_loss'],
            'estimated_full_time_hours': best_result['estimated_full_time']/3600,
            'all_results': results,
            'tuning_settings': {
                'subset_ratio': subset_ratio,
                'max_iter_tune': max_iter_tune,
                'n_radius_bins_tune': n_radius_bins_tune,
                'n_subset_samples': n_subset,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Save tuning results
        tuning_file = f'{self.save_dir}/hyperparameter_tuning_results.json'
        with open(tuning_file, 'w') as f:
            json.dump(tuning_results, f, indent=2)
        print(f"💾 Tuning results saved to: {tuning_file}")
        
        return tuning_results


"""
USAGE EXAMPLE: Fast Hyperparameter Tuning

# Initialize trainer
trainer = GPTrainer(
    sim_indices_total=[0, 1, 2, 10, 20, 50, 100],  # Small set for demo
    filterType='CAP', 
    ptype='gas',
    train_ratio=0.7, val_ratio=0.2, test_ratio=0.1
)

# Fast hyperparameter search (10% data, 3 kernels, 4 learning rates)
tuning_results = trainer.tune_hyperparameters(
    subset_ratio=0.1,                              # Use 10% of data for speed  
    lr_candidates=[1e-4, 3e-4, 1e-3, 3e-3],      # Learning rates to test
    kernel_types=['hierarchical', 'robust', 'physics_informed'],  # Kernels to test
    max_iter_tune=500,                             # Max iterations per run
    early_stop_patience=50,                        # Early stopping patience
    n_radius_bins_tune=5                           # Only tune on first 5 bins
)

# Get best hyperparameters
best_config = tuning_results['best_config']
print(f"Best kernel: {best_config['kernel_type']}")
print(f"Best learning rate: {best_config['learning_rate']}")
print(f"Estimated full training time: {tuning_results['estimated_full_time_hours']:.1f} hours")

# Train full model with best hyperparameters
training_info = trainer.train(
    kernel_type=best_config['kernel_type'],
    maxiter=5000,  # Use more iterations for final training
    lr=best_config['learning_rate'],
    save=True, plot=True
)

Expected timing:
- Hyperparameter tuning: ~10-30 minutes (3 kernels × 4 LRs × 5 bins × subset)
- Full training with best params: 3-5 days (estimated from subset scaling)
"""
