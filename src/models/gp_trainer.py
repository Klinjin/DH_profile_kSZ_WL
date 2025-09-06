"""
Gaussian Process training utilities for cosmological analysis.

This module provides functions for training GP models to predict halo profiles
based on cosmological parameters, halo mass, and power spectrum ratios.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jaxopt
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os
from datetime import datetime

from tinygp import GaussianProcess, kernels, transforms
from src.config.config import GP_TRAINING_DEFAULTS, TRAINED_MODELS_DIR, ensure_directory_exists


def prepare_gp_training_data(sim_indices_train, filterType='CAP', ptype='gas', log_transform_mass=True):
    """
    Prepare data for GP training with halo mass as an additional input feature.
    
    Args:
        sim_indices_train: List of simulation indices for training
        filterType: Type of filter to apply ('CAP', 'cumulative', 'dsigma')
        ptype: Particle type ('gas', 'dm', 'star', 'bh', 'total', 'baryon')
        log_transform_mass: Whether to log-transform the halo mass
    
    Returns:
        Tuple containing (X_combined, y, r_bins, k_bins)
    """
    # Import here to avoid circular imports
    from GP_dataloader import load_simulation_data
    
    # Load data using unified function
    r_bins, profiles_ptype, mass_halos, param_halos, k, PkRatio = load_simulation_data(
        sim_indices_train, filterType=filterType, ptype=ptype,
        include_params=True, include_pk=True, include_mass=True,
        aggregate_method='extend'
    )
    
    print(f'Profiles shape: {profiles_ptype.shape}, Mass shape: {mass_halos.shape}, '
          f'Params shape: {param_halos.shape}, PkRatio shape: {PkRatio.shape}')
    
    # Transform data
    if log_transform_mass:
        mass = np.log10(mass_halos).reshape(-1, 1)
        profiles_ptype_safe = np.where(profiles_ptype < 0, 1e-10, profiles_ptype)
        profiles = np.log10(profiles_ptype_safe + 1e-10)  # Avoid log(0)
    else:
        mass = mass_halos.reshape(-1, 1) / 1e13
        profiles = profiles_ptype / 1e13
    
    # Combine all input features
    X_combined = np.concatenate([np.concatenate([param_halos, mass], axis=1), PkRatio], axis=1)
    
    return (jnp.array(X_combined), jnp.array(profiles), jnp.array(r_bins), jnp.array(k[0]))


def initialize_gp_parameters(n_cosmo_params, n_k_bins):
    """
    Initialize GP hyperparameters with sensible defaults.
    
    Args:
        n_cosmo_params: Number of cosmological parameters
        n_k_bins: Number of k bins for power spectrum
    
    Returns:
        Dictionary of initialized parameters
    """
    return {
        "cosmo_amplitude": jnp.float32(0.0),
        "cosmo_length_scales": jnp.zeros(n_cosmo_params),
        "log_mass_amplitude": jnp.float32(0.0),
        "mass_length_scale": jnp.float32(0.0),
        "pk_amplitude": jnp.float32(0.0),
        "pk_length_scale": jnp.zeros(n_k_bins),
        "noise": jnp.float32(GP_TRAINING_DEFAULTS['noise_level'])
    }


def train_single_gp_model(build_gp, X_train, y_train_bin, initial_params, maxiter=5000, lr=3e-4):
    """
    Train a single GP model for one radial bin.
    
    Args:
        build_gp: Function to build GP from parameters
        X_train: Training input features
        y_train_bin: Training targets for this bin
        initial_params: Initial hyperparameters
        maxiter: Maximum iterations for scipy optimizer
        lr: Learning rate for Adam optimizer
    
    Returns:
        Tuple containing (trained_gp, best_params, losses)
    """
    
    @jax.jit
    def loss_fn(params):
        return -build_gp(params, X_train).log_probability(y_train_bin)
    
    # Step 1: Scipy optimization
    solver = jaxopt.ScipyMinimize(fun=loss_fn, maxiter=maxiter)
    soln = solver.run(initial_params)
    best_params = soln.params
    scipy_final_loss = soln.state.fun_val
    
    # Step 2: Adam fine-tuning
    opt = optax.adamw(learning_rate=lr)
    opt_state = opt.init(best_params)
    adam_losses = []
    
    for i in range(100):
        loss_val, grads = jax.value_and_grad(loss_fn)(best_params)
        adam_losses.append(float(loss_val))
        updates, opt_state = opt.update(grads, opt_state, best_params)
        best_params = optax.apply_updates(best_params, updates)
    
    # Build final GP
    trained_gp = build_gp(best_params, X_train)
    combined_losses = [float(scipy_final_loss)] + adam_losses
    
    return trained_gp, best_params, combined_losses


def train_gp_models_all_bins(build_gp, X_train, y_train, r_bins, k_bins, 
                             maxiter=5000, lr=3e-4, log_transform_mass=True):
    """
    Train GP models for all radial bins.
    
    Args:
        build_gp: Function to build GP from parameters
        X_train: Training input features  
        y_train: Training targets (all bins)
        r_bins: Radial bin centers
        k_bins: k-space bins
        maxiter: Maximum iterations for optimization
        lr: Learning rate
        log_transform_mass: Whether mass was log-transformed
    
    Returns:
        Tuple containing (gp_models, best_params_list, losses_all)
    """
    n_cosmo_params = 35  # Fixed for CAMELS
    n_k_bins = k_bins.shape[0]
    
    gp_models = []
    best_params_list = []
    losses_all = []
    
    for r_bin_idx in tqdm(range(len(r_bins)), desc="Training GP for each r_bin"):
        # Get targets for this bin
        y_train_bin = y_train[:, r_bin_idx]
        
        # Initialize parameters
        initial_params = initialize_gp_parameters(n_cosmo_params, n_k_bins)
        
        # Train single model
        trained_gp, best_params, losses = train_single_gp_model(
            build_gp, X_train, y_train_bin, initial_params, maxiter, lr
        )
        
        # Store results
        gp_models.append(trained_gp)
        best_params_list.append(best_params)
        losses_all.append(losses)
    
    return gp_models, best_params_list, losses_all


def save_trained_models(gp_models, best_params_list, model_info, save_dir=None):
    """
    Save trained GP models and metadata to disk.
    
    Args:
        gp_models: List of trained GP models
        best_params_list: List of optimized parameters
        model_info: Dictionary containing model metadata
        save_dir: Directory to save to (optional)
    
    Returns:
        Path where models were saved
    """
    if save_dir is None:
        save_dir = TRAINED_MODELS_DIR
    
    ensure_directory_exists(save_dir)
    
    # Save components
    with open(os.path.join(save_dir, "gp_models.pkl"), "wb") as f:
        pickle.dump(gp_models, f)
    
    with open(os.path.join(save_dir, "best_params_list.pkl"), "wb") as f:
        pickle.dump(best_params_list, f)
    
    with open(os.path.join(save_dir, "model_info.pkl"), "wb") as f:
        pickle.dump(model_info, f)
    
    print(f"Models saved to {save_dir}")
    return save_dir


def plot_training_curves(losses_all, save_path=None):
    """
    Plot training loss curves for all radial bins.
    
    Args:
        losses_all: List of loss curves for each bin
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    for i, losses in enumerate(losses_all):
        plt.plot(losses, label=f"r_bin[{i}]" if i < 5 else "", alpha=0.7)
    
    plt.ylabel("Negative Log Likelihood")
    plt.xlabel("Step Number")
    plt.title("GP Training Loss Curves")
    if len(losses_all) <= 5:
        plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def create_model_info(filterType, ptype, r_bins, k_bins, log_transform_mass, 
                     X_train, y_train, maxiter, lr):
    """
    Create model information dictionary for saving.
    
    Args:
        filterType: Filter type used
        ptype: Particle type
        r_bins: Radial bins
        k_bins: k-space bins  
        log_transform_mass: Whether mass was log-transformed
        X_train: Training inputs
        y_train: Training targets
        maxiter: Max iterations used
        lr: Learning rate used
    
    Returns:
        Dictionary with model metadata
    """
    return {
        'timestamp': datetime.now().isoformat(),
        'optimizer': 'scipy + adamw',
        'maxiter': maxiter,
        'learning_rate': lr,
        'filterType': filterType,
        'ptype': ptype,
        'r_bins': r_bins,
        'k_bins': k_bins,
        'log_mass_transform': log_transform_mass,
        'n_training_points': X_train.shape[0],
        'n_features': X_train.shape[1],
        'n_radial_bins': len(r_bins),
        'n_cosmo_params': 35,
        'n_k_bins': k_bins.shape[0] if k_bins is not None else 0
    }