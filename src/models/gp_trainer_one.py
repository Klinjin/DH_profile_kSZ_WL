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
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime
from functools import partial
from tqdm import tqdm, trange
import pickle
import json
from typing import Dict, List, Tuple, Optional, Callable, Any

# JAX and optimization imports
import jax
import jax.numpy as jnp
from jax import random
import jaxopt
import optax

# TinyGP imports
import tinygp
from tinygp import GaussianProcess, kernels, transforms

# Flax (neural network) imports
import flax.linen as nn
from flax.linen.initializers import zeros

# Local imports
from src.config.config import GP_TRAINING_DEFAULTS, NN_GP_DEFAULTS, N_COSMO_PARAMS, TRAINED_MODELS_DIR
from src.data.sim_dataloader import SimulationDataLoader, DataLoaderConfig, prepare_gp_training_data

# Import kernel builders
try:
    from src.models.kernels import build_hierarchical_gp, get_kernel_builder, initialize_gp_parameters, initialize_physics_informed_params
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


##########################################################
#### DEPRECATED FUNCTIONS FOR STANDALONE GP TRAINING ####
##########################################################

def train_single_gp_model(build_gp, X_train, y_train_bin, initial_params, maxiter=5000, lr=3e-4):

    """Robust two-stage optimization with error handling."""

    @jax.jit
    def loss_fn(params):
        gp = build_gp(params, X_train)
        return -gp.condition(y_train_bin).log_probability

    # Stage 1: Scipy global optimization for structure
    solver = jaxopt.ScipyMinimize(fun=loss_fn, maxiter=maxiter//4, method='L-BFGS-B')
    soln = solver.run(initial_params)
    best_params = soln.params
    scipy_final_loss = soln.state.fun_val

    
    # Stage 2: Adam fine-tuning with gradient clipping
    opt = optax.adamw(learning_rate=lr, weight_decay=1e-5)
    opt_state = opt.init(best_params)
    adam_losses = []
    best_loss = scipy_final_loss
    no_improve_count = 0
    
    # Training loop with early stopping
    for step in range(maxiter//2):
        loss_val, grads = jax.value_and_grad(loss_fn)(best_params)
        adam_losses.append(loss_val)
        # Numerical stability checks
        if jnp.isnan(loss_val):
            print(f"NaN detected at step {step}, stopping")
            break
            
        # Early stopping based on improvement
        if loss_val < best_loss:
            best_loss = loss_val
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count > maxiter//20:
            break
            
        updates, opt_state = opt.update(grads, opt_state, best_params)
        best_params = optax.apply_updates(best_params, updates)
    
    # Build final GP with best parameters
    optimized_gp = build_gp(best_params, X_train)
    combined_losses = [float(scipy_final_loss)] + adam_losses

    
    return optimized_gp, best_params, combined_losses


def train_single_gp_model_with_validation(build_gp, X_train, y_train_bin, X_val, y_val_bin, 
                                         initial_params, maxiter=5000, lr=3e-4, patience=50):
    """
    Train a single GP model with validation-based early stopping.
    
    Args:
        build_gp: GP builder function
        X_train, y_train_bin: Training data
        X_val, y_val_bin: Validation data  
        initial_params: Initial parameter values
        maxiter: Maximum iterations
        lr: Learning rate
        patience: Early stopping patience (iterations without improvement)
    
    Returns:
        Tuple: (optimized_gp, best_params, train_losses, val_losses)
    """
    
    @jax.jit
    def loss_fn(params):
        gp = build_gp(params, X_train)
        return -gp.condition(y_train_bin).log_probability
    
    @jax.jit
    def val_loss_fn(params):
        # Build GP with training data and evaluate on validation data
        gp = build_gp(params, X_train)
        conditioned_gp = gp.condition(y_train_bin)
        pred_mean, pred_var = conditioned_gp.predict(X_val)
        return jnp.mean((pred_mean - y_val_bin)**2)  # MSE on validation set
    
    # Stage 1: Scipy global optimization for structure
    solver = jaxopt.ScipyMinimize(fun=loss_fn, maxiter=maxiter//4, method='L-BFGS-B')
    soln = solver.run(initial_params)
    best_params = soln.params
    scipy_final_loss = soln.state.fun_val
    
    # Stage 2: Adam fine-tuning with validation-based early stopping
    opt = optax.adamw(learning_rate=lr, weight_decay=1e-5)
    opt_state = opt.init(best_params)
    
    train_losses = [float(scipy_final_loss)]
    val_losses = [float(val_loss_fn(best_params))]
    
    best_val_loss = val_losses[0]
    best_params_val = best_params
    no_improve_count = 0
    
    # Training loop with validation-based early stopping
    for step in range(maxiter//2):
        # Training step
        train_loss_val, grads = jax.value_and_grad(loss_fn)(best_params)
        train_losses.append(float(train_loss_val))
        
        # Numerical stability checks
        if jnp.isnan(train_loss_val):
            print(f"NaN detected at step {step}, stopping")
            break
            
        # Update parameters
        updates, opt_state = opt.update(grads, opt_state, best_params)
        best_params = optax.apply_updates(best_params, updates)
        
        # Validation evaluation (every few steps to save computation)
        if step % 5 == 0 or step < 10:  # Check validation more frequently at start
            val_loss_val = val_loss_fn(best_params)
            val_losses.append(float(val_loss_val))
            
            # Early stopping based on validation loss improvement
            if val_loss_val < best_val_loss:
                best_val_loss = val_loss_val
                best_params_val = best_params  # Save best validation parameters
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            if no_improve_count > patience:
                print(f"Early stopping at step {step} (patience={patience})")
                break
    
    # Build final GP with best validation parameters
    optimized_gp = build_gp(best_params_val, X_train)
    
    return optimized_gp, best_params_val, train_losses, val_losses


def train_gp_models_all_bins(build_gp, X_train, y_train, r_bins, k_bins, 
                           maxiter=5000, lr=3e-4, verbose=True):

    n_cosmo_params = N_COSMO_PARAMS
    n_k_bins = len(k_bins)
    
    # Initialize parameters once
    initial_params = initialize_gp_parameters(n_cosmo_params, n_k_bins)
    
    gp_models = []
    best_params_list = []
    losses_all = []
    
    iterator = tqdm(range(len(r_bins)), desc="Training GP models") if verbose else range(len(r_bins))
    
    for i in iterator:
        y_train_bin = y_train[:, i]
        
        # Skip bins with all NaN values
        if jnp.all(jnp.isnan(y_train_bin)):
            print(f"Warning: Bin {i} has all NaN values, skipping...")
            gp_models.append(None)
            best_params_list.append(None)
            losses_all.append([])
            continue
        
        # Train model for this bin
        gp, best_params, losses = train_single_gp_model(
            build_gp, X_train, y_train_bin, initial_params, maxiter, lr
        )
        
        gp_models.append(gp)
        best_params_list.append(best_params)
        losses_all.append(losses)
        
        if verbose:
            final_loss = losses[-1]
            iterator.set_postfix({"bin": i, "loss": f"{final_loss:.4f}"})
    
    return gp_models, best_params_list, losses_all


def save_trained_models(gp_models, best_params_list, model_info, save_dir=None):
    """
    Save trained GP models and metadata.
    
    Args:
        gp_models: List of trained GP models
        best_params_list: List of optimized parameters
        model_info: Dictionary with model metadata
        save_dir: Directory to save models (auto-generated if None)
    """
    if save_dir is None:
        save_dir = f"{TRAINED_MODELS_DIR}/gp_models_{datetime.now().strftime('%m%d%y_%H%M')}"
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save parameters and metadata
    with open(f"{save_dir}/best_params_list.pkl", "wb") as f:
        pickle.dump(best_params_list, f)
    
    with open(f"{save_dir}/model_info.pkl", "wb") as f:
        pickle.dump(model_info, f)
    
    print(f"Models saved to: {save_dir}")
    return save_dir


def plot_training_curves(losses_all, save_path=None):
    """
    Plot training loss curves for all radial bins.
    
    Args:
        losses_all: List of loss arrays for each bin
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    for i, losses in enumerate(losses_all):
        if len(losses) > 0:
            plt.plot(losses, label=f'Bin {i}', alpha=0.7)
    
    plt.xlabel('Iteration')
    plt.ylabel('Negative Log Likelihood')
    plt.title('GP Training Loss Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.tight_layout()
    plt.show()


def create_model_info(filterType, ptype, r_bins, k_bins, log_transform_mass,
                     X_train, y_train, maxiter, lr):
    """
    Create metadata dictionary for trained models.
    
    Returns:
        Dictionary with model training information
    """
    return {
        'timestamp': datetime.now().isoformat(),
        'filterType': filterType,
        'ptype': ptype,
        'n_radius_bins': len(r_bins),
        'n_k_bins': len(k_bins),
        'log_transform_mass': log_transform_mass,
        'n_training_samples': X_train.shape[0],
        'n_features': X_train.shape[1],
        'maxiter': maxiter,
        'learning_rate': lr,
        'optimizer': 'adamw',
        'r_bins': r_bins,
        'k_bins': k_bins
    }


def train_conditional_gp(sim_indices_train, build_gp, params=None, maxiter=5000, 
                        filterType='CAP', ptype='gas', log_transform_mass=True, save=False):
    """
    Main entry point for GP training. Train GP that learns f(cosmology_params, log_mass, pk_ratio) -> profile_value
    
    Args:
        sim_indices_train: List of simulation indices for training
        build_gp: Function to build GP from parameters
        params: Unused (kept for backward compatibility)
        maxiter: Maximum iterations for optimization
        filterType: Filter type ('CAP', 'cumulative', 'dsigma')
        ptype: Particle type ('gas', 'dm', 'star', 'bh', 'total', 'baryon')
        log_transform_mass: Whether to log-transform halo mass
        save: Whether to save trained models
    
    Returns:
        Tuple containing (gp_models, best_params_list, model_info)
        Notes: when applying on test data, use GaussianProcess(kernel, X_train).condition(y_train, X_test)
    """
    # Step 1: Prepare training data
    X_train, y_train, r_bins, k_bins = prepare_gp_training_data(
        sim_indices_train, filterType=filterType, ptype=ptype, 
        log_transform_mass=log_transform_mass
    )
    
    # Step 2: Train GP models for all radial bins
    lr = GP_TRAINING_DEFAULTS.get('learning_rate', 3e-4)
    gp_models, best_params_list, losses_all = train_gp_models_all_bins(
        build_gp, X_train, y_train, r_bins, k_bins, maxiter=maxiter, lr=lr
    )
    
    # Step 3: Create model metadata
    model_info = create_model_info(
        filterType, ptype, r_bins, k_bins, log_transform_mass,
        X_train, y_train, maxiter, lr
    )
    # Add legacy fields for backward compatibility
    model_info.update({
        'gp_params': best_params_list,
        'gp_builder': str(build_gp),
        'optimizer': 'scipy + adamw'
    })
    
    # Step 4: Plot training curves
    plot_training_curves(losses_all)
    
    # Step 5: Save models if requested
    if save:
        save_trained_models(gp_models, best_params_list, model_info)
    
    return gp_models, best_params_list, model_info



# Neural Network + GP Hybrid Models
class MLP(nn.Module):
    """A small MLP used to non-linearly transform the input data from (cosmo_params + mass + PkRatio) to a 1D feature (r_bins)."""
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=50)(x)
        x = nn.relu(x)
        x = nn.Dense(features=21)(x) 
        return x


class build_NN_gp(nn.Module):
    @nn.compact
    def __call__(self, x_train, y_train, t_test=None):
        #  Notes: when applying on test data, use GaussianProcess(kernel, X_train).condition(y_train, X_test)

        n_cosmo_params=35
        n_k_bins=len(x_train[0]) - n_cosmo_params - 1  # Assuming last dimension is (n_cosmo + 1 + n_k)

        # Set up a base  kernel --> CANNOT SPECIFY n_params or n_k_bins here since they are not static
        cosmo_amplitude = self.param("cosmo_amplitude", zeros, ())
        log_mass_amplitude = self.param("log_mass_amplitude", zeros, ())
        log_jitter = self.param("log_jitter", zeros, ())
        noise = self.param("noise", zeros, ())
        cosmo_length_scales = self.param("cosmo_length_scales", zeros, ())
        mass_length_scale = self.param("mass_length_scale", zeros, ())
        pk_amplitude = self.param("pk_amplitude", zeros, ())
        pk_length_scale = self.param("pk_length_scale", zeros, ())
        

        total_amplitude = jnp.exp(cosmo_amplitude) + \
                        jnp.exp(log_mass_amplitude) + \
                        jnp.exp(pk_amplitude)

        base_kernel = total_amplitude * \
                transforms.Linear(jnp.exp(-cosmo_length_scales), 
                                kernels.Matern52(distance=kernels.distance.L2Distance())
        )

        mlp = MLP(parent=None)
        mlp_params = self.param('mlp_params', mlp.init, x_train[:1,:])

        apply_fn = lambda x: mlp.apply(mlp_params, x)
        kernel = transforms.Transform(apply_fn, base_kernel)


        # Evaluate and return the GP negative log likelihood as usual with the
        # transformed features
        gp = GaussianProcess(
            kernel, x_train, diag=noise**2 + jnp.exp(2 * log_jitter)
        )

        if t_test is None:
            cond = gp.condition(y_train)
            log_prob = cond.log_probability
            gp_cond = cond.gp
        else:
            log_prob, gp_cond = gp.condition(y_train, t_test)

        # We return the loss, the conditional mean and variance, and the
        # transformed input parameters
        return (
            -log_prob,
            gp_cond,
            apply_fn(x_train),
        )


def train_NN_gp(sim_indices_train, filterType='CAP', ptype='gas', 
                             log_transform_mass=True, save=True, model_name='NN_gp'):
    """
    Train Neural Network + GP hybrid models for all radial bins.
    
    Args:
        sim_indices_train: List of simulation indices for training
        filterType: Filter type ('CAP', 'cumulative', 'dsigma')
        ptype: Particle type ('gas', 'dm', 'star', 'bh', 'total', 'baryon')
        log_transform_mass: Whether to log-transform halo mass
        save: Whether to save trained models
        model_name: Name prefix for saved models
    
    Returns:
        Tuple containing (gp_models, best_params_list, model_info)
    """
    # Prepare data with mass
    X_train, y_train, r_bins, k_bins = prepare_gp_training_data(sim_indices_train, filterType=filterType, ptype=ptype, log_transform_mass=log_transform_mass)
    
    today_str = datetime.now().strftime("%m%d%H")
    save_dir = f"trained_gp_models/{model_name}_{today_str}"

    # Collect model info and hyperparameters
    model_info = {
            'save_dir': save_dir,
            'optimizer': 'adamw',
            'learning_rate_init': 1e-3,
            'decay_rate': 0.9,
            'decay_steps': 500,
            'weight_decay': 1e-4,
            'clip_norm': 1.0,
            'epochs': 2000,
            'patience': 200,
            'r_bins': r_bins,
            'k_bins': k_bins,
            'log_mass_transform': log_transform_mass,
            'filterType': filterType,
            'ptype': ptype,
            'X_train': X_train,
            'y_train': y_train
        }
    gp_models = []
    best_params_list = []
    losses_all = []
    
    with trange(len(r_bins), desc="Training GP for each r_bin") as t:
        for r_bin_idx in t:
            # Prepare data for this r_bin
            y_train_bin = y_train[:, r_bin_idx]

            model = build_NN_gp()
            lr_schedule = optax.exponential_decay(
                init_value=model_info['learning_rate_init'],
                transition_steps=model_info['decay_steps'],
                decay_rate=model_info['decay_rate']
            )
            optimizer = optax.chain(
                optax.clip_by_global_norm(model_info['clip_norm']),  # Gradient clipping
                optax.adamw(learning_rate=lr_schedule, weight_decay=model_info['weight_decay'])
            )

            # Initialize parameters
            @jax.jit
            def loss(params):
                return model.apply(params, X_train, y_train_bin)[0]
            
            @jax.jit
            def update_step(params, opt_state):
                loss_val, grads = jax.value_and_grad(
                lambda p: model.apply(p, X_train, y_train_bin)[0], 
                has_aux=False
                )(params)
                
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                
                return params, opt_state, loss_val

            params = model.init(jax.random.PRNGKey(1234 + r_bin_idx), X_train, y_train_bin)

            init_loss = loss(params)
            print(f"Start Adamw training for r_bin {r_bin_idx}: Initial loss = {init_loss}")

            start_solver = time.time()

            opt_state = optimizer.init(params)
            # Training loop
            losses = []
            best_loss = float('inf')
            best_params = params.copy()
            patience_counter = 0

            for epoch in range(model_info['epochs']):
                params, opt_state, loss_val = update_step(params, opt_state)
                losses.append(float(loss_val))
                
                # Early stopping
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_params = params.copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Update progress
                if epoch % 100 == 0 or patience_counter >= model_info['patience']:
                    t.set_postfix({
                        "Step": epoch,
                        "Loss": f"{loss_val:.6f}",
                        "Best": f"{best_loss:.6f}"
                    })
                
                if patience_counter >= model_info['patience']:
                    print(f"Early stopping at epoch {epoch}")
                    break

            end_solver = time.time()
            print(f"r_bin {r_bin_idx} in {end_solver - start_solver:.2f}s: Final loss = {best_loss:.6f}")

            # Store results
            gp_models.append(model)
            best_params_list.append(best_params)
            losses_all.append(losses)

    # Save results if requested
    if save:
        os.makedirs(save_dir, exist_ok=True)
        
        with open(f"{save_dir}/best_params_nn_list.pkl", "wb") as f:
            pickle.dump(best_params_list, f)
        
        with open(f"{save_dir}/model_info_nn_list.pkl", "wb") as f:
            pickle.dump(model_info, f)
        
        print(f"NN+GP models saved to: {save_dir}")

    return gp_models, best_params_list, model_info