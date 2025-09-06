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

# sklearn imports (minimal usage)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.interpolate import CubicSpline
from scipy.optimize import differential_evolution

# Local imports
from GP_dataloader import *
from src.config.config import GP_TRAINING_DEFAULTS, NN_GP_DEFAULTS, N_COSMO_PARAMS, TRAINED_MODELS_DIR

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

"""
#### Conditional GP Training Code ####
"""

def prepare_GP_data(sim_indices, filterType='CAP', ptype='gas', log_transform_mass=True):
    """
    Legacy wrapper for prepare_gp_training_data for backward compatibility.
    
    DEPRECATED: Use src.models.gp_trainer.prepare_gp_training_data instead.
    """
    from src.models.gp_trainer import prepare_gp_training_data
    return prepare_gp_training_data(sim_indices, filterType, ptype, log_transform_mass)


def train_conditional_gp(sim_indices_train, build_gp, params=None, maxiter=5000, 
                        filterType='CAP', ptype='gas', log_transform_mass=True, save=False):
    """
    Train GP that learns f(cosmology_params, log_mass, pk_ratio) -> profile_value
    
    This is the main entry point for GP training. It has been refactored to use
    modular functions for better maintainability.
    
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
    """
    # Import modular training functions
    from src.models.gp_trainer import (
        prepare_gp_training_data, train_gp_models_all_bins,
        create_model_info, plot_training_curves, save_trained_models
    )
    
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
    

def build_hierarchical_gp(params, X):
    """
    More sophisticated kernel that treats cosmology and mass differently
    """
    # n_params = params['n_params']
    
    # # Split input dimensions
    # X_cosmo = X[:, :n_params]  # Cosmological parameters
    # X_mass = X[:, n_params]        # Log halo mass
    # X_pk = X[:, n_params+1:]       # PkRatio features

    all_length_scales = jnp.concatenate([
        params["cosmo_length_scales"],
        jnp.array([params["mass_length_scale"]]),
        params["pk_length_scale"]
    ])
    
    total_amplitude = jnp.exp(params["cosmo_amplitude"]) + \
                     jnp.exp(params["log_mass_amplitude"]) + \
                     jnp.exp(params["pk_amplitude"])
    
    kernel = total_amplitude * \
             transforms.Linear(jnp.exp(-all_length_scales), 
                             kernels.Matern52(distance=kernels.distance.L2Distance())
    )

    return GaussianProcess(kernel, X, diag=1e-4)

#### New Flax Config for neural network transformation ####

class MLP(nn.Module):
    """A small MLP used to non-linearly transform the input data from (cosmo_params + mass + PkRatio) to a 1D feature (r_bins)."""
    @nn.compact
    def __call__(self, x):
        # x = nn.Dense(features=200)(x)
        # x = nn.relu(x)
        x = nn.Dense(features=50)(x)
        x = nn.relu(x)
        x = nn.Dense(features=21)(x) 
        return x


class build_NN_gp(nn.Module):
    @nn.compact
    def __call__(self, x_train, y_train, t_test=None):
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
    Train GP that learns f(cosmology_params, log_mass) -> profile_value
    """ 

    
    # Prepare data with mass
    X_train, y_train, r_bins, k_bins = prepare_GP_data(sim_indices_train, filterType=filterType, ptype=ptype, log_transform_mass=log_transform_mass)
    
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

            # Initialize parameters: amplitude -- degree of variation; length_scale -- sensitivity to input changes
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
            best_params = params
            patience = model_info['patience']
            no_improve_count = 0
            
            for step in range(model_info['epochs']):
                params, opt_state, loss_val = update_step(params, opt_state)
                losses.append(float(loss_val))
                # Early stopping
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_params = params
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if step % 200 == 0:
                    t.set_postfix({ "Step": step, "Loss": f"{loss_val:.6f}", "Best": f"{best_loss:.6f}"}, refresh=True)

                if no_improve_count > patience:
                    print(f"  Early stopping at step {step}")
                    break

            print(f"r_bin {r_bin_idx} in {time.time()-start_solver:.2f}s: Final loss = {best_loss:.6f}")

            # Store results
            best_params_list.append(best_params)
            losses_all.append(losses)
            
            # Get the trained GP
            opt_gp = model.apply(best_params, X_train, y_train_bin)[1]
            gp_models.append(opt_gp)  # Store model + params
    
    # Plot training curves: all r_bins on one plot
    plt.figure(figsize=(10, 6))
    for i, losses in enumerate(losses_all):
        plt.plot(losses, label=f"r_bin[{i}]")
    plt.title("Training Loss for All r_bins")
    plt.xlabel("Step")
    plt.ylabel("Negative Log Likelihood")
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    if save:
        os.makedirs(save_dir, exist_ok=True)
   
        with open(os.path.join(save_dir, "best_params_list.pkl"), "wb") as f:
            pickle.dump(best_params_list, f)
        with open(os.path.join(save_dir, "losses_all.pkl"), "wb") as f:
            pickle.dump(losses_all, f)
        plt.savefig(os.path.join(save_dir, "learning_curve.png"))
        with open(os.path.join(save_dir, "model_info.pkl"), "wb") as f:
            pickle.dump(model_info, f)
    
    return gp_models, best_params_list, model_info