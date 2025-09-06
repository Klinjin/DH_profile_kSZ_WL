"""
Improved GP training with better kernel designs and robust optimization.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jaxopt
import optax
from tqdm import tqdm
import time
import warnings

try:
    from src.models.gp_trainer import (
        prepare_gp_training_data, save_trained_models, create_model_info
    )
except ImportError:
    # Fallback to train_GP functions
    from train_GP import prepare_GP_data as prepare_gp_training_data
    def save_trained_models(*args, **kwargs):
        print("save_trained_models not available - models not saved")
    def create_model_info(*args, **kwargs):
        return {'timestamp': time.time()}

try:
    from src.models.improved_kernels import (
        initialize_physics_informed_params, get_kernel_builder, KERNEL_REGISTRY
    )
except ImportError:
    print("Warning: improved_kernels not available, some features may not work")


def preprocess_data_robust(X_train, y_train, remove_outliers=True, outlier_threshold=4.0):
    """
    Robust data preprocessing with outlier detection and NaN handling.
    
    Args:
        X_train: Training inputs
        y_train: Training targets
        remove_outliers: Whether to remove statistical outliers
        outlier_threshold: Z-score threshold for outlier detection
        
    Returns:
        Processed X_train, y_train, and boolean mask of valid samples
    """
    print("Applying robust data preprocessing...")
    
    # Handle NaN values
    nan_mask_y = ~jnp.isnan(y_train)
    nan_mask_X = ~jnp.any(jnp.isnan(X_train), axis=1)
    valid_mask = nan_mask_y & nan_mask_X
    
    if remove_outliers:
        # Remove statistical outliers using z-score
        z_scores = jnp.abs((y_train - jnp.nanmean(y_train)) / jnp.nanstd(y_train))
        outlier_mask = z_scores < outlier_threshold
        valid_mask = valid_mask & outlier_mask
        
    n_removed = len(y_train) - jnp.sum(valid_mask)
    if n_removed > 0:
        print(f"Removed {n_removed} samples ({n_removed/len(y_train)*100:.1f}%) due to NaN/outliers")
    
    return X_train[valid_mask], y_train[valid_mask], valid_mask


def train_improved_gp_single_bin(kernel_builder, X_train, y_train_bin, r_bin_idx=0, 
                                maxiter=2000, lr=1e-3, use_robust_preprocessing=True):
    """
    Train a single GP model with improved optimization and preprocessing.
    
    Args:
        kernel_builder: Function to build GP from parameters
        X_train: Training inputs
        y_train_bin: Training targets for this radial bin
        r_bin_idx: Index of radial bin (for adaptive noise)
        maxiter: Maximum optimization iterations
        lr: Learning rate
        use_robust_preprocessing: Whether to apply robust preprocessing
        
    Returns:
        Tuple of (trained_gp, best_params, training_info)
    """
    start_time = time.time()
    
    # Robust preprocessing
    if use_robust_preprocessing:
        X_clean, y_clean, valid_mask = preprocess_data_robust(X_train, y_train_bin)
        if len(y_clean) < 0.5 * len(y_train_bin):
            warnings.warn(f"Bin {r_bin_idx}: Only {len(y_clean)}/{len(y_train_bin)} samples remain after preprocessing")
    else:
        X_clean, y_clean = X_train, y_train_bin
        
    # Data-driven parameter initialization
    initial_params = initialize_physics_informed_params(X_clean, y_clean)
    
    # Handle adaptive noise for radial dependence
    if hasattr(kernel_builder, '__name__') and 'adaptive' in kernel_builder.__name__:
        build_gp_func = lambda params: kernel_builder(params, X_clean, r_bin_idx)
    else:
        build_gp_func = lambda params: kernel_builder(params, X_clean)
    
    @jax.jit
    def loss_fn(params):
        try:
            gp = build_gp_func(params)
            return -gp.log_probability(y_clean)
        except Exception:
            return 1e6  # Return high loss for numerical issues
    
    # Multi-stage optimization: scipy first, then Adam
    print(f"  Optimizing bin {r_bin_idx} with {len(y_clean)} samples...")
    
    # Stage 1: Scipy optimization (global structure)
    try:
        solver = jaxopt.ScipyMinimize(fun=loss_fn, maxiter=maxiter//4, method='L-BFGS-B')
        soln = solver.run(initial_params)
        best_params = soln.params
        scipy_loss = float(soln.state.fun_val)
    except Exception as e:
        print(f"    Scipy optimization failed: {e}, using initial params")
        best_params = initial_params  
        scipy_loss = float(loss_fn(initial_params))
    
    # Stage 2: Adam fine-tuning (local optimization)
    opt = optax.adamw(learning_rate=lr, weight_decay=1e-5)
    opt_state = opt.init(best_params)
    
    # JAX tree compatibility - determine once outside loop
    try:
        tree_leaves = getattr(jax.tree, 'leaves', None) or getattr(jax.tree_util, 'tree_leaves', jax.tree_leaves)
    except AttributeError:
        tree_leaves = jax.tree_util.tree_leaves
    
    losses = [scipy_loss]
    best_loss = scipy_loss
    patience = 100
    no_improve_count = 0
    
    for step in range(maxiter//2):
        loss_val, grads = jax.value_and_grad(loss_fn)(best_params)
        
        # Check for numerical issues  
        if jnp.isnan(loss_val) or jnp.any(jnp.isnan(jnp.concatenate([
            jnp.atleast_1d(g) if jnp.ndim(g) == 0 else g.flatten() 
            for g in tree_leaves(grads)
        ]))):
            print(f"    NaN detected at step {step}, stopping optimization")
            break
            
        losses.append(float(loss_val))
        
        if loss_val < best_loss:
            best_loss = float(loss_val)
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count > patience:
            break
            
        updates, opt_state = opt.update(grads, opt_state, best_params)
        best_params = optax.apply_updates(best_params, updates)
    
    # Build final GP
    final_gp = build_gp_func(best_params)
    
    training_info = {
        'r_bin_idx': r_bin_idx,
        'training_time': time.time() - start_time,
        'final_loss': best_loss,
        'n_samples': len(y_clean),
        'n_iterations': len(losses),
        'losses': losses
    }
    
    print(f"    Completed in {training_info['training_time']:.2f}s, loss: {best_loss:.2f}")
    
    return final_gp, best_params, training_info


def train_improved_conditional_gp(sim_indices_train, kernel_name='multiscale', 
                                 filterType='CAP', ptype='gas', log_transform_mass=True,
                                 maxiter=2000, lr=1e-3, save=False, model_name=None):
    """
    Train improved conditional GP with modern kernel designs.
    
    Args:
        sim_indices_train: Training simulation indices
        kernel_name: Name of kernel to use ('multiscale', 'physics_informed', etc.)
        filterType: Filter type for profiles
        ptype: Particle type
        log_transform_mass: Whether to log-transform mass
        maxiter: Maximum optimization iterations
        lr: Learning rate
        save: Whether to save trained models
        model_name: Name for saved model
        
    Returns:
        Tuple of (gp_models, best_params_list, model_info)
    """
    print(f"\n=== Training Improved GP with {kernel_name} kernel ===")
    
    # Get kernel builder
    kernel_builder = get_kernel_builder(kernel_name)
    
    # Prepare training data
    X_train, y_train, r_bins, k_bins = prepare_gp_training_data(
        sim_indices_train, filterType=filterType, ptype=ptype, 
        log_transform_mass=log_transform_mass
    )
    
    print(f"Training data: {X_train.shape[0]} samples, {len(r_bins)} radial bins")
    print(f"Feature dimensions: {X_train.shape[1]} (cosmo: 35, mass: 1, pk: {X_train.shape[1]-36})")
    
    # Train GP for each radial bin
    gp_models = []
    best_params_list = []
    training_info_list = []
    
    for r_bin_idx in tqdm(range(len(r_bins)), desc=f"Training {kernel_name} GP"):
        y_train_bin = y_train[:, r_bin_idx]
        
        gp_model, best_params, training_info = train_improved_gp_single_bin(
            kernel_builder, X_train, y_train_bin, r_bin_idx, maxiter, lr
        )
        
        gp_models.append(gp_model)
        best_params_list.append(best_params)
        training_info_list.append(training_info)
    
    # Create model info
    total_time = sum(info['training_time'] for info in training_info_list)
    final_losses = [info['final_loss'] for info in training_info_list]
    
    model_info = create_model_info(
        filterType, ptype, r_bins, k_bins, log_transform_mass,
        X_train, y_train, maxiter, lr
    )
    
    # Add improvement-specific info
    model_info.update({
        'kernel_name': kernel_name,
        'total_training_time': total_time,
        'final_losses': final_losses,
        'mean_loss': float(jnp.mean(jnp.array(final_losses))),
        'training_info_per_bin': training_info_list,
        'optimizer': 'scipy + adamw (improved)',
        'data_preprocessing': 'robust'
    })
    
    print(f"\nTraining completed in {total_time:.1f}s")
    print(f"Mean final loss: {model_info['mean_loss']:.2f}")
    print(f"Loss range: [{min(final_losses):.2f}, {max(final_losses):.2f}]")
    
    # Save if requested
    if save:
        if model_name is None:
            model_name = f"improved_gp_{kernel_name}"
        save_trained_models(gp_models, best_params_list, model_info, 
                          save_dir=f"trained_gp_models/{model_name}")
    
    return gp_models, best_params_list, model_info


def compare_kernel_performance(sim_indices_train, sim_indices_test=None, 
                             kernel_names=None, **kwargs):
    """
    Compare performance of different kernel designs.
    
    Returns dictionary with trained models and performance metrics for each kernel.
    """
    if kernel_names is None:
        kernel_names = ['hierarchical', 'multiscale', 'physics_informed', 'robust']
    
    if sim_indices_test is None:
        # Use a subset of training data for quick testing
        sim_indices_test = sim_indices_train[:min(10, len(sim_indices_train)//4)]
    
    results = {}
    
    for kernel_name in kernel_names:
        print(f"\n{'='*50}")
        print(f"Testing {kernel_name} kernel")
        print(f"{'='*50}")
        
        try:
            gp_models, best_params_list, model_info = train_improved_conditional_gp(
                sim_indices_train, kernel_name=kernel_name, **kwargs
            )
            
            results[kernel_name] = {
                'gp_models': gp_models,
                'best_params_list': best_params_list,
                'model_info': model_info,
                'success': True
            }
            
        except Exception as e:
            print(f"Failed to train {kernel_name}: {e}")
            results[kernel_name] = {
                'error': str(e),
                'success': False
            }
    
    return results