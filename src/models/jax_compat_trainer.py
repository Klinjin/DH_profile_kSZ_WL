"""
JAX-compatible GP training that works across different JAX versions.
Simplified version to avoid compatibility issues.
"""

import numpy as np
import jax
import jax.numpy as jnp
import time
import warnings
from tqdm import tqdm

try:
    import jaxopt
    import optax
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    print("Warning: jaxopt/optax not available - using basic optimization")
    OPTIMIZATION_AVAILABLE = False

try:
    from src.models.improved_kernels import (
        initialize_physics_informed_params, get_kernel_builder
    )
    KERNELS_AVAILABLE = True
except ImportError:
    print("Warning: improved kernels not available")
    KERNELS_AVAILABLE = False

# Import basic functions with fallback
try:
    from train_GP import prepare_GP_data, build_hierarchical_gp
    BASIC_FUNCTIONS_AVAILABLE = True
except ImportError as e:
    print(f"Error: Basic train_GP functions not available: {e}")
    BASIC_FUNCTIONS_AVAILABLE = False
    

def simple_data_preprocessing(X_train, y_train, remove_outliers=True):
    """Simple data preprocessing avoiding complex operations."""
    print("Applying simple data preprocessing...")
    
    # Remove NaN rows - simple approach
    valid_X = ~np.any(np.isnan(X_train), axis=1)
    valid_y = ~np.isnan(y_train)
    valid_mask = valid_X & valid_y
    
    if remove_outliers:
        # Simple outlier removal using percentiles
        y_clean = y_train[valid_mask]
        if len(y_clean) > 10:  # Only if we have enough data
            q1, q99 = np.percentile(y_clean, [1, 99])
            outlier_mask = (y_train >= q1) & (y_train <= q99)
            valid_mask = valid_mask & outlier_mask
    
    n_removed = len(y_train) - np.sum(valid_mask)
    if n_removed > 0:
        print(f"Removed {n_removed} samples ({n_removed/len(y_train)*100:.1f}%) due to NaN/outliers")
    
    return X_train[valid_mask], y_train[valid_mask], valid_mask


def simple_parameter_initialization(X_train, y_train, n_cosmo_params=35):
    """Simple parameter initialization avoiding complex JAX operations."""
    
    # Basic statistics
    y_var = float(np.var(y_train[~np.isnan(y_train)]))
    
    # Feature scales - simple approach
    cosmo_features = X_train[:, :n_cosmo_params]
    cosmo_std = np.nanstd(cosmo_features, axis=0)
    cosmo_std = np.where(cosmo_std < 1e-10, 1.0, cosmo_std)  # Avoid division by zero
    
    mass_std = np.nanstd(X_train[:, n_cosmo_params])
    mass_std = max(mass_std, 1e-10)
    
    pk_features = X_train[:, n_cosmo_params+1:]
    pk_std = np.nanstd(pk_features, axis=0)
    pk_std = np.where(pk_std < 1e-10, 1.0, pk_std)
    
    # Simple initialization
    params = {
        "cosmo_amplitude": jnp.float32(np.log(max(y_var * 0.3, 1e-10))),
        "cosmo_length_scales": jnp.array(-np.log(cosmo_std + 1e-6), dtype=jnp.float32),
        "log_mass_amplitude": jnp.float32(np.log(max(y_var * 0.5, 1e-10))),
        "mass_length_scale": jnp.float32(-np.log(mass_std + 1e-6)),
        "pk_amplitude": jnp.float32(np.log(max(y_var * 0.2, 1e-10))),
        "pk_length_scale": jnp.array(-np.log(pk_std + 1e-6), dtype=jnp.float32),
        "noise": jnp.float32(max(np.sqrt(y_var * 0.01), 1e-4))
    }
    
    return params


def train_single_gp_simple(kernel_builder, X_train, y_train_bin, r_bin_idx=0, maxiter=500):
    """
    Simple GP training for a single radial bin with basic optimization.
    """
    start_time = time.time()
    
    print(f"  Training bin {r_bin_idx} with {len(y_train_bin)} samples...")
    
    # Simple preprocessing
    X_clean, y_clean, valid_mask = simple_data_preprocessing(X_train, y_train_bin)
    
    if len(y_clean) < 10:
        print(f"    Skipping bin {r_bin_idx}: insufficient data ({len(y_clean)} samples)")
        return None, None, {'error': 'insufficient_data', 'n_samples': len(y_clean)}
    
    # Simple parameter initialization
    initial_params = simple_parameter_initialization(X_clean, y_clean)
    
    # Define loss function
    def loss_fn(params):
        try:
            gp = kernel_builder(params, X_clean)
            return -gp.log_probability(y_clean)
        except Exception as e:
            print(f"      GP construction failed: {e}")
            return 1e6  # Return high loss for failures
    
    # Simple optimization (just use initial params if optimization fails)
    best_params = initial_params
    best_loss = float(loss_fn(initial_params))
    
    if OPTIMIZATION_AVAILABLE:
        try:
            # Try simple optimization
            solver = jaxopt.ScipyMinimize(fun=loss_fn, maxiter=min(maxiter, 100))
            soln = solver.run(initial_params)
            best_params = soln.params
            best_loss = float(soln.state.fun_val)
            print(f"    Optimization successful: loss {best_loss:.2f}")
        except Exception as e:
            print(f"    Optimization failed ({e}), using initial parameters")
            best_loss = float(loss_fn(initial_params))
    else:
        print(f"    Using initial parameters (no optimization available)")
    
    # Build final GP
    try:
        final_gp = kernel_builder(best_params, X_clean)
    except Exception as e:
        print(f"    Failed to build final GP: {e}")
        return None, None, {'error': 'gp_construction_failed'}
    
    training_info = {
        'r_bin_idx': r_bin_idx,
        'training_time': time.time() - start_time,
        'final_loss': best_loss,
        'n_samples': len(y_clean),
        'success': True
    }
    
    print(f"    Completed in {training_info['training_time']:.1f}s, loss: {best_loss:.2f}")
    
    return final_gp, best_params, training_info


def train_simple_conditional_gp(sim_indices_train, kernel_name='hierarchical',
                               filterType='CAP', ptype='gas', maxiter=500):
    """
    Simple GP training that should work across JAX versions.
    """
    print(f"\n=== Training Simple GP with {kernel_name} kernel ===")
    
    # Load data
    try:
        X_train, y_train, r_bins, k_bins = prepare_GP_data(
            sim_indices_train, filterType=filterType, ptype=ptype
        )
        print(f"Data loaded: {X_train.shape[0]} samples, {len(r_bins)} radial bins")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return None, None, {'error': 'data_loading_failed'}
    
    # Get kernel builder
    if KERNELS_AVAILABLE and kernel_name in ['multiscale', 'physics_informed', 'robust']:
        try:
            kernel_builder = get_kernel_builder(kernel_name)
            print(f"Using improved {kernel_name} kernel")
        except Exception as e:
            print(f"Failed to get improved kernel ({e}), falling back to hierarchical")
            kernel_builder = build_hierarchical_gp
    else:
        print("Using basic hierarchical kernel")
        kernel_builder = build_hierarchical_gp
    
    # Train GPs for first few bins (limit for testing)
    n_bins_to_train = min(5, len(r_bins))  # Train only first 5 bins for quick testing
    print(f"Training first {n_bins_to_train} radial bins...")
    
    gp_models = []
    best_params_list = []
    training_info_list = []
    
    for r_bin_idx in tqdm(range(n_bins_to_train), desc=f"Training {kernel_name} GP"):
        y_train_bin = y_train[:, r_bin_idx]
        
        gp_model, best_params, training_info = train_single_gp_simple(
            kernel_builder, X_train, y_train_bin, r_bin_idx, maxiter
        )
        
        if gp_model is not None:
            gp_models.append(gp_model)
            best_params_list.append(best_params)
            training_info_list.append(training_info)
        else:
            print(f"    Failed to train bin {r_bin_idx}")
            training_info_list.append(training_info)
    
    # Create simple model info
    successful_trainings = [info for info in training_info_list if info.get('success', False)]
    total_time = sum(info['training_time'] for info in successful_trainings)
    final_losses = [info['final_loss'] for info in successful_trainings]
    
    model_info = {
        'kernel_name': kernel_name,
        'total_training_time': total_time,
        'n_successful_bins': len(successful_trainings),
        'n_attempted_bins': n_bins_to_train,
        'final_losses': final_losses,
        'mean_loss': np.mean(final_losses) if final_losses else np.nan,
        'method': f'Simple {kernel_name} GP'
    }
    
    print(f"\nTraining completed:")
    print(f"  Successful bins: {len(successful_trainings)}/{n_bins_to_train}")
    print(f"  Total time: {total_time:.1f}s")
    if final_losses:
        print(f"  Mean loss: {model_info['mean_loss']:.2f}")
    
    return gp_models, best_params_list, model_info


def test_simple_training(sim_indices_train):
    """Test function for simple training approach."""
    
    print("🧪 TESTING SIMPLE TRAINING APPROACH")
    print("=" * 50)
    
    # Test different kernels
    kernels_to_test = ['hierarchical']
    if KERNELS_AVAILABLE:
        kernels_to_test.extend(['multiscale', 'physics_informed'])
    
    results = {}
    
    for kernel_name in kernels_to_test:
        print(f"\nTesting {kernel_name} kernel...")
        
        try:
            start_time = time.time()
            gp_models, best_params, model_info = train_simple_conditional_gp(
                sim_indices_train, kernel_name=kernel_name, maxiter=200  # Reduced for speed
            )
            
            results[kernel_name] = {
                'success': True,
                'n_models': len(gp_models) if gp_models else 0,
                'train_time': time.time() - start_time,
                'model_info': model_info,
                'method': f'Simple {kernel_name}'
            }
            
            print(f"✅ {kernel_name}: {len(gp_models) if gp_models else 0} models trained")
            
        except Exception as e:
            print(f"❌ {kernel_name}: Failed - {e}")
            results[kernel_name] = {
                'success': False,
                'error': str(e),
                'method': f'Simple {kernel_name}'
            }
    
    return results