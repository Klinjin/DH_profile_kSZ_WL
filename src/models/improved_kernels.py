"""
Improved kernel designs for GP training with better physics-informed structure.

This module implements several kernel improvements to address the GP accuracy issues:
1. Multi-scale kernels for different radius ranges
2. Physics-informed kernel structures
3. Adaptive noise modeling
4. Better hyperparameter initialization
"""

import jax
import jax.numpy as jnp
import numpy as np
from tinygp import GaussianProcess, kernels, transforms
try:
    from src.config.config import N_COSMO_PARAMS
except ImportError:
    N_COSMO_PARAMS = 35  # Fallback value


def initialize_physics_informed_params(X_train, y_train, n_cosmo_params=35):
    """
    Initialize GP hyperparameters using data-driven statistics instead of zeros.
    
    Args:
        X_train: Training input features [n_samples, n_features]  
        y_train: Training targets [n_samples]
        n_cosmo_params: Number of cosmological parameters
        
    Returns:
        Dictionary with sensible hyperparameter initialization
    """
    # Compute data statistics
    y_var = float(jnp.var(y_train))
    y_std = float(jnp.std(y_train))
    
    # Split features: [cosmo_params, mass, pk_ratios]
    cosmo_features = X_train[:, :n_cosmo_params]
    mass_feature = X_train[:, n_cosmo_params]
    pk_features = X_train[:, n_cosmo_params+1:]
    
    # Compute feature scales for length scale initialization
    cosmo_scales = jnp.std(cosmo_features, axis=0)
    mass_scale = jnp.std(mass_feature)
    pk_scales = jnp.std(pk_features, axis=0)
    
    # Initialize amplitudes based on data variance
    cosmo_amp_init = jnp.log(y_var * 0.3)  # 30% of variance from cosmology
    mass_amp_init = jnp.log(y_var * 0.5)   # 50% from halo mass (most important)
    pk_amp_init = jnp.log(y_var * 0.2)     # 20% from power spectrum
    
    # Initialize length scales inversely proportional to feature scales
    cosmo_length_init = -jnp.log(cosmo_scales + 1e-6)
    mass_length_init = -jnp.log(mass_scale + 1e-6)
    pk_length_init = -jnp.log(pk_scales + 1e-6)
    
    # Initialize noise as small fraction of data variance
    noise_init = jnp.sqrt(y_var * 0.01)  # 1% noise level
    
    return {
        "cosmo_amplitude": cosmo_amp_init,
        "cosmo_length_scales": cosmo_length_init,
        "log_mass_amplitude": mass_amp_init,
        "mass_length_scale": mass_length_init, 
        "pk_amplitude": pk_amp_init,
        "pk_length_scale": pk_length_init,
        "noise": noise_init
    }


def build_multiscale_kernel_gp(params, X):
    """
    Multi-scale kernel that handles different physics scales appropriately.
    
    This kernel uses:
    - Matern52 for cosmology (intermediate correlations)  
    - ExpSquared for mass (smooth correlations)
    - RationalQuadratic for power spectrum (multi-scale correlations)
    """
    n_cosmo_params = 35
    
    # Split input dimensions  
    X_cosmo = X[:, :n_cosmo_params]
    X_mass = X[:, n_cosmo_params:n_cosmo_params+1]
    X_pk = X[:, n_cosmo_params+1:]
    
    # Multi-scale kernels for different physics
    cosmo_kernel = jnp.exp(params["cosmo_amplitude"]) * \
                   transforms.Linear(jnp.exp(-params["cosmo_length_scales"]), 
                                   kernels.Matern52(distance=kernels.distance.L2Distance()))
    
    mass_kernel = jnp.exp(params["log_mass_amplitude"]) * \
                  kernels.ExpSquared(scale=jnp.exp(params["mass_length_scale"]))
    
    pk_kernel = jnp.exp(params["pk_amplitude"]) * \
                transforms.Linear(jnp.exp(-params["pk_length_scale"]),
                                kernels.RationalQuadratic(alpha=2.0, distance=kernels.distance.L2Distance()))
    
    # Combine kernels multiplicatively (interactions between scales)
    combined_kernel = cosmo_kernel * mass_kernel * pk_kernel
    
    return GaussianProcess(combined_kernel, X, diag=params["noise"]**2 + 1e-6)


def build_physics_informed_gp(params, X):
    """
    Physics-informed kernel incorporating known halo profile behavior.
    
    Uses physical priors:
    - Mass dependence follows approximate power-law scaling
    - Cosmological parameters have structured correlations
    - Power spectrum ratios have scale-dependent effects
    """
    n_cosmo_params = 35
    
    # Structured cosmological parameter kernel
    # Group related parameters (e.g., matter/baryon density, feedback parameters)
    cosmo_block_kernel = jnp.exp(params["cosmo_amplitude"]) * \
                        transforms.Linear(jnp.exp(-params["cosmo_length_scales"]),
                                        kernels.Matern32(distance=kernels.distance.L2Distance()))
    
    # Mass kernel with physical scaling (halo mass function scaling)
    mass_kernel = jnp.exp(params["log_mass_amplitude"]) * \
                  kernels.ExpSquared(scale=jnp.exp(params["mass_length_scale"]))
    
    # Power spectrum kernel with scale-dependent correlations
    pk_kernel = jnp.exp(params["pk_amplitude"]) * \
                transforms.Linear(jnp.exp(-params["pk_length_scale"]),
                                kernels.Matern52(distance=kernels.distance.L2Distance()))
    
    # Additive combination to allow independent contributions
    total_kernel = cosmo_block_kernel + mass_kernel + pk_kernel
    
    return GaussianProcess(total_kernel, X, diag=params["noise"]**2 + 1e-6)


def build_adaptive_noise_gp(params, X, r_bin_idx=0):
    """
    Adaptive noise model that accounts for radial dependence.
    
    Noise increases with radius due to:
    - Fewer halos at large radii
    - Increased measurement uncertainty
    - Model inadequacy at large scales
    """
    n_cosmo_params = 35
    
    # Standard multi-scale kernel
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
                             kernels.Matern52(distance=kernels.distance.L2Distance()))
    
    # Adaptive noise: increases with radius bin
    base_noise = params["noise"]**2
    radius_factor = 1.0 + 0.1 * r_bin_idx  # 10% increase per bin
    adaptive_noise = base_noise * radius_factor
    
    return GaussianProcess(kernel, X, diag=adaptive_noise + 1e-6)


def build_robust_kernel_gp(params, X):
    """
    Robust kernel designed to handle outliers and missing data.
    
    Uses Student-t process approximation via heavy-tailed kernels
    and robust noise modeling.
    """
    n_cosmo_params = 35
    
    # Heavy-tailed kernels for robustness
    cosmo_kernel = jnp.exp(params["cosmo_amplitude"]) * \
                   transforms.Linear(jnp.exp(-params["cosmo_length_scales"]),
                                   kernels.RationalQuadratic(alpha=1.0, distance=kernels.distance.L2Distance()))
    
    mass_kernel = jnp.exp(params["log_mass_amplitude"]) * \
                  kernels.RationalQuadratic(alpha=2.0, scale=jnp.exp(params["mass_length_scale"]))
    
    pk_kernel = jnp.exp(params["pk_amplitude"]) * \
                transforms.Linear(jnp.exp(-params["pk_length_scale"]),
                                kernels.RationalQuadratic(alpha=1.5, distance=kernels.distance.L2Distance()))
    
    # Multiplicative combination for robustness
    robust_kernel = cosmo_kernel * mass_kernel * pk_kernel
    
    # Robust noise with outlier tolerance
    robust_noise = params["noise"]**2 + jnp.exp(params.get("log_outlier_noise", jnp.log(0.1)))**2
    
    return GaussianProcess(robust_kernel, X, diag=robust_noise + 1e-6)


def build_hierarchical_gp_improved(params, X):
    """
    Improved version of the original hierarchical GP with better initialization.
    """
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
                             kernels.Matern52(distance=kernels.distance.L2Distance()))

    return GaussianProcess(kernel, X, diag=params["noise"]**2 + 1e-6)


# Mapping of kernel names to functions for easy selection
KERNEL_REGISTRY = {
    'hierarchical': lambda params, X: build_hierarchical_gp_improved(params, X),
    'multiscale': build_multiscale_kernel_gp,
    'physics_informed': build_physics_informed_gp, 
    'adaptive_noise': build_adaptive_noise_gp,
    'robust': build_robust_kernel_gp
}


def get_kernel_builder(kernel_name):
    """Get kernel builder function by name."""
    if kernel_name not in KERNEL_REGISTRY:
        raise ValueError(f"Unknown kernel: {kernel_name}. Available: {list(KERNEL_REGISTRY.keys())}")
    return KERNEL_REGISTRY[kernel_name]