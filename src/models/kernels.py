"""
Comprehensive kernel implementations for cosmological GP training.

This module consolidates all kernel designs including:
- Hierarchical kernels (standard approach)
- Multi-scale kernels for different physics scales
- Physics-informed kernels with domain knowledge
- Robust kernels for outlier handling
- Adaptive noise models

Replaces improved_kernels.py and kernel_integration.py for better organization.
"""

import jax
import jax.numpy as jnp
import numpy as np
from tinygp import GaussianProcess, kernels, transforms

try:
    from src.config.config import N_COSMO_PARAMS
    from src.data.profile_loader import getParamsFiducial
except ImportError:
    N_COSMO_PARAMS = 35  # Fallback value

# =============================================================================
# INITIALIZE KERNELS 
# =============================================================================


def initialize_physics_informed_params(X_train, y_train):
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
    cosmo_features = X_train[:, :N_COSMO_PARAMS]
    mass_feature = X_train[:, N_COSMO_PARAMS]
    pk_features = X_train[:, N_COSMO_PARAMS+1:]
    
    param_names, fiducial_values, maxdiff, minVal, maxVal = getParamsFiducial()

    # Compute feature scales for length scale initialization
    cosmo_scales = jnp.array(maxdiff * 0.5)  # Half the max difference
    mass_scale = jnp.std(mass_feature)
    pk_scales = jnp.std(pk_features, axis=0)
    
    # Initialize amplitudes based on data variance
    cosmo_amp_init = jnp.log(y_var * 0.3)  # 30% of variance from cosmology
    mass_amp_init = jnp.log(y_var * 0.5)   # 50% from halo mass (most important)
    pk_amp_init = jnp.log(y_var * 0.2)     # 20% from power spectrum
    
    # Initialize length scales inversely proportional to feature scales
    cosmo_length_init = -jnp.log(maxdiff + 1e-6)
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

def initialize_gp_parameters(n_cosmo_params, n_k_bins):
    """
    Initialize GP hyperparameters for hierarchical kernel.
    
    Args:
        n_cosmo_params: Number of cosmological parameters
        n_k_bins: Number of power spectrum ratio bins
    
    Returns:
        Dictionary of initial parameters
    """
    # Ensure n_k_bins is at least 1 to avoid empty arrays
    n_k_bins = max(1, n_k_bins)
    
    return {
        "cosmo_amplitude": 0.1,
        "log_mass_amplitude": 0.1, 
        "pk_amplitude": 0.1,
        "cosmo_length_scales": jnp.ones(n_cosmo_params) * 0.5,
        "mass_length_scale": 0.5,
        "pk_length_scale": jnp.ones(n_k_bins) * 0.5
    }

# =============================================================================
# HIERARCHICAL KERNELS - Standard approach used in main analysis
# =============================================================================

def build_hierarchical_gp(params, X):
    """
    More sophisticated kernel that treats cosmology and mass differently
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
                             kernels.Matern52(distance=kernels.distance.L2Distance())
    )

    return GaussianProcess(kernel, X, diag=1e-4)


def build_hierarchical_gp_improved(params, X):
    """
    Improved version of the hierarchical GP with better noise handling.
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
                             kernels.Matern52(distance=kernels.distance.L2Distance())
    )

    return GaussianProcess(kernel, X, diag=params.get("noise", 0.1)**2 + 1e-6)


# =============================================================================
# MULTI-SCALE KERNELS - Different kernels for different physics scales
# =============================================================================

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
    
    return GaussianProcess(combined_kernel, X, diag=params.get("noise", 0.1)**2 + 1e-6)


def build_multiscale_gp_integrated(params, X):
    """
    Integrated multiscale kernel using separate physics kernels with proper dimensionality.
    Each kernel operates on full feature space but only activates for relevant physics.
    """
    n_cosmo = 35
    n_mass = 1  
    n_pk = len(params["pk_length_scale"])
    
    # Create length scales for each physics component (full dimensionality)
    # Cosmology kernel: active for first 35 features, zero elsewhere
    cosmo_length_scales = jnp.concatenate([
        params["cosmo_length_scales"],  # Active for cosmology (35 features)
        jnp.full(n_mass + n_pk, 1e10)  # Effectively inactive for mass + pk (huge length scales)
    ])
    
    # Mass kernel: active only for mass feature (index 35)
    mass_length_scales = jnp.concatenate([
        jnp.full(n_cosmo, 1e10),           # Effectively inactive for cosmology
        jnp.array([params["mass_length_scale"]]),  # Active for mass (1 feature)
        jnp.full(n_pk, 1e10)               # Effectively inactive for pk
    ])
    
    # Power spectrum kernel: active only for last n_pk features
    pk_length_scales = jnp.concatenate([
        jnp.full(n_cosmo + n_mass, 1e10),  # Effectively inactive for cosmology + mass
        params["pk_length_scale"]          # Active for power spectrum (n_pk features)
    ])
    
    # Build separate physics kernels (each sees full feature space)
    cosmo_kernel = jnp.exp(params["cosmo_amplitude"]) * \
                   transforms.Linear(jnp.exp(-cosmo_length_scales), 
                                   kernels.Matern52(distance=kernels.distance.L2Distance()))
    
    mass_kernel = jnp.exp(params["log_mass_amplitude"]) * \
                  transforms.Linear(jnp.exp(-mass_length_scales),
                                  kernels.ExpSquared())  # Smooth mass correlations
    
    pk_kernel = jnp.exp(params["pk_amplitude"]) * \
                transforms.Linear(jnp.exp(-pk_length_scales),
                                kernels.RationalQuadratic(alpha=2.0, distance=kernels.distance.L2Distance()))  # Multi-scale pk
    
    # Combine kernels multiplicatively (physics interactions)
    combined_kernel = cosmo_kernel * mass_kernel * pk_kernel
    
    return GaussianProcess(combined_kernel, X, diag=params.get("noise", 0.1)**2 + 1e-6)


# =============================================================================
# PHYSICS-INFORMED KERNELS - Incorporate domain knowledge
# =============================================================================

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
    
    return GaussianProcess(total_kernel, X, diag=params.get("noise", 0.1)**2 + 1e-6)


def build_physics_informed_gp_integrated(params, X):
    """
    Integrated physics-informed kernel using separate physics kernels with proper dimensionality.
    Uses additive combination for independent physics contributions.
    """
    n_cosmo = 35
    n_mass = 1  
    n_pk = len(params["pk_length_scale"])
    
    # Create length scales for each physics component (full dimensionality)
    # Cosmology kernel: active for first 35 features, zero elsewhere
    cosmo_length_scales = jnp.concatenate([
        params["cosmo_length_scales"],  # Active for cosmology (35 features)
        jnp.full(n_mass + n_pk, 1e10)  # Effectively inactive for mass + pk
    ])
    
    # Mass kernel: active only for mass feature (index 35)
    mass_length_scales = jnp.concatenate([
        jnp.full(n_cosmo, 1e10),           # Effectively inactive for cosmology
        jnp.array([params["mass_length_scale"]]),  # Active for mass (1 feature)
        jnp.full(n_pk, 1e10)               # Effectively inactive for pk
    ])
    
    # Power spectrum kernel: active only for last n_pk features
    pk_length_scales = jnp.concatenate([
        jnp.full(n_cosmo + n_mass, 1e10),  # Effectively inactive for cosmology + mass
        params["pk_length_scale"]          # Active for power spectrum (n_pk features)
    ])
    
    # Build separate physics kernels with different characteristics
    cosmo_kernel = jnp.exp(params["cosmo_amplitude"]) * \
                   transforms.Linear(jnp.exp(-cosmo_length_scales), 
                                   kernels.Matern32(distance=kernels.distance.L2Distance()))  # Structured correlations
    
    mass_kernel = jnp.exp(params["log_mass_amplitude"]) * \
                  transforms.Linear(jnp.exp(-mass_length_scales),
                                  kernels.ExpSquared())  # Smooth halo mass function
    
    pk_kernel = jnp.exp(params["pk_amplitude"]) * \
                transforms.Linear(jnp.exp(-pk_length_scales),
                                kernels.Matern52(distance=kernels.distance.L2Distance()))  # Scale-dependent effects
    
    # Additive combination (independent physics contributions)
    combined_kernel = cosmo_kernel + mass_kernel + pk_kernel
    
    return GaussianProcess(combined_kernel, X, diag=params.get("noise", 0.1)**2 + 1e-6)


# =============================================================================
# ROBUST KERNELS - Handle outliers and missing data
# =============================================================================

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
    robust_noise = params.get("noise", 0.1)**2 + jnp.exp(params.get("log_outlier_noise", jnp.log(0.1)))**2
    
    return GaussianProcess(robust_kernel, X, diag=robust_noise + 1e-6)


def build_robust_gp_integrated(params, X):
    """
    Integrated robust kernel using separate physics kernels with proper dimensionality.
    Uses heavy-tailed kernels for robustness against outliers.
    """
    n_cosmo = 35
    n_mass = 1  
    n_pk = len(params["pk_length_scale"])
    
    # Create length scales for each physics component (full dimensionality)
    # Cosmology kernel: active for first 35 features, zero elsewhere
    cosmo_length_scales = jnp.concatenate([
        params["cosmo_length_scales"],  # Active for cosmology (35 features)
        jnp.full(n_mass + n_pk, 1e10)  # Effectively inactive for mass + pk
    ])
    
    # Mass kernel: active only for mass feature (index 35)
    mass_length_scales = jnp.concatenate([
        jnp.full(n_cosmo, 1e10),           # Effectively inactive for cosmology
        jnp.array([params["mass_length_scale"]]),  # Active for mass (1 feature)
        jnp.full(n_pk, 1e10)               # Effectively inactive for pk
    ])
    
    # Power spectrum kernel: active only for last n_pk features
    pk_length_scales = jnp.concatenate([
        jnp.full(n_cosmo + n_mass, 1e10),  # Effectively inactive for cosmology + mass
        params["pk_length_scale"]          # Active for power spectrum (n_pk features)
    ])
    
    # Build robust physics kernels (all heavy-tailed for outlier resistance)
    cosmo_kernel = jnp.exp(params["cosmo_amplitude"]) * \
                   transforms.Linear(jnp.exp(-cosmo_length_scales), 
                                   kernels.RationalQuadratic(alpha=1.0, distance=kernels.distance.L2Distance()))  # Heavy tails
    
    mass_kernel = jnp.exp(params["log_mass_amplitude"]) * \
                  transforms.Linear(jnp.exp(-mass_length_scales),
                                  kernels.RationalQuadratic(alpha=2.0))  # Heavy tails for mass
    
    pk_kernel = jnp.exp(params["pk_amplitude"]) * \
                transforms.Linear(jnp.exp(-pk_length_scales),
                                kernels.RationalQuadratic(alpha=1.5, distance=kernels.distance.L2Distance()))  # Heavy tails for pk
    
    # Multiplicative combination (robust interactions)
    combined_kernel = cosmo_kernel * mass_kernel * pk_kernel
    
    return GaussianProcess(combined_kernel, X, diag=params.get("noise", 0.1)**2 + 1e-6)


# =============================================================================
# ADAPTIVE KERNELS - Radial dependence and adaptive noise
# =============================================================================

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
    base_noise = params.get("noise", 0.1)**2
    radius_factor = 1.0 + 0.1 * r_bin_idx  # 10% increase per bin
    adaptive_noise = base_noise * radius_factor
    
    return GaussianProcess(kernel, X, diag=adaptive_noise + 1e-6)


# =============================================================================
# KERNEL REGISTRY AND HELPER FUNCTIONS
# =============================================================================

# Mapping of kernel names to functions for easy selection
KERNEL_REGISTRY = {
    # Standard hierarchical kernels
    'hierarchical': build_hierarchical_gp_improved,
    
    # Multi-scale kernels
    'multiscale': build_multiscale_gp_integrated,
    
    # Physics-informed kernels
    'physics_informed': build_physics_informed_gp_integrated,
    
    # Robust kernels
    'robust': build_robust_gp_integrated,
    
    # Adaptive kernels
    'adaptive_noise': build_adaptive_noise_gp,
}


def get_kernel_builder(kernel_name):
    """
    Get kernel builder function by name.
    
    Args:
        kernel_name: Name of kernel from KERNEL_REGISTRY
        
    Returns:
        Kernel builder function
    """
    if kernel_name not in KERNEL_REGISTRY:
        available_kernels = list(KERNEL_REGISTRY.keys())
        raise ValueError(f"Unknown kernel: {kernel_name}. Available kernels: {available_kernels}")
    return KERNEL_REGISTRY[kernel_name]


def list_available_kernels():
    """List all available kernel implementations."""
    return list(KERNEL_REGISTRY.keys())


# =============================================================================
# TRAINING CONVENIENCE FUNCTIONS
# =============================================================================

def test_multiscale_training(sim_indices_train):
    """Test multiscale kernel with the training framework."""
    from src.models.gp_trainer_one import train_conditional_gp
    print("Testing multiscale kernel with training framework...")
    return train_conditional_gp(
        sim_indices_train, 
        build_multiscale_gp_integrated,
        maxiter=1000,
        filterType='CAP', 
        ptype='gas'
    )


def test_physics_informed_training(sim_indices_train):
    """Test physics-informed kernel with the training framework."""
    from src.models.gp_trainer_one import train_conditional_gp
    print("Testing physics-informed kernel with training framework...")
    return train_conditional_gp(
        sim_indices_train,
        build_physics_informed_gp_integrated,
        maxiter=1000,
        filterType='CAP', 
        ptype='gas'
    )


def test_robust_training(sim_indices_train):
    """Test robust kernel with the training framework."""
    from src.models.gp_trainer_one import train_conditional_gp
    print("Testing robust kernel with training framework...")
    return train_conditional_gp(
        sim_indices_train,
        build_robust_gp_integrated,
        maxiter=1000,
        filterType='CAP', 
        ptype='gas'
    )