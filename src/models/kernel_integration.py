"""
Integration functions to use improved kernels with the existing train_GP.py framework.
This bridges the gap between your proven training methods and the new kernel designs.

The key insight: Make improved kernels work with train_GP.py's parameter format,
not the other way around.
"""

import jax.numpy as jnp
from tinygp import GaussianProcess, kernels, transforms
from train_GP import train_conditional_gp


def build_multiscale_gp_integrated(params, X):
    """
    Multiscale kernel using separate physics kernels with proper dimensionality.
    Each kernel operates on full feature space but only activates for relevant physics.
    """
    n_cosmo = 35
    n_mass = 1  
    n_pk = len(params["pk_length_scale"])
    total_features = n_cosmo + n_mass + n_pk
    
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
    
    return GaussianProcess(combined_kernel, X, diag=params["noise"]**2 + 1e-6)


def build_physics_informed_gp_integrated(params, X):
    """
    Physics-informed kernel using separate physics kernels with proper dimensionality.
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
    
    return GaussianProcess(combined_kernel, X, diag=params["noise"]**2 + 1e-6)


def build_robust_gp_integrated(params, X):
    """
    Robust kernel using separate physics kernels with proper dimensionality.
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
    
    return GaussianProcess(combined_kernel, X, diag=params["noise"]**2 + 1e-6)
    


# Simple usage functions - direct replacements for build_hierarchical_gp
def test_multiscale_training(sim_indices_train):
    """Test multiscale kernel with your proven training framework."""
    print("Testing multiscale kernel with train_GP.py framework...")
    return train_conditional_gp(
        sim_indices_train, 
        build_multiscale_gp_integrated,
        maxiter=1000,
        filterType='CAP', 
        ptype='gas'
    )


def test_physics_informed_training(sim_indices_train):
    """Test physics-informed kernel with your proven training framework."""
    print("Testing physics-informed kernel with train_GP.py framework...")
    return train_conditional_gp(
        sim_indices_train,
        build_physics_informed_gp_integrated,
        maxiter=1000,
        filterType='CAP', 
        ptype='gas'
    )


def test_robust_training(sim_indices_train):
    """Test robust kernel with your proven training framework."""
    print("Testing robust kernel with train_GP.py framework...")
    return train_conditional_gp(
        sim_indices_train,
        build_robust_gp_integrated,
        maxiter=1000,
        filterType='CAP', 
        ptype='gas'
    )