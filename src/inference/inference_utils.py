#!/usr/bin/env python3
"""
Shared utilities for GP-based parameter inference.

This module contains common functionality used by both gp_hmc_inference.py 
and gp_coverage_test.py to avoid code duplication.
"""

import sys
import os
import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Any, Callable

# Add project root
sys.path.append('/pscratch/sd/l/lindajin/DH_profile_kSZ_WL')

from src.data.profile_loader import load_simulation_mean_profiles


def load_target_simulation_data(sim_id: int, filterType: str = 'CAP', ptype: str = 'gas') -> Dict[str, Any]:
    """
    Load target simulation data for inference.
    
    Args:
        sim_id: Simulation ID to load
        filterType: Filter type ('CAP', 'cumulative', 'dsigma')
        ptype: Particle type ('gas', 'dm', 'star', 'bh', 'total', 'baryon')
        
    Returns:
        Dictionary containing simulation data or None if failed
    """
    try:
        r_bins, profiles, masses, params, k, pk_ratios = load_simulation_mean_profiles(
            [sim_id], filterType=filterType, ptype=ptype
        )
        
        if profiles is None or len(profiles) == 0:
            return None
        
        obs_profile = profiles[0]
        true_params = params[0]
        true_mass = float(masses[0])
        true_pk_ratios = pk_ratios[0]
        
        n_cosmo_params = len(true_params)
        n_mass_features = 1

        n_k_features = len(k)
        # pk_ratio has same dimension as k (pk_ratio vs k array)
        n_pk_features = n_k_features
        n_features = n_cosmo_params + n_mass_features + n_pk_features
        
        return {
            'sim_id': sim_id,
            'obs_profile': obs_profile,
            'true_params': true_params,
            'true_mass': true_mass,
            'true_pk_ratios': true_pk_ratios,  # Full pk_ratios vs k array
            'r_bins': r_bins,
            'k_bins': k,
            'n_features': n_features,
            'n_cosmo_params': n_cosmo_params,
            'n_k_features': n_k_features
        }
        
    except Exception as e:
        print(f"Failed to load simulation {sim_id}: {e}")
        return None


def create_basic_gp_likelihood_function(
    gp_models: List[Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    obs_profile: np.ndarray,
    true_mass: float,
    true_pk_ratios: np.ndarray,
    n_features: int,
    obs_noise_fraction: float = 0.357
) -> Callable[[np.ndarray], float]:
    """
    Create a GP likelihood function for parameter inference.

    Args:
        gp_models: List of trained GP models (one per radius bin)
        X_train: Training input features
        y_train: Training targets
        obs_profile: Observed profile to match
        true_mass: True halo mass (if known)
        true_pk_ratios: True pk_ratios array vs k
        n_features: Number of input features expected
        obs_noise_fraction: Observational noise as fraction of signal

    Returns:
        Log-likelihood function that takes parameter array and returns log-likelihood
    """
    
    def gp_log_likelihood(test_params: np.ndarray) -> float:
        """Evaluate log-likelihood using GP predictions."""
        try:
            # Construct GP input: [cosmo_params, log10(mass), log10(pk_ratio), k_features...]
            gp_input = np.zeros(n_features)
            n_cosmo = min(len(test_params), n_features, 35)
            gp_input[:n_cosmo] = test_params[:n_cosmo]
            
            if n_features > n_cosmo:
                gp_input[n_cosmo] = np.log10(true_mass)  # log mass
            if n_features > n_cosmo + 1:
                # Use mean of pk_ratios array for this legacy function
                true_pk_ratio_mean = np.mean(true_pk_ratios)
                gp_input[n_cosmo + 1] = np.log10(true_pk_ratio_mean)  # log pk_ratio  
            # k_features remain zero
            
            # Get GP predictions for all radius bins
            test_input_jnp = jnp.array(gp_input).reshape(1, -1)
            pred_means = []
            pred_vars = []
            
            for i, gp_model in enumerate(gp_models):
                y_train_i = jnp.array(y_train[:, i])
                _, cond_gp = gp_model.condition(y_train_i, test_input_jnp)
                pred_means.append(float(cond_gp.mean[0]))
                pred_vars.append(float(cond_gp.variance[0]))
            
            # Convert to arrays and handle shape
            pred_means_flat = np.array(pred_means)
            pred_vars_flat = np.array(pred_vars)
            
            # Gaussian likelihood with GP uncertainty + observational noise
            obs_noise = obs_noise_fraction * np.abs(obs_profile)
            total_var = pred_vars_flat + obs_noise**2
            
            residuals = obs_profile - pred_means_flat
            log_like = -0.5 * np.sum(residuals**2 / total_var + np.log(2 * np.pi * total_var))

            return log_like if np.isfinite(log_like) else -1e10
            
        except Exception:
            return -1e10
    
    return gp_log_likelihood


def run_simple_mcmc_sampling(
    log_likelihood_fn: Callable[[np.ndarray], float],
    param_names: List[str],
    minVal: np.ndarray,
    maxVal: np.ndarray,
    selected_indices: List[int],
    n_samples: int = 2000,
    burnin: int = 500,
    step_size_factor: float = 0.02,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run simple MCMC sampling (Metropolis-Hastings).
    
    Args:
        log_likelihood_fn: Function that takes parameters and returns log-likelihood
        param_names: List of parameter names
        minVal: Minimum values for parameters
        maxVal: Maximum values for parameters  
        selected_indices: Indices of parameters to vary
        n_samples: Total number of MCMC samples
        burnin: Number of burn-in samples to discard
        step_size_factor: Step size as fraction of parameter range
        verbose: Whether to print progress
        
    Returns:
        Tuple of (samples, log_likes, acceptance_rate)
    """
    
    if verbose:
        print(f"🔄 Running MCMC sampling...")
        print(f"   • Samples: {n_samples}, Burn-in: {burnin}")
        print(f"   • Parameters: {len(selected_indices)}")
    
    # Initialize chain
    current_params = np.array([0.5 * (minVal[i] + maxVal[i]) for i in range(len(minVal))])
    samples = np.zeros((n_samples, len(minVal)))
    log_likes = np.zeros(n_samples)
    n_accepted = 0
    
    # Step sizes
    step_sizes = {}
    for param_idx in selected_indices:
        param_range = maxVal[param_idx] - minVal[param_idx]
        step_sizes[param_idx] = step_size_factor * param_range
    
    # Current state
    current_log_like = log_likelihood_fn(current_params)
    
    if verbose:
        print(f"   • Initial log-likelihood: {current_log_like:.2f}")
    
    for i in range(n_samples):
        if verbose and i % 1000 == 0:
            acc_rate = n_accepted / (i + 1) if i > 0 else 0
            print(f"     Sample {i:5d}/{n_samples}, acceptance: {acc_rate:.1%}, log_like: {current_log_like:.2f}")
        
        # Propose new parameters
        proposal_params = current_params.copy()
        
        # Update only selected parameters
        for param_idx in selected_indices:
            proposal_params[param_idx] += np.random.normal(0, step_sizes[param_idx])
            # Enforce bounds
            proposal_params[param_idx] = np.clip(
                proposal_params[param_idx], minVal[param_idx], maxVal[param_idx]
            )
        
        # Evaluate proposal
        proposal_log_like = log_likelihood_fn(proposal_params)
        
        # Accept/reject (Metropolis-Hastings)
        if np.log(np.random.rand()) < (proposal_log_like - current_log_like):
            current_params = proposal_params.copy()
            current_log_like = proposal_log_like
            n_accepted += 1
        
        samples[i] = current_params
        log_likes[i] = current_log_like
    
    acceptance_rate = n_accepted / n_samples
    
    if verbose:
        print(f"   ✅ Sampling completed! Final acceptance rate: {acceptance_rate:.1%}")
    
    return samples, log_likes, acceptance_rate


def compute_mcmc_diagnostics(samples: np.ndarray, burnin: int = 1000) -> Dict[str, Any]:
    """
    Compute MCMC diagnostics including ESS and convergence metrics.
    
    Args:
        samples: MCMC samples array (n_samples, n_params)
        burnin: Number of burn-in samples to skip
        
    Returns:
        Dictionary with diagnostic statistics
    """
    samples_clean = samples[burnin:]
    n_samples, n_params = samples_clean.shape
    
    diagnostics = {}
    
    # Effective Sample Size (ESS) - simplified calculation
    def autocorr_func(x, max_lag=200):
        """Compute autocorrelation function."""
        n = len(x)
        x_centered = x - np.mean(x)
        autocorr = np.correlate(x_centered, x_centered, mode='full')
        autocorr = autocorr[n-1:n-1+max_lag]
        autocorr = autocorr / autocorr[0]
        return autocorr
    
    ess_values = []
    for param_idx in range(n_params):
        autocorr = autocorr_func(samples_clean[:, param_idx])
        # Find first negative value or use max_lag
        tau_idx = np.where(autocorr < 0)[0]
        tau = tau_idx[0] if len(tau_idx) > 0 else len(autocorr)
        ess = n_samples / (1 + 2 * np.sum(autocorr[:tau]))
        ess_values.append(max(1, ess))  # ESS should be at least 1
    
    diagnostics['ess'] = np.array(ess_values)
    diagnostics['mean_ess'] = np.mean(ess_values)
    diagnostics['min_ess'] = np.min(ess_values)
    
    return diagnostics


def run_single_simulation_inference(
    sim_id: int,
    gp_models: List[Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    selected_indices: List[int],
    param_names: List[str],
    minVal: np.ndarray,
    maxVal: np.ndarray,
    n_samples: int = 2000,
    burnin: int = 500,
    filterType: str = 'CAP',
    ptype: str = 'gas',
    obs_noise_fraction: float = 0.02,
    verbose: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Run full inference pipeline for a single simulation.
    
    This is a high-level function that combines data loading, likelihood creation,
    and MCMC sampling for testing purposes.
    
    Args:
        sim_id: Simulation ID to test
        gp_models: List of trained GP models
        X_train: Training input features 
        y_train: Training targets
        selected_indices: Parameter indices to vary
        param_names: Parameter names
        minVal: Parameter minimum values
        maxVal: Parameter maximum values
        n_samples: Number of MCMC samples
        burnin: Burn-in samples
        filterType: Filter type
        ptype: Particle type
        obs_noise_fraction: Observational noise fraction
        verbose: Print progress
        
    Returns:
        Dictionary with inference results or None if failed
    """
    
    try:
        # Load simulation data
        sim_data = load_target_simulation_data(sim_id, filterType, ptype)
        if sim_data is None:
            return None
        
        # Create likelihood function
        log_likelihood_fn = create_basic_gp_likelihood_function(
            gp_models, X_train, y_train,
            sim_data['obs_profile'], sim_data['true_mass'], sim_data['true_pk_ratios'],
            sim_data['n_features'], obs_noise_fraction
        )
        
        # Run MCMC sampling
        samples, log_likes, acceptance_rate = run_simple_mcmc_sampling(
            log_likelihood_fn, param_names, minVal, maxVal, selected_indices,
            n_samples, burnin, verbose=verbose
        )
        
        # Return results
        samples_clean = samples[burnin:]
        
        return {
            'sim_id': sim_id,
            'samples': samples_clean,
            'true_params': sim_data['true_params'],
            'acceptance_rate': acceptance_rate,
            'success': True,
            'obs_profile': sim_data['obs_profile'],
            'true_mass': sim_data['true_mass'],
            'true_pk_ratios': sim_data['true_pk_ratios']
        }
        
    except Exception as e:
        if verbose:
            print(f"   ❌ Simulation {sim_id} failed: {str(e)[:50]}...")
        return {
            'sim_id': sim_id,
            'success': False,
            'error': str(e)
        }