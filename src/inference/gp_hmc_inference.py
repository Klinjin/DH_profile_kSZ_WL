#!/usr/bin/env python3
"""
GP-based Parameter Inference using HMC Sampling with mass and pk_ratio inference.
Infers cosmological parameters + halo mass + pk_ratio simultaneously.
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
import warnings
import logging

# Suppress JAX CUDA plugin warnings/errors
logging.getLogger('jax._src.xla_bridge').setLevel(logging.CRITICAL)

sys.path.append('/pscratch/sd/l/lindajin/DH_profile_kSZ_WL')

# Force JAX to use CPU only - must be set BEFORE importing JAX
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'False'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide CUDA devices completely

import numpy as np
import matplotlib.pyplot as plt
import pickle
warnings.filterwarnings('ignore')

try:
    # Import JAX and force CPU-only configuration
    import jax
    jax.config.update('jax_platforms', 'cpu')
    jax.config.update('jax_enable_x64', False)
    import jax.numpy as jnp
    print(f"JAX configured for CPU execution: {jax.devices()}")
except ImportError:
    print("Warning: JAX not available, using numpy instead")
    import numpy as jnp
except Exception as e:
    print(f"Warning: JAX CUDA error ({e}), falling back to numpy")
    import numpy as jnp

from src.models.gp_trainer import GPTrainer
from src.data.profile_loader import getParamsFiducial
from src.inference.inference_utils import load_target_simulation_data, compute_mcmc_diagnostics


def create_gp_likelihood_function(gp_model_name, obs_profile, fiducial_params, cosmo_only=False, true_mass=None, true_pk_ratios=None):
    """Create GP-based log-likelihood function.

    Args:
        cosmo_only: If True, only infer cosmological parameters, fix mass and pk_ratio to true values
        true_mass: True halo mass (required if cosmo_only=True)
        true_pk_ratios: True pk_ratio array (required if cosmo_only=True)
    """

    trainer = GPTrainer(
        sim_indices_total=list(range(1024)),
        train_test_val_split=(0.7, 0.2, 0.1),
        filterType='CAP',
        ptype='gas'
    )

    if gp_model_name is None:
        trainer._load_pretrained('/pscratch/sd/l/lindajin/DH_profile_kSZ_WL/trained_gp_models/GPTrainer_091025_2209_CAP_gas_full_bins/')
    else:
        trainer._load_pretrained(f'/pscratch/sd/l/lindajin/DH_profile_kSZ_WL/trained_gp_models/{gp_model_name}/')

    expected_features = trainer.n_features

    def gp_log_likelihood(params):
        """Evaluate log-likelihood + priors."""
        if cosmo_only:
            # Only cosmological parameters are varied
            n_cosmo_varied = len(params)

            # Construct full GP input
            full_input = np.zeros(expected_features)
            full_input[:35] = np.array(fiducial_params)[:35]
            full_input[:n_cosmo_varied] = params  # Override selected cosmo params

            if expected_features > 35:
                full_input[35] = np.log10(true_mass)
            if expected_features > 36:
                n_pk_fill = min(len(true_pk_ratios), expected_features - 36)
                full_input[36:36 + n_pk_fill] = np.log10(np.clip(true_pk_ratios[:n_pk_fill], 1e-10, 10))
        else:
            # Infer cosmo params + mass + pk_ratio array
            n_k_features = trainer.n_features - 35 - 1
            n_cosmo_varied = len(params) - 1 - n_k_features

            if n_cosmo_varied >= len(params):
                return -np.inf

            log10_mass = params[n_cosmo_varied]
            pk_ratios = params[n_cosmo_varied + 1:]

            if len(pk_ratios) == 0:
                return -np.inf

            # Construct full GP input
            full_input = np.zeros(expected_features)
            full_input[:35] = np.array(fiducial_params)[:35]
            full_input[:n_cosmo_varied] = params[:n_cosmo_varied]

            if expected_features > 35:
                full_input[35] = log10_mass
            if expected_features > 36:
                n_pk_fill = min(len(pk_ratios), expected_features - 36)
                full_input[36:36 + n_pk_fill] = np.log10(np.clip(pk_ratios[:n_pk_fill], 1e-10, 10))

            # Priors for non-cosmo parameters
            if not (11.62 <= log10_mass <= 14.82):
                return -np.inf
            for pk_ratio in pk_ratios:
                if not (0.8 <= pk_ratio <= 1.02):
                    return -np.inf

        # GP prediction
        test_input_jnp = jnp.array(full_input).reshape(1, -1)
        pred_means, pred_vars = trainer.pred(test_input_jnp)
        pred_means_flat = np.array(pred_means).flatten()
        pred_vars_flat = np.array(pred_vars).flatten()

        # Likelihood
        obs_noise = 0.357 * np.abs(obs_profile)
        total_var = pred_vars_flat + obs_noise**2
        residuals = obs_profile - pred_means_flat
        log_like = -0.5 * np.sum(residuals**2 / total_var + np.log(2 * np.pi * total_var))

        if not np.isfinite(log_like):
            return -np.inf

        return log_like

    return gp_log_likelihood


def run_hmc_sampling(log_likelihood_fn, param_bounds, n_samples=5000, burnin=1000, step_size=0.01):
    """Run HMC sampling for selected cosmological params + mass + pk_ratio."""

    n_params = len(param_bounds)

    # Initialize chain at midpoint of bounds
    current_params = np.array([0.5 * (bounds[0] + bounds[1]) for bounds in param_bounds])
    samples = np.zeros((n_samples, n_params))
    log_likes = np.zeros(n_samples)
    n_accepted = 0

    # Step sizes proportional to parameter ranges
    step_sizes = [step_size * (bounds[1] - bounds[0]) for bounds in param_bounds]

    current_log_like = log_likelihood_fn(current_params)

    for i in range(n_samples):
        if i % 1000 == 0 and i > 0:
            acc_rate = n_accepted / i
            print(f"Sample {i}/{n_samples}, acceptance: {acc_rate:.1%}, log_like: {current_log_like:.2f}")

        # Simple Metropolis proposal (not full HMC)
        proposal_params = current_params + np.random.normal(0, step_sizes)

        # Enforce bounds
        for j, (min_val, max_val) in enumerate(param_bounds):
            proposal_params[j] = np.clip(proposal_params[j], min_val, max_val)

        proposal_log_like = log_likelihood_fn(proposal_params)

        # Accept/reject
        if np.log(np.random.rand()) < (proposal_log_like - current_log_like):
            current_params = proposal_params.copy()
            current_log_like = proposal_log_like
            n_accepted += 1

        samples[i] = current_params
        log_likes[i] = current_log_like

    acceptance_rate = n_accepted / n_samples
    return samples, log_likes, acceptance_rate


def create_corner_plot(samples, true_params, param_labels, acceptance_rate, target_sim, save_path, n_cosmo_params):
    """Create corner plot showing only cosmological parameters."""

    # Only plot cosmological parameters
    cosmo_samples = samples[:, :n_cosmo_params]
    cosmo_true = true_params[:n_cosmo_params] if true_params is not None else None
    cosmo_labels = param_labels[:n_cosmo_params]

    n_params = cosmo_samples.shape[1]
    fig, axes = plt.subplots(n_params, n_params, figsize=(10, 10))

    if n_params == 1:
        axes = np.array([[axes]])

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]

            if i == j:
                ax.hist(cosmo_samples[:, i], bins=30, alpha=0.7, density=True, color='lightblue')
                if cosmo_true is not None:
                    ax.axvline(cosmo_true[i], color='red', linestyle='--', linewidth=2)

                mean_val = np.mean(cosmo_samples[:, i])
                std_val = np.std(cosmo_samples[:, i])
                ax.set_title(f'{cosmo_labels[i]}\nμ={mean_val:.3f}±{std_val:.3f}', fontsize=10)

            elif i > j:
                ax.scatter(cosmo_samples[:, j], cosmo_samples[:, i], alpha=0.3, s=1, color='blue')
                if cosmo_true is not None:
                    ax.plot(cosmo_true[j], cosmo_true[i], 'r*', markersize=10)

            else:
                ax.set_visible(False)

            if i == n_params - 1 and j <= i:
                ax.set_xlabel(cosmo_labels[j], fontsize=10)
            if j == 0 and i > 0:
                ax.set_ylabel(cosmo_labels[i], fontsize=10)

    plt.suptitle(f'GP HMC Cosmological Parameters: Sim {target_sim}, Acceptance: {acceptance_rate:.1%}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_pk_ratio_plot(samples, true_pk_ratios, k_bins, burnin, sim_id, save_path):
    """Create pk_ratio vs k plot with inferred pk_ratio(k) and true values."""

    if k_bins is None or len(k_bins) == 0:
        # Create placeholder plot if no k data
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, 'No k-dependent pk_ratio data\navailable for this simulation',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(f'Power Spectrum Suppression - Sim {sim_id}')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return {'pk_median': None, 'pk_std': None, 'pk_16': None, 'pk_84': None, 'true_pk_mean': None}

    # Determine how many pk_ratio parameters we have
    # Parameters are: [cosmo_params..., log10_mass, pk_ratio_k0, pk_ratio_k1, ...]
    n_k_bins = len(k_bins)

    # Extract pk_ratio samples for each k bin (already in linear scale)
    # Assuming the last n_k_bins parameters are the pk_ratios
    samples_clean = samples[burnin:]
    pk_samples = samples_clean[:, -n_k_bins:]  # Last n_k_bins parameters, already linear scale

    # Compute posterior statistics for each k bin
    pk_medians = np.median(pk_samples, axis=0)
    pk_16s = np.percentile(pk_samples, 16, axis=0)
    pk_84s = np.percentile(pk_samples, 84, axis=0)
    pk_means = np.mean(pk_samples, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left panel: pk_ratio vs k
    if true_pk_ratios is not None:
        ax1.plot(k_bins, true_pk_ratios, 'r-', linewidth=2, label='True pk_ratio(k)', alpha=0.8)

    # Plot inferred pk_ratio(k) with uncertainty
    ax1.plot(k_bins, pk_medians, 'b-', linewidth=2, label='Inferred pk_ratio(k)')
    ax1.fill_between(k_bins, pk_16s, pk_84s, alpha=0.3, color='blue', label='68% CI')
    ax1.axhline(y=1.0, color='k', linestyle=':', alpha=0.5, label='No suppression')

    ax1.set_xlabel('k [h/Mpc]')
    ax1.set_ylabel('pk_ratio(k)')
    ax1.set_title(f'Power Spectrum Suppression - Sim {sim_id}')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right panel: Sample of individual pk_ratio posteriors for a few k bins
    # Show a few representative k bins to avoid clutter
    k_indices = [0, len(k_bins)//4, len(k_bins)//2, 3*len(k_bins)//4, -1]
    k_indices = [i for i in k_indices if i < len(k_bins)]

    for i, k_idx in enumerate(k_indices):
        if k_idx < len(k_bins):
            pk_samples_k = pk_samples[:, k_idx]
            ax2.hist(pk_samples_k, bins=30, alpha=0.4, density=True,
                    label=f'k={k_bins[k_idx]:.3f}', color=f'C{i}')

    ax2.axvline(1.0, color='k', linestyle=':', alpha=0.5, label='No suppression')
    ax2.set_xlabel('pk_ratio')
    ax2.set_ylabel('Posterior density')
    ax2.set_title('pk_ratio Posteriors (Sample of k bins)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Power Spectrum Suppression Analysis - Simulation {sim_id}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Return summary statistics
    return {
        'pk_medians': pk_medians,
        'pk_16s': pk_16s,
        'pk_84s': pk_84s,
        'pk_means': pk_means,
        'true_pk_mean': np.mean(true_pk_ratios) if true_pk_ratios is not None else None,
        'k_bins': k_bins
    }


def save_inference_results(samples, param_labels, acceptance_rate, diagnostics, target_sim, output_dir):
    """Save inference results."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = {
        'samples': samples,
        'param_labels': param_labels,
        'acceptance_rate': acceptance_rate,
        'diagnostics': diagnostics,
        'target_sim': target_sim,
        'timestamp': timestamp
    }

    results_path = output_dir / f"gp_hmc_results_sim{target_sim}_{timestamp}.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    # Summary report
    report_path = output_dir / f"gp_hmc_summary_sim{target_sim}_{timestamp}.md"
    with open(report_path, 'w') as f:
        f.write(f"# GP HMC Inference - Simulation {target_sim}\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Acceptance rate: {acceptance_rate:.1%}\n")
        f.write(f"Mean ESS: {diagnostics['mean_ess']:.0f}\n\n")

        f.write(f"## Parameter Posteriors\n")
        samples_clean = samples[1000:]  # Remove burn-in
        for i, label in enumerate(param_labels):
            post_mean = np.mean(samples_clean[:, i])
            post_std = np.std(samples_clean[:, i])
            f.write(f"- {label}: {post_mean:.4f} ± {post_std:.4f}\n")

    return results_path, report_path


def run_inference(sim_id, gp_name='GPTrainer_091025_2209_CAP_gas_full_bins',
                 filter_type='CAP', ptype='gas', n_samples=5000, burnin=1000,
                 n_cosmo_params=6, output_dir=None, save_plots=True, cosmo_only=False):
    """Callable function for GP HMC inference on a single simulation.

    Args:
        cosmo_only: If True, only infer cosmological parameters, fix mass and pk_ratio to true values
    """

    if output_dir is None:
        output_dir = Path("inference_results")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load target simulation data
    sim_data = load_target_simulation_data(sim_id, filter_type, ptype)
    if sim_data is None:
        return None

    obs_profile = sim_data['obs_profile']
    true_params = sim_data['true_params']
    true_mass = sim_data['true_mass']
    true_pk_ratios = sim_data['true_pk_ratios']
    k_bins = sim_data.get('k_bins', None)

    # Load parameter information
    param_names, fiducial_values, maxdiff, minVal, maxVal = getParamsFiducial()

    # Set up parameter bounds and labels
    param_bounds = []
    param_labels = []
    true_values_inference = []

    # Always add cosmological parameters
    for i in range(n_cosmo_params):
        param_bounds.append((minVal[i], maxVal[i]))
        param_labels.append(param_names[i])
        true_values_inference.append(true_params[i])

    if not cosmo_only:
        # Add halo mass bounds
        param_bounds.append((11.62, 14.82))
        param_labels.append('log10(M_halo)')
        true_values_inference.append(np.log10(true_mass))

        # Add pk_ratio array bounds
        pk_min, pk_max = 0.8, 1.02
        if k_bins is not None and true_pk_ratios is not None:
            n_k_bins = len(k_bins)
            for i in range(n_k_bins):
                param_bounds.append((pk_min, pk_max))
                param_labels.append(f'pk_ratio_k{i}')
                true_values_inference.append(true_pk_ratios[i])

    # Create GP likelihood function
    log_likelihood_fn = create_gp_likelihood_function(
        gp_name, obs_profile, fiducial_values,
        cosmo_only=cosmo_only, true_mass=true_mass, true_pk_ratios=true_pk_ratios
    )

    # Run HMC sampling
    samples, log_likes, acceptance_rate = run_hmc_sampling(
        log_likelihood_fn, param_bounds, n_samples=n_samples, burnin=burnin
    )

    # Compute diagnostics
    diagnostics = compute_mcmc_diagnostics(samples, burnin=burnin)

    # Create plots if requested
    corner_plot_path = None
    pk_ratio_plot_path = None
    pk_ratio_stats = None

    if save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Corner plot (cosmological parameters only)
        corner_plot_path = output_dir / f"gp_hmc_corner_sim{sim_id}_{timestamp}.png"
        create_corner_plot(
            samples[burnin:], true_values_inference, param_labels,
            acceptance_rate, sim_id, corner_plot_path, n_cosmo_params
        )

        # pk_ratio vs k plot (only if not cosmo_only)
        if not cosmo_only:
            pk_ratio_plot_path = output_dir / f"gp_hmc_pk_ratio_sim{sim_id}_{timestamp}.png"
            pk_ratio_stats = create_pk_ratio_plot(
                samples, true_pk_ratios, k_bins, burnin, sim_id, pk_ratio_plot_path
            )

    return {
        'sim_id': sim_id,
        'samples': samples,
        'log_likes': log_likes,
        'acceptance_rate': acceptance_rate,
        'diagnostics': diagnostics,
        'param_labels': param_labels,
        'param_bounds': param_bounds,
        'true_values': true_values_inference,
        'corner_plot_path': corner_plot_path,
        'pk_ratio_plot_path': pk_ratio_plot_path,
        'pk_ratio_stats': pk_ratio_stats,
        'burnin': burnin,
        'cosmo_only': cosmo_only,
        'n_cosmo_params': n_cosmo_params
    }


def run_batch_inference(sim_ids, **kwargs):
    """Run inference on multiple simulations and collect results."""

    results = {}
    all_samples = []
    all_log_likes = []
    all_true_values = []

    print(f"Running batch inference on {len(sim_ids)} simulations")

    for i, sim_id in enumerate(sim_ids):
        print(f"Processing simulation {sim_id} ({i+1}/{len(sim_ids)})")

        # Run inference with plots only for first simulation (unless explicitly disabled)
        save_plots_for_sim = (i == 0) and kwargs.get('save_plots', True)  # Only save plots for first sim to avoid clutter
        kwargs_copy = kwargs.copy()
        kwargs_copy['save_plots'] = save_plots_for_sim
        result = run_inference(sim_id, **kwargs_copy)

        if result is not None:
            results[sim_id] = result
            all_samples.append(result['samples'])
            all_log_likes.append(result['log_likes'])
            all_true_values.append(result['true_values'])
        else:
            print(f"  Warning: No data for simulation {sim_id}")

    # Combine results for coverage analysis
    if all_samples:
        combined_results = {
            'individual_results': results,
            'all_samples': all_samples,
            'all_log_likes': all_log_likes,
            'all_true_values': all_true_values,
            'param_labels': results[list(results.keys())[0]]['param_labels'],
            'n_simulations': len(results)
        }

        print(f"Batch inference completed: {len(results)}/{len(sim_ids)} simulations successful")
        return combined_results
    else:
        print("No successful inferences")
        return None


def main():
    """Main function for command-line usage."""

    parser = argparse.ArgumentParser(description='GP HMC Parameter Inference')
    parser.add_argument('--sim_id', type=int, default=777, help='Target simulation ID')
    parser.add_argument('--gp_name', type=str,
                       default='GPTrainer_091025_2209_CAP_gas_full_bins',
                       help='GP model directory')
    parser.add_argument('--filter_type', type=str, default='CAP', help='Filter type')
    parser.add_argument('--ptype', type=str, default='gas', help='Particle type')
    parser.add_argument('--n_samples', type=int, default=5000, help='Number of MCMC samples')
    parser.add_argument('--burnin', type=int, default=1000, help='Burn-in samples')
    parser.add_argument('--n_cosmo_params', type=int, default=6, help='Number of cosmological parameters to vary')
    parser.add_argument('--cosmo_only', action='store_true', help='Only infer cosmological parameters (fix mass and pk_ratio to true values)')

    args = parser.parse_args()

    print(f"GP HMC Parameter Inference - Simulation {args.sim_id}")
    if args.cosmo_only:
        print("Mode: Cosmological parameters only (mass and pk_ratio fixed to true values)")
    else:
        print("Mode: Full inference (cosmological parameters + mass + pk_ratio)")

    result = run_inference(
        sim_id=args.sim_id,
        gp_name=args.gp_name,
        filter_type=args.filter_type,
        ptype=args.ptype,
        n_samples=args.n_samples,
        burnin=args.burnin,
        n_cosmo_params=args.n_cosmo_params,
        cosmo_only=args.cosmo_only
    )

    if result is not None:
        print(f"Inference completed!")
        print(f"Acceptance rate: {result['acceptance_rate']:.1%}")
        print(f"Mean ESS: {result['diagnostics']['mean_ess']:.0f}")

        # Save results
        output_dir = Path("inference_results")
        save_inference_results(
            result['samples'], result['param_labels'], result['acceptance_rate'],
            result['diagnostics'], args.sim_id, output_dir
        )
    else:
        print(f"No data found for simulation {args.sim_id}")


if __name__ == "__main__":
    main()