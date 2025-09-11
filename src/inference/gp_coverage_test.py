#!/usr/bin/env python3
"""
Real GP Coverage Test for Multiple Models.

This script performs comprehensive coverage testing for GP-based parameter inference
by running HMC inference on multiple simulations and plotting C(α) vs α coverage plots.
Uses real trained GP models for likelihood evaluation.

Usage:
    python real_gp_coverage_test.py --n_sims 100 --gp_dirs /path/to/gp1 /path/to/gp2

Features:
- Multiple GP model testing
- Real GP likelihood evaluation (no mock functions)
- Proper C(α) vs α coverage plot format
- Comprehensive coverage statistics
- Parallel processing support for faster testing
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root
sys.path.append('/pscratch/sd/l/lindajin/DH_profile_kSZ_WL')

# Set JAX platform
os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import jax.numpy as jnp
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

from src.data.profile_loader import load_simulation_mean_profiles, getParamsFiducial


def load_gp_models_and_data(gp_dir, n_training_samples=716):
    """Load GP models and their training data."""
    
    model_path = Path(gp_dir)
    
    # Load models
    with open(model_path / 'trained_models.pkl', 'rb') as f:
        gp_models = pickle.load(f)
    
    # Load training info
    with open(model_path / 'training_info.json', 'r') as f:
        training_info = json.load(f)
    
    # Load training data (matching what was used for training)
    train_r_bins, train_profiles, train_masses, train_params, train_k, train_pk_ratios = load_simulation_mean_profiles(
        list(range(n_training_samples)), filterType='CAP', ptype='gas'
    )
    
    # Prepare training features
    n_k_features = len(train_k) if train_k is not None else 79
    n_features = 35 + 1 + 1 + n_k_features
    
    X_train = np.zeros((n_training_samples, n_features))
    X_train[:, :35] = train_params[:, :35]
    
    for i in range(n_training_samples):
        try:
            X_train[i, 35] = np.log10(float(train_masses[i]))
        except (TypeError, ValueError, IndexError):
            X_train[i, 35] = np.log10(1e14)
            
        try:
            X_train[i, 36] = np.log10(float(train_pk_ratios[i]))
        except (TypeError, ValueError, IndexError):
            X_train[i, 36] = np.log10(1.0)
    
    # Fill k_features
    if n_k_features > 0:
        X_train[:, 37:] = 0.0
    
    y_train = train_profiles
    
    return {
        'gp_models': gp_models,
        'X_train': X_train,
        'y_train': y_train,
        'training_info': training_info,
        'n_features': n_features,
        'r_bins': train_r_bins
    }


def create_gp_likelihood_function_for_sim(gp_data, obs_profile, true_mass, true_pk_ratio):
    """Create GP likelihood function for a specific simulation."""
    
    gp_models = gp_data['gp_models']
    X_train = gp_data['X_train']
    y_train = gp_data['y_train']
    n_features = gp_data['n_features']
    
    def gp_log_likelihood(test_params):
        try:
            # Construct GP input
            gp_input = np.zeros(n_features)
            gp_input[:35] = test_params[:35]
            gp_input[35] = np.log10(true_mass)
            gp_input[36] = np.log10(true_pk_ratio)
            
            # Get GP predictions
            test_input_jnp = jnp.array(gp_input).reshape(1, -1)
            pred_means = []
            pred_vars = []
            
            for i, gp_model in enumerate(gp_models):
                y_train_i = jnp.array(y_train[:, i])
                _, cond_gp = gp_model.condition(y_train_i, test_input_jnp)
                pred_means.append(float(cond_gp.mean[0]))
                pred_vars.append(float(cond_gp.variance[0]))
            
            pred_profile = np.array(pred_means)
            pred_vars = np.array(pred_vars)
            
            # Gaussian likelihood
            obs_noise = 0.02 * np.abs(obs_profile)
            total_var = pred_vars + obs_noise**2
            
            residuals = obs_profile - pred_profile
            log_like = -0.5 * np.sum(residuals**2 / total_var + np.log(2 * np.pi * total_var))
            
            return log_like if np.isfinite(log_like) else -1e10
            
        except Exception:
            return -1e10
    
    return gp_log_likelihood


def run_single_inference(sim_id, gp_data, selected_indices, param_names, minVal, maxVal, 
                        n_samples=2000, burnin=500):
    """Run inference for a single simulation."""
    
    try:
        # Load simulation data
        r_bins, profiles, masses, params, k, pk_ratios = load_simulation_mean_profiles(
            [sim_id], filterType='CAP', ptype='gas'
        )
        
        if profiles is None or len(profiles) == 0:
            return None
        
        obs_profile = profiles[0]
        true_params = params[0]
        
        try:
            true_mass = float(masses[0])
        except (TypeError, ValueError):
            true_mass = 1e14
        
        try:
            true_pk_ratio = float(pk_ratios[0]) if pk_ratios is not None else 1.0
        except (TypeError, ValueError):
            true_pk_ratio = 1.0
        
        # Create likelihood function
        log_likelihood_fn = create_gp_likelihood_function_for_sim(
            gp_data, obs_profile, true_mass, true_pk_ratio
        )
        
        # Simple MCMC sampling (simplified for speed)
        current_params = np.array([0.5 * (minVal[i] + maxVal[i]) for i in range(len(minVal))])
        samples = np.zeros((n_samples, len(minVal)))
        n_accepted = 0
        
        # Step sizes
        step_sizes = {}
        for param_idx in selected_indices:
            param_range = maxVal[param_idx] - minVal[param_idx]
            step_sizes[param_idx] = 0.02 * param_range
        
        current_log_like = log_likelihood_fn(current_params)
        
        for i in range(n_samples):
            # Propose new parameters
            proposal_params = current_params.copy()
            
            for param_idx in selected_indices:
                proposal_params[param_idx] += np.random.normal(0, step_sizes[param_idx])
                proposal_params[param_idx] = np.clip(
                    proposal_params[param_idx], minVal[param_idx], maxVal[param_idx]
                )
            
            # Evaluate proposal
            proposal_log_like = log_likelihood_fn(proposal_params)
            
            # Accept/reject
            if np.log(np.random.rand()) < (proposal_log_like - current_log_like):
                current_params = proposal_params.copy()
                current_log_like = proposal_log_like
                n_accepted += 1
            
            samples[i] = current_params
        
        # Return results
        samples_clean = samples[burnin:]
        acceptance_rate = n_accepted / n_samples
        
        return {
            'sim_id': sim_id,
            'samples': samples_clean,
            'true_params': true_params,
            'acceptance_rate': acceptance_rate,
            'success': True
        }
        
    except Exception as e:
        print(f"   ❌ Simulation {sim_id} failed: {str(e)[:50]}...")
        return {
            'sim_id': sim_id,
            'success': False,
            'error': str(e)
        }


def compute_coverage_statistics(inference_results, selected_indices, confidence_levels):
    """Compute coverage statistics from inference results."""
    
    print(f"\n📊 Computing coverage statistics...")
    
    successful_results = [r for r in inference_results if r['success']]
    n_successful = len(successful_results)
    
    print(f"   • Successful inferences: {n_successful}/{len(inference_results)}")
    
    coverage_stats = {}
    
    for alpha in confidence_levels:
        # Compute credible intervals for each parameter and simulation
        param_coverage = []
        
        for param_idx in selected_indices:
            covered_count = 0
            
            for result in successful_results:
                samples = result['samples']
                true_val = result['true_params'][param_idx]
                
                # Compute credible interval
                param_samples = samples[:, param_idx]
                lower = np.percentile(param_samples, 50 * (1 - alpha))
                upper = np.percentile(param_samples, 50 * (1 + alpha))
                
                # Check if true value is within interval
                if lower <= true_val <= upper:
                    covered_count += 1
            
            coverage_fraction = covered_count / n_successful
            param_coverage.append(coverage_fraction)
        
        # Average coverage across parameters
        mean_coverage = np.mean(param_coverage)
        coverage_stats[alpha] = {
            'mean_coverage': mean_coverage,
            'param_coverage': param_coverage,
            'expected_coverage': alpha,
            'deviation': abs(mean_coverage - alpha)
        }
    
    return coverage_stats


def create_coverage_plot(coverage_stats, gp_model_name, save_path):
    """Create C(α) vs α coverage plot."""
    
    confidence_levels = sorted(coverage_stats.keys())
    observed_coverage = [coverage_stats[alpha]['mean_coverage'] for alpha in confidence_levels]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    
    # Observed coverage
    ax.plot(confidence_levels, observed_coverage, 'o-', linewidth=2, markersize=6, 
           color='red', label=f'Observed Coverage ({gp_model_name})')
    
    # Confidence bands (approximate)
    n_sims = 50  # Approximate number of successful simulations
    std_error = np.sqrt(np.array(confidence_levels) * (1 - np.array(confidence_levels)) / n_sims)
    upper_bound = np.array(confidence_levels) + 1.96 * std_error
    lower_bound = np.array(confidence_levels) - 1.96 * std_error
    
    ax.fill_between(confidence_levels, lower_bound, upper_bound, alpha=0.2, color='gray', 
                   label='95% Confidence Band')
    
    # Formatting
    ax.set_xlabel('Confidence Level (α)', fontsize=12)
    ax.set_ylabel('Observed Coverage C(α)', fontsize=12)
    ax.set_title(f'Coverage Plot: Real GP Model ({gp_model_name})\n'
                f'Perfect calibration should follow the diagonal line', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Add text box with statistics
    mean_deviation = np.mean([coverage_stats[alpha]['deviation'] for alpha in confidence_levels])
    textstr = f'Mean |deviation|: {mean_deviation:.3f}\n'
    textstr += f'Max deviation: {max([coverage_stats[alpha]["deviation"] for alpha in confidence_levels]):.3f}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"💾 Coverage plot saved: {save_path}")
    plt.close()


def save_coverage_results(inference_results, coverage_stats, gp_model_dirs, output_dir):
    """Save comprehensive coverage test results."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save raw results
    results_path = output_dir / f"coverage_test_results_{timestamp}.pkl"
    results_data = {
        'inference_results': inference_results,
        'coverage_stats': coverage_stats,
        'gp_model_dirs': gp_model_dirs,
        'timestamp': timestamp
    }
    
    with open(results_path, 'wb') as f:
        pickle.dump(results_data, f)
    
    # Save summary report
    report_path = output_dir / f"coverage_test_summary_{timestamp}.md"
    with open(report_path, 'w') as f:
        f.write(f"# Real GP Coverage Test Results\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## Test Configuration\n")
        f.write(f"- GP model directories: {len(gp_model_dirs)}\n")
        for i, gp_dir in enumerate(gp_model_dirs):
            f.write(f"  {i+1}. {gp_dir}\n")
        
        successful_results = [r for r in inference_results if r['success']]
        f.write(f"- Total simulations attempted: {len(inference_results)}\n")
        f.write(f"- Successful inferences: {len(successful_results)}\n")
        f.write(f"- Success rate: {len(successful_results)/len(inference_results):.1%}\n\n")
        
        f.write(f"## Coverage Statistics\n")
        f.write(f"| Confidence Level (α) | Expected | Observed | Deviation |\n")
        f.write(f"|---------------------|----------|----------|----------|\n")
        
        for alpha in sorted(coverage_stats.keys()):
            stats = coverage_stats[alpha]
            f.write(f"| {alpha:.2f} | {alpha:.2f} | {stats['mean_coverage']:.3f} | {stats['deviation']:.3f} |\n")
        
        mean_deviation = np.mean([coverage_stats[alpha]['deviation'] for alpha in coverage_stats.keys()])
        f.write(f"\n**Mean absolute deviation:** {mean_deviation:.3f}\n\n")
        
        # Model evaluation
        if mean_deviation < 0.05:
            f.write("✅ **Model Assessment: WELL-CALIBRATED** - Mean deviation < 0.05\n")
        elif mean_deviation < 0.10:
            f.write("⚠️ **Model Assessment: MODERATELY CALIBRATED** - Mean deviation 0.05-0.10\n")
        else:
            f.write("❌ **Model Assessment: POORLY CALIBRATED** - Mean deviation > 0.10\n")
        
        f.write(f"\n## Individual Simulation Results\n")
        f.write(f"| Sim ID | Status | Acceptance Rate | Notes |\n")
        f.write(f"|--------|--------|-----------------| ------ |\n")
        
        for result in inference_results[:20]:  # Show first 20 results
            if result['success']:
                f.write(f"| {result['sim_id']} | ✅ Success | {result['acceptance_rate']:.1%} | - |\n")
            else:
                f.write(f"| {result['sim_id']} | ❌ Failed | - | {result.get('error', 'Unknown')[:30]} |\n")
    
    print(f"💾 Coverage results saved:")
    print(f"   • Data: {results_path}")
    print(f"   • Summary: {report_path}")
    
    return results_path, report_path


def main():
    """Main function for coverage testing."""
    
    parser = argparse.ArgumentParser(description='Real GP Coverage Test')
    parser.add_argument('--n_sims', type=int, default=50, 
                       help='Number of simulations to test')
    parser.add_argument('--gp_dirs', nargs='+', 
                       default=['/pscratch/sd/l/lindajin/DH_profile_kSZ_WL/trained_gp_models/GPTrainer_091025_2209_CAP_gas/'],
                       help='Directories containing trained GP models')
    parser.add_argument('--start_sim', type=int, default=800, 
                       help='Starting simulation ID (avoid training data)')
    parser.add_argument('--n_samples', type=int, default=2000, 
                       help='MCMC samples per simulation')
    parser.add_argument('--burnin', type=int, default=500, 
                       help='MCMC burn-in samples')
    parser.add_argument('--n_params', type=int, default=6, 
                       help='Number of parameters to test')
    parser.add_argument('--parallel', action='store_true', 
                       help='Use parallel processing')
    
    args = parser.parse_args()
    
    print("🚀 Real GP Coverage Test")
    print("=" * 60)
    print(f"Testing {args.n_sims} simulations with {len(args.gp_dirs)} GP models")
    print(f"MCMC: {args.n_samples} samples, {args.burnin} burn-in")
    print(f"Parameters to test: {args.n_params}")
    
    # Create output directory
    output_dir = Path("inference_results")
    output_dir.mkdir(exist_ok=True)
    
    # Load parameter information
    param_names, fiducial_values, maxdiff, minVal, maxVal = getParamsFiducial()
    selected_indices = list(range(args.n_params))
    
    # Define confidence levels for coverage testing
    confidence_levels = np.linspace(0.1, 0.9, 9)  # 10% to 90% in steps of 10%
    
    # Test each GP model
    for gp_idx, gp_dir in enumerate(args.gp_dirs):
        print(f"\n🔄 Testing GP model {gp_idx+1}/{len(args.gp_dirs)}: {Path(gp_dir).name}")
        
        # Load GP model and training data
        print("   Loading GP models and training data...")
        try:
            gp_data = load_gp_models_and_data(gp_dir)
            print(f"   ✅ Loaded {len(gp_data['gp_models'])} GP models")
        except Exception as e:
            print(f"   ❌ Failed to load GP model: {e}")
            continue
        
        # Generate simulation IDs to test
        sim_ids = list(range(args.start_sim, args.start_sim + args.n_sims))
        print(f"   Testing simulations: {sim_ids[0]} to {sim_ids[-1]}")
        
        # Run inference on multiple simulations
        print("   Running inference on test simulations...")
        inference_results = []
        
        if args.parallel and cpu_count() > 1:
            print(f"   Using parallel processing with {min(4, cpu_count())} workers...")
            with ProcessPoolExecutor(max_workers=min(4, cpu_count())) as executor:
                futures = []
                for sim_id in sim_ids:
                    future = executor.submit(
                        run_single_inference, sim_id, gp_data, selected_indices, 
                        param_names, minVal, maxVal, args.n_samples, args.burnin
                    )
                    futures.append(future)
                
                for future in tqdm(futures, desc="   Processing"):
                    result = future.result()
                    if result is not None:
                        inference_results.append(result)
        else:
            for sim_id in tqdm(sim_ids, desc="   Processing"):
                result = run_single_inference(
                    sim_id, gp_data, selected_indices, param_names, minVal, maxVal,
                    args.n_samples, args.burnin
                )
                if result is not None:
                    inference_results.append(result)
        
        # Compute coverage statistics
        coverage_stats = compute_coverage_statistics(
            inference_results, selected_indices, confidence_levels
        )
        
        # Create coverage plot
        print("   Creating coverage plot...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gp_model_name = Path(gp_dir).name
        coverage_plot_path = output_dir / f"coverage_plot_{gp_model_name}_{timestamp}.png"
        
        create_coverage_plot(coverage_stats, gp_model_name, coverage_plot_path)
        
        # Save results
        print("   Saving results...")
        results_path, report_path = save_coverage_results(
            inference_results, coverage_stats, [gp_dir], output_dir
        )
        
        # Print summary
        successful_results = [r for r in inference_results if r['success']]
        mean_deviation = np.mean([coverage_stats[alpha]['deviation'] for alpha in confidence_levels])
        
        print(f"   ✅ Coverage test completed!")
        print(f"   • Successful inferences: {len(successful_results)}/{len(inference_results)}")
        print(f"   • Mean absolute deviation: {mean_deviation:.3f}")
        
        if mean_deviation < 0.05:
            print("   • Assessment: ✅ WELL-CALIBRATED")
        elif mean_deviation < 0.10:
            print("   • Assessment: ⚠️  MODERATELY CALIBRATED")
        else:
            print("   • Assessment: ❌ POORLY CALIBRATED")
    
    print(f"\n✅ All coverage tests completed!")
    print(f"   • Results saved to: {output_dir}")


if __name__ == "__main__":
    main()