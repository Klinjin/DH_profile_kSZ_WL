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

from src.data.profile_loader import getParamsFiducial
from src.inference.gp_hmc_inference import run_batch_inference
from src.models.gp_trainer import GPTrainer


def load_gp_trainer_and_get_test_indices(gp_dir):
    """Load GP trainer and get test simulation indices."""

    model_path = Path(gp_dir)

    # Load GP trainer
    trainer = GPTrainer(
        sim_indices_total=list(range(1024)),
        train_test_val_split=(0.7, 0.2, 0.1),
        filterType='CAP',
        ptype='gas'
    )
    trainer._load_pretrained(model_path)

    # Get test indices from trainer
    test_indices = trainer.training_info.get('test_indices', [])

    return trainer, test_indices




def compute_coverage_statistics_from_batch(batch_results, confidence_levels):
    """Compute coverage statistics from batch inference results."""

    print(f"\n📊 Computing coverage statistics...")

    if batch_results is None:
        print("   ❌ No batch results available")
        return {}

    all_samples = batch_results['all_samples']
    all_true_values = batch_results['all_true_values']
    param_labels = batch_results['param_labels']
    n_successful = batch_results['n_simulations']

    print(f"   • Successful inferences: {n_successful}")

    coverage_stats = {}

    for alpha in confidence_levels:
        param_coverage = []

        for param_idx in range(len(param_labels)):
            covered_count = 0

            for sim_idx in range(len(all_samples)):
                # Get samples after burnin (samples already cleaned in batch_results)
                samples = all_samples[sim_idx]
                true_val = all_true_values[sim_idx][param_idx]

                # Compute credible interval
                param_samples = samples[:, param_idx]
                lower = np.percentile(param_samples, 50 * (1 - alpha))
                upper = np.percentile(param_samples, 50 * (1 + alpha))

                # Check if true value is within interval
                if lower <= true_val <= upper:
                    covered_count += 1

            coverage_fraction = covered_count / n_successful if n_successful > 0 else 0
            param_coverage.append(coverage_fraction)

        # Average coverage across parameters
        mean_coverage = np.mean(param_coverage) if param_coverage else 0
        coverage_stats[alpha] = {
            'mean_coverage': mean_coverage,
            'param_coverage': param_coverage,
            'expected_coverage': alpha,
            'deviation': abs(mean_coverage - alpha)
        }

    return coverage_stats


def create_coverage_plot(coverage_stats, gp_model_name, save_path, n_simulations):
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
    
    # Confidence bands (exact based on actual number of simulations)
    std_error = np.sqrt(np.array(confidence_levels) * (1 - np.array(confidence_levels)) / n_simulations)
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


def save_coverage_results(batch_results, coverage_stats, gp_model_dir, output_dir):
    """Save comprehensive coverage test results."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save raw results
    results_path = output_dir / f"coverage_test_results_{timestamp}.pkl"
    results_data = {
        'batch_results': batch_results,
        'coverage_stats': coverage_stats,
        'gp_model_dir': gp_model_dir,
        'timestamp': timestamp
    }

    with open(results_path, 'wb') as f:
        pickle.dump(results_data, f)

    # Save summary report
    report_path = output_dir / f"coverage_test_summary_{timestamp}.md"
    with open(report_path, 'w') as f:
        f.write(f"# GP Coverage Test Results - {Path(gp_model_dir).name}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"## Test Configuration\n")
        f.write(f"- GP model: {gp_model_dir}\n")
        f.write(f"- Total simulations attempted: {len(batch_results['individual_results']) if batch_results else 0}\n")
        f.write(f"- Successful inferences: {batch_results['n_simulations'] if batch_results else 0}\n")
        if batch_results and len(batch_results['individual_results']) > 0:
            success_rate = batch_results['n_simulations'] / len(batch_results['individual_results'])
            f.write(f"- Success rate: {success_rate:.1%}\n")
        f.write(f"- Parameters tested: {', '.join(batch_results['param_labels']) if batch_results else 'N/A'}\n\n")

        f.write(f"## Coverage Statistics\n")
        f.write(f"| Confidence Level (α) | Expected | Observed | Deviation |\n")
        f.write(f"|---------------------|----------|----------|----------|\n")

        for alpha in sorted(coverage_stats.keys()):
            stats = coverage_stats[alpha]
            f.write(f"| {alpha:.2f} | {alpha:.2f} | {stats['mean_coverage']:.3f} | {stats['deviation']:.3f} |\n")

        if coverage_stats:
            mean_deviation = np.mean([coverage_stats[alpha]['deviation'] for alpha in coverage_stats.keys()])
            f.write(f"\n**Mean absolute deviation:** {mean_deviation:.3f}\n\n")

            # Model evaluation
            if mean_deviation < 0.05:
                f.write("✅ **Model Assessment: WELL-CALIBRATED** - Mean deviation < 0.05\n")
            elif mean_deviation < 0.10:
                f.write("⚠️ **Model Assessment: MODERATELY CALIBRATED** - Mean deviation 0.05-0.10\n")
            else:
                f.write("❌ **Model Assessment: POORLY CALIBRATED** - Mean deviation > 0.10\n")

        # Individual simulation results
        f.write(f"\n## Individual Simulation Results (First 20)\n")
        f.write(f"| Sim ID | Status | Acceptance Rate |\n")
        f.write(f"|--------|--------|-----------------|\n")

        if batch_results and batch_results['individual_results']:
            for i, (sim_id, result) in enumerate(list(batch_results['individual_results'].items())[:20]):
                f.write(f"| {sim_id} | ✅ Success | {result['acceptance_rate']:.1%} |\n")

    print(f"💾 Coverage results saved:")
    print(f"   • Data: {results_path}")
    print(f"   • Summary: {report_path}")

    return results_path, report_path


def main():
    """Main function for coverage testing."""
    
    parser = argparse.ArgumentParser(description='Real GP Coverage Test')
    parser.add_argument('--n_sims', type=int, default=50, 
                       help='Number of simulations to test')
    parser.add_argument('--gp_dir', type=str,
                       default='/pscratch/sd/l/lindajin/DH_profile_kSZ_WL/trained_gp_models/GPTrainer_091025_2209_CAP_gas_full_bins/',
                       help='Directory containing trained GP model')
    parser.add_argument('--start_sim', type=int, default=800, 
                       help='Starting simulation ID (avoid training data)')
    parser.add_argument('--n_samples', type=int, default=2000, 
                       help='MCMC samples per simulation')
    parser.add_argument('--burnin', type=int, default=500, 
                       help='MCMC burn-in samples')
    parser.add_argument('--n_params', type=int, default=6,
                       help='Number of parameters to test')
    parser.add_argument('--cosmo_only', action='store_true',
                       help='Only infer cosmological parameters (fix mass and pk_ratio to true values)')
    parser.add_argument('--parallel', action='store_true',
                       help='Use parallel processing')
    
    args = parser.parse_args()
    
    print("🚀 Real GP Coverage Test")
    print("=" * 60)
    print(f"Testing {args.n_sims} simulations with GP model: {Path(args.gp_dir).name}")
    print(f"MCMC: {args.n_samples} samples, {args.burnin} burn-in")
    print(f"Parameters to test: {args.n_params}")
    if args.cosmo_only:
        print("Mode: Cosmological parameters only (mass and pk_ratio fixed to true values)")
    else:
        print("Mode: Full inference (cosmological parameters + mass + pk_ratio)")
    
    # Create output directory
    output_dir = Path("inference_results")
    output_dir.mkdir(exist_ok=True)
    
    # Load parameter information (for reference, not used in batch processing)
    _ = getParamsFiducial()
    
    # Define confidence levels for coverage testing
    confidence_levels = np.linspace(0, 1, 20)  # 0% to 100% with 20 points total
    
    # Test the single GP model
    gp_dir = args.gp_dir
    print(f"\n🔄 Testing GP model: {Path(gp_dir).name}")

    # Load GP trainer and get test indices
    print("   Loading GP trainer and getting test indices...")
    try:
        _, test_indices = load_gp_trainer_and_get_test_indices(gp_dir)
        print(f"   ✅ Loaded GP trainer with {len(test_indices)} test simulations available")
    except Exception as e:
        print(f"   ❌ Failed to load GP trainer: {e}")
        return

    # Use test indices or fallback to specified range
    if test_indices and len(test_indices) >= args.n_sims:
        sim_ids = test_indices[:args.n_sims]
        print(f"   Using GP test indices: {len(sim_ids)} simulations")
    else:
        sim_ids = list(range(args.start_sim, args.start_sim + args.n_sims))
        print(f"   Using fallback range: {sim_ids[0]} to {sim_ids[-1]}")

    # Run batch inference using the new integrated function
    print("   Running batch inference on test simulations...")
    batch_results = run_batch_inference(
        sim_ids=sim_ids,
        n_samples=args.n_samples,
        burnin=args.burnin,
        n_cosmo_params=args.n_params,
        gp_name=Path(gp_dir).name,
        cosmo_only=args.cosmo_only,
        save_plots=False  # Don't save individual plots for coverage testing
    )

    if batch_results is None:
        print(f"   ❌ Batch inference failed for GP model: {Path(gp_dir).name}")
        return

    print(f"   ✅ Batch inference completed: {batch_results['n_simulations']}/{len(sim_ids)} successful")

    # Compute coverage statistics using batch results
    coverage_stats = compute_coverage_statistics_from_batch(
        batch_results, confidence_levels
    )

    # Create coverage plot
    print("   Creating coverage plot...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gp_model_name = Path(gp_dir).name
    coverage_plot_path = output_dir / f"coverage_plot_{gp_model_name}_{timestamp}.png"

    create_coverage_plot(coverage_stats, gp_model_name, coverage_plot_path, batch_results['n_simulations'])

    # Save results
    print("   Saving results...")
    save_coverage_results(
        batch_results, coverage_stats, gp_dir, output_dir
    )

    # Print summary
    mean_deviation = np.mean([coverage_stats[alpha]['deviation'] for alpha in confidence_levels]) if coverage_stats else 0

    print(f"   ✅ Coverage test completed!")
    print(f"   • Successful inferences: {batch_results['n_simulations']}/{len(sim_ids)}")
    print(f"   • Mean absolute deviation: {mean_deviation:.3f}")
    print(f"   • Coverage plot saved: {coverage_plot_path}")

    if mean_deviation < 0.05:
        print("   • Assessment: ✅ WELL-CALIBRATED")
    elif mean_deviation < 0.10:
        print("   • Assessment: ⚠️  MODERATELY CALIBRATED")
    else:
        print("   • Assessment: ❌ POORLY CALIBRATED")

    print(f"\n✅ Coverage test completed!")
    print(f"   • Results saved to: {output_dir}")


if __name__ == "__main__":
    main()