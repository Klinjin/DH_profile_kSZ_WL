#!/usr/bin/env python3
"""
Example showing how to use the new GP HMC inference functions for coverage testing.
This demonstrates the usage for gp_coverage_test.py integration.
"""

import sys
import numpy as np
sys.path.append('/pscratch/sd/l/lindajin/DH_profile_kSZ_WL')

# Import the new callable functions
from src.inference.gp_hmc_inference import run_inference, run_batch_inference


def example_single_inference():
    """Example of single simulation inference."""

    print("=== Single Simulation Inference ===")

    # Run inference on a single simulation
    result = run_inference(
        sim_id=777,
        n_samples=500,  # Small for demo
        burnin=100,
        n_cosmo_params=4,
        save_plots=True
    )

    if result:
        print(f"✅ Inference successful for sim {result['sim_id']}")
        print(f"   • Acceptance rate: {result['acceptance_rate']:.1%}")
        print(f"   • Parameters: {len(result['param_labels'])}")
        print(f"   • Sample shape: {result['samples'].shape}")
        print(f"   • Corner plot: {result['corner_plot_path']}")
    else:
        print("❌ Inference failed")

    return result


def example_batch_inference():
    """Example of batch inference for coverage testing."""

    print("\n=== Batch Inference for Coverage Testing ===")

    # Example: simulate getting test indices from a trained GP model
    # In real usage: sim_ids = trainer.training_info['test_indices']
    sim_ids = [777, 778, 779, 780]  # Small subset for demo

    # Run batch inference
    batch_results = run_batch_inference(
        sim_ids=sim_ids,
        n_samples=200,  # Even smaller for demo
        burnin=50,
        n_cosmo_params=4,
        gp_name='GPTrainer_091025_2209_CAP_gas_full_bins'
    )

    if batch_results:
        print(f"✅ Batch inference completed!")
        print(f"   • Successful simulations: {batch_results['n_simulations']}/{len(sim_ids)}")
        print(f"   • Parameters: {batch_results['param_labels']}")
        print(f"   • Combined samples shape: {len(batch_results['all_samples'])} x {batch_results['all_samples'][0].shape if batch_results['all_samples'] else 'N/A'}")

        # Example coverage analysis
        if batch_results['n_simulations'] > 0:
            analyze_coverage(batch_results)
    else:
        print("❌ Batch inference failed")

    return batch_results


def analyze_coverage(batch_results):
    """Example coverage analysis function."""

    print("\n--- Coverage Analysis ---")

    all_samples = batch_results['all_samples']
    all_true_values = batch_results['all_true_values']
    param_labels = batch_results['param_labels']
    burnin = 50  # Use same as above

    coverage_stats = {}

    for param_idx, param_label in enumerate(param_labels):
        within_ci = 0
        total_sims = 0

        for sim_idx in range(len(all_samples)):
            if len(all_samples[sim_idx]) > burnin:
                # Get posterior samples for this parameter
                samples = all_samples[sim_idx][burnin:, param_idx]
                true_value = all_true_values[sim_idx][param_idx]

                # 95% credible interval
                ci_lower = np.percentile(samples, 2.5)
                ci_upper = np.percentile(samples, 97.5)

                # Check if true value is within CI
                if ci_lower <= true_value <= ci_upper:
                    within_ci += 1
                total_sims += 1

        coverage_rate = within_ci / total_sims if total_sims > 0 else 0
        coverage_stats[param_label] = {
            'coverage_rate': coverage_rate,
            'n_sims': total_sims,
            'within_ci': within_ci
        }

        print(f"   • {param_label}: {coverage_rate:.1%} ({within_ci}/{total_sims})")

    return coverage_stats


def example_gp_coverage_test_integration():
    """Example showing how to integrate with gp_coverage_test.py"""

    print("\n=== Integration with gp_coverage_test.py ===")

    # This is how you would use it in gp_coverage_test.py:
    example_code = '''
    # In gp_coverage_test.py:
    from src.inference.gp_hmc_inference import run_batch_inference

    # Get test simulation indices from trained GP
    trainer = GPTrainer(...)
    trainer._load_pretrained(model_path)
    test_sim_ids = trainer.training_info['test_indices']

    # Run batch inference
    batch_results = run_batch_inference(
        sim_ids=test_sim_ids,
        n_samples=2000,
        burnin=500,
        n_cosmo_params=6,
        gp_name=model_name
    )

    # Generate coverage plots and statistics
    if batch_results:
        coverage_stats = analyze_coverage(batch_results)
        create_coverage_plots(batch_results, coverage_stats)
        save_coverage_report(coverage_stats)
    '''

    print("Example integration code:")
    print(example_code)


if __name__ == "__main__":
    print("GP HMC Inference Coverage Testing Examples\n")

    # Run examples (commented out to avoid long execution)
    print("To run examples, uncomment the following lines:")
    print("# single_result = example_single_inference()")
    print("# batch_result = example_batch_inference()")

    # Show integration example
    example_gp_coverage_test_integration()

    print("\n✅ Example code ready for gp_coverage_test.py integration!")