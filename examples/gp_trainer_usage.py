"""
Example usage of the GPTrainer class for cosmological simulation analysis.

This script demonstrates how to use the new GPTrainer class with:
- Automatic data loading and splitting
- Multiple kernel types (hierarchical, robust, physics-informed)
- Training with progress visualization
- Testing with comprehensive metrics and plots
- Model saving and loading

Run this example with different configurations to compare GP performance.
"""

import sys
import numpy as np
import os
from datetime import datetime

# Add project root to path
sys.path.append('/pscratch/sd/l/lindajin/DH_profile_kSZ_WL')

from src.models.gp_trainer import GPTrainer

def basic_gp_training_example():
    """Basic GPTrainer usage example."""
    print("=== Basic GP Training Example ===")
    
    # Define simulation indices
    sim_indices_train = [0, 1, 2, 10, 20]  # Small subset for quick example
    sim_indices_test = [3, 4, 5, 11, 21]   # Different sims for testing
    
    # Initialize GPTrainer
    trainer = GPTrainer(
        sim_indices_train=sim_indices_train,
        sim_indices_test=sim_indices_test,
        filterType='CAP',
        ptype='gas',
        val_ratio=0.0  # Train/test only
    )
    
    # Train GP model
    training_info = trainer.train(
        kernel_type='hierarchical',
        maxiter=500,  # Reduced for quick example
        plot=True,
        save=True
    )
    
    print(f"Training completed in {training_info['train_time']:.1f}s")
    
    # Test the model
    metrics = trainer.test(plot=True, save_plots=True)
    
    print(f"Test MAPE: {metrics['mape']:.1f}%")
    print(f"Results saved to: {trainer.save_dir}")
    
    return trainer

def kernel_comparison_example():
    """Compare different GP kernel types."""
    print("\n=== Kernel Comparison Example ===")
    
    # Use sparse sampling indices for realistic example
    try:
        train_indices = np.load('data/sparse_sampling_train_indices_random.npy')[:15]
        test_indices = np.load('data/sparse_sampling_test_indices_random.npy')[:10]
    except FileNotFoundError:
        print("Sparse sampling files not found, using sequential indices")
        train_indices = list(range(15))
        test_indices = list(range(15, 25))
    
    kernel_types = ['hierarchical', 'robust', 'physics_informed']
    results = {}
    
    for kernel_type in kernel_types:
        print(f"\n--- Training with {kernel_type} kernel ---")
        
        # Create trainer with timestamped save directory
        timestamp = datetime.now().strftime('%H%M%S')
        trainer = GPTrainer(
            sim_indices_train=train_indices,
            sim_indices_test=test_indices,
            filterType='CAP',
            ptype='gas',
            save_dir=f'trained_gp_models/kernel_comparison_{kernel_type}_{timestamp}'
        )
        
        try:
            # Train model
            training_info = trainer.train(
                kernel_type=kernel_type,
                maxiter=800,
                plot=False,  # Skip plots for comparison
                save=True
            )
            
            # Test model
            metrics = trainer.test(plot=False, save_plots=False)
            
            results[kernel_type] = {
                'train_time': training_info['train_time'],
                'mape': metrics['mape'],
                'mse': metrics['mse'],
                'r2': metrics['r2'],
                'trainer': trainer
            }
            
            print(f"{kernel_type}: MAPE={metrics['mape']:.1f}%, Time={training_info['train_time']:.1f}s")
            
        except Exception as e:
            print(f"❌ {kernel_type} kernel failed: {e}")
            results[kernel_type] = {'error': str(e)}
    
    # Print comparison summary
    print(f"\n=== Kernel Comparison Results ===")
    print(f"{'Kernel':<15} {'MAPE%':<8} {'R²':<8} {'Time(s)':<10}")
    print("-" * 45)
    
    for kernel, result in results.items():
        if 'error' not in result:
            print(f"{kernel:<15} {result['mape']:<8.1f} {result['r2']:<8.3f} {result['train_time']:<10.1f}")
        else:
            print(f"{kernel:<15} ERROR")
    
    # Find best performing kernel
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        best_kernel = min(valid_results, key=lambda x: valid_results[x]['mape'])
        print(f"\nBest performing kernel: {best_kernel} (MAPE: {valid_results[best_kernel]['mape']:.1f}%)")
        
        # Generate detailed plots for best kernel
        print("Generating detailed plots for best kernel...")
        best_trainer = valid_results[best_kernel]['trainer']
        best_trainer.test(plot=True, save_plots=True)
    
    return results

def model_loading_example():
    """Example of loading previously trained models."""
    print("\n=== Model Loading Example ===")
    
    # Try to find a previously saved model
    base_dir = 'trained_gp_models'
    if not os.path.exists(base_dir):
        print("No previous models found. Run training examples first.")
        return
    
    # Find most recent GPTrainer directory
    gp_dirs = [d for d in os.listdir(base_dir) if d.startswith('GPTrainer')]
    if not gp_dirs:
        print("No GPTrainer models found. Run basic_gp_training_example() first.")
        return
    
    latest_dir = max(gp_dirs)
    model_path = os.path.join(base_dir, latest_dir)
    
    print(f"Loading model from: {model_path}")
    
    # Create trainer and load models
    trainer = GPTrainer(
        sim_indices_train=[0, 1, 2],  # Dummy indices for loading
        filterType='CAP',
        ptype='gas'
    )
    
    try:
        trainer.load_models(model_path)
        print(f"Successfully loaded {len(trainer.trained_models)} GP models")
        
        # Make predictions on new data (using training data as example)
        trainer._load_data()  # Load data for predictions
        pred_means, pred_vars = trainer.pred(trainer.X_test[:10])  # Predict on first 10 samples
        
        print(f"Generated predictions: {pred_means.shape}")
        print(f"Prediction range: [{np.min(pred_means):.2e}, {np.max(pred_means):.2e}]")
        
    except Exception as e:
        print(f"Failed to load models: {e}")

def validation_set_example():
    """Example using validation set for hyperparameter tuning."""
    print("\n=== Validation Set Example ===")
    
    trainer = GPTrainer(
        sim_indices_train=list(range(20)),
        sim_indices_test=list(range(20, 30)),
        filterType='CAP',
        ptype='gas',
        val_ratio=0.2  # Use 20% of training data for validation
    )
    
    # Train with validation monitoring
    training_info = trainer.train(
        kernel_type='hierarchical',
        maxiter=300,
        plot=True
    )
    
    # Test on held-out test set
    metrics = trainer.test(plot=True)
    
    print(f"Validation example completed")
    print(f"Training data: {trainer.X_train.shape}")
    print(f"Validation data: {trainer.X_val.shape}")
    print(f"Test data: {trainer.X_test.shape}")
    
    return trainer

if __name__ == "__main__":
    print("GPTrainer Usage Examples")
    print("=" * 50)
    
    try:
        # Run examples
        print("1. Running basic training example...")
        basic_trainer = basic_gp_training_example()
        
        print("\n2. Running kernel comparison...")
        comparison_results = kernel_comparison_example()
        
        print("\n3. Running model loading example...")
        model_loading_example()
        
        print("\n4. Running validation set example...")
        val_trainer = validation_set_example()
        
        print("\n" + "=" * 50)
        print("✅ All GPTrainer examples completed successfully!")
        print("\nKey features demonstrated:")
        print("  - Automatic data loading and splitting")
        print("  - Multiple kernel types (hierarchical, robust, physics-informed)")
        print("  - Training with progress visualization")
        print("  - Comprehensive testing with metrics and plots")
        print("  - Model saving and loading")
        print("  - Validation set support")
        
    except Exception as e:
        print(f"❌ Error in examples: {e}")
        import traceback
        traceback.print_exc()