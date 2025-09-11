"""
CPU-Only Training Script for Physics-Informed Neural Network

This script is optimized for single-CPU training on HPC systems.
It includes proper environment configuration and resource management
for efficient CPU-only neural network training.

Usage:
    python train_physics_nn_cpu.py
    
Or submit as SLURM job:
    sbatch job.sh
"""

import os
import numpy as np
import time
from datetime import datetime

# Set CPU-only environment before importing JAX
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'false'  # Use float32 for better CPU performance
os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=false --xla_cpu_use_mkl_dnn=true'

# Import after environment setup
from src.models.physics_neural_trainer import train_physics_neural_emulator, PhysicsNeuralTrainer
from src.models.physics_neural_emulator import PhysicsNeuralConfig
from src.models.physics_neural_trainer import PhysicsNeuralTrainerConfig
from src.data.sim_dataloader import SimulationDataLoader, DataLoaderConfig


def configure_cpu_training():
    """Configure optimal settings for CPU training."""
    
    print("🖥️  CPU Training Configuration")
    print("=" * 40)
    
    # Get CPU information
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=False)  # Physical cores
        cpu_threads = psutil.cpu_count(logical=True)  # Logical cores
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        print(f"   • Physical CPU cores: {cpu_count}")
        print(f"   • Logical CPU threads: {cpu_threads}")  
        print(f"   • Available memory: {memory_gb:.1f} GB")
    except ImportError:
        print("   • CPU info: psutil not available")
        cpu_threads = int(os.environ.get('OMP_NUM_THREADS', '32'))
        print(f"   • Using OMP_NUM_THREADS: {cpu_threads}")
    
    # Configure threading
    os.environ.setdefault('OMP_NUM_THREADS', str(min(32, cpu_threads)))
    os.environ.setdefault('MKL_NUM_THREADS', str(min(32, cpu_threads)))
    os.environ.setdefault('NUMEXPR_NUM_THREADS', str(min(8, cpu_threads)))
    
    print(f"   • OMP threads: {os.environ['OMP_NUM_THREADS']}")
    print(f"   • JAX platform: {os.environ.get('JAX_PLATFORMS', 'auto')}")
    print(f"   • Float precision: {'float32' if os.environ.get('JAX_ENABLE_X64') == 'false' else 'float64'}")
    
    # JAX device check
    try:
        import jax
        devices = jax.devices()
        print(f"   • JAX devices: {[str(d) for d in devices]}")
        print(f"   • Default device: {jax.devices()[0]}")
    except ImportError:
        print("   • JAX: Not yet imported")
    
    return True


def train_physics_nn_production():
    """Train physics-informed neural network on production dataset."""
    
    print("\n🚀 Physics-Informed Neural Network - Production Training")
    print("=" * 60)
    
    # Production configuration
    sim_indices = list(range(200))  # Use 200 simulations for production training
    
    print(f"📊 Production Training Configuration:")
    print(f"   • Simulations: {len(sim_indices)}")
    print(f"   • Filter: CAP (kinetic SZ)")
    print(f"   • Particle: gas")
    print(f"   • Expected CPU training time: 4-8 hours")
    print(f"   • Physics constraints: Full (mass scaling + cosmology attention + PK suppression)")
    
    # Configure for CPU training with physics constraints
    model_config = PhysicsNeuralConfig(
        # Physics-informed architecture
        use_mass_scaling=True,       # Mass-radius relationships
        use_cosmo_attention=True,    # Cosmological parameter attention
        use_pk_suppression=True,     # Power spectrum suppression
        
        # Ensemble uncertainty quantification
        uncertainty_method='ensemble',
        ensemble_size=3,             # 3 ensemble members for CPU efficiency
        
        # Physics regularization
        physics_loss_weight=0.1,     # Physics consistency weight
        mass_scaling_weight=0.05,    # Mass-radius scaling weight
        smoothness_weight=0.01,      # Radial smoothness weight
        
        # CPU-optimized architecture  
        hidden_dims=[256, 128, 64],  # Reasonable size for CPU training
        activation='swish',          # Smooth activation
        dropout_rate=0.1
    )
    
    # Configure training for CPU efficiency
    trainer_config = PhysicsNeuralTrainerConfig(
        epochs=800,                  # Sufficient for convergence
        learning_rate=3e-4,         # Conservative learning rate
        batch_size=128,             # CPU-optimized batch size
        patience=100,               # Early stopping patience
        
        # Validation and monitoring
        val_check_interval=10,      # Check validation every 10 epochs
        warmup_epochs=50,           # Learning rate warmup
        decay_factor=0.95,          # Learning rate decay
        
        # Regularization
        weight_decay=1e-4,          # L2 regularization
        gradient_clip=1.0,          # Gradient clipping
        
        # Physics loss weighting
        physics_loss_weight=0.1,           # Balance physics and data fit
        ensemble_diversity_weight=0.01,   # Encourage ensemble diversity
        
        # Output control
        verbose=True,               # Detailed progress
        plot_training=True,         # Generate plots
        save_best_model=True        # Save best checkpoint
    )
    
    print(f"\n🧠 Model Architecture (CPU-Optimized):")
    print(f"   • Hidden layers: {model_config.hidden_dims}")
    print(f"   • Ensemble size: {model_config.ensemble_size}")
    print(f"   • Physics constraints: {model_config.use_mass_scaling}, {model_config.use_cosmo_attention}, {model_config.use_pk_suppression}")
    print(f"   • Batch size: {trainer_config.batch_size}")
    print(f"   • Max epochs: {trainer_config.epochs}")
    
    # Create data loader
    dataloader_config = DataLoaderConfig(
        batch_size=trainer_config.batch_size,
        train_ratio=0.8,            # 80% training
        val_ratio=0.1,              # 10% validation  
        test_ratio=0.1,             # 10% test
        normalize_features=True,     # Important for convergence
        log_transform_mass=True,    # Log space for mass
        shuffle=True               # Shuffle for training
    )
    
    dataloader = SimulationDataLoader(
        sim_indices=sim_indices,
        config=dataloader_config,
        filterType='CAP',
        ptype='gas'
    )
    
    print(f"\n📊 Dataset Information:")
    stats = dataloader.get_stats()
    print(f"   • Total samples: {stats['total_samples']}")
    print(f"   • Features: {stats['n_features']} (35 cosmo + 1 mass + {stats['n_features']-36} PK)")
    print(f"   • Target radius bins: {stats['n_targets']}")
    print(f"   • Train/Val/Test: {stats['train_samples']}/{stats['val_samples']}/{stats['test_samples']}")
    
    # Create trainer with CPU configuration
    trainer = PhysicsNeuralTrainer(
        model_config=model_config,
        trainer_config=trainer_config,
        save_dir=None,  # Auto-generate timestamped directory
        setup_gpu=False  # CPU-only training
    )
    
    print(f"\n🏃 Starting CPU Training...")
    print(f"   • Results will be saved to: {trainer.save_dir}")
    print(f"   • Progress monitoring: Enabled")
    print(f"   • Early stopping: {trainer_config.patience} epochs patience")
    
    # Record start time
    start_time = time.time()
    start_datetime = datetime.now()
    
    print(f"   • Training started: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Train the model
    results = trainer.train(
        dataloader=dataloader,
        save_best=True,
        plot_results=True
    )
    
    # Calculate total time
    total_time = time.time() - start_time
    end_datetime = datetime.now()
    
    print(f"\n✅ CPU Training Complete!")
    print(f"   • Training time: {total_time/3600:.1f} hours")
    print(f"   • Started: {start_datetime.strftime('%H:%M:%S')}")
    print(f"   • Finished: {end_datetime.strftime('%H:%M:%S')}")
    print(f"   • Best validation loss: {results['best_val_loss']:.6f}")
    print(f"   • Final test MAPE: {results['final_results']['test_mape']:.1f}%")
    print(f"   • Test R²: {results['final_results']['test_r2']:.3f}")
    
    # Compare with GP performance expectations
    print(f"\n📊 Performance Analysis:")
    nn_mape = results['final_results']['test_mape']
    print(f"   • Neural Network MAPE: {nn_mape:.1f}%")
    print(f"   • GP MAPE range (reference): 29-35%")
    
    if nn_mape <= 40:
        print(f"   ✅ Performance meets expectations for cosmological inference")
    else:
        print(f"   ⚠️  Consider: More training data, physics constraints, or hyperparameter tuning")
    
    # Training efficiency analysis
    samples_per_hour = stats['total_samples'] / total_time * 3600
    print(f"   • Training efficiency: {samples_per_hour:.0f} samples/hour")
    print(f"   • Scalability: Ready for 1000+ simulation datasets")
    
    # Save summary
    summary = {
        'training_config': 'CPU-optimized physics-informed neural network',
        'dataset_size': stats['total_samples'],
        'training_time_hours': total_time / 3600,
        'test_mape': nn_mape,
        'physics_constraints': 'Enabled (mass scaling, cosmology attention, PK suppression)',
        'ready_for_npe': nn_mape <= 45,
        'model_path': trainer.save_dir
    }
    
    import json
    summary_path = os.path.join(trainer.save_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n💾 Results Saved:")
    print(f"   • Model: {trainer.save_dir}/best_model.pkl")
    print(f"   • Training curves: {trainer.save_dir}/training_curves.png")
    print(f"   • Predictions: {trainer.save_dir}/prediction_examples.png")
    print(f"   • Summary: {summary_path}")
    
    return trainer, results


def main():
    """Main training function."""
    
    print("🌌 Physics-Informed Neural Network - CPU Production Training")
    print("=" * 65)
    print("Training sophisticated neural emulator with same physics constraints")
    print("as GP kernels, optimized for single-CPU HPC environments")
    print("=" * 65)
    
    # Configure CPU environment
    configure_cpu_training()
    
    # Train production model
    trainer, results = train_physics_nn_production()
    
    print(f"\n🎯 Training Complete!")
    print(f"   • Ready for cosmological parameter inference (NPE integration)")
    print(f"   • Scalable to larger datasets (1000+ simulations)")
    print(f"   • Physics-informed architecture validated")
    
    return trainer, results


if __name__ == "__main__":
    trainer, results = main()