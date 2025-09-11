"""
Physics-Informed Neural Network Trainer

This module provides a comprehensive training pipeline for the physics-informed
neural network emulator, designed to be compatible with the existing data pipeline
while incorporating the same domain knowledge as successful GP kernels.

Key Features:
- Compatible with SimulationDataLoader pipeline
- Physics-informed loss functions and regularization  
- Uncertainty quantification via deep ensembles
- Training monitoring and early stopping
- Model comparison with GP baselines
- Interpretable training metrics
"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
import pickle
import json
import os
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.models.physics_neural_emulator import (
    PhysicsNeuralConfig, PhysicsInformedEnsemble, PhysicsInformedMLP,
    create_physics_informed_emulator, compute_physics_regularization
)
from src.data.sim_dataloader import SimulationDataLoader, DataLoaderConfig
from src.config.config import TRAINED_MODELS_DIR


def setup_gpu_training():
    """
    Configure JAX for optimal GPU training.
    
    Returns:
        dict: GPU configuration info
    """
    # Set JAX to use GPU if available
    os.environ.setdefault('JAX_PLATFORMS', 'gpu,cpu')  # Prefer GPU, fallback to CPU
    
    # Check available devices
    devices = jax.devices()
    gpu_available = any('gpu' in str(device).lower() for device in devices)
    
    # Configure memory allocation
    if gpu_available:
        # Prevent JAX from pre-allocating entire GPU memory
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
        os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.8')  # Use 80% of GPU memory
        
    config_info = {
        'devices': [str(device) for device in devices],
        'gpu_available': gpu_available,
        'default_device': str(jax.devices()[0]),
        'device_count': len(devices)
    }
    
    print(f"🖥️  JAX Device Configuration:")
    print(f"   • Available devices: {config_info['device_count']}")
    print(f"   • Default device: {config_info['default_device']}")
    print(f"   • GPU available: {config_info['gpu_available']}")
    
    if gpu_available:
        print(f"   ⚡ GPU training enabled - expect 2-5x speedup!")
    else:
        print(f"   💻 CPU training (consider GPU for faster training)")
        
    return config_info


@dataclass
class PhysicsNeuralTrainerConfig:
    """Configuration for physics-informed neural network trainer."""
    
    # Training parameters
    epochs: int = 1000
    learning_rate: float = 3e-4
    batch_size: int = 256
    patience: int = 100
    min_lr: float = 1e-6
    
    # Scheduler parameters
    warmup_epochs: int = 50
    decay_factor: float = 0.9
    decay_patience: int = 20
    
    # Regularization
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    
    # Physics loss weights
    physics_loss_weight: float = 0.1
    ensemble_diversity_weight: float = 0.01
    
    # Validation and monitoring
    val_check_interval: int = 5    # Check validation every N epochs
    save_best_model: bool = True
    plot_training: bool = True
    verbose: bool = True


class PhysicsNeuralTrainer:
    """
    Comprehensive trainer for physics-informed neural network emulators.
    
    This trainer is designed to work seamlessly with the existing data pipeline
    while incorporating advanced physics constraints and uncertainty quantification.
    """
    
    def __init__(
        self,
        model_config: PhysicsNeuralConfig = None,
        trainer_config: PhysicsNeuralTrainerConfig = None,
        save_dir: Optional[str] = None,
        setup_gpu: bool = True
    ):
        """
        Initialize physics-informed neural trainer.
        
        Args:
            model_config: Configuration for the neural network model
            trainer_config: Configuration for the training process
            save_dir: Directory to save trained models and results
            setup_gpu: Whether to configure JAX for GPU training automatically
        """
        # Configure GPU training if requested
        if setup_gpu:
            self.gpu_config = setup_gpu_training()
        else:
            self.gpu_config = None
            
        self.model_config = model_config or PhysicsNeuralConfig()
        self.trainer_config = trainer_config or PhysicsNeuralTrainerConfig()
        
        # Create save directory
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(TRAINED_MODELS_DIR, f'physics_nn_{timestamp}')
        
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize model and training state
        self.model = None
        self.params = None
        self.opt_state = None
        self.training_history = {
            'train_loss': [], 'val_loss': [], 'physics_loss': [],
            'learning_rate': [], 'epoch': []
        }
        self.best_val_loss = float('inf')
        self.best_params = None
        
        # Performance tracking
        self.training_time = 0
        self.is_trained = False
        
    def setup_model_and_optimizer(self, sample_input: jnp.ndarray):
        """Initialize model and optimizer with sample input."""
        
        # Create physics-informed model
        self.model = create_physics_informed_emulator(self.model_config)
        
        # Initialize model parameters
        key = jax.random.PRNGKey(42)
        
        if self.model_config.uncertainty_method == 'ensemble':
            # Initialize ensemble
            self.params = self.model.init(key, sample_input, training=False)
        else:
            # Initialize single model
            self.params = self.model.init(key, sample_input, training=False)
        
        # Setup learning rate scheduler
        warmup_schedule = optax.linear_schedule(
            init_value=0.0,
            end_value=self.trainer_config.learning_rate,
            transition_steps=self.trainer_config.warmup_epochs * 10  # Rough batch estimate
        )
        
        decay_schedule = optax.exponential_decay(
            init_value=self.trainer_config.learning_rate,
            decay_rate=self.trainer_config.decay_factor,
            transition_steps=self.trainer_config.decay_patience * 10
        )
        
        schedule = optax.join_schedules([warmup_schedule, decay_schedule], 
                                       [self.trainer_config.warmup_epochs * 10])
        
        # Setup optimizer with gradient clipping and weight decay
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.trainer_config.gradient_clip),
            optax.adamw(
                learning_rate=schedule,
                weight_decay=self.trainer_config.weight_decay
            )
        )
        
        self.opt_state = self.optimizer.init(self.params)
        
        if self.trainer_config.verbose:
            print(f"🚀 Model initialized:")
            print(f"  - Architecture: Physics-informed {'ensemble' if self.model_config.uncertainty_method == 'ensemble' else 'single'}")
            print(f"  - Physics constraints: {self.model_config.use_mass_scaling}, {self.model_config.use_cosmo_attention}, {self.model_config.use_pk_suppression}")
            print(f"  - Uncertainty method: {self.model_config.uncertainty_method}")
            if self.model_config.uncertainty_method == 'ensemble':
                print(f"  - Ensemble size: {self.model_config.ensemble_size}")
    
    @jax.jit
    def train_step(self, params, opt_state, batch_x, batch_y, key):
        """Single training step with physics-informed loss."""
        
        def loss_fn(params):
            # Forward pass
            if self.model_config.uncertainty_method == 'ensemble':
                pred_mean, pred_var = self.model.apply(params, batch_x, training=True, rngs={'dropout': key})
                
                # Ensemble loss (negative log likelihood + ensemble diversity)
                mse_loss = jnp.mean((pred_mean - batch_y)**2)
                
                # Uncertainty loss (encourage reasonable variance estimates)
                uncertainty_loss = jnp.mean(jnp.maximum(pred_var - 10.0, 0)) + jnp.mean(jnp.maximum(0.01 - pred_var, 0))
                
                prediction_loss = mse_loss + 0.1 * uncertainty_loss
                
                # Get individual member predictions for diversity loss
                individual_preds = self.model.apply(params, batch_x, training=True, return_individual=True, rngs={'dropout': key})
                diversity_loss = -jnp.mean(jnp.var(individual_preds, axis=0))  # Encourage diversity
                
            else:
                # Single model with uncertainty estimation
                if hasattr(self.model, 'apply') and 'logvar' in str(self.model):
                    pred_mean, logvar = self.model.apply(params, batch_x, training=True, rngs={'dropout': key})
                    pred_var = jnp.exp(logvar)
                    
                    # Negative log likelihood with uncertainty
                    prediction_loss = 0.5 * jnp.mean((batch_y - pred_mean)**2 / pred_var + logvar)
                else:
                    pred_mean = self.model.apply(params, batch_x, training=True, rngs={'dropout': key})
                    prediction_loss = jnp.mean((pred_mean - batch_y)**2)
                
                diversity_loss = 0.0
            
            # Physics-informed regularization losses
            physics_losses = compute_physics_regularization(pred_mean, batch_x, self.model_config)
            total_physics_loss = sum(physics_losses.values())
            
            # Total loss
            total_loss = (prediction_loss + 
                         self.trainer_config.physics_loss_weight * total_physics_loss +
                         self.trainer_config.ensemble_diversity_weight * diversity_loss)
            
            return total_loss, {
                'prediction_loss': prediction_loss,
                'physics_loss': total_physics_loss,
                'diversity_loss': diversity_loss,
                'total_loss': total_loss
            }
        
        # Compute loss and gradients
        (loss_val, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        
        # Update parameters
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss_val, loss_dict
    
    @jax.jit
    def eval_step(self, params, batch_x, batch_y):
        """Evaluation step without training-specific operations."""
        
        if self.model_config.uncertainty_method == 'ensemble':
            pred_mean, pred_var = self.model.apply(params, batch_x, training=False)
            eval_loss = jnp.mean((pred_mean - batch_y)**2)
        else:
            pred_mean = self.model.apply(params, batch_x, training=False)
            eval_loss = jnp.mean((pred_mean - batch_y)**2)
            pred_var = None
        
        return eval_loss, pred_mean, pred_var
    
    def train(
        self,
        dataloader: SimulationDataLoader,
        save_best: bool = True,
        plot_results: bool = True
    ) -> Dict[str, Any]:
        """
        Train the physics-informed neural network.
        
        Args:
            dataloader: Configured SimulationDataLoader with train/val/test splits
            save_best: Whether to save the best model during training
            plot_results: Whether to plot training curves and results
            
        Returns:
            Dictionary containing training results and metrics
        """
        
        if self.trainer_config.verbose:
            print(f"🔬 Training Physics-Informed Neural Network")
            print(f"  - Dataset: {dataloader.get_stats()['total_samples']} samples")
            print(f"  - Features: {dataloader.get_stats()['n_features']}")
            print(f"  - Targets: {dataloader.get_stats()['n_targets']} radius bins")
            print(f"  - Physics constraints: Mass scaling, Cosmology attention, PK suppression")
            print(f"  - Training config: {self.trainer_config.epochs} epochs, {self.trainer_config.patience} patience")
        
        # Get data splits
        X_train, y_train = dataloader.get_split_data('train')
        X_val, y_val = dataloader.get_split_data('val')
        
        if X_val is None:
            # Use a portion of training data for validation if no validation set
            val_split = int(0.15 * len(X_train))
            X_val, y_val = X_train[-val_split:], y_train[-val_split:]
            X_train, y_train = X_train[:-val_split], y_train[:-val_split]
            print("⚠️  No validation set provided, using 15% of training data")
        
        # Initialize model and optimizer
        self.setup_model_and_optimizer(X_train[:1])
        
        # Training loop
        start_time = time.time()
        patience_counter = 0
        
        n_train_batches = len(X_train) // self.trainer_config.batch_size
        n_val_batches = len(X_val) // self.trainer_config.batch_size
        
        key = jax.random.PRNGKey(42)
        
        for epoch in tqdm(range(self.trainer_config.epochs), desc="Training"):
            
            # Training epoch
            epoch_train_losses = []
            epoch_physics_losses = []
            
            # Shuffle training data
            perm = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]
            
            for batch_idx in range(n_train_batches):
                batch_start = batch_idx * self.trainer_config.batch_size
                batch_end = batch_start + self.trainer_config.batch_size
                
                batch_x = X_train_shuffled[batch_start:batch_end]
                batch_y = y_train_shuffled[batch_start:batch_end]
                
                # Training step
                key, subkey = jax.random.split(key)
                self.params, self.opt_state, train_loss, loss_dict = self.train_step(
                    self.params, self.opt_state, batch_x, batch_y, subkey
                )
                
                epoch_train_losses.append(float(train_loss))
                epoch_physics_losses.append(float(loss_dict['physics_loss']))
            
            avg_train_loss = np.mean(epoch_train_losses)
            avg_physics_loss = np.mean(epoch_physics_losses)
            
            # Validation evaluation
            val_losses = []
            if epoch % self.trainer_config.val_check_interval == 0:
                for batch_idx in range(n_val_batches):
                    batch_start = batch_idx * self.trainer_config.batch_size
                    batch_end = batch_start + self.trainer_config.batch_size
                    
                    batch_x = X_val[batch_start:batch_end]
                    batch_y = y_val[batch_start:batch_end]
                    
                    val_loss, _, _ = self.eval_step(self.params, batch_x, batch_y)
                    val_losses.append(float(val_loss))
                
                avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
            else:
                avg_val_loss = self.training_history['val_loss'][-1] if self.training_history['val_loss'] else float('inf')
            
            # Learning rate tracking
            current_lr = self.trainer_config.learning_rate  # Could be made more sophisticated
            
            # Store training history
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)
            self.training_history['physics_loss'].append(avg_physics_loss)
            self.training_history['learning_rate'].append(current_lr)
            self.training_history['epoch'].append(epoch + 1)
            
            # Early stopping logic
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.best_params = self.params
                patience_counter = 0
                
                if save_best:
                    self._save_checkpoint('best_model')
            else:
                patience_counter += 1
                
            if patience_counter >= self.trainer_config.patience:
                if self.trainer_config.verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Progress reporting
            if self.trainer_config.verbose and (epoch + 1) % 50 == 0:
                elapsed_time = time.time() - start_time
                print(f"Epoch {epoch + 1:4d}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}, "
                      f"Physics={avg_physics_loss:.6f}, Time={elapsed_time/3600:.1f}h")
        
        # Training complete
        total_time = time.time() - start_time
        self.training_time = total_time
        self.is_trained = True
        
        # Load best parameters
        if self.best_params is not None:
            self.params = self.best_params
        
        # Final evaluation
        final_results = self._evaluate_model(dataloader)
        
        if self.trainer_config.verbose:
            print(f"\n✅ Training completed in {total_time/3600:.1f} hours")
            print(f"   Best validation loss: {self.best_val_loss:.6f}")
            print(f"   Final test MAPE: {final_results['test_mape']:.1f}%")
        
        # Save final results
        training_results = {
            'training_time_hours': total_time / 3600,
            'best_val_loss': self.best_val_loss,
            'epochs_trained': len(self.training_history['train_loss']),
            'final_results': final_results,
            'model_config': self.model_config,
            'trainer_config': self.trainer_config,
            'training_history': self.training_history
        }
        
        self._save_training_results(training_results)
        
        if plot_results:
            self._plot_training_results()
            self._plot_predictions(dataloader)
        
        return training_results
    
    def _evaluate_model(self, dataloader: SimulationDataLoader) -> Dict[str, float]:
        """Comprehensive model evaluation on test set."""
        
        X_test, y_test = dataloader.get_split_data('test')
        
        if self.model_config.uncertainty_method == 'ensemble':
            pred_mean, pred_var = self.model.apply(self.params, X_test, training=False)
        else:
            pred_mean = self.model.apply(self.params, X_test, training=False)
            pred_var = None
        
        # Compute evaluation metrics
        mse = float(jnp.mean((pred_mean - y_test)**2))
        mae = float(jnp.mean(jnp.abs(pred_mean - y_test)))
        
        # MAPE (mean absolute percentage error) - comparable to GP metrics
        mape = float(jnp.mean(jnp.abs((pred_mean - y_test) / (y_test + 1e-8))) * 100)
        
        # R² score
        y_mean = jnp.mean(y_test)
        ss_tot = jnp.sum((y_test - y_mean)**2)
        ss_res = jnp.sum((y_test - pred_mean)**2)
        r2_score = float(1 - ss_res / ss_tot)
        
        results = {
            'test_mse': mse,
            'test_mae': mae,
            'test_mape': mape,
            'test_r2': r2_score
        }
        
        if pred_var is not None:
            # Uncertainty calibration metrics
            uncertainty_quality = float(jnp.mean(pred_var))
            results['mean_uncertainty'] = uncertainty_quality
        
        return results
    
    def predict(self, X_new: jnp.ndarray) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Make predictions on new data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.model_config.uncertainty_method == 'ensemble':
            pred_mean, pred_var = self.model.apply(self.params, X_new, training=False)
            return pred_mean, pred_var
        else:
            pred_mean = self.model.apply(self.params, X_new, training=False)
            return pred_mean, None
    
    def _save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.save_dir, f'{checkpoint_name}.pkl')
        with open(checkpoint_path, 'wb') as f:
            pickle.dump({
                'params': self.params,
                'training_history': self.training_history,
                'model_config': self.model_config,
                'trainer_config': self.trainer_config
            }, f)
    
    def _save_training_results(self, results: Dict[str, Any]):
        """Save comprehensive training results."""
        results_path = os.path.join(self.save_dir, 'training_results.json')
        
        # Convert JAX arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, jnp.ndarray):
                json_results[key] = value.tolist()
            elif hasattr(value, '__dict__'):  # Handle dataclass objects
                json_results[key] = value.__dict__
            else:
                json_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
    
    def _plot_training_results(self):
        """Plot training curves and diagnostics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = self.training_history['epoch']
        
        # Training and validation loss
        axes[0, 0].plot(epochs, self.training_history['train_loss'], 'b-', label='Train Loss', alpha=0.7)
        axes[0, 0].plot(epochs, self.training_history['val_loss'], 'r-', label='Val Loss', alpha=0.7)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # Physics loss
        axes[0, 1].plot(epochs, self.training_history['physics_loss'], 'g-', label='Physics Loss', alpha=0.7)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Physics Loss')
        axes[0, 1].set_title('Physics-Informed Regularization')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 0].plot(epochs, self.training_history['learning_rate'], 'orange', alpha=0.7)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # Loss improvement
        val_loss = np.array(self.training_history['val_loss'])
        val_improvement = np.maximum.accumulate(val_loss[0] - val_loss)
        axes[1, 1].plot(epochs, val_improvement, 'purple', alpha=0.7)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Validation Loss Improvement')
        axes[1, 1].set_title('Cumulative Validation Improvement')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
        plt.show()
    
    def _plot_predictions(self, dataloader: SimulationDataLoader, n_examples: int = 3):
        """Plot prediction examples compared to ground truth."""
        X_test, y_test = dataloader.get_split_data('test')
        
        # Get predictions
        pred_mean, pred_var = self.predict(X_test[:n_examples])
        
        fig, axes = plt.subplots(1, n_examples, figsize=(5*n_examples, 4))
        if n_examples == 1:
            axes = [axes]
        
        r_bins = dataloader.r_bins
        
        for i in range(n_examples):
            # Plot ground truth
            axes[i].plot(r_bins, y_test[i], 'k-', linewidth=2, label='Ground Truth', alpha=0.8)
            
            # Plot prediction
            axes[i].plot(r_bins, pred_mean[i], 'r--', linewidth=2, label='NN Prediction', alpha=0.8)
            
            # Plot uncertainty if available
            if pred_var is not None:
                uncertainty = np.sqrt(pred_var[i])
                axes[i].fill_between(r_bins, pred_mean[i] - uncertainty, pred_mean[i] + uncertainty,
                                   alpha=0.3, color='red', label='Uncertainty')
            
            # Calculate error
            mape = np.mean(np.abs((pred_mean[i] - y_test[i]) / (y_test[i] + 1e-8))) * 100
            
            axes[i].set_xlabel('Radius [Mpc/h]')
            axes[i].set_ylabel(f'{dataloader.ptype} Profile')
            axes[i].set_title(f'Example {i+1} (MAPE: {mape:.1f}%)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'prediction_examples.png'), dpi=150, bbox_inches='tight')
        plt.show()


# Convenience function for quick training
def train_physics_neural_emulator(
    sim_indices: List[int],
    filterType: str = 'CAP',
    ptype: str = 'gas',
    epochs: int = 1000,
    batch_size: int = 256,
    save_dir: Optional[str] = None,
    use_gpu: bool = True
) -> PhysicsNeuralTrainer:
    """
    Quick training function for physics-informed neural emulator.
    
    Args:
        sim_indices: List of simulation indices to use
        filterType: Filter type for profiles
        ptype: Particle type
        epochs: Number of training epochs
        batch_size: Training batch size
        save_dir: Directory to save results
        use_gpu: Whether to configure and use GPU training (if available)
        
    Returns:
        Trained PhysicsNeuralTrainer instance
    """
    
    # Create data loader
    dataloader_config = DataLoaderConfig(
        batch_size=batch_size,
        train_ratio=0.8,
        val_ratio=0.1, 
        test_ratio=0.1,
        normalize_features=True,
        log_transform_mass=True
    )
    
    dataloader = SimulationDataLoader(
        sim_indices=sim_indices,
        config=dataloader_config,
        filterType=filterType,
        ptype=ptype
    )
    
    # Configure model and trainer
    model_config = PhysicsNeuralConfig(
        ensemble_size=3,  # Smaller ensemble for faster training
        uncertainty_method='ensemble',
        use_mass_scaling=True,
        use_cosmo_attention=True,
        use_pk_suppression=True
    )
    
    trainer_config = PhysicsNeuralTrainerConfig(
        epochs=epochs,
        batch_size=batch_size,
        patience=100,
        physics_loss_weight=0.1
    )
    
    # Create and train
    trainer = PhysicsNeuralTrainer(
        model_config=model_config,
        trainer_config=trainer_config,
        save_dir=save_dir,
        setup_gpu=use_gpu
    )
    
    # Train model
    results = trainer.train(dataloader, save_best=True, plot_results=True)
    
    print(f"\n🎯 Physics-Informed Neural Network Training Complete!")
    print(f"   Training time: {results['training_time_hours']:.1f} hours")
    print(f"   Test MAPE: {results['final_results']['test_mape']:.1f}%")
    print(f"   Saved to: {trainer.save_dir}")
    
    return trainer