# Physics Neural Network Training: Troubleshooting & Advanced Techniques

**Created**: September 10, 2025  
**Context**: Physics-informed neural network training for cosmological profile prediction  
**Issue**: Document complications and advanced techniques for future implementation

## 🚨 **Core Problem Solved**

### Initial Issue
- **JAX Tuple Unpacking Error**: Model outputs inconsistent between single/ensemble modes
- **Gradient Explosion**: Training loss hitting 10^10+ causing numerical instability  
- **Infinite Validation Loss**: Complete failure to generalize to validation set
- **Early Stopping**: Training terminated at epoch 200 due to no improvement

### Root Cause Analysis
1. **JAX Version Compatibility**: JAX v0.6.0+ deprecated `jax.tree_map` → `jax.tree.map`
2. **Model Output Inconsistency**: Both single and ensemble models return tuples, not scalars
3. **Numerical Instability**: Large gradients and predictions causing overflow
4. **Architecture Complexity**: Physics-informed constraints making training unstable

## ✅ **Solutions Implemented**

### 1. **JAX Compatibility Fixes**
```python
# OLD (deprecated)
clipped_grads = jax.tree_map(lambda g: ..., grads)

# NEW (JAX v0.6.0+)  
clipped_grads = jax.tree.map(lambda g: ..., grads)
```

### 2. **Model Output Consistency**
```python
# Fixed trainer to always expect tuples
# Both ensemble and single models return tuples (mean, var)
pred_mean, pred_var = model_output
```

### 3. **Gradient Clipping & Numerical Stability**
```python
# Very aggressive gradient clipping
grad_norm = optax.global_norm(grads)
clipped_grads = jax.tree.map(
    lambda g: jnp.where(grad_norm > 0.1, g * (0.1 / grad_norm), g), grads
)

# Loss clipping for numerical stability
diff = jnp.clip(pred_mean - batch_y, -100.0, 100.0)
prediction_loss = jnp.mean(diff**2)
total_loss = jnp.clip(total_loss, 0.0, 1e10)
```

### 4. **Architectural Simplification**
```python
class SimpleMLP(nn.Module):
    """Ultra-simple feedforward network without physics constraints"""
    
    @nn.compact  # Required for Dropout in Flax
    def __call__(self, x, training: bool = False):
        # 2-layer network: [64, 32] hidden dims
        # Xavier initialization, conservative dropout
        # No physics constraints, no complex attention mechanisms
```

### 5. **Hyperparameter Tuning for Stability**
```python
# Ultra-conservative settings
learning_rate=1e-5       # Very low learning rate
weight_decay=1e-3        # Strong regularization  
gradient_clip=0.1        # Very conservative clipping
hidden_dims=[64, 32]     # Simple 2-layer architecture
```

## 📊 **Results After All Fixes**

| Metric | Before Fixes | After All Fixes | Improvement |
|--------|--------------|-----------------|-------------|
| **Training Loss** | 10^10 (clipped) | 10^4 | **6 orders of magnitude** ✅ |
| **JAX Crashes** | Frequent | None | **Completely stable** ✅ |
| **Training Speed** | 175x vs GP | 359x vs GP | **2x faster** ✅ |
| **Validation Loss** | inf | inf | **Still problematic** ❌ |

## 🔬 **Advanced Techniques for Future Implementation**

### A. **Physics-Informed Constraints** (Removed for Stability)
```python
class PhysicsInformedCore(nn.Module):
    """Advanced physics constraints - use after baseline works"""
    
    def setup(self):
        # Mass scaling with concentration relations
        self.mass_scaling = MassScalingLayer()
        
        # Cosmological parameter attention
        self.cosmo_attention = CosmologyAttention(
            n_cosmo_params=35, attention_dim=32
        )
        
        # Power spectrum suppression physics
        self.pk_processor = PowerSpectrumProcessor(
            pk_dim=79, output_dim=32
        )

    def __call__(self, x, training=False):
        # Physics-informed feature engineering
        cosmo_params = x[:, :35]
        mass = x[:, 35:36] 
        pk_ratios = x[:, 36:]
        
        # Apply physics-based transformations
        cosmo_features = self.cosmo_attention(cosmo_params)
        mass_features = self.mass_scaling(mass)  
        pk_features = self.pk_processor(pk_ratios)
        
        return jnp.concatenate([cosmo_features, mass_features, pk_features], axis=-1)
```

### B. **Advanced Regularization Techniques**
```python
def compute_physics_loss(predictions, mass_inputs):
    """Multi-component physics loss for advanced training"""
    losses = {}
    
    # 1. Monotonic decrease constraint
    radial_gradients = jnp.diff(predictions, axis=-1) 
    monotonic_loss = jnp.mean(jnp.maximum(radial_gradients, 0)**2)
    
    # 2. Mass scaling consistency  
    mass_scaling_loss = check_mass_scaling_relations(predictions, mass_inputs)
    
    # 3. Smoothness regularization
    second_derivatives = jnp.diff(radial_gradients, axis=-1)
    smoothness_loss = jnp.mean(second_derivatives**2)
    
    # 4. Physical boundary conditions
    boundary_loss = enforce_boundary_conditions(predictions)
    
    return {
        'monotonic': monotonic_loss,
        'mass_scaling': mass_scaling_loss, 
        'smoothness': smoothness_loss,
        'boundary': boundary_loss
    }
```

### C. **Ensemble Uncertainty Quantification** 
```python
class PhysicsInformedEnsemble(nn.Module):
    """Deep ensemble for uncertainty estimation"""
    
    def setup(self):
        self.ensemble_members = [
            SimpleMLP(self.config) for _ in range(self.config.ensemble_size)
        ]
        
    def __call__(self, x, training=False):
        # Get predictions from all members
        predictions = []
        for member in self.ensemble_members:
            pred = member(x, training=training)
            predictions.append(pred)
            
        predictions = jnp.stack(predictions, axis=0)
        
        # Compute ensemble statistics  
        ensemble_mean = jnp.mean(predictions, axis=0)
        ensemble_var = jnp.var(predictions, axis=0)  # Epistemic uncertainty
        
        return ensemble_mean, ensemble_var
```

### D. **Advanced Training Strategies**
```python
# 1. Curriculum Learning
def curriculum_schedule(epoch):
    """Gradually introduce complexity during training"""
    if epoch < 100:
        return {'use_physics_loss': False, 'dropout_rate': 0.0}
    elif epoch < 300: 
        return {'use_physics_loss': True, 'physics_weight': 0.01}
    else:
        return {'use_physics_loss': True, 'physics_weight': 0.1}

# 2. Learning Rate Scheduling with Warmup
warmup_schedule = optax.linear_schedule(
    init_value=0.0,
    end_value=1e-4, 
    transition_steps=100
)

decay_schedule = optax.exponential_decay(
    init_value=1e-4,
    decay_rate=0.95,
    transition_steps=100
)

# 3. Advanced Optimizers
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),      # Gradient clipping
    optax.adamw(                         # AdamW with weight decay
        learning_rate=schedule,
        weight_decay=1e-4,
        b1=0.9, b2=0.999
    )
)
```

### E. **Data Preprocessing & Augmentation**
```python
# 1. Advanced feature normalization
def robust_normalize(X):
    """Robust normalization using median/MAD instead of mean/std"""
    median = jnp.median(X, axis=0)
    mad = jnp.median(jnp.abs(X - median), axis=0)
    return (X - median) / (1.4826 * mad + 1e-8)

# 2. Feature engineering
def engineer_physics_features(cosmo_params, masses, pk_ratios):
    """Create physics-motivated feature combinations"""
    
    # Cosmological ratios
    omega_ratio = cosmo_params[:, 0] / cosmo_params[:, 1]  # Ωm/Ωb
    
    # Mass-dependent features
    log_mass = jnp.log10(masses)
    mass_deviation = log_mass - jnp.mean(log_mass)
    
    # PK suppression features
    pk_mean = jnp.mean(pk_ratios, axis=-1, keepdims=True)
    pk_std = jnp.std(pk_ratios, axis=-1, keepdims=True)
    
    return jnp.concatenate([
        cosmo_params, omega_ratio, mass_deviation, 
        pk_mean, pk_std, pk_ratios
    ], axis=-1)

# 3. Data augmentation for robustness  
def augment_training_data(X, y, noise_level=0.01):
    """Add small amount of noise for regularization"""
    key = jax.random.PRNGKey(42)
    noise = jax.random.normal(key, X.shape) * noise_level
    return X + noise, y
```

## 🎯 **Debugging Workflow for Future Issues**

### 1. **Numerical Stability Checklist**
- [ ] Check for `inf` or `NaN` values in loss computation
- [ ] Verify gradient norms (should be < 1.0 after clipping)
- [ ] Monitor parameter magnitudes during training
- [ ] Ensure all activations are bounded (use `jnp.clip` if needed)

### 2. **Architecture Debugging**
- [ ] Start with simplest possible model (1-2 layers)
- [ ] Verify model can overfit to small dataset (5-10 samples)  
- [ ] Test with synthetic data where ground truth is known
- [ ] Gradually add complexity only after baseline works

### 3. **Training Diagnosis**
- [ ] Plot training/validation curves in real-time
- [ ] Monitor individual loss components (prediction, physics, regularization)
- [ ] Check prediction distributions (should not be extreme)
- [ ] Validate data loading and preprocessing pipeline

## 🚀 **Implementation Priority for Future Work**

### Phase 1: Fix Validation Loss (Immediate)
1. **Data Quality**: Ensure validation set is representative
2. **Feature Engineering**: Better input preprocessing  
3. **Loss Function**: Try robust losses (Huber, quantile)
4. **Architecture**: Even simpler models (1 layer, 16 hidden units)

### Phase 2: Add Basic Physics (After Validation Works)  
1. **Mass Scaling**: Simple log-linear relationships
2. **Smoothness**: Basic L2 regularization on derivatives
3. **Monotonicity**: Soft constraints on profile shape

### Phase 3: Advanced Techniques (After Basic Physics Works)
1. **Attention Mechanisms**: Cosmological parameter attention
2. **Ensemble Methods**: Deep ensembles for uncertainty
3. **Curriculum Learning**: Gradual complexity introduction
4. **Advanced Optimizers**: AdamW with sophisticated scheduling

## 📝 **Key Lessons Learned**

1. **Start Simple**: Physics constraints can destabilize training - establish baseline first
2. **JAX Compatibility**: Keep track of version-specific syntax changes  
3. **Numerical Stability**: Aggressive clipping is better than instability
4. **Model Output Consistency**: Ensure consistent tuple returns across architectures
5. **Validation is Critical**: Training metrics can be misleading without proper validation

## 🔗 **Related Files Modified**
- `src/models/physics_neural_emulator.py`: Model architectures
- `src/models/physics_neural_trainer.py`: Training pipeline  
- `examples/physics_neural_demo.py`: Demo configuration
- All SLURM output files documenting the debugging journey

---
**Next Steps**: Use this as reference when adding back physics constraints after achieving stable validation loss convergence.