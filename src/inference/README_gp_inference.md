# Real GP Parameter Inference

This directory contains **2 consolidated scripts** for real GP-based parameter inference, replacing all previous diagnostic and testing files.

## Key Features

✅ **Real GP Models Only** - No mock or simulated functions  
✅ **Proper TinyGP Conditioning** - Correct use of trained GP models  
✅ **Comprehensive Diagnostics** - ESS, R-hat, coverage statistics  
✅ **Production Ready** - Command-line interface, automatic saving, error handling  

## Files

### 1. `real_gp_hmc_inference.py`
**Single simulation parameter inference using HMC sampling**

**Usage:**
```bash
python real_gp_hmc_inference.py --sim_id 777 --gp_dir /path/to/trained/gp/models
```

**Features:**
- Loads actual trained GP models from directory
- Performs Bayesian parameter inference using HMC
- Creates corner plots with parameter recovery statistics
- Computes MCMC diagnostics (ESS, R-hat, Geweke tests)
- Saves results with comprehensive summary reports

**Key Parameters:**
- `--sim_id`: Target simulation ID for inference
- `--gp_dir`: Directory containing trained GP models
- `--n_samples`: Number of MCMC samples (default: 5000)
- `--n_params`: Number of parameters to vary (default: 8)

### 2. `real_gp_coverage_test.py` 
**Coverage testing across multiple simulations**

**Usage:**
```bash
python real_gp_coverage_test.py --n_sims 100 --gp_dirs /path/to/gp1 /path/to/gp2
```

**Features:**
- Tests multiple GP models simultaneously
- Runs inference on many simulations (50-100+)
- Creates proper C(α) vs α coverage plots 
- Computes calibration statistics
- Supports parallel processing for speed

**Key Parameters:**
- `--n_sims`: Number of test simulations (default: 50)
- `--gp_dirs`: List of GP model directories to test
- `--start_sim`: Starting simulation ID to avoid training data overlap
- `--parallel`: Enable parallel processing

## Technical Implementation

### GP Model Loading
Both scripts properly load:
- Trained TinyGP models (`trained_models.pkl`)
- Training parameters (`trained_params.pkl`) 
- Training metadata (`training_info.json`)

### GP Conditioning
Uses correct TinyGP API:
```python
# Load training data matching original dimensions (716 samples)
X_train, y_train = load_training_data(n_samples=716)

# Condition GP for predictions
for i, gp_model in enumerate(gp_models):
    y_train_i = jnp.array(y_train[:, i])  # Training targets for radius bin i
    _, cond_gp = gp_model.condition(y_train_i, test_input_jnp)
    pred_mean = float(cond_gp.mean[0])
    pred_var = float(cond_gp.variance[0])
```

### Likelihood Function
Real GP-based likelihood evaluation:
```python
def gp_log_likelihood(test_params):
    # Get GP predictions for all radius bins
    pred_profile, pred_vars = gp_predict(test_params)
    
    # Gaussian likelihood with GP uncertainty
    obs_noise = 0.02 * np.abs(obs_profile)
    total_var = pred_vars + obs_noise**2
    
    residuals = obs_profile - pred_profile
    log_like = -0.5 * np.sum(residuals**2 / total_var + np.log(2π * total_var))
    return log_like
```

## Expected Results

### Well-Calibrated Model
- Coverage plot should follow diagonal (perfect calibration)
- Mean absolute deviation < 0.05
- ESS > 100 for good mixing

### Poorly-Calibrated Model  
- Coverage plot deviates from diagonal
- Mean absolute deviation > 0.10
- May indicate overfitting issues

## File Structure

```
examples/
├── real_gp_hmc_inference.py          # Single simulation HMC inference
├── real_gp_coverage_test.py          # Multi-simulation coverage testing  
├── gp_trainer_usage.py               # GP training examples (kept)
└── README_real_gp_inference.md       # This documentation

inference_results/                     # Auto-generated results directory
├── real_gp_hmc_corner_sim777_*.png   # Corner plots
├── real_gp_hmc_summary_sim777_*.md   # Inference summaries  
├── coverage_plot_*.png               # Coverage plots
└── coverage_test_results_*.pkl       # Raw coverage data
```

## Migration from Old Files

**Removed files** (replaced by consolidated scripts):
- `direct_gp_corner_sim777.py`
- `quick_real_gp_corner_sim777.py`
- `real_gp_corner_plot*.py`
- `simple_*_gp_corner.py`
- `gp_*_inference.py`
- `working_gp_mcmc_inference.py`
- `coverage_test_gp_mcmc.py`

**Key improvements:**
- ✅ Real GP models instead of mock functions
- ✅ Proper TinyGP conditioning API usage  
- ✅ Correct training data loading (716 samples)
- ✅ Production-ready command-line interface
- ✅ Comprehensive error handling and diagnostics
- ✅ Automatic result saving with timestamps

## Usage Examples

### Basic parameter inference:
```bash
python real_gp_hmc_inference.py --sim_id 777
```

### Coverage test with multiple models:
```bash
python real_gp_coverage_test.py --n_sims 50 \
  --gp_dirs /path/to/hierarchical_gp /path/to/robust_gp \
  --parallel
```

### Custom configuration:
```bash
python real_gp_hmc_inference.py \
  --sim_id 888 \
  --n_samples 10000 \
  --burnin 2000 \
  --n_params 6 \
  --gp_dir /path/to/custom/gp/model
```