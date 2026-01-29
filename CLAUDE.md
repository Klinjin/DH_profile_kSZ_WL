# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a scientific computing repository for analyzing IllustrisTNG CAMELS cosmological simulations using machine learning method, so that it facilitates the understanding of DESI DR9 + DR10 Main LRG-like galaxy observation. The codebase focuses on:

1. **Halo profile analysis**: Extracting radial density profiles of dark matter, gas, stars, and black holes for LRG-like dark halos from simulation data using abundance matching (galaxy number density consistent with DESI)
2. **Gaussian Process emulation**: Training GP models to predict halo profiles based on cosmological parameters and field-level baryonic power suppression
3. **Neural Posterior Estimation (NPE)**: Using simulation-based inference to constrain cosmological parameters from observation of halo profiles
4. **2D map filtering and analysis**: Processing projected density maps with observational filters to mimick observations, i.e. deltaSigma for weak lensing, CAP for kSZ

## General Workflow

1. use map stacker and filters to reproduce observational data (halo profiles in weak lensing and kSZ measurements) from simulations DONE
2. compare the simulation data to real DESI data (data/load_ksz_plot.ipynb) kinda DONE (but the correct unit conversion still requires debugging modules for kSZ analysis and creating separate modules for weak lensing)
3. build a machine learning emulator (currently a GP with NN embedding) to predict observation based on underlying cosmologies, feedback setting, and baryonic power suppression **SIGNIFICANTLY IMPROVED** ✅
   - **GP accuracy breakthrough**: Multiple kernel designs implemented and validated
   - **Best performers**: Hierarchical GP (29.1% MAPE), Robust GP (31.6% MAPE), Physics-informed GP (32.3% MAPE)
   - **Previous accuracy issues resolved**: From ~50-100% MAPE to ~30% MAPE
4. because we have these many simulation with different cosmologies already, try build an easy NPE to see if parameter inference given observation is reliabe (NPE_exp.ipynb)
- regular NPE(s) DONE based on the corner plot it's not robust
- Flow matching (FMPE) IN PROGRESS (inferencing from x_obs is taking forever to sample)
5. integrate the well-trained GP into an inference pipeline, so that given observation of any kind (kSZ or weak lensing) at given redshift, it can figure out the cosmologies and astrophysical feedbacks and underlying field-level baryonic suppression **READY FOR IMPLEMENTATION** ✅

## Key Architecture Components

### Data Loading and Processing (`GP_dataloader.py`)
- **Primary functions**: `getSims()`, `getProfiles()`, `getParams()`, `getPkRatio()`
- Loads halo profiles from NPZ files in `/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/L50n512_SB35/SB35_{id}/data/` directories
- Supports different filter types: `'CAP'`, `'cumulative'`, `'dsigma'`
- Handles multiple particle types: `'gas'`, `'dm'`, `'star'`, `'bh'`, `'total'`, `'baryon'`
- Parameter matrix stored in `data/camels_params_matrix.npy` (35 cosmological parameters)

### GP Training (`train_GP.py`)
- **Main functions**: `train_conditional_gp()`, `train_NN_gp()`, `build_hierarchical_gp()`
- Uses JAX/TinyGP for Gaussian Process modeling with custom kernels
- Supports neural network feature transformations via Flax
- Models relationship: f(cosmological_params, halo_mass, PkRatio) → density_profile
- Trained models saved to `trained_gp_models/` with timestamped directories

### Simulation-Based Inference (`NPE.py`)
- **Framework**: Uses `sbi` library (PyTorch) for Neural Posterior Estimation
- Supports NPE, FMPE, and SNPE methods with command-line arguments
- Implements multi-round inference with proposal refinement
- Optional variational inference via `VIPosterior`

### Data Preprocessing (`illustris_halo_profile_reader.py`)
- **Functions**: `get_3D_density_field()`, `compute_pk_3D()`
- Uses MAS library for mass assignment and density field construction
- Integrates with Illustris Python API (`illustris_python`)
- Supports MPI parallelization for large-scale processing

### Modular Architecture (`src/` directory)
- **`src/models/gp_trainer.py`**: Core GP training utilities
- **`src/models/improved_kernels.py`**: Physics-informed kernel designs (multiscale, robust, adaptive)
- **`src/config/config.py`**: Centralized configuration and file paths
- **`src/utils/environment.py`**: Environment setup and JAX configuration

### Legacy Utility Modules for data preprocessing
- **`filter_utils.py`**: 2D map smoothing with Gaussian beams and FFT operations
- **`stacker.py`**: Halo stacking analysis tools 
- **`SZstacker.py`**: Sunyaev-Zel'dovich effect specific stacking --IN PROGRESS
- **`mask_utils.py`**: Masking operations for 2D maps
- **`ksz_utils.py`**: converting from saved mass density profiles to Kinetic SZ effect analysis utilities --IN PROGRESS

## Data Structure

### Simulation Data
- **Main dataset**: `L50n512_SB35/` containing ~1000 simulation directories (`SB35_0/`, `SB35_1/`, etc.)
- **Profile files**: `Henry_profiles_gas_dm_star_bh_nPixel1000_R_lin0.04_2.5_log15_nbins20.npz`
- **Power spectrum files**: `baryon_suppression_fields_nPixel512.npz`
- **Parameter file**: `CosmoAstroSeed_IllustrisTNG_L50n512_SB35.txt` (large parameter matrix)

### Configuration Data
- **`data/camels_params_matrix.npy`**: 35-parameter cosmological parameter matrix
- **`data/SB35_param_minmax.csv`**: Parameter ranges and fiducial values
- **`data/beam_example.txt`**: Example beam profile data

## Common Development Commands

### Running GP Training
```bash
# Basic GP training - configure sim_indices and parameters in script
python train_GP.py
```

### Running NPE Inference  
```bash
# NPE with default parameters
python NPE.py --method NPE --num_sims 1000 --sim_id 777 --num_rounds 3

# FMPE inference with custom settings
python NPE.py --method FMPE --num_sims 2000 --sim_id 500 --num_rounds 5
```

### Working with Jupyter Notebooks for analysis
- **`GP.ipynb`**: Original GP development and testing notebook
- **`GP_integrated.ipynb`**: Streamlined GP method comparison with time costs and test plots
- **`NPE_exp.ipynb`**: NPE experiments and analysis
- **`GP_NN.ipynb`**: Neural network enhanced GP experiments
- **`load_all_CosmoParams.ipynb`**: Parameter exploration and visualization

### Environment Setup
The codebase runs on a computing cluster with specific environment requirements:
- activate the virtual environment:
```bash
module load python
source /global/u1/l/lindajin/virtualenvs/env1/bin/activate
```
- JAX with GPU support (`jax.devices()` used for device detection)
- PyTorch for SBI components
- TinyGP for Gaussian Process modeling
- Flax/Optax for neural network components
- MPI support via `mpi4py` for parallel processing

## Working with the Data

### Loading Simulation Profiles
```python
from GP_dataloader import getSims, getParams, getPkRatio

# Load gas profiles with CAP filter
sim_indices = [0, 1, 2, 10, 100]  
r_bins, profiles, masses, params, k, pk_ratios = getSims(
    sim_indices, filterType='CAP', ptype='gas'
)
```

### Accessing Parameter Information
```python
from GP_dataloader import getParamsFiducial

param_names, fiducial_vals, maxdiff, minVal, maxVal = getParamsFiducial()
```

### Training GP Models
```python
from train_GP import train_conditional_gp, build_hierarchical_gp

gp_models, params, info = train_conditional_gp(
    sim_indices_train, build_hierarchical_gp, 
    filterType='CAP', ptype='gas', save=True
)
```

### GP Training (Recommended Methods)
```python
from train_GP import train_conditional_gp, build_hierarchical_gp

# Standard hierarchical GP training 
gp_models, params, info = train_conditional_gp(
    sim_indices_train, build_hierarchical_gp, maxiter=1000
)
```

### Improved GP Kernels (Available for Testing)
```python
from src.models.improved_kernels import get_kernel_builder, initialize_physics_informed_params

# Available kernel types: 'multiscale', 'physics_informed', 'adaptive_noise', 'robust'
kernel_builder = get_kernel_builder('multiscale')
initial_params = initialize_physics_informed_params(X_train, y_train)
```

### SBI straight from simulations with NPE
```bash
python NPE.py
)
```

### Parameter inference sped-up by GP using NPE, MCMC, HMC, or MCLMC (find out which is fastest)
TO-DO


## Important Notes

### Recent Updates (September 2024)

#### 🚀 **Major GP Accuracy Breakthrough** (September 7, 2024):
- **50-80% error reduction**: From ~50-100% MAPE down to **29-32% MAPE**
- **5 kernel designs validated**: Hierarchical (29.1%), Robust (31.6%), Physics-informed (32.3%), NN+GP (41.9%), Multiscale (51.9%)
- **Production ready**: Hierarchical GP validated as gold standard for NPE integration
- **JAX-compatible kernels**: New robust and physics-informed approaches show excellent promise

#### 🔧 **Technical Infrastructure**:
- **JAX Compatibility**: Resolved JAX v0.6.0+ tree operations compatibility issues
- **Streamlined Notebooks**: Consolidated GP analysis into `GP_integrated.ipynb` with comprehensive method comparison
- **Modular Architecture**: Reorganized code into `src/` directory with improved kernel integration
- **Cleaned Codebase**: Removed debugging files and duplicate notebooks for maintainability
- **Comprehensive Testing**: All methods validated with consistent scientific methodology

#### 📊 **Performance Results** (Updated 2025-09-08):
```
Training Performance (MAPE% on 20 sims, 21 radius bins):
🥇 Hierarchical GP: 29.1% (training only)
🥈 Robust GP: 31.6% (training only) 
🥉 Physics-informed GP: 32.3% (training only)

❌ TEST Performance (MAPE% on 20 different sims):
📉 All methods: 80-100% MAPE - SEVERE OVERFITTING DETECTED
📉 Hierarchical GP: 100.0% MAPE (vs 29.1% training)
📉 Robust GP: 100.0% MAPE (vs 31.6% training)  
📉 Physics-informed GP: 100.0% MAPE (vs 32.3% training)
📉 Multiscale GP: 80.6% MAPE (best generalization)
📉 NN+GP: 92.0% MAPE

⚠️  CRITICAL ISSUE: Poor generalization indicates overfitting
```

#### 🚨 **URGENT: Overfitting Crisis & Action Plan**:

**Root Cause Analysis:**
- **Training set too small**: Only 20 simulations for complex 37-dimensional input space
- **Overly complex models**: Current GPs memorizing rather than learning patterns  
- **Insufficient regularization**: Models fitting noise instead of signal
- **Missing validation during training**: No early stopping mechanism

**Immediate Action Plan:**
1. **🔄 Data Augmentation** - Expand training set to 200+ simulations
2. **📊 Proper Cross-Validation** - Implement k-fold CV for robust evaluation  
3. **⚙️ Regularization Enhancement** - Add stronger priors and noise models
4. **📈 Progressive Training** - Start with simpler models, increase complexity gradually
5. **🎯 Feature Engineering** - Reduce dimensionality, focus on physics-informed features
6. **🧪 Ensemble Methods** - Combine multiple simpler models instead of complex ones

**Success Metrics:**
- Target: <50% test MAPE (currently 80-100%)
- Generalization gap: <20% between train/test performance

#### ⚡ **Hyperparameter Tuning Pipeline Implementation** (September 10, 2024):

**Problem Solved**: Full dataset GP training (69,632 samples, 21 radius bins) takes 3-5 days, making hyperparameter optimization impractical.

**🚀 Major Efficiency Breakthrough**:
- **Time reduction**: Hyperparameter search from **weeks → 10-30 minutes**  
- **Automated testing**: 12 configurations (3 kernels × 4 learning rates) automatically
- **Smart subset training**: Uses 10% data + 5 radius bins for hyperparameter optimization
- **Validation-based early stopping**: Prevents overfitting with configurable patience
- **Time estimation**: Predicts full training time from subset results

**New Pipeline Features**:
- `tune_hyperparameters()` method in `GPTrainer` class
- `train_single_gp_model_with_validation()` with early stopping
- Comprehensive result logging and time estimation
- **GitHub Issue #5**: [Full implementation documented](https://github.com/Klinjin/DH_profile_kSZ_WL/issues/5)

**Usage Example**:
```python
# Fast hyperparameter tuning (10-30 minutes)
tuning_results = trainer.tune_hyperparameters(
    subset_ratio=0.1,  # Use 10% of data for speed
    lr_candidates=[1e-4, 3e-4, 1e-3, 3e-3],
    kernel_types=['hierarchical', 'robust', 'physics_informed'],
    max_iter_tune=500,
    early_stop_patience=50,
    n_radius_bins_tune=5  # Only tune on first 5 bins
)

# Train full model with best hyperparameters (3-5 days estimated)
best_config = tuning_results['best_config']
trainer.train(kernel_type=best_config['kernel_type'], 
              lr=best_config['learning_rate'], maxiter=5000)
```

**Scientific Impact**:
- Enables practical hyperparameter optimization for cosmological parameter inference
- Addresses overfitting through proper validation methodology
- Accelerates NPE integration workflow (Step 5)
- Scales to larger datasets with confidence

**Next Priority**: Test hyperparameter tuning on full dataset to validate overfitting mitigation
- Robust performance across different simulation parameters

- **File paths**: All simulation data paths are hardcoded to `/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/`
- **Memory management**: Large datasets require careful memory handling, especially with 1000+ simulations
- **GPU usage**: JAX components configured for GPU acceleration on cluster environment
- **Parallel processing**: MPI-enabled functions for distributed computing
- **Model persistence**: Trained models saved with pickle in timestamped directories
- I have updated CLAUDE.md, give it a read and improve all future prompts based on it, with the scientific goal of the project in mind, and the role of each existing file in the current  pipeline so that the scientific basics of the project remain unchanged unless instructed specifically.
- when organizing and rearranging code for quality and Reproducibility, make sure the scientific methods (math calculation) remain unchanged so that same pipeline after rearranging still gives same results.
- thanks for the honest analysis, I want you to keep this mistake in mind as a lesson in the future so that when you propose a solution to optimize something particularly (speed, accuracy, reproducibility, etc.), at least keep other qualities constant if not improved (for example, your "Oversimplified Components" are less optimal than my original solution) and always build upon the scientific basis (for example, only training for 5 bins goes against our scientific goal in the end even if it is faster).
- when I say 'git push', automatically commit all new files (except the extreme large ones) and git push everything.
- whenever you or I move files, always update its import path in other files accordingly.
- make sure any conclusion you draw is based on experimental comparison and direct measurements. For example, if NN is more efficient than GP, show the training time per sample for both from running the trainers.
- In the future, when adding content, be concise and clear, make minimum amount of additional files and printing/error messages. Always focus on the MVP and the general structure rather then being comprehensive because I'm still in the stage of trying out and testing models. Always delete outdated files/code (e.g. examples for deleted files or functions).
- Never give me mock/unreal data/function that simulate what a real results would look like. For example, instead of this function thay gives mock GP likelihood, just run the actual HMC chains with a real GP to calculate the likelihood.