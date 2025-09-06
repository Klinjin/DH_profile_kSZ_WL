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
3. build a machine learning emulator (currently a GP with NN embedding) to predict observation based on underlying cosmologies, feedback setting, and baryonic power suppression DONE (improve: accuracy too low)
4. because we have these many simulation with different cosmologies already, try build an easy NPE to see if parameter inference given observation is reliabe (NPE_exp.ipynb)
- regular NPE(s) DONE based on the corner plot it's not
- Flow matching (FMPE) IN PROGRESS (inferencing from x_obs is taking forever to sample)
5. integrate the well-trained GP into an inference pipeline, so that given observation of any kind (kSZ or weak lensing) at given redshift, it can figure out the cosmologies and astrophysical feedbacks and underlying field-level baryonic suppression  TO-DO

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
- **`src/models/improved_gp_trainer.py`**: Enhanced training with JAX compatibility and robust optimization
- **`src/models/improved_kernels.py`**: Physics-informed kernel designs (multiscale, robust, adaptive)
- **`src/models/jax_compat_trainer.py`**: JAX-compatible fallback trainer for stability
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

### JAX-Compatible GP Training (Recommended)
```python
from src.models.jax_compat_trainer import train_simple_conditional_gp

# Stable training across JAX versions
gp_models, params, info = train_simple_conditional_gp(
    sim_indices_train, kernel_name='hierarchical', maxiter=500
)
```

### Improved GP Kernels (Advanced)
```python
from src.models.improved_gp_trainer import train_improved_conditional_gp

# Physics-informed kernels for better accuracy
gp_models, params, info = train_improved_conditional_gp(
    sim_indices_train, kernel_name='multiscale', maxiter=1500
)
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
- **JAX Compatibility**: Resolved JAX v0.6.0+ tree operations compatibility issues
- **Streamlined Notebooks**: Consolidated GP analysis into `GP_integrated.ipynb` with time comparison and test plots
- **Modular Architecture**: Reorganized code into `src/` directory with improved structure
- **Cleaned Codebase**: Removed debugging files and duplicate notebooks for maintainability

- **File paths**: All simulation data paths are hardcoded to `/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/`
- **Memory management**: Large datasets require careful memory handling, especially with 1000+ simulations
- **GPU usage**: JAX components configured for GPU acceleration on cluster environment
- **Parallel processing**: MPI-enabled functions for distributed computing
- **Model persistence**: Trained models saved with pickle in timestamped directories
- I have updated CLAUDE.md, give it a read and improve all future prompts based on it, with the scientific goal of the project in mind, and the role of each existing file in the current  pipeline so that the scientific basics of the project remain unchanged unless instructed specifically.
- when organizing and rearranging code for quality and Reproducibility, make sure the scientific methods (math calculation) remain unchanged so that same pipeline after rearranging still gives same results.