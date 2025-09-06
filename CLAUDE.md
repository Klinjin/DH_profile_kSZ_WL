# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a scientific computing repository for analyzing IllustrisTNG CAMELS cosmological simulations. The codebase focuses on:

1. **Halo profile analysis**: Extracting radial density profiles of dark matter, gas, stars, and black holes from simulation data
2. **Gaussian Process emulation**: Training GP models to predict halo profiles based on cosmological parameters
3. **Neural Posterior Estimation (NPE)**: Using simulation-based inference to constrain cosmological parameters
4. **2D map filtering and analysis**: Processing projected density maps with observational filters

## Key Architecture Components

### Data Loading and Processing (`GP_dataloader.py`)
- **Primary functions**: `getSims()`, `getProfiles()`, `getParams()`, `getPkRatio()`
- Loads halo profiles from NPZ files in `L50n512_SB35/SB35_{id}/data/` directories
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

### Profile Reading (`illustris_halo_profile_reader.py`)
- **Functions**: `get_3D_density_field()`, `compute_pk_3D()`
- Uses MAS library for mass assignment and density field construction
- Integrates with Illustris Python API (`illustris_python`)
- Supports MPI parallelization for large-scale processing

### Utility Modules
- **`filter_utils.py`**: 2D map smoothing with Gaussian beams and FFT operations
- **`stacker.py`**: Halo stacking analysis tools
- **`SZstacker.py`**: Sunyaev-Zel'dovich effect specific stacking
- **`mask_utils.py`**: Masking operations for 2D maps
- **`ksz_utils.py`**: Kinetic SZ effect analysis utilities

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

### Working with Jupyter Notebooks
- **`GP.ipynb`**: Main GP development and testing notebook
- **`NPE_exp.ipynb`**: NPE experiments and analysis
- **`GP_NN.ipynb`**: Neural network enhanced GP experiments
- **`load_all_CosmoParams.ipynb`**: Parameter exploration and visualization

### Environment Setup
The codebase runs on a computing cluster with specific environment requirements:
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

## Important Notes

- **File paths**: All simulation data paths are hardcoded to `/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/`
- **Memory management**: Large datasets require careful memory handling, especially with 1000+ simulations
- **GPU usage**: JAX components configured for GPU acceleration on cluster environment
- **Parallel processing**: MPI-enabled functions for distributed computing
- **Model persistence**: Trained models saved with pickle in timestamped directories