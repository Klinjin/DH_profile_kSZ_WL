"""
Configuration constants for CAMELS cosmological simulation analysis.

This module centralizes all hard-coded paths, filenames, and configuration 
parameters used throughout the codebase.
"""

import os
import numpy as np

# ============================================================================
# Data Paths and File Names
# ============================================================================

# Base paths
CAMELS_BASE_PATH = '/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/L50n512_SB35'
PROJECT_ROOT = '/pscratch/sd/l/lindajin/DH_profile_kSZ_WL'
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Simulation data file names
PROFILE_FILENAME = 'Henry_profiles_gas_dm_star_bh_nPixel1000_R_lin0.04_2.5_log15_nbins20.npz'
POWER_SPECTRUM_FILENAME = 'baryon_suppression_fields_nPixel512.npz'

# Parameter files
PARAM_MATRIX_PATH = os.path.join(DATA_DIR, 'camels_params_matrix.npy')
PARAM_MINMAX_PATH = os.path.join(DATA_DIR, 'SB35_param_minmax.csv')
BEAM_EXAMPLE_PATH = os.path.join(DATA_DIR, 'beam_example.txt')

# Model save directories
TRAINED_MODELS_DIR = os.path.join(PROJECT_ROOT, 'trained_gp_models')

# ============================================================================
# Simulation Parameters
# ============================================================================

# Cosmological parameters
N_COSMO_PARAMS = 35

# Supported filter types
FILTER_TYPES = ['CAP', 'cumulative', 'dsigma']

# Supported particle types
PARTICLE_TYPES = ['gas', 'dm', 'star', 'bh', 'total', 'baryon']

# Profile indices in simulation files
FILTER_INDICES = {
    'dsigma': 0,
    'CAP': 1, 
    'cumulative': 2
}

# ============================================================================
# Model Training Configuration
# ============================================================================

# GP Training defaults
GP_TRAINING_DEFAULTS = {
    'maxiter': 5000,
    'learning_rate': 3e-4,
    'log_transform_mass': True,
    'noise_level': 1e-2
}

# Neural Network GP defaults
NN_GP_DEFAULTS = {
    'learning_rate_init': 1e-3,
    'decay_rate': 0.9,
    'decay_steps': 500,
    'weight_decay': 1e-4,
    'clip_norm': 1.0,
    'epochs': 2000,
    'patience': 200
}

# NPE/SBI defaults
SBI_DEFAULTS = {
    'num_sims': 1000,
    'num_rounds': 3,
    'sim_id': 777,
    'method': 'NPE'
}

# ============================================================================
# JAX Configuration
# ============================================================================

# GPU configuration
JAX_CONFIG = {
    'cuda_visible_devices': "0",
    'enable_x64': False  # Use float32 for speed
}

# ============================================================================
# Utility Functions
# ============================================================================

def get_simulation_path(sim_id):
    """Get the full path to a simulation directory."""
    return os.path.join(CAMELS_BASE_PATH, f'SB35_{sim_id}', 'data')

def get_profile_file_path(sim_id):
    """Get the full path to a simulation's profile file."""
    return os.path.join(get_simulation_path(sim_id), PROFILE_FILENAME)

def get_power_spectrum_file_path(sim_id):
    """Get the full path to a simulation's power spectrum file."""
    return os.path.join(get_simulation_path(sim_id), POWER_SPECTRUM_FILENAME)

def validate_filter_type(filter_type):
    """Validate that filter_type is supported."""
    if filter_type not in FILTER_TYPES:
        raise ValueError(f"Invalid filter_type: {filter_type}. Must be one of {FILTER_TYPES}")
    return filter_type

def validate_particle_type(ptype):
    """Validate that particle type is supported."""
    if ptype not in PARTICLE_TYPES:
        raise ValueError(f"Invalid ptype: {ptype}. Must be one of {PARTICLE_TYPES}")
    return ptype

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)
    return directory