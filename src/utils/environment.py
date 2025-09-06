"""
Environment setup and JAX configuration for CAMELS analysis.

This module handles JAX configuration, GPU setup, and other environment
initialization that needs to be done once at the start of the application.
"""

import os
import jax
import jax.numpy as jnp
from src.config.config import JAX_CONFIG

def setup_jax_environment():
    """
    Configure JAX environment for optimal performance.
    
    Sets up GPU usage, precision, and other JAX configurations.
    """
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = JAX_CONFIG['cuda_visible_devices']
    
    # Configure JAX precision
    jax.config.update("jax_enable_x64", JAX_CONFIG['enable_x64'])
    
    # Print device information
    print(f"JAX devices: {jax.devices()}")
    if jax.devices():
        print(f"Using device: {jax.devices()[0]}")
    else:
        print("Warning: No JAX devices found!")
    
    return jax.devices()

def get_default_device():
    """Get the default JAX device."""
    devices = jax.devices()
    return devices[0] if devices else None