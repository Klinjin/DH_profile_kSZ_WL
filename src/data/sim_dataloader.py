"""
Comprehensive DataLoader class for cosmological simulation data.

This module provides a PyTorch-style DataLoader with JAX compatibility for:
- Train/validation/test splits
- Batching with configurable batch size  
- Data transformations and normalization
- JAX array outputs for efficient GPU computation
"""

import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Tuple, Optional, Callable, Dict, Any, List
from dataclasses import dataclass
import warnings

from src.data.profile_loader import load_simulation_data


@dataclass
class DataLoaderConfig:
    """Configuration for SimulationDataLoader."""
    # Data splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Batching
    batch_size: int = 128
    shuffle: bool = True
    drop_last: bool = False
    
    # Data processing
    log_transform_mass: bool = True
    normalize_features: bool = True
    normalize_targets: bool = False
    
    # Random seed for reproducible splits
    random_seed: int = 42


class SimulationDataLoader:
    """
    Advanced DataLoader for cosmological simulation data with train/val/test splits.
    
    Features:
    - Automatic train/validation/test splitting
    - Configurable batch sizes and transformations
    - JAX-compatible outputs for GPU acceleration
    - Feature normalization and standardization
    - Flexible data filtering and preprocessing
    
    Example:
        ```python
        config = DataLoaderConfig(
            train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
            batch_size=64, normalize_features=True
        )
        
        dataloader = SimulationDataLoader(
            sim_indices=[1, 2, 3, 10, 20], 
            config=config,
            filterType='CAP', 
            ptype='gas'
        )
        
        # Get dataloaders
        train_dl, val_dl, test_dl = dataloader.get_dataloaders()
        
        # Iterate through batches
        for batch_X, batch_y in train_dl:
            # batch_X, batch_y are JAX arrays
            pass
        ```
    """
    
    def __init__(
        self,
        sim_indices: List[int] = np.arange(1024).tolist(),
        config: DataLoaderConfig = None,
        filterType: str = 'CAP',
        ptype: str = 'gas',
        custom_transforms: Optional[Dict[str, Callable]] = None,
        func: str = 'mean', # 'mean' or 'median' or 'extend'
    ):
        """
        Initialize SimulationDataLoader.
        
        Args:
            sim_indices: List of total simulation indices to load and split
            config: DataLoader configuration
            filterType: Type of filter ('CAP', 'cumulative', 'dsigma')
            ptype: Particle type ('gas', 'dm', 'star', 'bh', 'total', 'baryon')
            custom_transforms: Optional custom transformation functions
        """
        self.sim_indices = sim_indices
        self.config = config or DataLoaderConfig()
        self.filterType = filterType
        self.ptype = ptype
        self.func = func
        self.custom_transforms = custom_transforms or {}
        
        # Validate configuration
        self._validate_config()
        
        # Load and process data
        self._load_raw_data()
        self._apply_transformations()
        self._create_splits()
        
        print(f"✅ DataLoader initialized:")
        print(f"  - Dataset size: {self.X.shape[0]} samples, {self.X.shape[1]} features")
        print(f"  - Train/Val/Test: {len(self.train_indices)}/{len(self.val_indices)}/{len(self.test_indices)}")
        print(f"  - Batch size: {self.config.batch_size}")
    
    def _clip_extreme_profiles(self, profiles: jnp.ndarray, min_val: float = 1e4, max_val: float = 1e20) -> Tuple[jnp.ndarray, int, int]:
        """
        Clip extreme values in profiles and replace with mean profile values (vectorized).
        
        Args:
            profiles: Profile data array of shape (n_samples, n_radius_bins)
            min_val: Minimum allowed value (default: 1e-6)
            max_val: Maximum allowed value (default: 1e20)
            
        Returns:
            Tuple of (clipped_profiles, n_profiles_affected, total_bins_clipped)
        """
        print(f'Clipping extrem measurements >{max_val}&<{min_val}')
        # Calculate mean profile across all simulations for substitution
        mean_profile = jnp.nanmean(profiles, axis=0)
        
        # Find all extreme values using vectorized operations
        extreme_mask = (profiles < min_val) | (profiles > max_val) | jnp.isnan(profiles) | jnp.isinf(profiles)
        
        # Count statistics
        n_profiles_affected = jnp.sum(jnp.any(extreme_mask, axis=1))
        total_bins_clipped = jnp.sum(extreme_mask)
        
        # Replace extreme values with corresponding mean profile values (vectorized)
        profiles_clipped = jnp.where(extreme_mask, mean_profile[None, :], profiles)
        
        if total_bins_clipped > 0:
            print(f"📊 {n_profiles_affected} profiles had {total_bins_clipped} bins clipped out with mean profile data")
        
        return profiles_clipped, int(n_profiles_affected), int(total_bins_clipped)
    
    def _validate_config(self):
        """Validate DataLoader configuration."""
        ratios_sum = self.config.train_ratio + self.config.val_ratio + self.config.test_ratio
        if not np.isclose(ratios_sum, 1.0, atol=1e-6):
            raise ValueError(f"Train/val/test ratios must sum to 1.0, got {ratios_sum}")
        
        if self.config.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.config.batch_size}")
    
    def _load_raw_data(self):
        """Load raw simulation data."""
        print(f"Loading data for {len(self.sim_indices)} simulations...")
        
        # Import here to avoid circular imports
        from src.data.profile_loader import load_simulation_data
        
        # Load data using mean profiles function (compatible with neural network training)
        data_tuple = load_simulation_data(
            self.sim_indices, filterType=self.filterType, ptype=self.ptype,
            include_params=True, include_pk=True, include_mass=True,
            aggregate_func= self.func
        )
        
        # Unpack based on what's included
        self.r_bins = data_tuple[0]
        profiles_ptype = data_tuple[1]
        
        # Handle variable returns based on includes
        idx = 2
        if True:  # include_mass=True
            mass_halos = data_tuple[idx]
            idx += 1
        if True:  # include_params=True  
            param_halos = data_tuple[idx]
            idx += 1
        if True:  # include_pk=True
            self.k_bins = data_tuple[idx]
            PkRatio = data_tuple[idx + 1]
        
        print(f'Raw data loaded:')
        print(f'  - Profiles: {profiles_ptype.shape}')
        print(f'  - Masses: {mass_halos.shape}')
        print(f'  - Params: {param_halos.shape}')
        print(f'  - PkRatio: {PkRatio.shape}')
        
        # Clip extreme values in profiles before storing
        profiles_ptype_clipped, n_profiles_clipped, n_bins_clipped = self._clip_extreme_profiles(profiles_ptype)
        
        # Store raw data
        self.raw_profiles = profiles_ptype_clipped
        self.raw_masses = mass_halos
        self.raw_params = param_halos
        self.raw_pk_ratios = PkRatio
        
        # Store metadata
        self.metadata = {
            'n_radius_bins': len(self.r_bins),
            'n_k_bins': len(self.k_bins),
            'n_cosmo_params': param_halos.shape[1],
            'filterType': self.filterType,
            'ptype': self.ptype,
            'sim_indices': self.sim_indices
        }
    
    def _apply_transformations(self):
        """Apply data transformations."""
        print("Applying data transformations...")
        
        # Transform masses
        if self.config.log_transform_mass:
            log_mass_halos = jnp.log10(self.raw_masses)
            print(f'Log mass range: [{jnp.min(log_mass_halos):.2f}, {jnp.max(log_mass_halos):.2f}]')
        else:
            log_mass_halos = self.raw_masses
        
        # Combine features: [cosmology_params, log_mass, pk_ratio]
        X_combined = jnp.concatenate([
            self.raw_params, 
            log_mass_halos[:, None], 
            self.raw_pk_ratios
        ], axis=1)
        
        y_combined = self.raw_profiles
        
        # Apply custom transformations if provided
        if 'features' in self.custom_transforms:
            X_combined = self.custom_transforms['features'](X_combined)
        
        if 'targets' in self.custom_transforms:
            y_combined = self.custom_transforms['targets'](y_combined)
        
        # Normalize features
        if self.config.normalize_features:
            self.feature_mean = jnp.mean(X_combined, axis=0)
            self.feature_std = jnp.std(X_combined, axis=0)
            X_combined = (X_combined - self.feature_mean) / (self.feature_std + 1e-8)
            print("✅ Features normalized (zero mean, unit variance)")
        
        # Normalize targets  
        if self.config.normalize_targets:
            self.target_mean = jnp.nanmean(y_combined, axis=0)
            self.target_std = jnp.nanstd(y_combined, axis=0)
            y_combined = (y_combined - self.target_mean) / (self.target_std + 1e-8)
            print("✅ Targets normalized (zero mean, unit variance)")
        
        self.X = X_combined
        self.y = y_combined
        
        print(f'Final data shapes:')
        print(f'  - Features: {self.X.shape}')
        print(f'  - Targets: {self.y.shape}')
    
    def _create_splits(self):
        """Create train/validation/test splits."""
        print("Creating train/validation/test splits...")
        
        # Set random seed for reproducible splits
        torch.manual_seed(self.config.random_seed)
        
        n_samples = self.X.shape[0]
        
        # Calculate split sizes
        n_train = int(self.config.train_ratio * n_samples)
        n_val = int(self.config.val_ratio * n_samples)
        n_test = n_samples - n_train - n_val  # Remainder goes to test
        
        # Create random split
        indices = torch.randperm(n_samples).tolist()
        
        self.train_indices = indices[:n_train]
        self.val_indices = indices[n_train:n_train + n_val]
        self.test_indices = indices[n_train + n_val:]
        
        print(f"Split created:")
        print(f"  - Train: {len(self.train_indices)} samples ({len(self.train_indices)/n_samples:.1%})")
        print(f"  - Val:   {len(self.val_indices)} samples ({len(self.val_indices)/n_samples:.1%})")
        print(f"  - Test:  {len(self.test_indices)} samples ({len(self.test_indices)/n_samples:.1%})")
    
    def get_split_data(self, split: str = 'train') -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get data for a specific split.
        
        Args:
            split: 'train', 'val', or 'test'
            
        Returns:
            Tuple of (X, y) as JAX arrays
        """
        if split == 'train':
            indices = self.train_indices
        elif split == 'val':
            indices = self.val_indices
            if len(indices) == 0:
                print("Validation set is empty (val_ratio=0).")
                return None, None  # Handle empty validation set
        elif split == 'test':
            indices = self.test_indices
        else:
            raise ValueError(f"Unknown split '{split}'. Use 'train', 'val', or 'test'")
        
        # Convert indices to numpy array for JAX indexing
        indices_np = indices.numpy() if hasattr(indices, 'numpy') else np.array(indices)
        return self.X[indices_np], self.y[indices_np]
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get PyTorch DataLoaders for train/val/test splits with JAX compatibility.
        
        Returns:
            Tuple of (train_dataloader, val_dataloader, test_dataloader)
            Note: val_dataloader will be None if val_ratio=0
        """
        def create_dataloader(indices: torch.Tensor, shuffle: bool = None) -> DataLoader:
            """Create a DataLoader for given indices."""
            if len(indices) == 0:
                return None
                
            if shuffle is None:
                shuffle = self.config.shuffle
                
            # Convert torch tensor indices to numpy array for JAX indexing
            indices_np = indices.numpy() if hasattr(indices, 'numpy') else np.array(indices)
            X_split = self.X[indices_np]
            y_split = self.y[indices_np]
            
            # Convert to torch tensors for DataLoader
            X_torch = torch.from_numpy(np.array(X_split)).float()
            y_torch = torch.from_numpy(np.array(y_split)).float()
            
            dataset = TensorDataset(X_torch, y_torch)
            
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                drop_last=self.config.drop_last,
                collate_fn=self._jax_collate_fn
            )
        
        train_dl = create_dataloader(self.train_indices, shuffle=True)
        val_dl = create_dataloader(self.val_indices, shuffle=False)
        test_dl = create_dataloader(self.test_indices, shuffle=False)
        
        return train_dl, val_dl, test_dl
    
    def _jax_collate_fn(self, batch) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Custom collate function that returns JAX arrays.
        
        Args:
            batch: List of (X, y) tensor pairs
            
        Returns:
            Tuple of (X_batch, y_batch) as JAX arrays
        """
        # Stack tensors
        X_batch = torch.stack([item[0] for item in batch])
        y_batch = torch.stack([item[1] for item in batch])
        
        # Convert to JAX arrays
        X_jax = jnp.array(X_batch.numpy())
        y_jax = jnp.array(y_batch.numpy())
        
        return X_jax, y_jax
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for interpretability."""
        feature_names = []
        
        # Cosmological parameters (assuming 35 parameters)
        for i in range(self.metadata['n_cosmo_params']):
            feature_names.append(f'cosmo_param_{i}')
        
        # Mass feature
        if self.config.log_transform_mass:
            feature_names.append('log_halo_mass')
        else:
            feature_names.append('halo_mass')
        
        # Power spectrum ratios
        for i in range(self.metadata['n_k_bins']):
            feature_names.append(f'pk_ratio_{i}')
        
        return feature_names
    
    def get_normalization_params(self) -> Dict[str, Any]:
        """Get normalization parameters for denormalizing predictions."""
        norm_params = {}
        
        if self.config.normalize_features:
            norm_params['feature_mean'] = self.feature_mean
            norm_params['feature_std'] = self.feature_std
        
        if self.config.normalize_targets:
            norm_params['target_mean'] = self.target_mean
            norm_params['target_std'] = self.target_std
        
        return norm_params
    
    def denormalize_targets(self, y_normalized: jnp.ndarray) -> jnp.ndarray:
        """
        Denormalize target predictions.
        
        Args:
            y_normalized: Normalized predictions
            
        Returns:
            Denormalized predictions
        """
        if not self.config.normalize_targets:
            return y_normalized
        
        return y_normalized * self.target_std + self.target_mean
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            'total_samples': self.X.shape[0],
            'n_features': self.X.shape[1],
            'n_targets': self.y.shape[1],
            'train_samples': len(self.train_indices),
            'val_samples': len(self.val_indices), 
            'test_samples': len(self.test_indices),
            'feature_mean': jnp.mean(self.X, axis=0) if hasattr(self, 'X') else None,
            'feature_std': jnp.std(self.X, axis=0) if hasattr(self, 'X') else None,
            'target_mean': jnp.nanmean(self.y, axis=0) if hasattr(self, 'y') else None,
            'target_std': jnp.nanstd(self.y, axis=0) if hasattr(self, 'y') else None,
            **self.metadata
        }


# Backward compatibility function

def prepare_gp_training_data(sim_indices_train, filterType='CAP', ptype='gas', log_transform_mass=True):
    """
    Prepare data for GP training with halo mass as an additional input feature.
    
    Args:
        sim_indices_train: List of simulation indices for training
        filterType: Type of filter to apply ('CAP', 'cumulative', 'dsigma')
        ptype: Particle type ('gas', 'dm', 'star', 'bh', 'total', 'baryon')
        log_transform_mass: Whether to log-transform the halo mass
    
    Returns:
        Tuple containing (X_combined, y, r_bins, k_bins)
    """        
    warnings.warn(
        "prepare_gp_training_data is deprecated. Use SimulationDataLoader instead.",
        DeprecationWarning,
        stacklevel=2
    )
    from src.data.profile_loader import load_simulation_data
    
    # Load data using unified function
    r_bins, profiles_ptype, mass_halos, param_halos, k, PkRatio = load_simulation_data(
        sim_indices_train, filterType=filterType, ptype=ptype,
        include_params=True, include_pk=True, include_mass=True,
        aggregate_method='extend'
    )
    
    print(f'Profiles shape: {profiles_ptype.shape}, Mass shape: {mass_halos.shape}, '
          f'Params shape: {param_halos.shape}, PkRatio shape: {PkRatio.shape}')
    
    # Transform data
    if log_transform_mass:
        mass = np.log10(mass_halos).reshape(-1, 1)
        profiles_ptype_safe = np.where(profiles_ptype < 0, 1e-10, profiles_ptype)
        profiles = np.log10(profiles_ptype_safe + 1e-10)  # Avoid log(0)
    else:
        mass = mass_halos.reshape(-1, 1) / 1e13
        profiles = profiles_ptype / 1e13
    
    # Combine all input features
    X_combined = np.concatenate([np.concatenate([param_halos, mass], axis=1), PkRatio], axis=1)
    
    return (jnp.array(X_combined), jnp.array(profiles), jnp.array(r_bins), jnp.array(k[0]))


# Convenience functions for common configurations
def create_nn_dataloader(
    sim_indices: List[int],
    batch_size: int = 128,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    **kwargs
) -> SimulationDataLoader:
    """Quick setup for common ML DataLoader configurations."""
    config = DataLoaderConfig(
        train_ratio=train_ratio,
        val_ratio=val_ratio, 
        test_ratio=test_ratio,
        batch_size=batch_size,
        **kwargs
    )
    
    return SimulationDataLoader(sim_indices=sim_indices, config=config)


def create_gp_dataloader(
    sim_indices: List[int],
    filterType: str = 'CAP',
    ptype: str = 'gas',
    normalize_features: bool = True,
    log_transform_mass: bool = True,
    **config_kwargs
) -> SimulationDataLoader:
    """Create DataLoader optimized for GP training."""
    config = DataLoaderConfig(
        normalize_features=normalize_features,
        log_transform_mass=log_transform_mass,
        shuffle=False,  # GPs don't need shuffling
        **config_kwargs
    )
    
    return SimulationDataLoader(
        sim_indices=sim_indices, 
        config=config,
        filterType=filterType,
        ptype=ptype
    )