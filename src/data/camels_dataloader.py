import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

class KSZDeltaSigmaDataset(Dataset):
    """
    DataLoader for kSZ/DeltaSigma emulator training.
    Loads kSZ_over_R2_DeltaSigma_ratios_nd_0_n_12.npz and Ptot_Pdm_ratio_k_le15.npz.
    
    ksz_data has shape (15, 921600) where each of 1024 samples uses 900 consecutive values.
    ptot_data has shape (1024, 255) where each row is a target.
    
    Creates 1024 paired samples with:
    - input: (15, 900*1024) kSZ/DeltaSigma ratio for halos per sample
    - target: (255,) Ptot/Pdm power suppression ratios
    - cosmo_params: (n_params,) cosmological parameters (optional)
    """
    def __init__(self, data_dir, include_cosmo_params=False, mean_prof=False):
        if mean_prof:
            self.ksz_path = os.path.join(data_dir, 'kSZ_over_T_DeltaSigma_profiles_nd_4_n_300.npz')
        else:
            self.ksz_path = os.path.join(data_dir, 'kSZ_over_R2_DeltaSigma_ratios_nd_4_n_300.npz')
        self.ptot_path = os.path.join(data_dir, 'Ptot_Pdm_ratio_k_le15.npz')
        self.cosmo_params_path = os.path.join(data_dir, "camels_params_matrix.npy")
        self.include_cosmo_params = include_cosmo_params
        self.n_samples = 1024
        self.n_features_per_sample = 1
        self.r_bins = None
        self.k_bins = None
        self.mean_prof = mean_prof
        self._load_data()

    def _load_data(self):
        # Load kSZ/DeltaSigma ratios - shape (15, 921600)
        ksz_npz = np.load(self.ksz_path)
        self.ksz_data = ksz_npz['prof']
        if self.mean_prof and 'prof_std' in ksz_npz:
            self.ksz_data = np.stack([ksz_npz['prof'], ksz_npz['prof_std']], axis=1).reshape(-1, ksz_npz['prof'].shape[1])
        self.r_bins = ksz_npz['radii']
        
        # Load Ptot/Pdm ratios - shape (1024, 255)
        ptot_npz = np.load(self.ptot_path)
        self.ptot_data = ptot_npz['Ptot_Pdm_ratio']
        self.k_bins = ptot_npz['k']

        # Load cosmological parameters only if needed
        if self.include_cosmo_params:
            self.cosmo_params = np.load(self.cosmo_params_path)
            print(f"Loaded cosmo params with shape: {self.cosmo_params.shape}")
        else:
            self.cosmo_params = None
    
        print(f"\n=== Data Loading Debug ===")
        print(f"Loaded ksz_data shape: {self.ksz_data.shape}")
        print(f"Loaded ptot_data shape: {self.ptot_data.shape}")
        print(f"Include cosmo params: {self.include_cosmo_params}")
        
        
        # Check if we have the expected amount of data
        if self.ksz_data.shape[0] != self.n_samples * self.n_features_per_sample:
            print(f"\nWARNING: ksz_data has {self.ksz_data.shape[0]} values, not {self.n_samples * self.n_features_per_sample}")
            print(f"Recalculating n_features_per_sample...")
            self.n_features_per_sample = self.ksz_data.shape[0] // self.n_samples
            print(f"Updated n_features_per_sample to: {self.n_features_per_sample}")
            print(f"Input feature dimension will be: {self.r_bins.shape[0] * self.n_features_per_sample}")
        
        assert self.ptot_data.shape[0] == self.n_samples, \
            f"Expected {self.n_samples} samples in ptot_data, got {self.ptot_data.shape[0]}"
        
        print(f"\nFinal configuration:")
        print(f"  n_samples: {self.n_samples}")
        print(f"  n_features_per_sample: {self.n_features_per_sample}")
        print(f"  Input dimension: {self.r_bins.shape[0] * self.n_features_per_sample}")
        print(f"  Output dimension: {self.ptot_data.shape[1]}")
        print(f"=========================\n")

    def __len__(self):
        return self.n_samples
    
    @property
    def n_radial_bins(self):
        """Number of radial bins in kSZ data."""
        return self.r_bins.shape[1]
    
    @property
    def n_spatial_positions(self):
        """Number of spatial positions per sample."""
        return self.n_features_per_sample
    
    @property
    def n_power_bins(self):
        """Number of power spectrum bins."""
        return self.ptot_data.shape[1]
    
    @property
    def input_dim(self):
        """Total input feature dimension (n_radial_bins * n_spatial_positions)."""
        return self.r_bins.shape[1] * self.n_features_per_sample

    def __getitem__(self, idx):
        """
        Get the idx-th paired sample.
        
        Returns:
            input_features: (900*15,) kSZ/DeltaSigma ratio normalized tensor
            target: (255,) normalized tensor
            cosmo_params: (n_params,) cosmological parameters or None
        """
        # Extract the corresponding 900 values for this sample from each of the 15 rows
        start_col = idx * self.n_features_per_sample
        end_col = (idx + 1) * self.n_features_per_sample
        
        # Shape: (900, 15) or (1,15) depending on n_features_per_sample
        ksz_sample = self.ksz_data[start_col:end_col, :]
        
        # Flatten to (900 * 15,)
        input_features = ksz_sample.flatten()
        
        # Get target - shape: (255,)
        target = self.ptot_data[idx]
        
        # Get cosmological parameters if included
        if self.include_cosmo_params:
            cosmo_params = torch.tensor(self.cosmo_params[idx], dtype=torch.float32)
        else:
            cosmo_params = None
        
        return torch.tensor(input_features, dtype=torch.float32), \
               torch.tensor(target, dtype=torch.float32), \
               cosmo_params


def get_dataloader(data_dir, batch_size=16, shuffle=True, num_workers=0, include_cosmo_params=False):
    """
    Convenience function to create a DataLoader.
    
    Args:
        data_dir: Path to directory containing data files
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        include_cosmo_params: Whether to include cosmological parameters
        
    Returns:
        DataLoader instance
    """
    dataset = KSZDeltaSigmaDataset(data_dir, include_cosmo_params=include_cosmo_params)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def custom_collate_fn(batch):
    """
    Custom collate function that handles None values for cosmo_params.
    """
    inputs, targets, cosmo_params = zip(*batch)
    
    # Stack inputs and targets normally
    inputs_tensor = torch.stack(inputs)
    targets_tensor = torch.stack(targets)
    
    # Handle cosmo_params - check if all are None
    if cosmo_params[0] is None:
        cosmo_params_tensor = None
    else:
        cosmo_params_tensor = torch.stack(cosmo_params)
    
    return inputs_tensor, targets_tensor, cosmo_params_tensor


def get_train_val_test_dataloaders(data_dir, batch_size=16, train_ratio=0.7, val_ratio=0.15, 
                                   test_ratio=0.15, random_state=42, num_workers=0, include_cosmo_params=False):
    """
    Create separate train, validation, and test dataloaders with proper splitting.
    
    Args:
        data_dir: Path to directory containing data files
        batch_size: Batch size for dataloaders
        train_ratio: Fraction of data for training (default: 0.7)
        val_ratio: Fraction of data for validation (default: 0.15)
        test_ratio: Fraction of data for testing (default: 0.15)
        random_state: Random seed for reproducibility
        num_workers: Number of worker processes for data loading
        include_cosmo_params: Whether to include cosmological parameters
        
    Returns:
        train_dataloader, val_dataloader, test_dataloader
    """
    # Create full dataset
    dataset = KSZDeltaSigmaDataset(data_dir, include_cosmo_params=include_cosmo_params)
    n_samples = len(dataset)
    
    # Generate indices
    indices = np.arange(n_samples)
    
    # Split into train and temp (val+test)
    train_indices, temp_indices = train_test_split(
        indices, 
        test_size=(val_ratio + test_ratio),
        random_state=random_state
    )
    
    # Split temp into val and test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1 - val_size),
        random_state=random_state
    )
    
    print(f"Dataset split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")
    
    # Create subset dataloaders with custom collate function
    def create_dataloader(subset_indices, shuffle):
        subset = Subset(dataset, subset_indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, 
                         num_workers=num_workers, collate_fn=custom_collate_fn)
    
    train_dl = create_dataloader(train_indices, shuffle=True)
    val_dl = create_dataloader(val_indices, shuffle=False)
    test_dl = create_dataloader(test_indices, shuffle=False)
    
    return train_dl, val_dl, test_dl
