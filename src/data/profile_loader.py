import numpy as np
import jax.numpy as jnp
import pandas as pd
import torch
import warnings
from src.config.config import (
    get_profile_file_path, get_power_spectrum_file_path,
    PARAM_MATRIX_PATH, PARAM_MINMAX_PATH,
    validate_filter_type, validate_particle_type,
    FILTER_INDICES, N_COSMO_PARAMS
)

def load_simulation_data(sim_indices, filterType='CAP', ptype='gas',
                        include_params=True, include_pk=True, include_mass=True,
                        aggregate_method='extend'):
    """
    Unified function to load simulation data with configurable outputs.
    
    Args:
        sim_indices: List of simulation indices to load
        filterType: Filter type ('CAP', 'cumulative', 'dsigma')
        ptype: Particle type ('gas', 'dm', 'star', 'bh', 'total', 'baryon')
        include_params: Whether to include cosmological parameters
        include_pk: Whether to include power spectrum ratios
        include_mass: Whether to include halo masses
        aggregate_method: How to aggregate profiles ('extend' or 'median')
    
    Returns:
        Tuple containing (r_bins, profiles, [mass], [params], [k], [pk_ratios])
        Optional returns depend on include_* parameters
    """
    # Validate inputs
    filterType = validate_filter_type(filterType)
    ptype = validate_particle_type(ptype)
    
    print(f'Getting {ptype} profiles with {filterType} filter for {len(sim_indices)} simulations...')
    
    # Initialize output containers
    profiles_ptype = []
    mass_halos = [] if include_mass else None
    param_halos = [] if include_params else None
    pkratio_halos = [] if include_pk else None
    k_vals = None
    
    for i, sim_id in enumerate(sim_indices):
        # Load profile data
        profile_file = get_profile_file_path(sim_id)
        Henry_profiles_sim = np.load(profile_file)
        
        r_bins = Henry_profiles_sim['r_bins']
        [m_min, m_max] = Henry_profiles_sim['m_halos_range']
        
        # Extract profiles based on filter type
        filter_idx = FILTER_INDICES[filterType]
        Henry_profiles = Henry_profiles_sim['profiles'][filter_idx]
        profiles_g, profiles_m, profiles_s, profiles_bh = Henry_profiles
        
        # Select particle type
        profiles = _select_particle_profiles(profiles_g, profiles_m, profiles_s, profiles_bh, ptype)
        
        # Aggregate profiles
        if aggregate_method == 'extend':
            profiles_ptype.extend(profiles)
            if include_mass:
                mass_halos.extend(np.linspace(m_min, m_max, len(profiles)))
        elif aggregate_method == 'median':
            profiles_ptype.append(np.median(profiles, axis=0))
            if include_mass:
                mass_halos.append(np.median(np.linspace(m_min, m_max, len(profiles))))
        
        # Load additional data if requested
        if include_params:
            param_sim = getParams(sim_id)
            if aggregate_method == 'extend':
                param_halos.extend(np.repeat(param_sim[np.newaxis, :], len(profiles), axis=0))
            else:
                param_halos.append(param_sim)
                
        if include_pk:
            k, PkRatio = getPkRatio(sim_id)
            if k_vals is None:
                k_vals = k
            if aggregate_method == 'extend':
                pkratio_halos.extend(np.repeat(PkRatio, len(profiles), axis=0))
            else:
                pkratio_halos.append(PkRatio[0])  # Take first (should be same for all halos in sim)
    
    # Print completion message
    if aggregate_method == 'extend':
        print(f'Finished getting profiles in {len(profiles_ptype)} halos.')
    else:
        print(f'Finished getting profiles in {len(profiles_ptype)} simulations.')
    
    # Prepare return values
    result = [jnp.array(r_bins), jnp.array(profiles_ptype)]
    
    if include_mass:
        result.append(jnp.array(mass_halos))
    if include_params:
        result.append(jnp.array(param_halos))
    if include_pk:
        # Ensure k_vals maintains array structure even for single simulation
        if len(k_vals) == 1:
            k_vals = k_vals[0]  # Extract the k array, not make it scalar
        result.extend([jnp.array(k_vals), jnp.array(pkratio_halos)])
        
    return tuple(result)


def load_simulation_mean_profiles(sim_indices, filterType='CAP', ptype='gas',
                                include_params=True, include_mass=True, include_pk=True, aggregate_func='mean'):
    """
    Load simulation data with mean profiles per simulation.
    
    This function returns mean profiles for each simulation instead of all individual
    halo profiles, resulting in shape (n_sims, n_bins) instead of (n_halos*n_sims, n_bins).
    
    Args:
        sim_indices: List of simulation indices to load
        filterType: Filter type ('CAP', 'cumulative', 'dsigma')
        ptype: Particle type ('gas', 'dm', 'star', 'bh', 'total', 'baryon')
        include_params: Whether to include cosmological parameters
        include_pk: Whether to include power spectrum ratios
        aggregate_func: Aggregation function ('mean', 'median', 'std')
    
    Returns:
        Tuple containing (r_bins, mean_profiles, [params], [k], [pk_ratios])
        - mean_profiles: shape (n_sims, n_bins) - one profile per simulation
        - params: shape (n_sims, n_params) if included
        - pk_ratios: shape (n_sims, n_k) if included
    """
    # Validate inputs
    filterType = validate_filter_type(filterType)
    ptype = validate_particle_type(ptype)
    
    print(f'Getting mean {ptype} profiles with {filterType} filter for {len(sim_indices)} simulations...')
    
    # Initialize output containers
    mean_profiles = []
    param_sims = [] if include_params else None
    pkratio_sims = [] if include_pk else None
    mass_halos = [] if include_mass else None
    k_vals = None
    
    for i, sim_id in enumerate(sim_indices):
        # Load profile data
        profile_file = get_profile_file_path(sim_id)
        Henry_profiles_sim = np.load(profile_file)
        
        r_bins = Henry_profiles_sim['r_bins']
        [m_min, m_max] = Henry_profiles_sim['m_halos_range']
        
        # Extract profiles based on filter type
        filter_idx = FILTER_INDICES[filterType]
        Henry_profiles = Henry_profiles_sim['profiles'][filter_idx]
        profiles_g, profiles_m, profiles_s, profiles_bh = Henry_profiles
        
        # Select particle type
        profiles = _select_particle_profiles(profiles_g, profiles_m, profiles_s, profiles_bh, ptype)
        
        # Compute aggregated profile for this simulation
        if aggregate_func == 'mean':
            agg_profile = np.mean(profiles, axis=0)
        elif aggregate_func == 'median':
            agg_profile = np.median(profiles, axis=0)
        elif aggregate_func == 'std':
            agg_profile = np.std(profiles, axis=0)
        else:
            raise ValueError(f"Unknown aggregate_func: {aggregate_func}. Use 'mean', 'median', or 'std'")
        
        mean_profiles.append(agg_profile)
        if include_mass:
            mass_halos.append(np.mean([m_min, m_max]))
        
        # Load additional data if requested
        if include_params:
            param_sim = getParams(sim_id)
            param_sims.append(param_sim)
                
        if include_pk:
            k, PkRatio = getPkRatio(sim_id)
            if k_vals is None:
                k_vals = k
            pkratio_sims.append(PkRatio[0])  # Take first (should be same for all halos in sim)
    
    # Print completion message
    print(f'Finished getting mean profiles from {len(mean_profiles)} simulations.')
    print(f'Output shape: ({len(mean_profiles)}, {len(r_bins)}) vs individual halo shape would be (n_halos*{len(sim_indices)}, {len(r_bins)})')
    
    # Prepare return values
    result = [jnp.array(r_bins), jnp.array(mean_profiles)]
    if include_mass:
        result.append(jnp.array(mass_halos))
    if include_params:
        result.append(jnp.array(param_sims))
    if include_pk:
        # Ensure k_vals maintains array structure
        if len(k_vals) == 1:
            k_vals = k_vals[0]  # Extract the k array, not make it scalar
        result.extend([jnp.array(k_vals), jnp.array(pkratio_sims)])
        
    return tuple(result)


def _select_particle_profiles(profiles_g, profiles_m, profiles_s, profiles_bh, ptype):
    """Helper function to select profiles based on particle type."""
    if ptype == 'gas':
        return profiles_g
    elif ptype == 'dm':
        return profiles_m
    elif ptype == 'star':
        return profiles_s
    elif ptype == 'bh':
        return profiles_bh
    elif ptype == 'total':
        return profiles_g + profiles_m + profiles_s + profiles_bh
    elif ptype == 'baryon':
        return profiles_g + profiles_s + profiles_bh


def getSims(sim_indices, filterType='CAP', ptype='gas'):
    """
    Get the radial profiles with observation-specific filters for a list of simulation indices.
    
    Legacy wrapper around load_simulation_data for backward compatibility.
    """
    return load_simulation_data(
        sim_indices, filterType=filterType, ptype=ptype,
        include_params=True, include_pk=True, include_mass=True,
        aggregate_method='extend'
    )

def getProfiles(sim_indices, filterType='CAP', ptype='gas'):
    """
    Get the radial profiles with observation-specific filters for a list of simulation indices.
    
    Legacy wrapper around load_simulation_data for backward compatibility.
    """
    r_bins, profiles_ptype = load_simulation_data(
        sim_indices, filterType=filterType, ptype=ptype,
        include_params=False, include_pk=False, include_mass=False,
        aggregate_method='extend'
    )
    return r_bins, profiles_ptype

def getProfilesParamsTensor(theta, filterType='CAP', ptype='gas'):
    """
    Get the radial profiles with observation-specific filters for given parameters.
    Input and output are torch Tensors.
    
    This function finds the nearest simulation parameters and returns median profiles.
    """
    print(f'Getting {ptype} profiles with {filterType} filter given parameters...')

    # Ensure theta is a torch Tensor
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, dtype=torch.float32)
    
    # If theta is 1D, reshape to (1, n_params)
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)

    # Load parameter matrix and find nearest simulations
    params_matrix = np.load(PARAM_MATRIX_PATH)
    params_matrix_torch = torch.tensor(params_matrix, dtype=torch.float32)

    sim_indices = []
    for i in range(theta.shape[0]):
        # Compute distances in torch
        distances = torch.norm(params_matrix_torch - theta[i], dim=1)
        nearest_index = torch.argmin(distances).item()
        sim_indices.append(nearest_index)

    # Load profiles using unified function
    r_bins, profiles_ptype = load_simulation_data(
        sim_indices, filterType=filterType, ptype=ptype,
        include_params=False, include_pk=False, include_mass=False,
        aggregate_method='median'
    )
    
    # Convert outputs to torch Tensors
    r_bins_tensor = torch.tensor(r_bins, dtype=torch.float32)
    profiles_ptype_tensor = torch.tensor(profiles_ptype, dtype=torch.float32)
    return r_bins_tensor, profiles_ptype_tensor

def getParams(sim_indices):
    """
    Get the CAMELS cosmological and astrophysical parameters (35 in total) for a list of simulation indices.
    """
    params_matrix = np.load(PARAM_MATRIX_PATH)
    return params_matrix[sim_indices]

def getPkRatio(sim_indices):
    '''
    Get the field-level baryonic power spectrum suppression from cross-correlation/auto-correlation ratios
    for a list of simulation indices.
    '''
    k_vals = []
    Pk_ratios = []
    for i, id in enumerate([sim_indices]):
        pk_file = get_power_spectrum_file_path(id)
        Henry_Pk_sim = np.load(pk_file)
        
        # Extract data
        k = Henry_Pk_sim['k']
        PX_dm_tot = Henry_Pk_sim['PX_dm_tot']
        P_dm = Henry_Pk_sim['P_dm']
        P_tot= Henry_Pk_sim['P_tot']

        k_vals.append(k[k<10])  # Keep k < 10 h/Mpc
        Pk_ratios.append((PX_dm_tot*jnp.sqrt(P_tot)/jnp.sqrt(P_dm))[k<10])

    return jnp.array(k_vals), jnp.array(Pk_ratios)

def getParamsFiducial():
    '''
    Get the CAMELS cosmological and astrophysical parameters (35 in total) for the fiducial simulation.
    '''
    try:
        # Read CSV file, skipping header
        df = pd.read_csv(PARAM_MINMAX_PATH, header=0)
        
        # Extract column A (param_names) and column D (fiducial_values)
        # Assuming 0-indexed: A=0, B=1, C=2, D=3
        param_names = df.iloc[:, 0].tolist()  # Column A
        fiducial_values = df.iloc[:, 3].values  # Column D
        maxdiff = df.iloc[:, 1].values # Column B
        minVal = df.iloc[:, 4].values # Column E
        maxVal = df.iloc[:, 5].values # Column F

        print(f"Read {len(param_names)} parameters from data/SB35_param_minmax.csv")

        return param_names, fiducial_values, maxdiff, minVal, maxVal

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None, None