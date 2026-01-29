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
                        aggregate_func='extend'):
    """
    Unified function to load simulation data with configurable outputs and aggregation.
    
    Args:
        sim_indices: List of simulation indices to load
        filterType: Filter type ('CAP', 'cumulative', 'dsigma')
        ptype: Particle type ('gas', 'dm', 'star', 'bh', 'total', 'baryon')
        include_params: Whether to include cosmological parameters
        include_pk: Whether to include power spectrum ratios
        include_mass: Whether to include halo masses
        aggregate_func: How to aggregate profiles:
            - 'extend': Return all individual halo profiles (shape: n_halos*n_sims, n_bins)
            - 'mean': Return mean profile per simulation (shape: n_sims, n_bins)
            - 'median': Return median profile per simulation (shape: n_sims, n_bins)
            - 'std': Return std profile per simulation (shape: n_sims, n_bins)
    
    Returns:
        Tuple containing (r_bins, profiles, [mass], [params], [k], [pk_ratios])
        Optional returns depend on include_* parameters
        Profile shape depends on aggregate_func
    """
    # Validate inputs
    filterType = validate_filter_type(filterType)
    ptype = validate_particle_type(ptype)
    
    if aggregate_func not in ['extend', 'mean', 'median', 'std']:
        raise ValueError(f"Unknown aggregate_func: {aggregate_func}. Use 'extend', 'mean', 'median', or 'std'")
    
    agg_name = aggregate_func if aggregate_func != 'extend' else 'individual halo'
    print(f'Getting {agg_name} {ptype} profiles with {filterType} filter for {len(sim_indices)} simulations...')
    
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
        
        # Aggregate profiles based on function
        if aggregate_func == 'extend':
            profiles_ptype.extend(profiles)
            if include_mass:
                mass_halos.extend(np.linspace(m_min, m_max, len(profiles)))
        elif aggregate_func == 'mean':
            agg_profile = np.mean(profiles, axis=0)
            profiles_ptype.append(agg_profile)
            if include_mass:
                mass_halos.append(np.mean([m_min, m_max]))
        elif aggregate_func == 'median':
            agg_profile = np.median(profiles, axis=0)
            profiles_ptype.append(agg_profile)
            if include_mass:
                mass_halos.append(np.median(np.linspace(m_min, m_max, len(profiles))))
        elif aggregate_func == 'std':
            agg_profile = np.std(profiles, axis=0)
            profiles_ptype.append(agg_profile)
            if include_mass:
                mass_halos.append(np.std(np.linspace(m_min, m_max, len(profiles))))
        
        # Load additional data if requested
        if include_params:
            param_sim = getParams(sim_id)
            if aggregate_func == 'extend':
                param_halos.extend(np.repeat(param_sim[np.newaxis, :], len(profiles), axis=0))
            else:
                param_halos.append(param_sim)
                
        if include_pk:
            k, PkRatio = getPkRatio(sim_id)
            if k_vals is None:
                k_vals = k
            if aggregate_func == 'extend':
                pkratio_halos.extend(np.repeat(PkRatio, len(profiles), axis=0))
            else:
                pkratio_halos.append(PkRatio[0])  # Take first (should be same for all halos in sim)
    
    # Print completion message
    if aggregate_func == 'extend':
        print(f'Finished getting profiles in {len(profiles_ptype)} halos.')
        print(f'Output shape: ({len(profiles_ptype)}, {len(r_bins)})')
    else:
        print(f'Finished getting {aggregate_func} profiles from {len(profiles_ptype)} simulations.')
        print(f'Output shape: ({len(profiles_ptype)}, {len(r_bins)})')
    
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
    Legacy wrapper for backward compatibility.
    
    This function provides the same interface as the old load_simulation_mean_profiles
    but now uses the unified load_simulation_data function.
    """
    return load_simulation_data(
        sim_indices=sim_indices, filterType=filterType, ptype=ptype,
        include_params=include_params, include_pk=include_pk, include_mass=include_mass,
        aggregate_func=aggregate_func
    )


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
        aggregate_func='extend'
    )

def getProfiles(sim_indices, filterType='CAP', ptype='gas'):
    """
    Get the radial profiles with observation-specific filters for a list of simulation indices.
    
    Legacy wrapper around load_simulation_data for backward compatibility.
    """
    r_bins, profiles_ptype = load_simulation_data(
        sim_indices, filterType=filterType, ptype=ptype,
        include_params=False, include_pk=False, include_mass=False,
        aggregate_func='extend'
    )
    return r_bins, profiles_ptype

def getMeanProfilesParamsTensor(theta, filterType='CAP', ptype='gas'):
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

    # Load profiles using unified function with mean aggregation
    data_tuple = load_simulation_data(
        sim_indices, filterType=filterType, ptype=ptype,
        include_params=False, include_pk=False, include_mass=False,
        aggregate_func='mean'
    )
    r_bins_tensor = torch.tensor(np.array(data_tuple[0]), dtype=torch.float32)
    profiles_ptype_tensor = torch.tensor(np.array(data_tuple[1]), dtype=torch.float32)

    return r_bins_tensor, profiles_ptype_tensor

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

    # Load profiles using unified function with median aggregation
    data_tuple = load_simulation_data(
        sim_indices, filterType=filterType, ptype=ptype,
        include_params=False, include_pk=False, include_mass=False,
        aggregate_func='median'
    )
    
    # Convert outputs to torch Tensors
    r_bins_tensor = torch.tensor(data_tuple[0], dtype=torch.float32)
    profiles_ptype_tensor = torch.tensor(data_tuple[1], dtype=torch.float32)
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