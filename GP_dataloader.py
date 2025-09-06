import numpy as np
import jax.numpy as jnp
import pandas as pd
import torch

def getSims(sim_indices, filterType='CAP', ptype='gas'):
    '''
    Get the radial profiles with observation-specific filters for a list of simulation indices.
    '''
    print(f'Getting {ptype} profiles with {filterType} filter for {len(sim_indices)} simulations...')
    mass_halos = []
    profiles_ptype = []
    param_halos = []
    pkratio_halos = []
    for i, id in enumerate(sim_indices):
        Henry_profiles_sim =  np.load(f'/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/L50n512_SB35/SB35_{id}/data/'+ f"Henry_profiles_gas_dm_star_bh_nPixel1000_R_lin0.04_2.5_log15_nbins20.npz")

        r_bins =  Henry_profiles_sim['r_bins']
        [m_min, m_max] = Henry_profiles_sim['m_halos_range']
        # mass_halos.extend(Henry_profiles_sim['m_halos'])
        
        if filterType == 'cumulative':
            Henry_profiles = Henry_profiles_sim['profiles'][2]
        elif filterType == 'CAP':
            Henry_profiles = Henry_profiles_sim['profiles'][1]
        elif filterType == 'dsigma':
            Henry_profiles = Henry_profiles_sim['profiles'][0]
        profiles_g, profiles_m, profiles_s, profiles_bh = Henry_profiles

        if ptype == 'gas':
            profiles = profiles_g
        elif ptype == 'dm':
            profiles = profiles_m
        elif ptype == 'star':
            profiles = profiles_s
        elif ptype == 'bh':
            profiles = profiles_bh
        elif ptype == 'total':
            profiles = profiles_g + profiles_m + profiles_s + profiles_bh
        elif ptype =='baryon':
            profiles = profiles_g + profiles_s + profiles_bh
        else:
            raise ValueError(f"Invalid ptype: {ptype}. Must be one of ['gas', 'dm', 'star', 'bh', 'total', 'baryon'].")

        profiles_ptype.extend(profiles)
        mass_halos.extend(np.linspace(m_min, m_max, len(profiles)))

        param_sim = getParams(id)
        k, PkRatio = getPkRatio(id)
        param_halos.extend(np.repeat(param_sim[np.newaxis, :], len(profiles), axis=0))
        pkratio_halos.extend(np.repeat(PkRatio, len(profiles), axis=0))


    print(f'Finished getting profiles in {len(profiles_ptype)} halos.')
    return  jnp.array(r_bins), jnp.array(profiles_ptype), jnp.array(mass_halos), jnp.array(param_halos), jnp.array(k), jnp.array(pkratio_halos)

def getProfiles(sim_indices, filterType='CAP', ptype='gas'):
    '''
    Get the radial profiles with observation-specific filters for a list of simulation indices.
    '''
    print(f'Getting {ptype} profiles with {filterType} filter for {len(sim_indices)} simulations...')
    profiles_ptype = []
    for i, id in enumerate(sim_indices):
        Henry_profiles_sim =  np.load(f'/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/L50n512_SB35/SB35_{id}/data/'+ f"Henry_profiles_gas_dm_star_bh_nPixel1000_R_lin0.04_2.5_log15_nbins20.npz")

        r_bins =  Henry_profiles_sim['r_bins']

        
        if filterType == 'cumulative':
            Henry_profiles = Henry_profiles_sim['profiles'][2]
        elif filterType == 'CAP':
            Henry_profiles = Henry_profiles_sim['profiles'][1]
        elif filterType == 'dsigma':
            Henry_profiles = Henry_profiles_sim['profiles'][0]
        profiles_g, profiles_m, profiles_s, profiles_bh = Henry_profiles

        if ptype == 'gas':
            profiles = profiles_g
        elif ptype == 'dm':
            profiles = profiles_m
        elif ptype == 'star':
            profiles = profiles_s
        elif ptype == 'bh':
            profiles = profiles_bh
        elif ptype == 'total':
            profiles = profiles_g + profiles_m + profiles_s + profiles_bh
        elif ptype =='baryon':
            profiles = profiles_g + profiles_s + profiles_bh
        else:
            raise ValueError(f"Invalid ptype: {ptype}. Must be one of ['gas', 'dm', 'star', 'bh', 'total', 'baryon'].")

        profiles_ptype.extend(profiles)

    print(f'Finished getting profiles in {len(profiles_ptype)} halos.')
    return  jnp.array(r_bins), jnp.array(profiles_ptype)

def getProfilesParamsTensor(theta, filterType='CAP', ptype='gas'):
    '''
    Get the radial profiles with observation-specific filters for a list of simulation indices.
    Input and output are torch Tensors.
    '''
    print(f'Getting {ptype} profiles with {filterType} filter given parameters...')

    # Ensure theta is a torch Tensor
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, dtype=torch.float32)
    
    # If theta is 1D, reshape to (1, n_params)
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)

    params_matrix = np.load("data/camels_params_matrix.npy")
    params_matrix_torch = torch.tensor(params_matrix, dtype=torch.float32)

    sim_indices = []
    for i in range(theta.shape[0]):
        # Compute distances in torch
        distances = torch.norm(params_matrix_torch - theta[i], dim=1)
        nearest_index = torch.argmin(distances).item()
        sim_indices.append(nearest_index)

    profiles_ptype = []

    for i, id in enumerate(sim_indices):
        Henry_profiles_sim =  np.load(f'/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/L50n512_SB35/SB35_{id}/data/'+ f"Henry_profiles_gas_dm_star_bh_nPixel1000_R_lin0.04_2.5_log15_nbins20.npz")

        r_bins =  Henry_profiles_sim['r_bins']

        if filterType == 'cumulative':
            Henry_profiles = Henry_profiles_sim['profiles'][2]
        elif filterType == 'CAP':
            Henry_profiles = Henry_profiles_sim['profiles'][1]
        elif filterType == 'dsigma':
            Henry_profiles = Henry_profiles_sim['profiles'][0]
        profiles_g, profiles_m, profiles_s, profiles_bh = Henry_profiles

        if ptype == 'gas':
            profiles = profiles_g
        elif ptype == 'dm':
            profiles = profiles_m
        elif ptype == 'star':
            profiles = profiles_s
        elif ptype == 'bh':
            profiles = profiles_bh
        elif ptype == 'total':
            profiles = profiles_g + profiles_m + profiles_s + profiles_bh
        elif ptype =='baryon':
            profiles = profiles_g + profiles_s + profiles_bh
        else:
            raise ValueError(f"Invalid ptype: {ptype}. Must be one of ['gas', 'dm', 'star', 'bh', 'total', 'baryon'].")

        #profiles_ptype.extend(profiles)
        profiles_ptype.append(np.median(profiles, axis=0))

    #print(f'Finished getting profiles in {len(profiles_ptype)} halos.')
    print(f'Finished getting profiles in {len(profiles_ptype)} simulations.')
    # Convert outputs to torch Tensors
    r_bins_tensor = torch.tensor(r_bins, dtype=torch.float32)
    profiles_ptype_tensor = torch.tensor(np.array(profiles_ptype), dtype=torch.float32)
    return r_bins_tensor, profiles_ptype_tensor

def getParams(sim_indices):
    '''
    Get the CAMELS cosmological and astrophysical parameters (35 in total) for a list of simulation indices.
    '''
    params_matrix = np.load("data/camels_params_matrix.npy")
    return params_matrix[sim_indices]

def getPkRatio(sim_indices):
    '''
    Get the field-level baryonic power spectrum suppression from cross-correlation/auto-correlation ratios
    for a list of simulation indices.
    '''
    k_vals = []
    Pk_ratios = []
    for i, id in enumerate([sim_indices]):
        Henry_Pk_sim= np.load(f'/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/L50n512_SB35/SB35_{id}/data/'
                                      + 'baryon_suppression_fields_nPixel512.npz')
        
        # Extract data
        k = Henry_Pk_sim['k']
        PX_dm_tot = Henry_Pk_sim['PX_dm_tot']
        P_dm = Henry_Pk_sim['P_dm']
        P_tot= Henry_Pk_sim['P_tot']

        k_vals.append(k)
        Pk_ratios.append((PX_dm_tot*jnp.sqrt(P_tot)/jnp.sqrt(P_dm)))

    return jnp.array(k_vals), jnp.array(Pk_ratios)

def getParamsFiducial():
    '''
    Get the CAMELS cosmological and astrophysical parameters (35 in total) for the fiducial simulation.
    '''
    try:
        # Read CSV file, skipping header
        df = pd.read_csv("data/SB35_param_minmax.csv", header=0)
        
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