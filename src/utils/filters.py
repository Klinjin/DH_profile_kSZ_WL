import numpy as np
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline

'''
This script contains many of the utility functions used for manipulation of 2D
maps for 
'''

################################################################################
# Functions for smoothing a density map on 2D with a Gaussian FWHM beam.
def fft_smoothed_map(D, fwhm_arcmin, pixel_size_arcmin):
    """
    Smooth a 2D map D with a Gaussian beam of FWHM (arcmin) using FFTs.
        Periodic boundary conditions are assumed (same as gaussian_filter(mode='wrap')).


    Args:
        D (np.ndarray): 2D map to smooth.
        fwhm_arcmin (float): Full width at half maximum of the Gaussian beam (in arcminutes).
        pixel_size_arcmin (float): Size of each pixel (in arcminutes).

    Returns:
        np.ndarray: Smoothed 2D map.

    TODO: 
        Test to make sure that this works as expected when compared to gaussian_smoothed_map
    """
    
    D = np.asarray(D, dtype=np.float32)
    n0, n1 = D.shape
    assert n0 == n1, "Map must be square for this implementation."
    n = n0

    # Build flat-sky multipole grid l_x, l_y (radian^-1)
    #TODO: this should be the physical rather than angular variant, incorrect
    theta_pix_rad = np.deg2rad(pixel_size_arcmin / 60.0) 
    lx = np.fft.fftfreq(n, d=theta_pix_rad) * 2.0 * np.pi
    ly = lx
    LX, LY = np.meshgrid(lx, ly, indexing='xy')
    ellsq = LX**2 + LY**2

    # Beam transfer function B_ell
    B_ell = gauss_beam(ellsq, fwhm_arcmin)  # uses radians internally

    # FFT, multiply, inverse FFT
    dfour = scipy.fft.fftn(D, workers=-1)
    dksmo = dfour * B_ell
    drsmo = np.real(scipy.fft.ifftn(dksmo, workers=-1))
    return drsmo
    
def gauss_beam(ellsq, fwhm):
    """
    Gaussian beam of size fwhm
    
    Args:
        ellsq: squared angular multipole term used in the Gaussian beam function's exponent. 
            In flat-sky this is defined as (l_x^2 + l_y^2), where l_x and l_y are the Fourier space frequencies in rad^-1.
        fwhm: Gaussian FWHM of the beam (in arcminutes)
        
    Returns:
        np.ndarray: The Gaussian beam transfer function.
    """
    tht_fwhm = np.deg2rad(fwhm/60.)
    return np.exp(-0.5*(tht_fwhm**2.)*(ellsq)/(8.*np.log(2.)))



################################################################################
# Functions for the various filters used on 2D cluster maps
    
def delta_sigma(mass_grid, r_grid, r, dr=0.1):
    '''
    Delta Sigma Filter, note that the amplitude of this filter is not necessarily
    correct
    '''

    mean_sigma = np.sum(mass_grid[r_grid<r]) / (np.pi*r**2)
    # mean_sigma = np.mean(mass_grid[r_grid<r]) 

    r_mask = np.logical_and((r_grid >= r), (r_grid < r+dr))
    sigma_value = np.sum(mass_grid[r_mask]) / (2*np.pi*r*dr)
    # sigma_value = np.mean(mass_grid[r_mask])

    return mean_sigma - sigma_value

def delta_sigma_kernel_map(
    mass_grid: np.ndarray,
    r_grid: np.ndarray,
    r: float,
    dr: float = 0.1,
    # pixsize: float = 1.0,
) -> float:
    """
    * `mass_grid` – 2‑D Σ map.
    * `r_grid`    – 2‑D array of radial distances (same shape as `mass_grid`).
    * `r`         – aperture radius.
    * `dr`        – thickness of the outer ring (R < r < R+dr).
    * `pixsize`   – linear size of one pixel in *same* units as `r`.

    The function builds the analytical compensated kernel and performs the
    dot‑product (a single `np.sum`). Setting `dr` small reproduces the mean‑over‑thin‑annulus
    estimator used in the original code; larger `dr` gives a thicker, lower‑noise ring.
    """

    if dr <= 0:
        raise ValueError("dr must be positive.")

    R_out = r + dr

    # Build compensated kernel analytically from r_grid
    kernel              = np.zeros_like(r_grid, dtype=float)
    kernel[r_grid < r]  = +1.0 / (np.pi * r**2)
    annulus             = (r_grid >= r) & (r_grid < R_out)
    kernel[annulus]     = -1.0 / (np.pi * (R_out**2 - r**2))
    # print(kernel.mean())
    kernel             -= kernel.mean()  # ensure ∫K dA = 0 numerically

    return float(np.sum(mass_grid * kernel) )#* pixsize**2)
    # return float(np.mean(mass_grid * kernel) * pixsize**2)

def total_mass(mass_grid, r_grid, r):
    '''
    Cumulative Mass Filter
    '''
    mass_tot = np.sum(mass_grid[r_grid<r])
    return mass_tot

def CAP(mass_grid, r_grid, r):
    '''
    Compensated Aperture Photometry (CAP) Filter, see papers on kSZ/tSZ stacking
    '''
    
    r1 = r * np.sqrt(2.)
    inDisk = 1.*(r_grid <= r)
    inRing = 1.*(r_grid > r)*(r_grid <= r1)
    inRing *= np.sum(inDisk) / np.sum(inRing) # Normalize the ring
    filterW = inDisk - inRing

    filtMap = np.sum(filterW * mass_grid)
    return filtMap


def DSigma_from_mass(r, radii_2D, M_2D, k=3):
    r = np.atleast_1d(r)

    if M_2D.ndim == 1:
        M_interp = InterpolatedUnivariateSpline(radii_2D, M_2D, k=k)
        dM_dr_interp = M_interp.derivative()
        return M_interp(r)/(np.pi*r**2) - dM_dr_interp(r)/(2*np.pi*r)

    elif M_2D.ndim == 2:
        result = []
        for i in range(M_2D.shape[1]):
            M_interp = InterpolatedUnivariateSpline(radii_2D, M_2D[:, i], k=k)
            dM_dr_interp = M_interp.derivative()
            dsigma = M_interp(r)/(np.pi*r**2) - dM_dr_interp(r)/(2*np.pi*r)
            result.append(dsigma)
        return np.stack(result, axis=-1)  # shape (n, l)

    else:
        raise ValueError("M_2D must be either a 1D or 2D array.")


################################################################################
# Functions for getting cutouts from 3D Cluster grids

def cutout_3d_periodic(array, center, length):
    """
    Returns a cubic cutout from a 3D array with periodic boundary conditions.

    Parameters:
    - array: 3D numpy array
    - center: tuple (x, y, z) center index
    - length: float or int, half-width of the cutout (will be rounded)

    Returns:
    - 3D numpy array cutout of shape (2*length+1, 2*length+1, 2*length+1)
    """
    length = int(round(length))
    x, y, z = center
    size = 2 * length + 1

    # Generate index ranges with wrapping
    x_indices = [(x + i) % array.shape[0] for i in range(-length, length + 1)]
    y_indices = [(y + j) % array.shape[1] for j in range(-length, length + 1)]
    z_indices = [(z + k) % array.shape[2] for k in range(-length, length + 1)]

    # Use np.ix_ to create a 3D index grid
    cutout = array[np.ix_(x_indices, y_indices, z_indices)]

    return cutout

def radial_distance_grid_3d(array, bounds):
    """
    array: 3D numpy array (only shape is used)
    bounds: tuple ((x_min, x_max), (y_min, y_max), (z_min, z_max)) representing physical bounds
    Returns a 3D array of radial distances from the origin.
    """
    nx, ny, nz = array.shape
    xyz_min, xyz_max = bounds

    # Generate coordinate values for each axis
    x_coords = np.linspace(xyz_min, xyz_max, nx)
    y_coords = np.linspace(xyz_min, xyz_max, ny)
    z_coords = np.linspace(xyz_min, xyz_max, nz)

    # Create meshgrid of coordinates
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

    # Calculate distances from the center (0,0,0)
    radial_distances = np.sqrt(X**2 + Y**2 + Z**2)

    return radial_distances


################################################################################
# Functions for getting cutouts from 2D Cluster maps
    

def cutout_2d(array, center, length):
    length = int(round(length))  # Round float to nearest int
    x, y = center

    # Check for out-of-bounds
    if (x - length < 0 or x + length >= array.shape[0] or
        y - length < 0 or y + length >= array.shape[1]):
        return None

    return array[x - length:x + length + 1, y - length:y + length + 1]

def cutout_2d_periodic(array, center, length):
    """
    Returns a square cutout from a 2D array with periodic boundary conditions.

    Parameters:
    - array: 2D numpy array
    - center: tuple (x, y) center index
    - length: float or int, half-width of the cutout (will be rounded)

    Returns:
    - 2D numpy array cutout of shape (2*length+1, 2*length+1)
    """
    length = int(round(length))
    x, y = center
    size = 2 * length + 1

    # Generate index ranges with wrapping
    row_indices = [(x + i) % array.shape[0] for i in range(-length, length + 1)]
    col_indices = [(y + j) % array.shape[1] for j in range(-length, length + 1)]

    # Use np.ix_ to create a 2D index grid
    cutout = array[np.ix_(row_indices, col_indices)]

    return cutout

def radial_distance_grid(array, bounds):
    """
    array: 2D numpy array (only shape is used)
    bounds: tuple ((x_min, x_max), (y_min, y_max)) representing physical bounds
    """
    rows, cols = array.shape
    xy_min, xy_max = bounds

    # Generate coordinate values for each axis
    x_coords = np.linspace(xy_min, xy_max, cols)
    y_coords = np.linspace(xy_min, xy_max, rows)

    # Create meshgrid of coordinates
    X, Y = np.meshgrid(x_coords, y_coords)

    # Calculate distances from the center (0,0)
    radial_distances = np.sqrt(X**2 + Y**2)
    
    return radial_distances

################################################################################
# Other Utils

def summarize_array_stats(arr, axis=1):
    """
    Compute the mean, first quartile (Q1), and third quartile (Q3) along axis 1 of a 2D array.

    Parameters:
    - arr: np.ndarray, shape (n, m)

    Returns:
    - mean: np.ndarray, shape (n,)
    - q1: np.ndarray, shape (n,)
    - q3: np.ndarray, shape (n,)
    """
    mean = np.mean(arr, axis=axis)
    q1 = np.percentile(arr, 25, axis=axis)
    q3 = np.percentile(arr, 75, axis=axis)
    return mean, q1, q3

def stack_haloes(field, filt_func, haloLocs, radii, rad_pixel, n_vir):
    combined_profiles = []
    for haloLoc in haloLocs:
        cutout = cutout_2d_periodic(field, haloLoc, n_vir * rad_pixel)
        profile = []
        r_grid = radial_distance_grid(cutout, (-n_vir, n_vir))
    
        for rad in radii:
            filt_result = filt_func(cutout, r_grid, rad)
            profile.append(filt_result)
        combined_profiles.append(np.array(profile))
    return np.array(combined_profiles)

def compute_radial_profiles(
    haloLocs,
    n_vir,
    R200_Pixel,
    radii,
    filt_func,
    gas_field=None,
    DM_field=None,
    star_field=None
):
    """
    Compute radial profiles for selected fields (gas, DM, stars) around each halo.

    Args:
        halo_mask: list of halo indices
        haloPos: array of halo positions
        kpcPerPixel: conversion factor
        n_vir: virial multiplier
        R200_Pixel: virial radius in pixels
        radii: list of radii to evaluate the profile
        filt_func: filter function to apply
        gas_field, DM_field, star_field: optional 2D fields

    Returns:
        Dictionary with keys 'gas', 'DM', and/or 'stars' containing lists of radial profiles.
    """
    results = {'gas': [], 'DM': [], 'Stars': []}

    for haloLoc in haloLocs:
        # haloLoc = np.round(haloPos[haloID] / kpcPerPixel).astype(int)[:2]

        cutout_g = cutout_2d_periodic(gas_field, haloLoc, n_vir * R200_Pixel) if gas_field is not None else None
        cutout_m = cutout_2d_periodic(DM_field, haloLoc, n_vir * R200_Pixel) if DM_field is not None else None
        cutout_s = cutout_2d_periodic(star_field, haloLoc, n_vir * R200_Pixel) if star_field is not None else None

        # Use the first available non-None cutout to generate the radial grid
        reference_cutout = next(c for c in (cutout_g, cutout_m, cutout_s) if c is not None)
        rr = radial_distance_grid(reference_cutout, (-n_vir, n_vir))

        if cutout_g is not None:
            profile_g = [filt_func(cutout_g, rr, rad) for rad in radii]
            results['gas'].append(np.array(profile_g))
        if cutout_m is not None:
            profile_m = [filt_func(cutout_m, rr, rad) for rad in radii]
            results['DM'].append(np.array(profile_m))
        if cutout_s is not None:
            profile_s = [filt_func(cutout_s, rr, rad) for rad in radii]
            results['Stars'].append(np.array(profile_s))

    # Convert non-empty lists to NumPy arrays
    return {k: np.array(v) for k, v in results.items() if v}