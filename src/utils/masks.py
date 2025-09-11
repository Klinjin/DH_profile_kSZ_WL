import numpy as np
import time


def precompute_offsets_3d(radius):
    """
    Precompute relative (dz, dy, dx) offsets within a sphere of the given radius.
    """
    r = int(np.ceil(radius))
    dz, dy, dx = np.meshgrid(
        np.arange(-r, r + 1),
        np.arange(-r, r + 1),
        np.arange(-r, r + 1),
        indexing='ij'
    )
    dist_sq = dz**2 + dy**2 + dx**2
    mask = dist_sq <= radius**2
    return np.stack((dz[mask], dy[mask], dx[mask]), axis=1)

def get_cutout_mask_3d(array, centers, radii):
    """
    For each center and corresponding radius, return a boolean mask of valid 3D indices within that radius.

    Parameters:
        array (np.ndarray): The 3D array (shape used for bounds).
        centers (List[Tuple[int, int, int]]): List of center coordinates (z, y, x).
        radii (List[float]): List of radii (same length as centers).

    Returns:
        numpy.ndarray: Boolean mask of the same shape as the input array.
    """
    _offset_cache = {}
    shape = np.array(array.shape)
    mask = np.zeros_like(array, dtype=bool)

    for center, radius in zip(centers, radii):
        center = np.array(center)

        if radius not in _offset_cache:
            _offset_cache[radius] = precompute_offsets_3d(radius)
        offsets = _offset_cache[radius]

        # Apply offsets and wrap with modulo
        candidate_indices = (center + offsets) % shape
        z, y, x = candidate_indices[:, 0], candidate_indices[:, 1], candidate_indices[:, 2]
        mask[z, y, x] = True

    return mask

def get_cutout_indices_3d(array, centers, radii):
    """
    For each center and corresponding radius, return a list of valid 3D indices within that radius.

    Parameters:
        array (np.ndarray): The 3D array (shape used for bounds).
        centers (List[Tuple[int, int, int]]): List of center coordinates (z, y, x).
        radii (List[float]): List of radii (same length as centers).

    Returns:
        List[np.ndarray]: List of arrays of indices for each center-radius pair.
    """
    _offset_cache = {}
    shape = np.array(array.shape)
    all_indices = []

    for center, radius in zip(centers, radii):
        center = np.array(center)

        if radius not in _offset_cache:
            _offset_cache[radius] = precompute_offsets_3d(radius)
        offsets = _offset_cache[radius]

        # Apply offsets and wrap with modulo
        candidate_indices = (center + offsets) % shape
        all_indices.append(candidate_indices)

    return all_indices

def sum_over_cutouts(array, all_indices):
    """
    Given a 3D array and a list of 3D index arrays, compute the sum of values
    in the array for each set of indices.

    Parameters:
        array (np.ndarray): The 3D input array.
        all_indices (List[np.ndarray]): List of index arrays.

    Returns:
        np.ndarray: Array of summed values, one per index set.
    """
    result = np.empty(len(all_indices), dtype=array.dtype)
    for i, inds in enumerate(all_indices):
        z, y, x = inds[:, 0], inds[:, 1], inds[:, 2]
        result[i] = array[z, y, x].sum()
    return result


def precompute_offsets_2d(radius):
    """
    Precompute relative (dy, dx) offsets within a circle of the given radius.
    """
    r = int(np.floor(radius))
    dy, dx = np.meshgrid(
        np.arange(-r, r + 1),
        np.arange(-r, r + 1),
        indexing='ij'
    )
    dist_sq = dy**2 + dx**2
    mask = dist_sq <= radius**2
    return np.stack((dy[mask], dx[mask]), axis=1)

def get_cutout_mask_2d(array, centers, radii):
    """
    For each center and radius, return a flat list of unique valid 2D indices within that radius.

    Parameters:
        array (np.ndarray): The 2D array (shape used for bounds).
        centers (List[Tuple[int, int]]): List of center coordinates (row, col).
        radii (List[float]): List of radii (same length as centers).

    Returns:
        DEPRECIATED - List[Tuple[int, int]]: Flat list of all unique valid indices across all centers.
        numpy.ndarray: Boolean mask of the same shape as the input array.
    """
    _offset_cache_2d = {}
    shape = np.array(array.shape)
    unique_indices = set()

    for center, radius in zip(centers, radii):
        center = np.array(center)

        # Cache and reuse offset grid
        if radius not in _offset_cache_2d:
            _offset_cache_2d[radius] = precompute_offsets_2d(radius)
        offsets = _offset_cache_2d[radius]

        # Add offsets to center
        candidate_indices = center + offsets

        # Check bounds
        in_bounds = np.all((candidate_indices >= 0) & (candidate_indices < shape), axis=1)
        valid = candidate_indices[in_bounds]

        unique_indices.update(map(tuple, valid))

    list_indices = list(unique_indices)
    arr_ind = np.array(list_indices)
    indices = (arr_ind[:, 0], arr_ind[:, 1])
    mask = np.zeros_like(array, dtype=bool)
    mask[indices] = True
    return mask

    