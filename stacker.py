# !pip install illustris_python
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
import matplotlib

import time
import json
import pprint 

sys.path.append('../illustrisPython/')
import illustris_python as il

# print('Import Parameters from JSON')
# JSON Parameters:
# param_Dict = json.loads(sys.argv[1])
# locals().update(param_Dict)

# print('Parameters:')
# pprint.pprint(param_Dict)

'''
JSON Parameters
sim = 'TNG300-1' # 'TNG300' or 'TNG100' for boxsize, '-1', '-2' for resolution
pType = 'gas' # particle type; 'gas' or 'DM' or 'Stars'
snapshot = 99 # Redshift snapshot; currently only 99 (z=0) or 67 (z=0.5)
nPixels = 10000 # size of the 2D output box
'''

class SimulationStacker(object):

    def __init__(self, SB35_sim_index, snapshot, nPixels):

        self.simPath =f'/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/L50n512_SB35/SB35_{SB35_sim_index}'
        self.snapshot = snapshot
        self.nPixels = nPixels
        
        
        with h5py.File(il.snapshot.snapPath(self.simPath, self.snapshot), 'r') as f:
        
            header = dict(f['Header'].attrs.items())
        
        self.Lbox = header['BoxSize'] # kpc/h
        
        self.kpcPerPixel = self.Lbox / self.nPixels


    def makeField(self, pType):

        with h5py.File(il.snapshot.snapPath(self.simPath, self.snapshot), 'r') as f:
            header = dict(f['Header'].attrs.items())
        
        Lbox = header['BoxSize'] # kpc/h
        
        # Get all particles
        if pType =='gas':
            particles = il.snapshot.loadSubset(self.simPath, self.snapshot, pType, fields=['Masses','Coordinates'])
        elif pType == 'DM':
            particles = il.snapshot.loadSubset(self.simPath, self.snapshot, pType, fields=['ParticleIDs','Coordinates'])
        elif pType == 'Stars':
            particles = il.snapshot.loadSubset(self.simPath, self.snapshot, pType, fields=['Masses','Coordinates'])
        
        if pType == 'gas':
            coordinates = particles['Coordinates']
            masses= particles['Masses']
        elif pType == 'DM':
            coordinates = particles['Coordinates']
            IDs = particles['ParticleIDs']
            DM_mass = header['MassTable'][1]
        elif pType == 'Stars':
            coordinates = particles['Coordinates']
            masses= particles['Masses']
        else:
            raise NotImplementedError('Particle Type not implemented')
        
        xx = coordinates[:,0]
        yy = coordinates[:,1]
        
        gridSize = [self.nPixels, self.nPixels]
        minMax = [0, self.Lbox]
        
        t0 = time.time()
        if pType == 'gas':
            result = binned_statistic_2d(xx, yy, masses, 'sum', bins=gridSize, range=[minMax, minMax])
            field = result.statistic
        elif pType == 'DM':
            result = binned_statistic_2d(xx, yy, IDs, 'count', bins=gridSize, range=[minMax, minMax])
            field = result.statistic * DM_mass
        elif pType == 'Stars':
            result = binned_statistic_2d(xx, yy, masses, 'sum', bins=gridSize, range=[minMax, minMax])
            field = result.statistic

        return field
        
    def stackField(self, pType):

        field = self.makeField(pType)

        haloes = il.groupcat.loadHalos(self.simPath, self.snapshot)
        haloMass = haloes['GroupMass'] * 1e10 / 0.6774
        haloPos = haloes['GroupPos']

        mass_min, mass_max, _ = self.halo_ind(2)
        
        halo_mask = np.where(np.logical_and((haloMass > mass_min), (haloMass < mass_max)))[0]
        print(halo_mask.shape)
        R200 = haloes['Group_R_Crit200'][halo_mask].mean()
        R200_Pixel = R200 / self.kpcPerPixel


        # Do stacking
        i = 0
        profiles = []
        n_vir = 10 # number of virial radii to stack
        radii = np.linspace(0.2, 9, 25)
        
        profiles = []
        # profiles_m = []
        # profiles_s = []
    
        for j, haloID in enumerate(halo_mask):
        
            # Load the snapshot for gas and DM around that halo:
            haloLoc = np.round(haloPos[haloID] / self.kpcPerPixel).astype(int)[:2]
            cutout = self.cutout_2d_periodic(field, haloLoc, n_vir*R200_Pixel)

            rr = self.radial_distance_grid(cutout, (-n_vir, n_vir))
            
            profile = []
    
            for rad in radii:
                # delta_sig_gas = delta_sigma(result_gas.statistic, rr, rad)
                filt_result = self.total_mass(cutout, rr, rad)
                profile.append(filt_result)
        
        
            profile = np.array(profile)
            profiles.append(profile)
            # print(i)
            i += 1
            
        profiles = np.array(profiles)

        return radii, profiles
        


    def halo_ind(self, ind):
        if ind == 0:
            mass_min = 5e11 # solar masses
            mass_max = 1e12 # solar masses
            title_str = r'$5\times 10^{11} M_\odot < M_{\rm halo} < 10^{12} M_\odot$, '
        elif ind == 1:
            mass_min = 1e12 # solar masses
            mass_max = 1e13 # solar masses
            title_str = r'$1\times 10^{12} M_\odot < M_{\rm halo} < 10^{13} M_\odot$, '
        elif ind == 2:
            mass_min = 1e13 # solar masses
            mass_max = 1e14 # solar masses
            title_str = r'$1\times 10^{13} M_\odot < M_{\rm halo} < 10^{14} M_\odot$, '
        elif ind == 3:
            mass_min = 1e14 # solar masses
            mass_max = 1e19 # solar masses
            title_str = r'$M_{\rm halo} > 10^{14} M_\odot$, '
        else:
            print('Wrong ind')
        return mass_min, mass_max, title_str


    def total_mass(self, mass_grid, r_grid, r):
        '''
        Cumulative Mass Filter
        '''
        mass_tot = np.sum(mass_grid[r_grid<r])
        return mass_tot        

    def delta_sigma(self, mass_grid, r_grid, r, dr=0.1):
        '''
        Delta Sigma Filter, note that the amplitude of this filter is not necessarily
        correct
        '''
    
        mean_sigma = np.sum(mass_grid[r_grid<r]) / (np.pi*r**2)
    
        r_mask = np.logical_and((r_grid >= r), (r_grid < r+dr))
        sigma_value = np.sum(mass_grid[r_mask]) / (2*np.pi*r*dr)
    
        return mean_sigma - sigma_value

    def CAP(self, mass_grid, r_grid, r):
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


    def cutout_2d_periodic(self, array, center, length):
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
    
    def radial_distance_grid(self, array, bounds):
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

