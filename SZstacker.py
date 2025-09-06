from illustris_halo_profile_reader import *

import sys

from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

class SZMapStacker(readHaloProfiles):
    
    def __init__(self,
                 SB35_sim_index=0,
                 snapNum='090', 
                 chunkNum=0):
        """Initialize the SZMapStacker consistent with readHaloProfiles.

        Args:
            SB35_sim_index (int): Simulation index
            snapNum (str): Snapshot number
            chunkNum (int): Chunk number
        """
        
        # Initialize parent class (readHaloProfiles)
        super().__init__(SB35_sim_index, snapNum, chunkNum)
        
        # Read basic data like readHaloProfiles does
        self.read_header()
        self.read_particles() 
        self.read_catalog(plot=False)
        
        # Calculate ACT-like pixel resolution
        self.calc_theta_arcmin_ACT_nPixels(PixelSize=0.5)
        
        # Additional SZ-specific attributes
        self.maps = {}  # Store computed maps

    def makeMap(self, pType='ksz', nPixels=None, z=None, projection='xy', beamsize=1.6, save=False):
        """Create a map from the simulation data.

        Args:
            pType (str): The type of particle to use for the map. Either 'tSZ', 'kSZ', or 'tau'.
            z (float, optional): The redshift to use for the map. Defaults to None.
            projection (str, optional): The projection to use for the map. Defaults to 'xy'.
            beamsize (float, optional): The size of the beam to use for the map. Defaults to 1.6.
            save (bool, optional): Whether to save the map to disk. Defaults to False.
            load (bool, optional): Whether to load the map from disk. Defaults to True.
            pixelSize (float, optional): The size of the pixels in the map. Defaults to 0.5.
        """ 

        if nPixels is None:
            nPixels = self.nPixels
            arcminPerPixel = self.theta_arcmin / self.nPixels
            print(f"Using default nPixels={nPixels} for ACT-like resolution of {arcminPerPixel:.3f} arcmin/pixel")
        else:
            arcminPerPixel = self.theta_arcmin / nPixels
            print(f"Using nPixels={nPixels} for arcmin/pixel of {arcminPerPixel:.3f}")


        # Compute the map using makeField
        map_ = self.makeField(pType, nPixels=nPixels, projection=projection, save=False)

        # Convolve the map with a Gaussian beam
        if beamsize is not None:
            map_ = self.convolveMap(map_, beamsize, arcminPerPixel)

        # Save if requested
        if save:
            filename = f'{pType}_{nPixels}_{projection}_map.npy'
            filepath = self.save_profile_path + filename
            np.save(filepath, map_)
            print(f"Saved map to {filepath}")

        return map_

    def makeField(self, 
                  pType, 
                  nPixels=None, 
                  projection='xy', 
                  save=False):
        """Create a field from the simulation data using readHaloProfiles particle data.

        Args:
            pType (str): The type of particle to use for the map. Either 'tSZ', 'kSZ', or 'tau'.
            nPixels: Size of the output map in pixels. Defaults to self.nPixels.
            projection (str, optional): The projection to use. Defaults to 'xy'.
            save (bool, optional): Whether to save the field to disk. Defaults to False.
            load (bool, optional): Whether to load the field from disk. Defaults to True.
        """
        if nPixels is None:
            nPixels = self.nPixels

        # Define physical constants (from SZ_TNG repo)
        gamma = 5/3.  # Adiabatic Index
        k_B = 1.3807e-16  # erg/K, Boltzmann constant
        m_p = 1.6726e-24  # g, mass of proton
        unit_c = 1.e10  # TNG unit conversion
        X_H = 0.76  # primordial hydrogen fraction
        sigma_T = 6.6524587158e-29*1.e2**2  # cm^2, thomson cross section
        m_e = 9.10938356e-28  # g, electron mass
        c = 29979245800.  # cm/s, speed of light
        const = k_B*sigma_T/(m_e*c**2)  # constant for compton y computation
        kpc_to_cm = 3.0857e21  # cm per kpc
        solar_mass = 1.989e33  # g

        # Simulation parameters consistent with readHaloProfiles
        h = self.h
        unit_mass = 1.e10*(solar_mass/h)
        unit_dens = 1.e10*(solar_mass/h)/(kpc_to_cm/h)**3
        unit_vol = (kpc_to_cm/h)**3

        z = self.redshift
        a = self.scale_factor
        Lbox_hkpc = self.BoxSize * 1000  # Convert Mpc/h to kpc/h

        # Get gas particle data from readHaloProfiles
        gas_particles = self.particles[0]
        Co = gas_particles['pos'] * 1000  # Convert Mpc/h to kpc/h for consistency
        M = gas_particles['masses']  # Already in 1e10 Msun/h
        
        # Get additional gas properties using illustris_python
        # We need to load these separately since readHaloProfiles doesn't load them
        EA = il.snapshot.loadSubset(self.basePath, self.snapNum, 0, fields=['ElectronAbundance'])
        IE = il.snapshot.loadSubset(self.basePath, self.snapNum, 0, fields=['InternalEnergy'])
        D = il.snapshot.loadSubset(self.basePath, self.snapNum, 0, fields=['Density'])
        V = il.snapshot.loadSubset(self.basePath, self.snapNum, 0, fields=['Velocities'])

        # Calculate SZ quantities
        dV = M/D  # Volume in ckpc/h^3
        D *= unit_dens  # Convert to g/cm^3

        # Electron temperature, number density and velocity
        Te = (gamma - 1.)*IE/k_B * 4*m_p/(1 + 3*X_H + 4*X_H*EA) * unit_c  # K
        ne = EA*X_H*D/m_p  # cm^-3
        Ve = V*np.sqrt(a)  # km/s

        # Compute SZ signals
        pixel_area = (a*Lbox_hkpc*(kpc_to_cm/h)/nPixels)**2
        dY = const*(ne*Te*dV)*unit_vol/pixel_area  # Compton Y parameter
        b = sigma_T*(ne[:, None]*(Ve/c)*dV[:, None])*unit_vol/pixel_area  # kSZ signal
        tau = sigma_T*(ne*dV)*unit_vol/pixel_area  # Optical depth

        # Set up projection coordinates
        if projection == 'xy':
            coordinates = Co[:, :2]
        elif projection == 'xz':
            coordinates = Co[:, [0, 2]]
        elif projection == 'yz':
            coordinates = Co[:, 1:]
        else:
            raise NotImplementedError('Projection type not implemented: ' + projection)

        # Create 2D histogram
        gridSize = [nPixels, nPixels]
        minMax = [0, Lbox_hkpc]

        if pType == 'tSZ':
            result = binned_statistic_2d(coordinates[:, 0], coordinates[:, 1], values=dY, 
                                        statistic='sum', bins=gridSize, range=[minMax, minMax])
        elif pType == 'kSZ':
            # For kSZ, we sum over the line-of-sight velocity component
            if projection == 'xy':
                b_los = b[:, 2]  # z-component
            elif projection == 'xz':
                b_los = b[:, 1]  # y-component
            elif projection == 'yz':
                b_los = b[:, 0]  # x-component
            
            result = binned_statistic_2d(coordinates[:, 0], coordinates[:, 1], values=b_los, 
                                        statistic='sum', bins=gridSize, range=[minMax, minMax])
        elif pType == 'tau':
            result = binned_statistic_2d(coordinates[:, 0], coordinates[:, 1], values=tau, 
                                        statistic='sum', bins=gridSize, range=[minMax, minMax])
        else:
            raise ValueError('Particle type not recognized: ' + pType)
        
        field = result.statistic

        # Save if requested
        if save:
            filename = f'{pType}_{nPixels}_{projection}_field.npy'
            filepath = self.save_profile_path + filename
            np.save(filepath, field)
            print(f"Saved field to {filepath}")

        return field
    
    def stackMap(self, pType='kSZ', filterType='CAP', 
                 nPixels=None, beamsize=1.6,
                 n_bins=20, theta_min=0.04, theta_max=15.0,
                 projection='xy', save=False, 
                 plot=False):
        """Stack the map using CAP filter, consistent with get_stacked_profiles methodology.
    
        Args:
            pType (str): Particle type to stack ('kSZ', 'tSZ', 'tau').
            filterType (str): Filter type - only 'CAP' supported.
            nPixels (int): Number of pixels, defaults to ACT resolution.
            beamsize (float): Beam FWHM in arcmin.
            n_bins (int): Number of radial bins.
            r_min (float): Minimum radius in cMpc/h.
            r_max (float): Maximum radius in cMpc/h.
            projection (str): Projection direction ('xy', 'xz', 'yz').
            save (bool): Save results.
            load (bool): Load existing results.
            pixelSize (float): Pixel size in arcmin.
            plot (bool): Make plots.
    
        Returns:
            tuple: (radii, profiles) where profiles is (n_halos, n_bins) array
        """
        
        if filterType != 'CAP':
            raise ValueError("Only 'CAP' filter is supported in this function")
        
        # Use ACT resolution if nPixels not specified
        if nPixels is None:
            nPixels = self.nPixels
            
        # Create the map using makeMap (equivalent to make_2D_maps for SZ)
        field = self.makeMap(pType, z=self.redshift, projection=projection,  nPixels=nPixels, 
                            beamsize=beamsize, save=save)
        
        # Calculate pixel size in physical units
        MpcPerPixel = self.BoxSize / nPixels
        arcminPerPixel = self.theta_arcmin / nPixels
        
        radii = np.linspace(theta_min, theta_max, n_bins) #arcmins
        
        r_mids_array = 0.5 * (radii[1:] + radii[:-1])
        radii_Pixel = radii / arcminPerPixel
        
        # Use CAP filter
        filt_func = CAP
        
        # Do stacking over halos (same structure as get_stacked_profiles)
        profiles = []
        
        for j, haloPOS in enumerate(self.pos_g):
            # Convert halo position to pixels (only use x,y for 2D projection)
            if projection == 'xy':
                haloLoc = np.round(haloPOS[:2] / MpcPerPixel).astype(int)
            elif projection == 'xz':
                haloLoc = np.round(haloPOS[[0,2]] / MpcPerPixel).astype(int)
            elif projection == 'yz':
                haloLoc = np.round(haloPOS[1:] / MpcPerPixel).astype(int)
            else:
                raise ValueError(f"Unknown projection: {projection}")
            
            # Cut out region around halo
            cutout = cutout_2d_periodic(field, haloLoc, radii_Pixel[-1])
            
            # Create radial distance grid
            rr = radial_distance_grid(cutout, (-r_max, r_max))
            
            # Apply CAP filter at each radius
            profile = []
            for rad in radii:
                filt_result = filt_func(cutout, rr, rad)
                profile.append(filt_result)
            
            profile = np.array(profile)
            profiles.append(profile)
        
        profiles = np.array(profiles)
        
        # Save results if requested (same format as get_stacked_profiles)
        if save:
            save_data = {
                'profiles': profiles,
                'radii_arcmin': radii,
                'r_arcmin_mids_array': r_mids_array,
                'filter_type': filterType,
                'particle_type': pType,
                'projection': projection,
                'm_halos': self.mass_selected,
                'r_vir': np.array([self.R_200[self.index_h_sh[n]] for n in range(self.n_halos)]),
                'n_halos': self.n_halos,
                'mass_bin_string': self.mass_bin_string
            }
            
            filename = f"{pType}_{filterType}_profiles_nPixel{nPixels}_fwhm_{beamsize}_theta_lin{r_min}_{r_max}_nbins{n_bins}.npz"
            np.savez(self.save_profile_path + filename, **save_data)
            print(f"Saved stacked profiles to {filename}")
        
        # Plot results if requested
        if plot:
            self._plot_stacked_results(radii, profiles, pType, filterType)
        
        return radii, profiles
    
    def sz_main(self,
                nPixel=512,    #if None, use ACT pixel resolution 0.5 arcmin/pixel: self.nPixels
                     beamsize=1.6, #arcmin
                     n_bins=30, 
                     theta_min=1,  #arcmin
                     theta_max=6, #arcmin
                     plot=True,
                     save=True,
                    pType='kSZ', 
                    filterType='CAP', 
                    projection='xy'):
        print(f"Starting SZ stacking analysis for {pType}")
        t0 = time.time()
        
        # Run stacking analysis
        radii, profiles = self.stackMap(
            pType=pType, filterType=filterType, 
                 nPixels=nPixel, beamsize=beamsize,
                 n_bins=n_bins, theta_min=theta_min, theta_max=theta_max,
                 projection=projection, save=save, 
                 plot=plot
        )
     
        print(f"SZ stacking completed in {time.time() - t0:.2f} seconds")
        return radii, profiles
