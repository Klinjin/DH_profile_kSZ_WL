import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from scipy.integrate import quad
from scipy.ndimage import gaussian_filter
import illustris_python as il
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import time
from tqdm import tqdm
import Pk_library as PKL
import MAS_library as MASL
from mpi4py import MPI
from numba import njit, prange
import pickle

from src.utils.filters import *
from src.utils.masks import *
import matplotlib

#### 3D density field and power spectrum utils ####
def get_3D_density_field(pos, mass, 
                         BoxSize,   #Mpc/h ; size of box
                         grid=512 #the 3D field will have grid x grid x grid voxels
                         ): 
    '''
    This function generates a 3D density field using the Mass Assignment Scheme (MAS) library.
    It creates a random set of particle positions and masses, and then computes the density field.
    '''

    # density field parameters
    MAS     = 'CIC'  #mass-assigment scheme
    verbose = False   #print information on progress

    # define 3D density field
    delta = np.zeros((grid,grid,grid), dtype=np.float32)  # use float64 for better precision

    # construct 3D density field
    MASL.MA(pos.astype(np.float32), delta, BoxSize, MAS, W=mass.astype(np.float32), verbose=verbose)

    # at this point, delta contains the effective gas mass in each voxel
    # now compute overdensity and density constrast
    delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0

    return delta

def compute_pk_3D(field, 
                  field_b=None,
                  BoxSize=50,   #Mpc/h ; size of box
                    grid=512,
                  threads = 1 # maximum number of cores per node
                  ):
    """
    Compute the 3D power spectrum for 3D fields.
    field: numpy array or torch tensor, shape (N, N, N), DM field if field_b is not None
    field_b: optional, baryonic field, for cross power spectrum, same shape as field
    boxsize: physical size of the box
    Returns: k, P(k)
    """

    assert field.ndim == 3, "Input field must have shape (N, N, N)"

    MAS = 'CIC'
    verbose = False
    axis    = 0

    k_nyq = np.pi * grid/BoxSize 

    Pk3D = PKL.Pk(field, BoxSize, axis=axis, MAS=MAS, threads=threads, verbose=verbose)
    k = Pk3D.k3D
    Pk_val = Pk3D.Pk[:, 0]  # monopole

    if field_b is not None:
        Pkb =  PKL.Pk(field_b, BoxSize, axis=axis, MAS=MAS, threads=threads, verbose=verbose)
        Pkb_val = Pkb.Pk[:, 0] 
        Pk3D = PKL.XPk([field, field_b], BoxSize, axis=axis, MAS=[MAS, MAS], threads=threads)
        k = Pk3D.k3D
        Pk0_X = Pk3D.XPk[:,0,0] #monopole of 1-2 cross P(k)
        Pk0_X /= np.sqrt(Pk_val*Pkb_val)  # monopole
        Pk_val = Pk0_X
    
    return k[k<=k_nyq], Pk_val[k<=k_nyq]

def compute_corr_3D(field, 
                  field_b=None,
                  BoxSize=50,   #Mpc/h ; size of box
                    grid=512,
                  threads = 1 # maximum number of cores per node
                  ):
    """
    Compute the 3D power spectrum for 3D fields.
    field: numpy array or torch tensor, shape (N, N, N), DM field if field_b is not None
    field_b: optional, baryonic field, for cross power spectrum, same shape as field
    boxsize: physical size of the box
    Returns: k, P(k)
    """

    assert field.ndim == 3, "Input field must have shape (N, N, N)"

    MAS = 'CIC'
    verbose = False
    axis    = 0

    CF     = PKL.Xi(field, BoxSize, MAS, axis, threads)
    
    # get the attributes
    r      = CF.r3D      #radii in Mpc/h
    xi0    = CF.xi[:,0]  #correlation function (monopole)

    if field_b is not None:
        CCF = PKL.XXi(field, field_b, BoxSize, axis=axis, MAS=[MAS, MAS], threads=threads)
        r      = CCF.r3D      #radii in Mpc/h
        xi0   = CCF.xi[:,0] 
    
    return r[r<=BoxSize], xi0[r<=BoxSize]

def take_shell_mass(a):
    '''
    a: (n_halo, n_bins)

    Returns the mass in each shell, where the first column is the mass at the r_min and the rest are the differences between consecutive shells.
    '''
    return np.concatenate([a[:, 0:1], np.diff(a, axis=1)], axis=1)



class readHaloProfiles:
    def __init__(self, 
                 SB35_sim_index = 0, 
                 snapNum = '090',
                chunkNum=0
                 ):

        self.sim_name = f'IllustrisTNG L50n512 SB35_{SB35_sim_index}_snap{snapNum}'

        
        self.snapNum = int(snapNum)
        self.basePath = f'/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/L50n512_SB35/SB35_{SB35_sim_index}'
        self.save_figure_path = f'/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/L50n512_SB35/SB35_{SB35_sim_index}/figs/'
        self.save_profile_path = f'/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/L50n512_SB35/SB35_{SB35_sim_index}/data/'
        self.catalog = f'/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/L50n512_SB35/SB35_{SB35_sim_index}/groups_{snapNum}/fof_subhalo_tab_{snapNum}.{chunkNum}.hdf5'
        self.snapshot  = f'/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/L50n512_SB35/SB35_{SB35_sim_index}/snapdir_{snapNum}/snap_{snapNum}.{chunkNum}.hdf5'


#### reading in simulations ####

    def read_header(self):
        f = h5py.File(self.snapshot, 'r')

        # read different attributes of the header
        self.BoxSize      = f['Header'].attrs[u'BoxSize']/1e3 #Mpc/h
        self.redshift     = f['Header'].attrs[u'Redshift'] 
        self.h            = f['Header'].attrs[u'HubbleParam'] #0.6711 100 km/s/(Mpc/h)
        self.Masses       = f['Header'].attrs[u'MassTable']*1e10 #Msun/h
        self.Np           = f['Header'].attrs[u'NumPart_Total']
        self.Omega_m      = f['Header'].attrs[u'Omega0']
        self.Omega_L      = f['Header'].attrs[u'OmegaLambda']
        self.Omega_b      = f['Header'].attrs[u'OmegaBaryon']
        self.scale_factor = f['Header'].attrs[u'Time'] #scale factor array([1.])
        self.C_OVER_HUBBLE = 2997.9

        f.close()

    def read_particles(self):
        particles = {}
        for ptype in [0,1,4,5]:
            particles[ptype] = {
                    'pos': None , 
                    'masses': None,  
                    'ne': None,
                    'id':None,
                    'nH': None
                }
        
        for ptype in [0,1,4,5]:
            particles[ptype]['pos'] =  il.snapshot.loadSubset(self.basePath, self.snapNum , ptype, fields=['Coordinates'], float32=True)/1e3
            particles[ptype]['id'] = il.snapshot.loadSubset(self.basePath, self.snapNum , ptype, fields=['ParticleIDs'])
            if ptype == 1:
                particles[ptype]['masses'] = np.full((self.Np[1],), self.Masses[1])
            else:
                particles[ptype]['masses'] = il.snapshot.loadSubset(self.basePath, self.snapNum , ptype, fields=['Masses'], float32=True)*1e10
        
            if ptype == 0:
                particles[ptype]['ne'] = il.snapshot.loadSubset(self.basePath, self.snapNum , ptype, fields=['ElectronAbundance'])
                particles[ptype]['nH'] = il.snapshot.loadSubset(self.basePath, self.snapNum , ptype, fields=['GFM_Metals'])[:,0]
                particles[ptype]['MetalAbundance'] = il.snapshot.loadSubset(self.basePath, self.snapNum , ptype, fields=['GFM_Metals'])
                particles[ptype]['nH+'] = 1- il.snapshot.loadSubset(self.basePath, self.snapNum , ptype, fields=['NeutralHydrogenAbundance'])[:]
                particles[ptype]['density'] = 1- il.snapshot.loadSubset(self.basePath, self.snapNum , ptype, fields=['Density'])[:]
        self.particles = particles

        X_H = particles[0]['MetalAbundance'][:,0]      
        X_He = particles[0]['MetalAbundance'][:,1]    

        m_u = 1.6605e-24  # grams
        # Number of electrons per gram
        self.n_e_per_gram = (X_H * 1/1 + X_He * 2/4) / m_u
        # Mass per electron
        self.m_baryon_per_electron = 1 / (X_H + 0.5 * X_He) * m_u # same result

        # Build KDTree for each particle type if not already built
        # kdtrees = {}
        # kdtrees_projected = {}
        # for ptype in [0, 1, 4, 5]:
        #     if particles[ptype]['pos'] is not None:
        #         kdtrees[ptype] = KDTree(particles[ptype]['pos'])
        #         kdtrees_projected[ptype] = KDTree(particles[ptype]['pos'][:,:2])
        # self.kdtrees_projected = kdtrees_projected
        # self.kdtrees = kdtrees
        
        #return particles, kdtrees
    
    def read_catalog(self, plot=False):
        fields = ['GroupFirstSub', 'GroupPos', 'GroupVel', 'GroupMass', 'GroupLen', 'GroupNsubs', 'Group_R_Mean200', 'Group_R_Crit500']
        halos = il.groupcat.loadHalos(self.basePath,self.snapNum ,fields=fields)
        sub_h = halos['GroupFirstSub'][:]
        pos_h  = halos['GroupPos'][:]/1e3           #center of mass of the FOF group in Mpc/h
        vel_h  = halos['GroupVel'][:]/self.scale_factor  #velocities in km/s
        mass_h = halos['GroupMass'][:]*1e10         #masses in Msun/h
        self.len_h  = halos['GroupLen'][:]
        self.Nsub_h = halos['GroupNsubs'][:] 
        self.R_200 =  halos['Group_R_Mean200'][:]/1e3 
        R_500 =  halos['Group_R_Crit500'][:]/1e3 

        w = np.where(sub_h >= 0) # value of -1 indicates no subhalo in this group
        central_subhalo_ids = sub_h[w] #total central halos: 113903

        fields = ['SubhaloMass','SubhaloMassInMaxRadType','SubhaloVmaxRad', 'SubhaloMassInMaxRad','SubhaloPos','SubhaloGrNr','SubhaloLen']
        subhalos = il.groupcat.loadSubhalos(self.basePath, self.snapNum , fields=fields)
        mass_msun   = subhalos['SubhaloMass'][central_subhalo_ids] * 1e10   

        M_star =  subhalos['SubhaloMassInMaxRadType'][central_subhalo_ids,3]*1e10 #stellar masses in Msun/h
        M_wind =  subhalos['SubhaloMassInMaxRadType'][central_subhalo_ids,4]*1e10
        M_gas =  subhalos['SubhaloMassInMaxRadType'][central_subhalo_ids,0]*1e10
        M_bh =  subhalos['SubhaloMassInMaxRadType'][central_subhalo_ids,5]*1e10
        M_dm =  subhalos['SubhaloMassInMaxRadType'][central_subhalo_ids,1]*1e10
        M_tot =  subhalos['SubhaloMassInMaxRad'][central_subhalo_ids]*1e10 

        mass_threshold = np.sort(mass_msun)[::-1][int(2.4e-2*self.BoxSize**3)]  #5.4e-4 for DESI Main LRG Samples (68 halos) 2.4e-2 (3000 halos)
        mass_condition = mass_msun >= mass_threshold # condition for massive halos
        central_subhalo_ids_massive_bins = central_subhalo_ids[mass_condition] #massive halos: 113
        self.pos_g  =  subhalos['SubhaloPos'][central_subhalo_ids_massive_bins]/1e3         #position of the central subhalos (the most massive galaxy in each group), in Mpc/h   (11113,)
        self.R_g =  subhalos['SubhaloVmaxRad'][central_subhalo_ids_massive_bins]/1e3 #𝑐M𝑝𝑐/ℎ 
        self.len_sh     =  subhalos['SubhaloLen'][central_subhalo_ids_massive_bins] 
        self.index_h_sh =  subhalos['SubhaloGrNr'][central_subhalo_ids_massive_bins]
        self.mass_threshold = mass_threshold
        self.mass_selected = mass_msun[mass_condition]

        self.mass_bin_string = rf'{central_subhalo_ids_massive_bins.shape[0]} halos {mass_threshold:.1e} \sim {mass_msun.max():.1e} M_\odot$'
        self.central_subhalo_ids_massive_bins = central_subhalo_ids_massive_bins
        self.n_halos = self.central_subhalo_ids_massive_bins.shape[0]

        if plot:
            fig = plt.figure(figsize=(10,8))
            ax = fig.add_subplot(111)
            ax.plot(mass_msun[mass_msun >=1e13], (M_gas[mass_msun >=1e13]+M_star[mass_msun >=1e13] + M_bh[mass_msun >=1e13] + M_wind[mass_msun >=1e13]) / M_tot[mass_msun >=1e13], 
                    '.', color = 'r',
                    markersize=1,
                    label=self.mass_bin_string)
            ax.plot(mass_msun[mass_msun <1e13], (M_gas[mass_msun <1e13]+M_star[mass_msun <1e13] + M_bh[mass_msun <1e13] + M_wind[mass_msun <1e13]) / M_tot[mass_msun <1e13], 
                    '.', 
                    markersize=0.5)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Total Mass Central Subhalos [$M_\odot$]')
            ax.set_ylabel('Radial Baryonic Mass Fraction')
            plt.legend()
            plt.title(f'{self.sim_name}: Radial Baryonic Mass Fraction vs. Halo Mass')
            plt.savefig(self.save_figure_path + f'{self.sim_name}_halo_mass_baryon_fraction.png', dpi=300)
            plt.close()
        #return self.pos_g, self.R_200

#### MPI parallelized halo analysis ####

    def get_field_baryon_suppression(self,
                                     particles=None,
                                     grid=512,
                                     plot=False,
                                     save=False,
                                     threads=None):
        if particles is None:
            particles = self.particles
        
        delta_dm = get_3D_density_field(particles[1]['pos'], particles[1]['masses'], self.BoxSize, grid)
        # Concatenate positions and masses for gas (0), stars (4), and black holes (5)
        pos_tot = np.concatenate([particles[0]['pos'], particles[1]['pos'], particles[4]['pos'], particles[5]['pos']], axis=0)
        mass_tot = np.concatenate([particles[0]['masses'], particles[1]['masses'], particles[4]['masses'], particles[5]['masses']], axis=0)
        delta_tot = get_3D_density_field(pos_tot, mass_tot, self.BoxSize, grid)

        # Calculate the baryon suppression field in Fourier space
        kX, PX_dm_tot = compute_pk_3D(delta_dm, delta_tot, self.BoxSize, grid, threads=256)
        k, P_dm = compute_pk_3D(delta_dm, BoxSize=self.BoxSize, grid=grid, threads=256)
        k, P_tot = compute_pk_3D(delta_tot, BoxSize=self.BoxSize, grid=grid, threads=256)


        rX, XX_dm_tot = compute_corr_3D(delta_dm, delta_tot, self.BoxSize, grid, threads=256)
        r, X_dm = compute_corr_3D(delta_dm,   BoxSize=self.BoxSize,  grid=grid, threads=256)
        r, X_tot = compute_corr_3D(delta_tot,   BoxSize=self.BoxSize,  grid=grid, threads=256)

        if plot:
            print('Plotting Baryon Suppression Field...')
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(k[k<=15], (PX_dm_tot)[k<=15], color='blue')
            ax.set_xscale('log')
            ax.set_xlabel('k (h/Mpc)')
            ax.set_ylabel(r'$R_{dm \times total} (k)$')
            ax.set_title(f'{self.sim_name}: Baryon Suppression Power Spectrum')
            ax.grid(True)
            if save:
                fig.savefig(self.save_figure_path + f'{self.sim_name}_baryon_suppression_fourier.png', dpi=300)
            plt.show()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(r[r<=10], (XX_dm_tot/X_dm)[r<=10],  color='blue')
            ax.set_xscale('log')
            ax.set_xlabel('r (Mpc/h)')
            ax.set_ylabel(r'$\xi_{dm \times total}/\xi_{dm} (r)$')
            ax.set_title(f'{self.sim_name}: Baryon Suppression Correlation Function')
            ax.grid(True)
            if save:
                fig.savefig(self.save_figure_path + f'{self.sim_name}_baryon_suppression_real.png', dpi=300)
            plt.show()
        if save:
            print('Saving Baryon Suppression Field...')
            np.savez(self.save_profile_path + f"baryon_suppression_fields_nPixel{grid}.npz",
                     k=k,
                     PX_dm_tot=PX_dm_tot,
                     P_dm=P_dm,
                     P_tot=P_tot,
                     r=r,
                     XX_dm_tot=XX_dm_tot,
                     X_dm=X_dm,
                     X_tot=X_tot)
    
    #@njit(parallel=True)
    def mpi_halo_analysis_kdtree(self, 
                        pos_g=None,
                        R_200=None,
                        particles=None, 
                        kdtrees=None,
                        kdtrees_projected=None,
                        n_bins=10, 
                        r_min=0.04, #cMpc/h
                        r_max=15,  #cMpc/h
                        plot=False,
                        save=False):
        """
        MPI-parallelized halo analysis using KDTree
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

     
        attrs = {
                'pos_g': self.pos_g,
                'R_200': self.R_200,
                'particles': self.particles,
                'n_halos': self.n_halos,
                'index_h_sh': self.index_h_sh,
                'Omega_b': self.Omega_b,
                'Omega_m': self.Omega_m,
                'mass_bin_string': self.mass_bin_string,
                'sim_name': self.sim_name,
            }

  
        pos_g = attrs['pos_g']
        R_200 = attrs['R_200']
        particles = attrs['particles']
        n_halos = attrs['n_halos']
        index_h_sh = attrs['index_h_sh']
        Omega_b = attrs['Omega_b']
        Omega_m = attrs['Omega_m']
        # mass_bin_string and sim_name are not used below, so no need to assign them

        # Rebuild KDTree objects on each process
        kdtrees = {}
        kdtrees_projected = {}
        for ptype in [0, 1, 4, 5]:
            if particles[ptype]['pos'] is not None:
                kdtrees[ptype] = KDTree(particles[ptype]['pos'])
                kdtrees_projected[ptype] = KDTree(particles[ptype]['pos'][:, :2])

        # Now, do not use any self attributes below this point!
        halo_vir_radius = np.array([R_200[index_h_sh[n]] for n in range(n_halos)])
        if rank == 0:
            print(f'Average virial radius is {halo_vir_radius.mean():.2e}cMpc/h')

        # Set radial bins
        r_bins = np.logspace(np.log10(r_min), np.log10(r_max), n_bins+1)
        r_mids_array = 0.5 * (r_bins[1:] + r_bins[:-1])
        r_bins = np.tile(r_bins, (n_halos, 1)).T  # shape: (n_bins+1, n_halos)



        if rank == 0:
            print(f"Starting MPI analysis with {size} processes")
            print(f"Processing {n_halos} halos with {n_bins} radial bins")

        # Distribute halos across MPI processes
        halos_per_rank = n_halos // size
        remainder = n_halos % size

        # Calculate start and end indices for this rank
        if rank < remainder:
            start_halo = rank * (halos_per_rank + 1)
            end_halo = start_halo + halos_per_rank + 1
        else:
            start_halo = rank * halos_per_rank + remainder
            end_halo = start_halo + halos_per_rank

        local_halos = end_halo - start_halo

        if rank == 0:
            print(f"Halo distribution: {halos_per_rank} base + {remainder} extra")
            for r in range(size):
                if r < remainder:
                    s = r * (halos_per_rank + 1)
                    e = s + halos_per_rank + 1
                else:
                    s = r * halos_per_rank + remainder
                    e = s + halos_per_rank
                print(f"  Rank {r}: halos {s} to {e-1} ({e-s} halos)")

        # Initialize results storage for this rank
        local_results = {
                'baryon_mass_profile': np.zeros((local_halos, n_bins)),
                'gas_mass_profile': np.zeros((local_halos, n_bins)),
                'total_mass_profile': np.zeros((local_halos, n_bins)),
                'CAP_baryon_mass_profile': np.zeros((local_halos, n_bins)),
                'CAP_gas_mass_profile': np.zeros((local_halos, n_bins)),
                'CAP_total_mass_profile': np.zeros((local_halos, n_bins)),
                'dS_total_mass_profile': np.zeros((local_halos, n_bins)),
                'dS_baryon_mass_profile': np.zeros((local_halos, n_bins)),
                'sum_radial_baryon_fraction': np.empty((local_halos, n_bins)),
                'sum_radial_gas_fraction': np.empty((local_halos, n_bins)),
                'CAP_baryon_fraction': np.empty((local_halos, n_bins)),
                'CAP_gas_fraction': np.empty((local_halos, n_bins)),
                'dS_baryon_fraction': np.empty((local_halos, n_bins)),
                'halo_indices': np.arange(start_halo, end_halo),
                'r_bins': r_bins,
                'halo_vir_radius': halo_vir_radius,
                'r_mids_array': r_mids_array,
            }

        # Process each particle type
        for ptype in [0, 1, 4, 5]:
            if rank == 0:
                print(f"Processing particle type {ptype}...")

            pos = particles[ptype]['pos']
            pos_2D = particles[ptype]['pos'][:,:2]
            mass = particles[ptype]['masses']
            
            # Build KDTree (each process builds the same tree)
            tree_3d = kdtrees[ptype]
            tree_2d = kdtrees_projected[ptype]
            
            
            # Do this for all local halos at once:
            halo_positions_3d = pos_g[start_halo:end_halo]
            halo_positions_2d = pos_g[start_halo:end_halo, :2]

            max_radius = r_bins[-1, start_halo:end_halo]
            max_radius_2d = max_radius * np.sqrt(2.)

            # Query all halos at once (returns a list of arrays, one per halo)
            indices_3d_list = tree_3d.query_ball_point(halo_positions_3d, max_radius)
            indices_2d_list = tree_2d.query_ball_point(halo_positions_2d, max_radius_2d)

            # Process halos assigned to this rank
            for local_idx, (indices_3d, indices_2d) in enumerate(zip(indices_3d_list, indices_2d_list)):
                if len(indices_3d) == 0:
                    continue
                
                halo_pos_3d = halo_positions_3d[local_idx]
                halo_pos_2d = halo_positions_2d[local_idx]

                # Get nearby particles data
                nearby_pos_3d = pos[indices_3d]
                nearby_pos_2d = pos_2D[indices_2d]
                nearby_mass_3d = mass[indices_3d]
                nearby_mass_2d = mass[indices_2d]
                
                # Compute distances
                dists_3d = np.linalg.norm(nearby_pos_3d - halo_pos_3d, axis=1)
                dists_2d = np.linalg.norm(nearby_pos_2d - halo_pos_2d, axis=1)
                
                # Process all radial bins for this halo
                for i in prange(n_bins):
                    r_outer = r_bins[i+1, local_idx]
                    
                    # 3D analysis
                    in_bin_3d = dists_3d < r_outer
                    local_results['total_mass_profile'][local_idx, i] += np.sum(nearby_mass_3d[in_bin_3d])
                    
                    # 2D projected analysis
                    in_bin_projected = dists_2d < r_outer
                    shell_bin_projected = ((dists_2d >= r_outer) & 
                                        (dists_2d < r_outer * np.sqrt(2.)))
                    
                    local_results['dS_total_mass_profile'][local_idx, i] += np.sum(nearby_mass_2d[in_bin_projected])
                    local_results['CAP_total_mass_profile'][local_idx, i] += np.sum(nearby_mass_2d[in_bin_projected]) - np.sum(nearby_mass_2d[shell_bin_projected])
                    
                    if ptype in [0, 4, 5]:
                        local_results['baryon_mass_profile'][local_idx, i] += np.sum(nearby_mass_3d[in_bin_3d])
                        local_results['dS_baryon_mass_profile'][local_idx, i] += np.sum(nearby_mass_2d[in_bin_projected])
                        local_results['CAP_baryon_mass_profile'][local_idx, i] += np.sum(nearby_mass_2d[in_bin_projected]) - np.sum(nearby_mass_2d[shell_bin_projected])
                        if ptype == 0:  # gas particles
                            local_results['gas_mass_profile'][local_idx, i] += np.sum(nearby_mass_3d[in_bin_3d])
                            local_results['CAP_gas_mass_profile'][local_idx, i] += np.sum(nearby_mass_2d[in_bin_projected]) - np.sum(nearby_mass_2d[shell_bin_projected])
            # Synchronize after each particle type
            comm.Barrier()
            if rank == 0:
                print(f"Completed particle type {ptype} on all ranks")
   
        # Calculate baryon fractions using local_results
        b_mass = local_results['baryon_mass_profile']
        t_mass = local_results['total_mass_profile']
        g_mass = local_results['gas_mass_profile']
        cap_b_mass = local_results['CAP_baryon_mass_profile']
        cap_t_mass = local_results['CAP_total_mass_profile']
        cap_g_mass = local_results['CAP_gas_mass_profile']
        ds_b_mass = local_results['dS_baryon_mass_profile']
        ds_t_mass = local_results['dS_total_mass_profile']

        # Excess surface mass density 
        area_outer = np.pi * r_bins[1:, start_halo:end_halo]**2
        area_inner = np.pi * r_bins[:-1, start_halo:end_halo]**2
        area_shell = area_outer - area_inner
        dS_total_profile = ds_t_mass / area_outer.T - (take_shell_mass(ds_t_mass) / area_shell.T)
        dS_baryon_profile = ds_b_mass / area_outer.T - (take_shell_mass(ds_b_mass) / area_shell.T)
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            sum_radial_baryon_fraction = np.divide(b_mass, t_mass, out=np.zeros_like(b_mass), where=t_mass!=0) / (Omega_b / Omega_m)
            CAP_baryon_fraction = np.divide(cap_b_mass, cap_t_mass, out=np.zeros_like(cap_b_mass), where=cap_t_mass!=0) / (Omega_b / Omega_m)
            dS_baryon_fraction = np.divide(dS_baryon_profile, dS_total_profile, out=np.zeros_like(ds_b_mass), where=ds_t_mass!=0) /(Omega_b / Omega_m)
            sum_radial_gas_fraction = np.divide(g_mass, t_mass, out=np.zeros_like(g_mass), where=t_mass!=0) / (Omega_b / Omega_m)
            CAP_gas_fraction = np.divide(local_results['CAP_gas_mass_profile'], cap_t_mass, out=np.zeros_like(local_results['CAP_gas_mass_profile']), where=cap_t_mass!=0) / (Omega_b / Omega_m)

        # Store in local_results for this ptype
        local_results['sum_radial_baryon_fraction'] = sum_radial_baryon_fraction
        local_results['CAP_baryon_fraction'] = CAP_baryon_fraction
        local_results['dS_baryon_fraction'] = dS_baryon_fraction
        local_results['sum_radial_gas_fraction'] = sum_radial_gas_fraction
        local_results['CAP_gas_fraction'] = CAP_gas_fraction
        
        return local_results

    def gather_results(self, local_results, comm):
        """
        Gather results from all MPI processes to root
        """
        rank = comm.Get_rank()
        
        # Gather all local results to root
        all_results = comm.gather(local_results, root=0)
        
        if rank == 0:
            # Combine results from all ranks
            combined_results = {}
            
            # Combine all results into a single dictionary (no [ptype] separation)
            combined_results = {
                'baryon_mass_profile': [],
                'gas_mass_profile': [],
                'total_mass_profile': [],
                'CAP_baryon_mass_profile': [],
                'CAP_gas_mass_profile': [],
                'CAP_total_mass_profile': [],
                'dS_total_mass_profile': [],
                'dS_baryon_mass_profile': [],
                'sum_radial_baryon_fraction': [],
                'sum_radial_gas_fraction': [],
                'CAP_baryon_fraction': [],
                'CAP_gas_fraction': [],
                'dS_baryon_fraction': [],
                'halo_indices': [],
                'halo_vir_radius': local_results['halo_vir_radius'],
                'r_mids_array': local_results['r_mids_array'],
            }

            for rank_results in all_results:
                combined_results['baryon_mass_profile'].append(rank_results['baryon_mass_profile'])
                combined_results['gas_mass_profile'].append(rank_results['gas_mass_profile'])
                combined_results['total_mass_profile'].append(rank_results['total_mass_profile'])
                combined_results['CAP_baryon_mass_profile'].append(rank_results['CAP_baryon_mass_profile'])
                combined_results['CAP_gas_mass_profile'].append(rank_results['CAP_gas_mass_profile'])
                combined_results['CAP_total_mass_profile'].append(rank_results['CAP_total_mass_profile'])
                combined_results['dS_total_mass_profile'].append(rank_results['dS_total_mass_profile'])
                combined_results['dS_baryon_mass_profile'].append(rank_results['dS_baryon_mass_profile'])
                combined_results['sum_radial_baryon_fraction'].append(rank_results['sum_radial_baryon_fraction'])
                combined_results['sum_radial_gas_fraction'].append(rank_results['sum_radial_gas_fraction'])
                combined_results['CAP_baryon_fraction'].append(rank_results['CAP_baryon_fraction'])
                combined_results['CAP_gas_fraction'].append(rank_results['CAP_gas_fraction'])
                combined_results['dS_baryon_fraction'].append(rank_results['dS_baryon_fraction'])
                combined_results['halo_indices'].append(rank_results['halo_indices'])

            # Concatenate arrays
            combined_results['baryon_mass_profile'] = np.concatenate(combined_results['baryon_mass_profile'], axis=0)
            combined_results['gas_mass_profile'] = np.concatenate(combined_results['gas_mass_profile'], axis=0)
            combined_results['total_mass_profile'] = np.concatenate(combined_results['total_mass_profile'], axis=0)
            combined_results['CAP_baryon_mass_profile'] = np.concatenate(combined_results['CAP_baryon_mass_profile'], axis=0)
            combined_results['CAP_gas_mass_profile'] = np.concatenate(combined_results['CAP_gas_mass_profile'], axis=0)
            combined_results['CAP_total_mass_profile'] = np.concatenate(combined_results['CAP_total_mass_profile'], axis=0)
            combined_results['dS_total_mass_profile'] = np.concatenate(combined_results['dS_total_mass_profile'], axis=0)
            combined_results['dS_baryon_mass_profile'] = np.concatenate(combined_results['dS_baryon_mass_profile'], axis=0)
            combined_results['sum_radial_baryon_fraction'] = np.concatenate(combined_results['sum_radial_baryon_fraction'], axis=0)
            combined_results['sum_radial_gas_fraction'] = np.concatenate(combined_results['sum_radial_gas_fraction'], axis=0)
            combined_results['CAP_baryon_fraction'] = np.concatenate(combined_results['CAP_baryon_fraction'], axis=0)
            combined_results['CAP_gas_fraction'] = np.concatenate(combined_results['CAP_gas_fraction'], axis=0)
            combined_results['dS_baryon_fraction'] = np.concatenate(combined_results['dS_baryon_fraction'], axis=0)
            combined_results['halo_indices'] = np.concatenate(combined_results['halo_indices'])
            return combined_results
        else:
            return None

    def read_all(self,  
                 n_bins=30, 
                r_min=0.04, #cMpc/h
                r_max=15, 
                 plot=False, 
                 save=False):
        self.read_header()
        self.read_particles()
        self.read_catalog(plot=plot)
        self.get_field_baryon_suppression(plot=plot, save=save)

    def mpi_main(self,  
                 n_bins=10, 
                r_min=0.04, #cMpc/h
                r_max=15, 
                 plot=False, 
                 save=False):
                
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        start_time = time.time()

        print("Loading data on all process...")
        self.read_header()
        self.read_particles()
        self.read_catalog(plot=False)
        if rank == 0:
            self.get_field_baryon_suppression(plot=plot, save=save)
            print(f"Data loaded. Starting MPI analysis with {size} processes...")

        # Run MPI analysis
        local_results = self.mpi_halo_analysis_kdtree(plot=plot, save=save, n_bins=n_bins, r_min=r_min, r_max=r_max)
        
        # Gather results
        if rank == 0:
            print("Gathering results from all processes...")
        
        final_results = self.gather_results(local_results, comm)
        
        # Save results on root
        if rank == 0:
            end_time = time.time()
            print(f"Analysis completed in {end_time - start_time:.2f} seconds")
            
            np.savez(self.save_profile_path + 'halo_analysis_results.npz', **final_results)
            print("Results saved to halo_analysis_results.npz")
           
            # Print summary
            for ptype in [0, 1, 4, 5]:
                print(f"Particle type {ptype}: processed {len(final_results['halo_indices'])} halos")

            
            if plot:
                print('plotting...')
                sum_radial_baryon_fraction = final_results['sum_radial_baryon_fraction']
                CAP_baryon_fraction = final_results['CAP_baryon_fraction']
                dS_baryon_fraction = final_results['dS_baryon_fraction']
                baryon_mass_profile = final_results['baryon_mass_profile']
                total_mass_profile = final_results['total_mass_profile']
                gas_mass_profile = final_results['gas_mass_profile']
                CAP_baryon_mass_profile = final_results['CAP_baryon_mass_profile']
                dS_total_profile = final_results['dS_total_mass_profile']
                halo_vir_radius = final_results['halo_vir_radius']
                r_mids_array = final_results['r_mids_array']

            # Convert cumulative profiles to shell profiles
                filtered_fractions = np.where((sum_radial_baryon_fraction == 0) | np.isnan(sum_radial_baryon_fraction), np.nan, sum_radial_baryon_fraction)
                mean_fraction_baryon = np.nanmean(filtered_fractions, axis=0)
                min_spectrum_baryon = np.nanpercentile(filtered_fractions, 16, axis=0)
                max_spectrum_baryon = np.nanpercentile(filtered_fractions, 84, axis=0)
                
                filtered_fractions = np.where((CAP_baryon_fraction == 0) | np.isnan(CAP_baryon_fraction), np.nan, CAP_baryon_fraction)
                mean_CAP_fraction_baryon = np.nanmean(filtered_fractions, axis=0)
                min_CAP_spectrum_baryon = np.nanpercentile(filtered_fractions, 16, axis=0)
                max_CAP_spectrum_baryon = np.nanpercentile(filtered_fractions, 84, axis=0)

                filtered_fractions = np.where((dS_baryon_fraction == 0) | np.isnan(dS_baryon_fraction), np.nan, dS_baryon_fraction)
                mean_dS_fraction_baryon = np.nanmean(filtered_fractions, axis=0)
                min_dS_spectrum_baryon = np.nanpercentile(filtered_fractions, 16, axis=0)
                max_dS_spectrum_baryon = np.nanpercentile(filtered_fractions, 84, axis=0)

                fig = plt.figure(figsize=(10,6))
                ax = fig.add_subplot(111)
                ax.plot(r_mids_array, mean_fraction_baryon, 'k-', linewidth=3, label='Baryon Mass Fraction')
                ax.fill_between(r_mids_array, min_spectrum_baryon, max_spectrum_baryon, color='k', alpha=0.1)
                ax.plot(r_mids_array, mean_CAP_fraction_baryon, ':', color='orange', linewidth=3, label='Baryon Mass Fraction + CAP filter')
                ax.fill_between(r_mids_array, min_CAP_spectrum_baryon, max_CAP_spectrum_baryon, color='orange', alpha=0.1)
                ax.plot(r_mids_array, mean_dS_fraction_baryon, ':', color='m', linewidth=3, label=r'Baryon Mass Fraction + $\Delta \Sigma$ filter')
                ax.fill_between(r_mids_array, min_dS_spectrum_baryon, max_dS_spectrum_baryon, color='m', alpha=0.1)

                #ax.set_xscale('log')
                ax.set_xlabel('$R (cMpc/h)$')
                ax.set_ylabel('Cumulative Mass Fraction ($f/(\Omega_b/\Omega_m)$)')
                ax.axvline(x=halo_vir_radius.mean(),color='r',label=r'average $R_{200}$='+f'{halo_vir_radius.mean():.2e}cMpc/h')
                ax.axvspan(halo_vir_radius.mean()-halo_vir_radius.std(), halo_vir_radius.mean()+halo_vir_radius.std(), color='red', alpha=0.3)
                plt.title(rf'{self.sim_name}: Cumulative Mass Fraction vs. Radius for '+  self.mass_bin_string , fontsize=14)
                plt.grid(True, alpha=0.3, which='both')
                ax.legend(ncol=2)
                plt.show()
                if save:
                    fig.savefig(self.save_figure_path + f'{self.sim_name}_cumulative_baryon_gas_star_fractioself.n_halos_{self.n_halos}_9r200.png', dpi=300)
                # Calculate baryon fraction

                fig = plt.figure(figsize=(10,6))
                ax = fig.add_subplot(111)
                
                filtered_fractions = np.where((baryon_mass_profile == 0) | np.isnan(baryon_mass_profile), np.nan, baryon_mass_profile)
                mean_fraction_baryon = np.nanmean(filtered_fractions, axis=0)
                min_spectrum_baryon = np.nanpercentile(filtered_fractions, 16, axis=0)
                max_spectrum_baryon = np.nanpercentile(filtered_fractions, 84, axis=0)
                
                filtered_fractions = np.where((CAP_baryon_mass_profile == 0) | np.isnan(CAP_baryon_mass_profile), np.nan, CAP_baryon_mass_profile)
                mean_CAP_fraction_baryon = np.nanmean(filtered_fractions, axis=0)
                min_CAP_spectrum_baryon = np.nanpercentile(filtered_fractions, 16, axis=0)
                max_CAP_spectrum_baryon = np.nanpercentile(filtered_fractions, 84, axis=0)
                
                mean_dS_fraction_baryon = np.nanmean(dS_total_profile, axis=0)
                min_dS_spectrum_baryon = np.nanpercentile(dS_total_profile, 16, axis=0)
                max_dS_spectrum_baryon = np.nanpercentile(dS_total_profile, 84, axis=0)
                
                ax.plot(r_mids_array, mean_CAP_fraction_baryon, '.', color='orange', linewidth=3, label='CAP filter on baryon profile')
                ax.fill_between(r_mids_array, min_CAP_spectrum_baryon, max_CAP_spectrum_baryon, color='orange', alpha=0.1)
                
                ax.plot(r_mids_array, mean_fraction_baryon, 'k-', linewidth=3, label='Cumulative baryon profile')
                ax.fill_between(r_mids_array, min_spectrum_baryon, max_spectrum_baryon, color='k', alpha=0.1)
                
                #ax.set_xscale('log')
                ax.set_xlabel('$R (cMpc/h)$')
                ax.axvline(x=halo_vir_radius.mean(),color='r',label=r'average $R_{200}$='+f'{halo_vir_radius.mean():.2e}cMpc/h')
                ax.axvspan(halo_vir_radius.mean()-halo_vir_radius.std(), halo_vir_radius.mean()+halo_vir_radius.std(), color='red', alpha=0.3)
                ax.set_ylabel('Cumulative Profile  [$M_\odot$/h]')
                plt.title(rf'{self.sim_name}: kSZ Baryon Profile vs. Radius for '+ self.mass_bin_string, fontsize=14)
                plt.grid(True, alpha=0.3, which='both')
                ax.legend()
                plt.show()
                if save:
                    fig.savefig(self.save_figure_path + f'{self.sim_name}_kSZ_baryon_profiles_{n_bins}_bins_{self.n_halos}_halos.png', dpi=300)


                fig = plt.figure(figsize=(5,8))
                ax = fig.add_subplot(212)

                ax.plot(r_mids_array[1:], (mean_dS_fraction_baryon*r_mids_array)[1:]/1e12, '.', color='m', linewidth=3)
                ax.fill_between(r_mids_array[1:], (min_dS_spectrum_baryon*r_mids_array)[1:]/1e12, (max_dS_spectrum_baryon*r_mids_array)[1:]/1e12, color='m', alpha=0.1)
                
                ax.set_xscale('log')
                ax.set_xlabel('$R (cMpc/h)$')
                ax.axvline(x=halo_vir_radius.mean(),color='r')
                ax.axvspan(halo_vir_radius.mean()-halo_vir_radius.std(), halo_vir_radius.mean()+halo_vir_radius.std(), color='red', alpha=0.3)
                ax.set_ylabel(r'R$\times \Delta \Sigma$ [Mpc $M_\odot$ pc$^{-2}$]')
                plt.grid(True, alpha=0.3, which='both')
                
                ax1 = fig.add_subplot(211)
                ax1.plot(r_mids_array[1:], (mean_dS_fraction_baryon)[1:]/1e12, '.', color='m', linewidth=3, label=f'[{r_min}, {r_max}] '+ r'on total mass profile')
                ax1.fill_between(r_mids_array[1:], (min_dS_spectrum_baryon)[1:]/1e12, (max_dS_spectrum_baryon)[1:]/1e12, color='m', alpha=0.1)
                
                #ax1.set_xscale('log')
                ax1.axvline(x=halo_vir_radius.mean(),color='r',label=r'average $R_{200}$='+f'{halo_vir_radius.mean():.2e}cMpc/h')
                ax1.axvspan(halo_vir_radius.mean()-halo_vir_radius.std(), halo_vir_radius.mean()+halo_vir_radius.std(), color='red', alpha=0.3)
                ax1.set_ylabel(r'$\Delta \Sigma$ [h $M_\odot$ pc$^{-2}$]')
                plt.grid(True, alpha=0.3, which='both')
                ax1.legend()
                ax.set_xlabel('$R (cMpc/h)$')
                ax1.sharex(ax)
                fig.subplots_adjust(hspace=0)
                plt.show()
                if save:
                    fig.savefig(self.save_figure_path + f'{self.sim_name}_WL_total_profiles_{n_bins}_bins_{self.n_halos}_halos.png', dpi=300)

#### utils functions ####

    def hub_func(self, z):
        Om = self.Omega_m
        Ol = self.Omega_L
        O_tot = Om + Ol
        return np.sqrt(Om*(1.0 + z)**3 + Ol + (1 - O_tot)*(1 + z)**2)

    def rho_cz(self, z):
        Ez2 = self.Omega_m*(1+z)**3. + (1-self.Omega_m)
        return self.rhocrit * Ez2

    def ComInt(self, z):
        return 1.0/self.hub_func(z)

    def ComDist(self, z):
        Om = self.Omega_m
        Ol = self.Omega_L
        O_tot = Om + Ol
        Dh = self.C_OVER_HUBBLE/self.h
        ans = Dh*quad(self.ComInt,0,z)[0]
        if (O_tot < 1.0): ans = Dh / np.sqrt(1.0-O_tot) *  np.sin(np.sqrt(1.0-O_tot) * quad(self.ComInt,0,z)[0])
        if (O_tot > 1.0): ans = Dh / np.sqrt(O_tot-1.0) *  np.sinh(np.sqrt(O_tot-1.0) * quad(self.ComInt,0,z)[0])
        return ans

    def AngDist(self, z):
        return self.ComDist(z) / (1.0 + z) #Mpc/h

    def convolveMap(self, map_, fwhm_arcmin, pixel_size_arcmin):
        """Convolve the map with a Gaussian beam.

        Args:
            map_ (np.ndarray): 2D numpy array of the field for the given particle type.
            fwhm_arcmin (float): Full width at half maximum of the Gaussian beam in arcminutes.
            pixel_size_arcmin (float): Size of the pixel in arcminutes.

        Returns:
            np.ndarray: Convolved 2D numpy array.
        """
        
        # Use log(2) for the FWHM-to-sigma conversion
        sigma_pixels = fwhm_arcmin / (2 * np.sqrt(2 * np.log(2)) * pixel_size_arcmin)

        # Apply Gaussian filter
        convolved_map = gaussian_filter(map_, sigma=sigma_pixels, mode='wrap')
        
        return convolved_map

    def calc_theta_arcmin_ACT_nPixels(self, PixelSize=0.5): #arcmin
        self.theta_arcmin = np.degrees(self.BoxSize / self.AngDist(self.redshift)) * 60
        self.nPixels = np.ceil(self.theta_arcmin / PixelSize).astype(int)

    
#### Henry's halo analysis functions####
    def make_2D_maps(self,
                    pType='gas',
                    nPixels=None, #if None, use ACT pixel resolution 0.5 arcmin/pixel: self.nPixels
                    save=False,
                    beamsize=None, #arcmin
                    ):
                                   # Get snapshot header:
        # Use self attributes for paths and snapshot info
        BoxSize = self.BoxSize  # Mpc/h
        if nPixels is None:
            nPixels = self.nPixels
            arcminPerPixel = self.theta_arcmin / self.nPixels
            print(f"Using default nPixels={nPixels} for ACT-like resolution of {arcminPerPixel:.3f} arcmin/pixel")
        else:
            arcminPerPixel = self.theta_arcmin / nPixels
            print(f"Using nPixels={nPixels} for arcmin/pixel of {arcminPerPixel:.3f}")


         # Get all particles from self.particles
        if pType == 'gas':
            coordinates = self.particles[0]['pos']
            masses = self.particles[0]['masses']
        elif pType == 'DM':
            coordinates = self.particles[1]['pos']
            IDs = self.particles[1]['id']
            DM_mass = self.Masses[1]
        elif pType == 'Stars':
            coordinates = self.particles[4]['pos']
            masses = self.particles[4]['masses']
        elif pType == 'BH':
            coordinates = self.particles[5]['pos']
            masses = self.particles[5]['masses']
        else:
            raise NotImplementedError('Particle Type not implemented')

        xx = coordinates[:, 0]
        yy = coordinates[:, 1]

        gridSize = [nPixels, nPixels]
        minMax = [0, BoxSize]

        if pType == 'gas':
            result = binned_statistic_2d(xx, yy, masses, 'sum', bins=gridSize, range=[minMax, minMax])
            field = result.statistic
        elif pType == 'DM':
            result = binned_statistic_2d(xx, yy, IDs, 'count', bins=gridSize, range=[minMax, minMax])
            field = result.statistic * DM_mass
        elif pType == 'Stars':
            result = binned_statistic_2d(xx, yy, masses, 'sum', bins=gridSize, range=[minMax, minMax])
            field = result.statistic
        elif pType == 'BH':
            result = binned_statistic_2d(xx, yy, masses, 'sum', bins=gridSize, range=[minMax, minMax])
            field = result.statistic
        else:
            raise NotImplementedError('Particle Type not implemented')

        # Convolve with beam if specified
        if beamsize is not None:
            try:
                field = self.convolveMap(field, beamsize, arcminPerPixel)
                print(f"Applied Gaussian convolution with FWHM={beamsize} arcmin")
            except Exception as e:
                print(f"Error in convolution: {e}")
                print("Fail to proceed with gaussian filter convolution.")
        
        # Save to self.save_profile_path
        if save:
            path_save = self.save_profile_path + f"{pType}_2D_{nPixels}.npy"
            np.save(path_save, field)
        return field
    

    def get_stacked_profiles(self,
                             nPixel=512,
                             beamsize=1.6, #arcmin
                             n_bins=10, 
                             r_min=0.04, #cMpc/h
                             r_max=15,  #cMpc/h
                             plot=False,
                             save=False):
        # Make 2D maps for each particle type
        gas_field = self.make_2D_maps(pType='gas', nPixels=nPixel, beamsize=beamsize)
        DM_field = self.make_2D_maps(pType='DM', nPixels=nPixel, beamsize=beamsize)
        star_field = self.make_2D_maps(pType='Stars', nPixels=nPixel, beamsize=beamsize)
        BH_field = self.make_2D_maps(pType='BH', nPixels=nPixel, beamsize=beamsize)

        if nPixel is None:
            nPixel = self.nPixels
        MpcPerPixel = self.BoxSize / nPixel

        # Do stacking
        filt_funcs = [delta_sigma, CAP, total_mass]
        filt_func_names = [r'$\Delta \Sigma$', 'CAP', '2D Cumulative Mass']

        profiles = []
        ## Set radial bins
        # radii = np.linspace(r_min, r_max, n_bins+1)
        # Split the bins into linear and logarithmic parts  
        n_linear = n_bins // 2
        n_log = n_bins - n_linear
        
        r_split = 2.5 #cMpc/h
        
        # Linear bins from r_min to r_split (inclusive)
        radii_linear = np.linspace(r_min, r_split, n_linear + 1)
        # Log bins from r_split to r_max (inclusive)
        radii_log = np.logspace(np.log10(r_split), np.log10(r_max), n_log + 1)
        
        # Concatenate, removing duplicate at r_split
        radii = np.concatenate([radii_linear, radii_log[1:]])

        r_mids_array = 0.5 * (radii[1:] + radii[:-1])
        radii_Pixel = radii / MpcPerPixel

        for i, filt_func in enumerate(filt_funcs):
            profiles_g = []
            profiles_m = []
            profiles_s = []
            profiles_bh = []

            for j, haloPOS in enumerate(self.pos_g):

                # Load the snapshot for gas and DM around that halo:
                haloLoc = np.round(haloPOS/ MpcPerPixel).astype(int)[:2]
                cutout_g = cutout_2d_periodic(gas_field, haloLoc, radii_Pixel[-1]) #in pixels
                cutout_m = cutout_2d_periodic(DM_field, haloLoc, radii_Pixel[-1])
                cutout_s = cutout_2d_periodic(star_field, haloLoc, radii_Pixel[-1])
                cutout_bh = cutout_2d_periodic(BH_field, haloLoc, radii_Pixel[-1])

                rr = radial_distance_grid(cutout_g, (-r_max, r_max)) #returns same units as in (bounds)

                
                profile_g = []
                profile_m = []
                profile_s = []
                profile_bh = []

                for rad in radii:
                    filt_result_g = filt_func(cutout_g, rr, rad)
                    profile_g.append(filt_result_g)
            
                    filt_result_m = filt_func(cutout_m, rr, rad)
                    profile_m.append(filt_result_m)

                    filt_result_bh = filt_func(cutout_bh, rr, rad)
                    profile_bh.append(filt_result_bh)

                    filt_result_s = filt_func(cutout_s, rr, rad)
                    profile_s.append(filt_result_s)
            
            
                profile_g = np.array(profile_g)
                profile_m = np.array(profile_m)
                profile_s = np.array(profile_s)
                profile_bh = np.array(profile_bh)
                profiles_g.append(profile_g)
                profiles_m.append(profile_m)
                profiles_s.append(profile_s)
                profiles_bh.append(profile_bh)
        
                
            profiles_g = np.array(profiles_g)
            profiles_m = np.array(profiles_m)
            profiles_s = np.array(profiles_s)
            profiles_bh = np.array(profiles_bh)

            profiles.append((profiles_g, profiles_m, profiles_s, profiles_bh))

        if save:
            np.savez(self.save_profile_path + f"Henry_profiles_gas_dm_star_bh_nPixel{nPixel}_fwhm_{beamsize}_R_lin{r_min}_{r_split}_log{r_max}_nbins{n_bins}.npz",
                     profiles=profiles,
                    profile_names=filt_func_names,
                     r_bins=radii,
                     r_mids_array=r_mids_array,
                     m_halos=self.mass_selected,
                     r_vir = np.array([self.R_200[self.index_h_sh[n]] for n in range(self.n_halos)])
)

        if plot:
            colourmap = matplotlib.colormaps['magma']
            colours = colourmap(np.linspace(0, 0.85, 3))

            fig, ax = plt.subplots(1, 2, figsize=(15,8))
            for i, name in enumerate(filt_func_names):
                profiles_g, profiles_m, profiles_s = profiles[i]

                upper_g = np.quantile(profiles_g, 0.25, axis=0)
                lower_g = np.quantile(profiles_g, 0.75, axis=0)
                mean_g = np.mean(profiles_g, axis=0)
                median_g = np.median(profiles_g, axis=0)

                upper_m = np.quantile(profiles_m, 0.25, axis=0)
                lower_m = np.quantile(profiles_m, 0.75, axis=0)
                mean_m = np.mean(profiles_m, axis=0)
                median_m = np.median(profiles_m, axis=0)

                ax[0].fill_between(radii, lower_g, upper_g, alpha=0.2, color=colours[i])
                ax[0].plot(radii, mean_g, label=name, c=colours[i])

                ax[1].fill_between(radii, lower_m, upper_m, alpha=0.2, color=colours[i])
                ax[1].plot(radii, mean_m, label=name, c=colours[i])

            fig.suptitle(f'Mass Profiles of Halo Clusters in {self.sim_name} with Different Filters\n'
                 f'{self.mass_bin_string} n={self.n_halos}, $R_{{200, \\rm{{mean}}}}$={self.R_200.mean():.1f} cMpc/h', fontsize=17)

            ax[0].set_title('Gas', fontsize=14)
            ax[1].set_title('DM', fontsize=14)
            ax[0].set_xlim((radii[0], radii[-1]))
            ax[1].set_xlim((radii[0], radii[-1]))
            ax[0].set_xlabel(r'$R$ [cMpc/h]', fontsize=15)
            ax[1].set_xlabel(r'$R$ [cMpc/h]', fontsize=15)
            ax[0].tick_params(axis='both', which='major', labelsize=16)
            ax[1].tick_params(axis='both', which='major', labelsize=16)
            ax[1].legend(fontsize=15)
            ax[0].grid()
            ax[1].grid()
            plt.tight_layout()
            plt.show()
            plt.savefig(self.save_figure_path + f"Henry_{self.sim_name}_FilterComparison.png", dpi=100)

            # Plot the ratio of gas to all matter in these filter profiles, normalized by Omega_b/Omega_m
            fig = plt.figure(figsize=(10,8))
            for i, name in enumerate(filt_func_names):
                profiles_g, profiles_m, profiles_s = profiles[i]
                gas_fraction = profiles_g / (profiles_g + profiles_m + profiles_s) / (self.Omega_b / self.Omega_m)

                upper = np.quantile(gas_fraction, 0.25, axis=0)
                lower = np.quantile(gas_fraction, 0.75, axis=0)
                mean = np.mean(gas_fraction, axis=0)
                median = np.median(gas_fraction, axis=0)

                plt.fill_between(radii, lower, upper, alpha=0.2, color=colours[i])
                plt.plot(radii, mean, label=name, c=colours[i])

            plt.axhline(1)
            plt.suptitle(r'Mean Gas Fraction $\left(\frac{\rm gas}{\rm gas + stars + DM}\right)$ of ' +
                 f'{self.sim_name} with Different Filters\n'
                 f'{self.mass_bin_string} n={self.n_halos}, $R_{{200, \\rm{{mean}}}}$={self.R_200.mean():.1f} cMpc/h', fontsize=17)
            plt.xlim((radii[0], radii[-1]))
            plt.ylim((-.5, 2.0))
            plt.xlabel(r'$R$ [cMpc/h]', fontsize=20)
            plt.ylabel(r'$\left(\frac{\rm gas}{\rm gas + stars + DM}\right) \times \frac{\Omega_m}{\Omega_{\rm baryon}}$', fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(fontsize=15)
            plt.grid()
            plt.tight_layout()
            plt.show()
            plt.savefig(self.save_figure_path + f"Henry_{self.sim_name}_Gas_fraction.png", dpi=100)

    def henry_main(self, 
                     nPixel=512,    #if None, use ACT pixel resolution 0.5 arcmin/pixel: self.nPixels
                     beamsize=1.6, #arcmin
                     n_bins=30, 
                     r_min=0.04, #cMpc/h
                     r_max=15,  #cMpc/h
                     plot=True,
                     save=True):
        """
        Main function to run Henry's halo analysis
        """
        t0 = time.time()
        print('Reading snapshot')
        self.read_header()
        self.read_particles()
        self.read_catalog(plot=False)
        # self.get_field_baryon_suppression(plot=plot, save=save, grid=512)
        self.calc_theta_arcmin_ACT_nPixels(PixelSize=0.5)
        self.get_stacked_profiles(nPixel=nPixel,  beamsize=beamsize, n_bins=n_bins, r_min=r_min, r_max=r_max, plot=plot, save=save)
        print(f"Henry's halo analysis completed in {time.time() - t0:.2f} seconds.")