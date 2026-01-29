from scipy.integrate import quad
import numpy as np
import h5py
import argparse

class KSZSimulator:
    def __init__(self, SB35_sim_index,nPixels=1000):
        self.snapNum = '074'
        self.chunkNum = 0
        self.SB35_sim_index = SB35_sim_index
        self.snapshot = f'/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/L50n512_SB35/SB35_{SB35_sim_index}/snapdir_{self.snapNum}/snap_{self.snapNum}.{self.chunkNum}.hdf5'

        with h5py.File(self.snapshot, 'r') as f:
            self.h = f['Header'].attrs[u'HubbleParam']
            self.z = f['Header'].attrs[u'Redshift']
            self.BoxSize = f['Header'].attrs[u'BoxSize']/1e3 #Mpc/h
            self.cosmo_params = {
                'Omega_m': f['Header'].attrs[u'Omega0'],
                'hh': self.h,
                'Omega_L': f['Header'].attrs[u'OmegaLambda'],
                'Omega_b': f['Header'].attrs[u'OmegaBaryon'],
                'C_OVER_HUBBLE': 2997.9
            }

        self.kpc_cgs = 3.086e21
        self.TCMB = 2.725
        self.v_rms = 1.06e-3
        self.MP_CGS = 1.6726219e-24
        self.ST_CGS = 6.65246e-25
        self.rhocrit = 1.87847e-29 * self.cosmo_params['hh']**2

        theta_arcmin = np.degrees(self.BoxSize / self.AngDist(self.z)) * 60
        self.nPixels = nPixels
        self.arcminPerPixel1000 = theta_arcmin / self.nPixels
        print(f"Using nPixels={self.nPixels} for arcmin/pixel of {self.arcminPerPixel1000:.3f}")

        theta_rad, beam = self.gaussian(1.6, self.arcminPerPixel1000)
        self.theta_rad = theta_rad
        self.beam = beam

    def f_beam(self, tht):
        return np.interp(tht, self.theta_rad, self.beam, left=0, right=0)

    def hub_func(self, z):
        Om = self.cosmo_params['Omega_m']
        Ol = self.cosmo_params['Omega_L']
        O_tot = Om + Ol
        return np.sqrt(Om*(1.0 + z)**3 + Ol + (1 - O_tot)*(1 + z)**2)

    def rho_cz(self, z):
        Ez2 = self.cosmo_params['Omega_m']*(1+z)**3. + (1-self.cosmo_params['Omega_m'])
        return self.rhocrit * Ez2

    def ComInt(self, z):
        return 1.0/self.hub_func(z)

    def ComDist(self, z):
        Om = self.cosmo_params['Omega_m']
        Ol = self.cosmo_params['Omega_L']
        O_tot = Om + Ol
        Dh = self.cosmo_params['C_OVER_HUBBLE']/self.cosmo_params['hh']
        ans = Dh*quad(self.ComInt,0,z)[0]
        if (O_tot < 1.0): ans = Dh / np.sqrt(1.0-O_tot) *  np.sin(np.sqrt(1.0-O_tot) * quad(self.ComInt,0,z)[0])
        if (O_tot > 1.0): ans = Dh / np.sqrt(O_tot-1.0) *  np.sinh(np.sqrt(O_tot-1.0) * quad(self.ComInt,0,z)[0])
        return ans

    def AngDist(self, z):
        return self.ComDist(z) / (1.0 + z) #Mpc/h

    @staticmethod
    def gaussian(fwhm_arcmin, pixel_size_arcmin, max_arcmin=None):
        fwhm_rad = np.deg2rad(fwhm_arcmin / 60)
        if max_arcmin is None:
            max_arcmin = 5 * fwhm_arcmin
        tht_in_arcmin = np.linspace(0, pixel_size_arcmin, 100)
        tht_in = np.deg2rad(tht_in_arcmin / 60)
        sigma_rad = fwhm_rad / (2 * np.sqrt(2 * np.log(2)))
        beam = np.exp(-0.5 * (tht_in / sigma_rad) ** 2)
        beam /= beam.sum()
        return tht_in, beam

    def make_a_obs_profile_sim_rho(self, thta_arc, rho_CAP, rint_CAP, gaussian_beam=True, f_beam=None):
        rho = np.zeros(len(thta_arc))
        for ii in range(len(thta_arc)):
            temp = self.project_prof_beam_sim_rho(thta_arc[ii], self.z, rho_CAP, rint_CAP, gaussian_beam, f_beam)
            rho[ii] = temp
        return rho

    def project_prof_beam_sim_rho(self, tht_arc, z, rho_CAP, rint_CAP, gaussian_beam=True, f_beam=None):
        XH = 0.76
        AngDis = self.AngDist(z)
        # print(f'AngDis={AngDis} for z={z}')
        tht_rad = np.radians(tht_arc/60.)
        thta_bins = np.arctan((rint_CAP) / AngDis)
        dthta = np.diff(thta_bins)
        dthta = np.append(dthta, dthta[-1])
        sig_total = 0.0
        if f_beam is None:
            f_beam = self.f_beam
        for i, (thta_bin, rho_val, dthta_val) in enumerate(zip(thta_bins, rho_CAP, dthta)):
            # print(f'angle diff: {abs(tht_rad - thta_bin)}')
            # print(f'i={i}, thta_bin={thta_bin:.3e}, rho_val={rho_val:.3f}, dthta_val={dthta_val:.3e}, beam_weight={beam_weight:.3f}')
            if gaussian_beam:
                beam_weight = f_beam(abs(tht_rad - thta_bin))
            else:
                beam_weight = 1.0 if abs(tht_rad - thta_bin) < self.arcminPerPixel1000/2. else 0.0
            sig_total += 2.0 * np.pi * thta_bin * dthta_val * rho_val * beam_weight
        sig_total_cgs = sig_total * 1.989e33 / (self.kpc_cgs*1e3)**2 / self.cosmo_params['hh']
        sig_all_beam = sig_total_cgs * self.v_rms * self.ST_CGS * self.TCMB * 1e6 * ((1. + XH)/2.) / self.MP_CGS
        return sig_all_beam

def main(SB35_sim_index):
    sim = KSZSimulator(SB35_sim_index)
    return sim.SB35_sim_index, sim.cosmo_params, sim.z, sim.h, sim.BoxSize, sim.f_beam, sim

if __name__ == "__main__":
    indices = np.load('ASN1_varying_sims.npy')#np.arange(1024)
    temp_ksz_sims = []
    mass_halos_ranges = []
    for i, id in enumerate(indices):
        KSZone = KSZSimulator(id)
        Henry_profiles_sim =  np.load(f'/pscratch/sd/l/lindajin/CAMELS/IllustrisTNG/L50n512_SB35/SB35_{KSZone.SB35_sim_index}/data/'+ f"Henry_profiles_gas_dm_star_bh_nPixel1000_R_lin0.04_2.5_log15_nbins20.npz")

        r_bins =  Henry_profiles_sim['r_bins']
        mass_halos_ranges.append(Henry_profiles_sim['m_halos_range'])
        ## CAP
        profiles_g, profiles_m, profiles_s, profiles_bh = Henry_profiles_sim['profiles'][1]
        #median = np.median(profiles_g, axis=0)

        theta = np.linspace(1, 6, 9)
        temp_ksz = []
        temp_ksz.extend(KSZone.make_a_obs_profile_sim_rho(theta, profile,r_bins) for profile in profiles_g)
        temp_ksz_sims.extend(temp_ksz)  # flatten the (68,9) into the main list
    np.save('ksz_profiles_sims_halos.npy', np.array(temp_ksz_sims))
    np.save('mass_halos_ranges_ASN1_varying_sims.npy', np.array(mass_halos_ranges))

