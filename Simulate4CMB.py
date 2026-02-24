#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# simulate radio backgrouns PS
# add CMB
# add Syncrotron
# add Dust
# add noise

def make_parallel_simu4CMB(in_parfile, out_file):
    from multiprocessing import Pool
    from datetime import datetime
    import json
    from functools import partial
    from numpy import zeros, vstack, shape, max
    import matplotlib.pyplot as plt

    print(str(datetime.now()))

    with open(in_parfile) as par_file: 
        Simulation_Parameters = json.loads(par_file.read())

    Lon, Lat = CreateRandomCatalogue(nmcat = None,
                                     Ns = Simulation_Parameters['n_sims'], ex_rad = 6.,
                                     cut = Simulation_Parameters["gal_cut"])

    pos = [[x,y] for x, y in zip(Lon,Lat)]
    
    fun = partial(MakeSimu, nm_parfile = in_parfile)

    p = Pool()
    sims = p.map(fun, pos)

    # sims is a list of MakeSimu outputs.
    # sims = [total_map_low, total_map, total_map_high, label]_1, ...

    total_map_list = zeros((len(sims), Simulation_Parameters['npix'],
                            Simulation_Parameters['npix'], 3))
    label_list = zeros((len(sims),Simulation_Parameters['npix'],Simulation_Parameters['npix'], 1))
    
    for i in range(len(sims)):
        # lower freq
        total_map_list[i,:,:,0] = sims[i][0]       
        # central freq
        total_map_list[i,:,:,1] = sims[i][1]       

        label_list[i,:,:,0] = sims[i][3]
        # higher freq
        total_map_list[i,:,:,2] = sims[i][2]       

    write2h5(total_map_list, label_list, out_file + '.h5')
     
    p.terminate()
    print(str(datetime.now()))

    pass


def MakeSimu(pos, nm_parfile):
    
    from corrsky_v0124 import corrsky, fluxassoc
    import astropy
    from astropy import units as u
    import json
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    from numpy import sqrt, log, zeros, random, std, zeros_like, mean, std, pi
    import healpy as hp
    
    with open(nm_parfile) as par_file: 
        Simulation_Parameters = json.loads(par_file.read())

    # make patch with random radio background sources
    
    Rbk_map, Rbk_mapp, R_bkcat, Rbk_catp = corrsky(nu = Simulation_Parameters['freq'],
                                                   npix = Simulation_Parameters['npix'],
                                                   pixsize = Simulation_Parameters['pixsize'],
                                                   sncmodel = 'tucci', pkmodel = 'lapi11',
                                                   Sn = Simulation_Parameters['Sn'],
                                                   Sx = Simulation_Parameters['Sx'])
    Rbk_map_low = zeros_like(Rbk_map)
    Rbk_map_high = zeros_like(Rbk_map)

    fac = conversion_factor_Jy_K(Simulation_Parameters["freq"],
                                            Simulation_Parameters["pixsize"]/60.) * 1e6
    fac_low = conversion_factor_Jy_K(Simulation_Parameters["freq_low"],
                                 Simulation_Parameters["pixsize"]/60.) * 1e6
    fac_high = conversion_factor_Jy_K(Simulation_Parameters["freq_high"],
                                 Simulation_Parameters["pixsize"]/60.) * 1e6

    alpha = 0.
    alpha = random.normal(Simulation_Parameters['mean_alpha_radio_low'], 0.05, 1)
    Rbk_map_low =  Rbk_map * (
            Simulation_Parameters['freq_low'] / Simulation_Parameters['freq']) ** alpha

    alpha = random.normal(Simulation_Parameters['mean_alpha_radio_high'], 0.05, 1)
    Rbk_map_high =  Rbk_map * (
            Simulation_Parameters['freq_high'] / Simulation_Parameters['freq']) ** alpha
    
    Rbk_map *=fac
    Rbk_map_low *=fac_low
    Rbk_map_high *=fac_high
            
    plt.figure(1)
    plt.imshow(Rbk_map), plt.colorbar()
    plt.savefig('Rbk_map.pdf')
    plt.close()

    plt.figure(1)
    plt.imshow(Rbk_map_low), plt.colorbar()
    plt.savefig('Rbk_map_low.pdf')
    plt.close()

    plt.figure(1)
    plt.imshow(Rbk_map_high), plt.colorbar()
    plt.savefig('Rbk_map_high.pdf')
    plt.close()

    # make patch with IR background sources 
    IRbk_map, IRbk_mapp,IR_bkcat, IRbk_catp = corrsky(nu = Simulation_Parameters['freq'],
                                                      npix = Simulation_Parameters['npix'],
                                                      pixsize = Simulation_Parameters['pixsize'],
                                                      sncmodel = 'lapi11', pkmodel = 'lapi11',
                                                      Sn = -4.5, 
                                                      Sx = min(Simulation_Parameters['Sx'],0.0))
    
    IRbk_map_low = zeros_like(IRbk_map)
    IRbk_map_high = zeros_like(IRbk_map)
    
    alpha = 0.

    alpha = random.normal(Simulation_Parameters['mean_alpha_IR_low'], 0.05, 1)
    IRbk_map_low =  IRbk_map * (
            Simulation_Parameters['freq_low'] / Simulation_Parameters['freq']) ** alpha

    alpha = random.normal(Simulation_Parameters['mean_alpha_IR_high'], 0.05, 1)
    IRbk_map_high = IRbk_map * (
            Simulation_Parameters['freq_high'] / Simulation_Parameters['freq']) ** alpha
    
    IRbk_map *=fac
    IRbk_map_low *=fac_low
    IRbk_map_high *=fac_high
            
    plt.figure(1)
    plt.imshow(IRbk_map), plt.colorbar()
    plt.savefig('IRbk_map.pdf')
    plt.close()

    plt.figure(1)
    plt.imshow(IRbk_map_low), plt.colorbar()
    plt.savefig('IRbk_map_low.pdf')
    plt.close()

    plt.figure(1)
    plt.imshow(IRbk_map_high), plt.colorbar()
    plt.savefig('IRbk_map_high.pdf')
    plt.close()

    # make patch with IRLT sources
    IRLTbk_map, IRLTbk_mapp, IRLT_bkcat, IRLTbk_catp = corrsky(nu = Simulation_Parameters['freq'],
                                                               npix = Simulation_Parameters['npix'],
                                                               pixsize = Simulation_Parameters['pixsize'],
                                                               sncmodel = 'IRLT', pkmodel = 'lapi11',
                                                               Sn = -4.5, Sx = Simulation_Parameters['Sx'])
    IRLTbk_map_low = zeros_like(IRLTbk_map)
    IRLTbk_map_high = zeros_like(IRLTbk_map)
    
    alpha = 0.

    alpha = random.normal(Simulation_Parameters['mean_alpha_IR_low'], 0.05, 1)
    IRLTbk_map_low =  IRLTbk_map * (
            Simulation_Parameters['freq_low'] / Simulation_Parameters['freq']) ** alpha

    alpha = random.normal(Simulation_Parameters['mean_alpha_IR_high'], 0.05, 1)
    IRLTbk_map_high = IRLTbk_map * (
            Simulation_Parameters['freq_high'] / Simulation_Parameters['freq']) ** alpha
            
    alpha = 0.

    IRLTbk_map *=fac
    IRLTbk_map_low *=fac_low
    IRLTbk_map_high *=fac_high

    plt.figure(1)
    plt.imshow(IRLTbk_map), plt.colorbar()
    plt.savefig('IRLTbk_map.pdf')
    plt.close()

    plt.figure(1)
    plt.imshow(IRLTbk_map_low), plt.colorbar()
    plt.savefig('IRLTbk_map_low.pdf')
    plt.close()

    plt.figure(1)
    plt.imshow(IRLTbk_map_high), plt.colorbar()
    plt.savefig('IRLTbk_map_high.pdf')
    plt.close()

    
    # Sum and smoothing
    sigma = Simulation_Parameters["fwhm"] / (2. * sqrt(2. * log(2.)))
    sigma = sigma / Simulation_Parameters["pixsize"]
    sigma_low = Simulation_Parameters["fwhm_low"] / (2. * sqrt(2. * log(2.))) 
    sigma_low = sigma / Simulation_Parameters["pixsize"]
    sigma_high = Simulation_Parameters["fwhm_high"] / (2. * sqrt(2. * log(2.))) 
    sigma_high = sigma / Simulation_Parameters["pixsize"]
    


    Tmap = Rbk_map + IRbk_map + IRLTbk_map
    Tmap_low = Rbk_map_low + IRbk_map_low + IRLTbk_map_low
    Tmap_high = Rbk_map_high + IRbk_map_high + IRLTbk_map_high
    
    Tmap *= fac
    Tmap = gaussian_filter(Tmap, sigma)

    Tmap_low *= fac_low
    Tmap_low = gaussian_filter(Tmap_low, sigma_low)

    Tmap_high *= fac_high
    Tmap_high = gaussian_filter(Tmap_high, sigma_high)

    plt.figure(1)
    plt.imshow(Tmap), plt.colorbar()
    plt.savefig('Tmap.pdf')
    plt.close()

    plt.figure(1)
    plt.imshow(Tmap_low), plt.colorbar()
    plt.savefig('Tmap_low.pdf')
    plt.close()

    plt.figure(1)
    plt.imshow(Tmap_high), plt.colorbar()
    plt.savefig('Tmap_high.pdf')
    plt.close()
    
    # add contaminants

    if not pos:
        Lon, Lat = CreateRandomCatalogue(nmcat = None, Ns = 1, ex_rad = 6.,
                                     cut = Simulation_Parameters["gal_cut"])
        pos = [Lon, Lat]
    
    patch = zeros((Simulation_Parameters['npix'], Simulation_Parameters['npix']))
    patch_low = zeros((Simulation_Parameters['npix'], Simulation_Parameters['npix']))
    patch_high = zeros((Simulation_Parameters['npix'], Simulation_Parameters['npix']))

    label = zeros((Simulation_Parameters['npix'], Simulation_Parameters['npix']))
    
    for comp in Simulation_Parameters["contaminants"]:

        # central freq       
        nm_map = Simulation_Parameters["channels"][0][comp]     
        imap = hp.fitsfunc.read_map(nm_map, field=0, memmap = True)
        omap = hp.visufunc.gnomview(imap, rot = pos, xsize = Simulation_Parameters['npix'],
                                    reso = Simulation_Parameters['pixsize'] / 60.,
                                    notext = True, return_projected_map = True,
                                    no_plot = True)

        if comp == "cmb":
            label = omap
            omap *= 1e6
            plt.figure(1)
            plt.imshow(omap), plt.colorbar()
            plt.savefig('Label.pdf')
            plt.close()

        if comp == "dust":
            label = omap
            omap *= 1e6
            plt.figure(1)
            plt.imshow(omap), plt.colorbar()
            plt.savefig('dust.pdf')
            plt.close()

        patch += omap

        
        # lower freq       
        nm_map = Simulation_Parameters["channels_low"][0][comp]
        imap = hp.fitsfunc.read_map(nm_map, field=0, memmap = True)
        omap = hp.visufunc.gnomview(imap, rot = pos, xsize = Simulation_Parameters['npix'],
                                    reso = Simulation_Parameters['pixsize'] / 60.,
                                    notext = True, return_projected_map = True,
                                    no_plot = True)

        if comp == "cmb":
            label = omap
            omap *= 1e6
            plt.figure(1)
            plt.imshow(omap), plt.colorbar()
            plt.savefig('Label_low.pdf')
            plt.close()

        if comp == "dust":
            label = omap
            omap *= 1e6
            plt.figure(1)
            plt.imshow(omap), plt.colorbar()
            plt.savefig('dust_low.pdf')
            plt.close()

        patch_low += omap

        # higher freq       
        nm_map = Simulation_Parameters["channels_high"][0][comp]
        imap = hp.fitsfunc.read_map(nm_map, field=0, memmap = True)
        omap = hp.visufunc.gnomview(imap, rot = pos, xsize = Simulation_Parameters['npix'],
                                    reso = Simulation_Parameters['pixsize'] / 60.,
                                    notext = True, return_projected_map = True,
                                    no_plot = True)

        if comp == "cmb":
            label = omap
            omap *= 1e6
            plt.figure(1)
            plt.imshow(omap), plt.colorbar()
            plt.savefig('Label_high.pdf')
            plt.close()

        if comp == "dust":
            label = omap
            omap *= 1e6
            plt.figure(1)
            plt.imshow(omap), plt.colorbar()
            plt.savefig('dust_high.pdf')
            plt.close()

        patch_high += omap


    plt.figure(1)
    plt.imshow(patch), plt.colorbar()
    plt.savefig('patch.pdf')
    plt.close()

    plt.figure(1)
    plt.imshow(patch_low), plt.colorbar()
    plt.savefig('patch_low.pdf')
    plt.close()

    plt.figure(1)
    plt.imshow(patch_high), plt.colorbar()
    plt.savefig('patch_high.pdf')
    plt.close()

    # add noise
  
    # central freq
    beam_size = sqrt(2. * pi * (Simulation_Parameters["fwhm"] / (2 * sqrt(2 * log(2.))))**2)   
    sigma_noise = Simulation_Parameters["spp"] * (60. * 60. / beam_size)       
    # sigma_noise = Simulation_Parameters["spp"] * 1e-6 * (60. * 60. / Simulation_Parameters["pixsize"])       
    Noise = random.randn(Simulation_Parameters['npix'],Simulation_Parameters['npix']) * sigma_noise
    patch += Noise

    plt.figure(1)
    plt.imshow(Noise), plt.colorbar()
    plt.savefig('Noise.pdf')
    plt.close()

    # lower freq
    beam_size = sqrt(2. * pi * (Simulation_Parameters["fwhm_low"] / (2 * sqrt(2 * log(2.))))**2)  
    sigma_noise = Simulation_Parameters["spp_low"] * (60. * 60. / beam_size)     
    # sigma_noise = Simulation_Parameters["spp_low"] * 1e-6 * (60. * 60. / Simulation_Parameters["pixsize"])  
    Noise = random.randn(Simulation_Parameters['npix'],Simulation_Parameters['npix']) * sigma_noise
    patch_low += Noise

    plt.figure(1)
    plt.imshow(Noise), plt.colorbar()
    plt.savefig('Noise_low.pdf') 
    plt.close()

    # higher freq
    beam_size = sqrt(2. * pi * (Simulation_Parameters["fwhm_high"] / (2 * sqrt(2 * log(2.))))**2) 
    sigma_noise = Simulation_Parameters["spp_high"] * (60. * 60. / beam_size)     
    # sigma_noise = Simulation_Parameters["spp_high"] * 1e-6 * (60. * 60. / Simulation_Parameters["pixsize"])
    Noise = random.randn(Simulation_Parameters['npix'],Simulation_Parameters['npix']) * sigma_noise
    patch_high += Noise

    plt.figure(1)
    plt.imshow(Noise), plt.colorbar()
    plt.savefig('Noise_high.pdf')
    plt.close()           
    
    total_map = patch + Tmap
    total_map_low = patch_low + Tmap_low
    total_map_high = patch_high + Tmap_high

    #print('mean', mean(total_map), 'std', std(total_map))
    #print('mean', mean(patch), 'std', std(patch))
    plt.figure(1)
    plt.imshow(total_map), plt.colorbar()
    plt.savefig('total_map.pdf')
    plt.close()

    plt.figure(1)
    plt.imshow(total_map_low), plt.colorbar()
    plt.savefig('total_map_low.pdf')
    plt.close()

    plt.figure(1)
    plt.imshow(total_map_high), plt.colorbar()
    plt.savefig('total_map_high.pdf')
    plt.close()

    return total_map_low, total_map, total_map_high, label

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def write2h5(total_maps_list, label_list, out_file):
    import h5py
    
    f = h5py.File(out_file,'w')
    
    f.create_dataset('M', data = total_maps_list)
    f.create_dataset('M0', data = label_list)
        
    f.close()
        
    pass

def conversion_factor_Jy_K(freq, pix_size):
    # freq: Planck frequency in GHz
    # pix_size: in arcmin
        
    from astropy.cosmology import Planck15
    from astropy import units as u
    from numpy import pi

    freq = freq * u.GHz
    equiv = u.thermodynamic_temperature(freq, Planck15.Tcmb0)
    fac = (1. * u.Jy / u.sr).to(u.K, equivalencies = equiv)
    fac = fac.value * (1/((pix_size / 60 / 180 * pi)**2))

    return fac

def CreateRandomCatalogue(nmcat=None, Ns=1e3, ex_rad=6., cut=15.):
    """ Create a random catalogue avoiding source positions
        from the PCCS input catalogue.
        Ns: Number of random positions
        ex_rad: Exclusion radius in arcmin
        cut = galactic cut in degxs
        output glon, glat in degrees
    """
    from astropy.io import fits
    from numpy import random, pi, where, cos, sin, vstack, array, ones, delete, arcsin, ones, pi
    from scipy import spatial as sp

    # Generate random all-sky catalogue
    glon = random.uniform(0, 360, Ns*3)
    glat = arcsin(random.uniform(sin(cut*pi/180.), 1, size=Ns*3)) * 180. / pi
    segno = ones(len(glat), dtype=int)
    segno[random.random(Ns*3) < 0.5] = -1
    glat *= segno # lat in degrees

    # # Remove auto close pairs
    # xyz = lonlat_to_xyz(vstack((glon, glat)).swapaxes(0, 1))
    # T = sp.cKDTree(xyz)
    # chord = 2. * sin(ex_rad / 60. / 180. * pi / 2.)
    # pairs = array(list(T.query_pairs(chord)))
    # if len(pairs) > 0:
    #     glon = delete(glon, pairs[:, 1])
    #     glat = delete(glat, pairs[:, 1])

    if nmcat:
        # Read input catalogue
        # "../../../../ancillary_data/catalogues/COM_PCCS_030_R2.04.fits"
        hdu = fits.open(nmcat)
        iglon = hdu[1].data.field("GLON")
        iglat = hdu[1].data.field("GLAT")

        xyz = lonlat_to_xyz(vstack((iglon, iglat)).swapaxes(0, 1))
        iT = sp.cKDTree(xyz)
        pairs = T.query_ball_tree(iT, chord)

        mask = ones(glon.size, dtype=bool)

        for ii in xrange(len(pairs)):
            if pairs[ii]:
                mask[pairs[ii]] = False

        glon = glon[mask]
        glat = glat[mask]

    return glon[:Ns], glat[:Ns]

def lonlat_to_xyz(pos_lonlat):
    """
    Transform angular position (lon;lat) into cartessian ones (xyz)
    """
    from numpy import hstack, array
    from astropy.coordinates import SkyCoord

    c = SkyCoord(pos_lonlat[:, 0], pos_lonlat[:, 1], frame='icrs',
                 unit='deg')
    x = array(c.cartesian.x, ndmin=2).swapaxes(0, 1)
    y = array(c.cartesian.y, ndmin=2).swapaxes(0, 1)
    z = array(c.cartesian.z, ndmin=2).swapaxes(0, 1)

    return hstack((x, y, z))

def apply_smoothing(infile, outfile, in_fwhm, out_fwhm, pixsize):
    #apply_smoothing('simu_cfreq143_r02/prova_U.h5', 'simu_cfreq143_r02/smo30_prova_U.h5', [7.22, 4.9, 4.92], 30., 90.)
    import h5py
    import numpy as np
    from scipy.ndimage import gaussian_filter
    import matplotlib.pyplot as plt

    inp_file = h5py.File(infile, 'r')
    inputs = inp_file["M"][:,:,:,:]
    labels = inp_file["M0"][:,:,:,:]

    plt.figure(1)
    plt.imshow(inputs[0,:,:,1]), plt.colorbar()
    plt.savefig('Map_U_inp_prova.pdf')
    plt.close()

    plt.figure(1)
    plt.imshow(labels[0,:,:,0]), plt.colorbar()
    plt.savefig('Map_U_inp_label.pdf')
    plt.close()

    sigma = np.zeros(np.size(in_fwhm))
    smo_inputs = inputs
    smo_labels = labels
    
    for i in range(len(in_fwhm)):
        sigma[i] = np.sqrt((out_fwhm**2.) - (in_fwhm[i]**2.))# fwhm in and out in arcmin
        sigma[i] = sigma[i] * 60. / (2. * np.sqrt(2. * np.log(2.)))
        sigma[i] = sigma[i] / pixsize # pixsize in arcsec

    print(sigma)

    for i in range(len(inputs)):
        smo_inputs[i,:,:,0] = gaussian_filter(inputs[i,:,:,0], sigma[0])
        smo_inputs[i,:,:,1] = gaussian_filter(inputs[i,:,:,1], sigma[1])
        smo_inputs[i,:,:,2] = gaussian_filter(inputs[i,:,:,2], sigma[2])
        smo_labels[i,:,:,0] = gaussian_filter(labels[i,:,:,0], sigma[1])
   
    plt.figure(1)
    plt.imshow(smo_inputs[0,:,:,1]), plt.colorbar()
    plt.savefig('Map_U_inp_prova_smo.pdf')
    plt.close()

    plt.figure(1)
    plt.imshow(smo_labels[0,:,:,0]), plt.colorbar()
    plt.savefig('Map_U_inp_label_smo.pdf')
    plt.close()
    
    write2h5(smo_inputs, smo_labels, outfile)

    pass

def plot_fullsky_from_parfile(parfile_path):
    import json
    import healpy as hp
    import matplotlib.pyplot as plt

    """
    Genera mapas de todo el cielo de CMB y Dust a 353 GHz usando mollview (usando healpy), leyendo los paths desde el archivo .par.

    Parametros:
    - parfile_path: Ruta al archivo .par (formato JSON).
    """

    with open(parfile_path, "r") as f:
        par = json.load(f)

    channels_high = par["channels_high"][0]

    cmb_path = channels_high.get("cmb", None)
    dust_path = channels_high.get("dust", None)

    if not cmb_path or not dust_path:
        raise ValueError("No se encontraron los paths de 'cmb' y/o 'dust' en channels_high.")

    # Leemos los mapas
    cmb_map = hp.read_map(cmb_path, field=0)
    dust_map = hp.read_map(dust_path, field=0)

    # Convertimos de Kelvin a microKelvin
    cmb_map *= 1e6
    dust_map *= 1e6

    # Pintamos con mollview el CMB completo
    hp.mollview(cmb_map, title="Mapa cielo completo CMB (353 GHz)", unit="μK", norm='hist', cmap="coolwarm")
    plt.savefig("cmb_353GHz.png")
    plt.close()

    # Pintamos con mollview el polvo
    hp.mollview(dust_map, title="Mapa completo de contaminación del polvo (353 GHz)", unit="μK", norm='hist', cmap="inferno")
    plt.savefig("dust_353GHz.png")
    plt.close()


def main():
  plot_fullsky_from_parfile('/content/gdrive/MyDrive/TFG Simulacion/sim_cfreq217.par')
  #make_parallel_simu4CMB('/content/gdrive/MyDrive/TFG Simulacion/sim_cfreq217.par', '/content/gdrive/MyDrive/TFG Simulacion/outs/output_sims')

if __name__ == "__main__":
        main()
