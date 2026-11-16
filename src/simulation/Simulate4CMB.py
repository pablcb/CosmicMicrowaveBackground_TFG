#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# simulate radio backgrouns PS
# add CMB
# add Syncrotron
# add Dust
# add noise

"""
Simulate4CMB.py — Generación de parches simulados del cielo de microondas.

Cada simulación es un parche de npix x npix píxeles (256x256, 90"/píxel) en una
posición aleatoria del cielo (fuera del corte galáctico) y en tres frecuencias
de Planck HFI (143, 217 y 353 GHz). El parche total incluye:
  - CMB y polvo térmico galáctico (mapas del Planck Legacy Archive, proyección
    gnomónica con healpy),
  - fuentes puntuales extragalácticas (radio, infrarrojo y galaxias infrarrojas
    de tipo tardío) simuladas con CORRSKY (González-Nuevo et al. 2005), usando
    los modelos de cuentas de la carpeta counts/,
  - ruido instrumental blanco con la sensibilidad de cada canal.

Cada parche se recorta con una rotación aleatoria alrededor de su centro.

Salida: un archivo .h5 con los mapas en K
  - "M":  (n_sims, npix, npix, 3) -> parche total en [low, central, high],
  - "M0": (n_sims, npix, npix, 1) -> etiqueta (CMB a la frecuencia central).

Dependencias externas no incluidas en el repositorio:
  - corrsky_v0124.py (simulador de fuentes puntuales; colocarlo en esta carpeta),
  - mapas FITS de CMB y polvo del PLA, cuyas rutas se indican en el .par.

Uso (desde esta carpeta):
    python Simulate4CMB.py simulate --out ../../data/output_sims_train --n_sims 10000
    python Simulate4CMB.py simulate --out ../../data/output_sims_test --n_sims 3000
    python Simulate4CMB.py simulate --out ../../data/output_sims_validate --n_sims 1000
    python Simulate4CMB.py fullsky     # mapas de cielo completo (figuras 10 y 11)
Con SAVE_DEBUG_PLOTS = True se guardan figuras de cada componente (figuras 12 y 13).
"""

# Guardar figuras de control de cada componente en cada simulación (lento y,
# en paralelo, todas sobrescriben los mismos archivos): activar solo para depurar
# o para generar las figuras del capítulo 2 con una única simulación.
SAVE_DEBUG_PLOTS = False


def make_parallel_simu4CMB(in_parfile, out_file, n_sims=None):
    """
    Genera n_sims simulaciones (por defecto, Simulation_Parameters['n_sims']) en
    paralelo y las guarda en out_file + '.h5'. Los mapas se guardan en K (como esperan los scripts de análisis).
    """
    from multiprocessing import Pool
    from datetime import datetime
    import json
    from functools import partial
    from numpy import zeros

    print(str(datetime.now()))

    with open(in_parfile) as par_file:
        Simulation_Parameters = json.loads(par_file.read())

    if n_sims is None:
        n_sims = Simulation_Parameters['n_sims']

    Lon, Lat = CreateRandomCatalogue(nmcat = None,
                                     Ns = n_sims, ex_rad = 6.,
                                     cut = Simulation_Parameters["gal_cut"])

    pos = [[x, y] for x, y in zip(Lon, Lat)]

    fun = partial(MakeSimu, nm_parfile = in_parfile)

    with Pool() as p:
        sims = p.map(fun, pos)

    # sims = [(total_map_low, total_map, total_map_high, label), ...] en µK
    npix = Simulation_Parameters['npix']
    total_map_list = zeros((len(sims), npix, npix, 3))
    label_list = zeros((len(sims), npix, npix, 1))

    for i in range(len(sims)):
        total_map_list[i, :, :, 0] = sims[i][0]    # frecuencia baja
        total_map_list[i, :, :, 1] = sims[i][1]    # frecuencia central
        total_map_list[i, :, :, 2] = sims[i][2]    # frecuencia alta
        label_list[i, :, :, 0] = sims[i][3]        # CMB (frecuencia central)

    # µK -> K
    write2h5(total_map_list * 1e-6, label_list * 1e-6, out_file + '.h5')

    print(str(datetime.now()))


def _debug_plot(image, filename):
    """Guarda una figura de control si SAVE_DEBUG_PLOTS está activado."""
    if not SAVE_DEBUG_PLOTS:
        return
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.imshow(image), plt.colorbar()
    plt.savefig(filename)
    plt.close()


def _point_sources(Simulation_Parameters, sncmodel, Sn, Sx, alpha_low_key, alpha_high_key, fac, fac_low, fac_high, name):
    """
    Simula un fondo de fuentes puntuales con CORRSKY a la frecuencia central y lo
    extrapola a las frecuencias baja y alta con un índice espectral aleatorio.
    Devuelve los tres mapas en µK.
    """
    from corrsky_v0124 import corrsky
    from numpy import random

    bk_map, _, _, _ = corrsky(nu = Simulation_Parameters['freq'],
                              npix = Simulation_Parameters['npix'],
                              pixsize = Simulation_Parameters['pixsize'],
                              sncmodel = sncmodel, pkmodel = 'lapi11',
                              Sn = Sn, Sx = Sx)

    alpha = random.normal(Simulation_Parameters[alpha_low_key], 0.05)
    bk_map_low = bk_map * (Simulation_Parameters['freq_low'] / Simulation_Parameters['freq']) ** alpha

    alpha = random.normal(Simulation_Parameters[alpha_high_key], 0.05)
    bk_map_high = bk_map * (Simulation_Parameters['freq_high'] / Simulation_Parameters['freq']) ** alpha

    # Jy/píxel -> µK (la conversión se aplica una sola vez)
    bk_map = bk_map * fac
    bk_map_low = bk_map_low * fac_low
    bk_map_high = bk_map_high * fac_high

    _debug_plot(bk_map, name + '.pdf')
    _debug_plot(bk_map_low, name + '_low.pdf')
    _debug_plot(bk_map_high, name + '_high.pdf')

    return bk_map_low, bk_map, bk_map_high


def MakeSimu(pos, nm_parfile):
    """
    Genera una simulación en la posición pos = [lon, lat] (grados galácticos).
    Devuelve (total_map_low, total_map, total_map_high, label) en µK, donde label
    es el CMB a la frecuencia central.
    """
    import json
    from scipy.ndimage import gaussian_filter
    from numpy import sqrt, log, zeros, random, pi, asarray
    import healpy as hp

    with open(nm_parfile) as par_file:
        Simulation_Parameters = json.loads(par_file.read())

    npix = Simulation_Parameters['npix']
    pixsize = Simulation_Parameters['pixsize']   # arcsec

    # Factores de conversión Jy/píxel -> µK_CMB
    fac = conversion_factor_Jy_K(Simulation_Parameters["freq"], pixsize / 60.) * 1e6
    fac_low = conversion_factor_Jy_K(Simulation_Parameters["freq_low"], pixsize / 60.) * 1e6
    fac_high = conversion_factor_Jy_K(Simulation_Parameters["freq_high"], pixsize / 60.) * 1e6

    # ------------------------------------------------------------------
    # Fuentes puntuales extragalácticas (CORRSKY)
    # ------------------------------------------------------------------
    # Fondo de radio (cuentas de Tucci et al. 2011)
    R_low, R, R_high = _point_sources(Simulation_Parameters, 'tucci',
                                      Simulation_Parameters['Sn'], Simulation_Parameters['Sx'],
                                      'mean_alpha_radio_low', 'mean_alpha_radio_high',
                                      fac, fac_low, fac_high, 'Rbk_map')
    # Fondo infrarrojo (Lapi et al. 2011)
    IR_low, IR, IR_high = _point_sources(Simulation_Parameters, 'lapi11',
                                         -4.5, min(Simulation_Parameters['Sx'], 0.0),
                                         'mean_alpha_IR_low', 'mean_alpha_IR_high',
                                         fac, fac_low, fac_high, 'IRbk_map')
    # Galaxias infrarrojas de tipo tardío
    IRLT_low, IRLT, IRLT_high = _point_sources(Simulation_Parameters, 'IRLT',
                                               -4.5, Simulation_Parameters['Sx'],
                                               'mean_alpha_IR_low', 'mean_alpha_IR_high',
                                               fac, fac_low, fac_high, 'IRLTbk_map')

    # Suma y convolución con el haz de cada canal (FWHM en arcsec -> sigma en píxeles)
    fwhm2sigma = 1. / (2. * sqrt(2. * log(2.)))
    sigma = Simulation_Parameters["fwhm"] * fwhm2sigma / pixsize
    sigma_low = Simulation_Parameters["fwhm_low"] * fwhm2sigma / pixsize
    sigma_high = Simulation_Parameters["fwhm_high"] * fwhm2sigma / pixsize

    Tmap = gaussian_filter(R + IR + IRLT, sigma)
    Tmap_low = gaussian_filter(R_low + IR_low + IRLT_low, sigma_low)
    Tmap_high = gaussian_filter(R_high + IR_high + IRLT_high, sigma_high)

    _debug_plot(Tmap, 'Tmap.pdf')
    _debug_plot(Tmap_low, 'Tmap_low.pdf')
    _debug_plot(Tmap_high, 'Tmap_high.pdf')

    # ------------------------------------------------------------------
    # CMB y contaminantes difusos (mapas HEALPix del PLA)
    # ------------------------------------------------------------------
    if not pos:
        Lon, Lat = CreateRandomCatalogue(nmcat = None, Ns = 1, ex_rad = 6.,
                                         cut = Simulation_Parameters["gal_cut"])
        pos = [Lon[0], Lat[0]]

    # Rotación aleatoria del parche alrededor de su centro
    rot = [pos[0], pos[1], random.uniform(0., 360.)]

    bands = {"central": "channels", "low": "channels_low", "high": "channels_high"}
    patches = {band: zeros((npix, npix)) for band in bands}
    label = None

    for comp in Simulation_Parameters["contaminants"]:
        for band, key in bands.items():
            nm_map = Simulation_Parameters[key][0][comp]
            imap = hp.fitsfunc.read_map(nm_map, field=0, memmap = True)
            omap = hp.visufunc.gnomview(imap, rot = rot, xsize = npix,
                                        reso = pixsize / 60.,
                                        notext = True, return_projected_map = True,
                                        no_plot = True)
            omap = asarray(omap, dtype=float) * 1e6   # K -> µK

            # La etiqueta es el CMB a la frecuencia central
            if comp == "cmb" and band == "central":
                label = omap.copy()

            _debug_plot(omap, comp + ('' if band == 'central' else '_' + band) + '.pdf')
            patches[band] += omap

    if label is None:
        raise ValueError('"cmb" debe estar en la lista "contaminants" del archivo de parámetros.')

    patch, patch_low, patch_high = patches["central"], patches["low"], patches["high"]

    # ------------------------------------------------------------------
    # Ruido instrumental blanco
    # ------------------------------------------------------------------
    def noise(spp, fwhm):
        beam_size = sqrt(2. * pi * (fwhm * fwhm2sigma) ** 2)       # arcsec
        sigma_noise = spp * (60. * 60. / beam_size)                # µK
        return random.randn(npix, npix) * sigma_noise

    noise_c = noise(Simulation_Parameters["spp"], Simulation_Parameters["fwhm"])
    noise_low = noise(Simulation_Parameters["spp_low"], Simulation_Parameters["fwhm_low"])
    noise_high = noise(Simulation_Parameters["spp_high"], Simulation_Parameters["fwhm_high"])

    _debug_plot(noise_c, 'Noise.pdf')
    _debug_plot(noise_low, 'Noise_low.pdf')
    _debug_plot(noise_high, 'Noise_high.pdf')

    total_map = patch + noise_c + Tmap
    total_map_low = patch_low + noise_low + Tmap_low
    total_map_high = patch_high + noise_high + Tmap_high

    _debug_plot(total_map, 'total_map.pdf')
    _debug_plot(total_map_low, 'total_map_low.pdf')
    _debug_plot(total_map_high, 'total_map_high.pdf')

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

def CreateRandomCatalogue(nmcat=None, Ns=1000, ex_rad=6., cut=15.):
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

    Ns = int(Ns)

    # Generate random all-sky catalogue (|b| > cut)
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

        # Árboles con las posiciones aleatorias y las del catálogo de entrada
        T = sp.cKDTree(lonlat_to_xyz(vstack((glon, glat)).swapaxes(0, 1)))
        iT = sp.cKDTree(lonlat_to_xyz(vstack((iglon, iglat)).swapaxes(0, 1)))
        chord = 2. * sin(ex_rad / 60. / 180. * pi / 2.)
        pairs = T.query_ball_tree(iT, chord)

        mask = ones(glon.size, dtype=bool)

        for ii in range(len(pairs)):
            if pairs[ii]:
                mask[ii] = False

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
    import argparse
    parser = argparse.ArgumentParser(description="Simulaciones de parches del cielo de microondas para CENN.")
    parser.add_argument("mode", nargs="?", default="simulate", choices=["simulate", "fullsky"],
                        help="simulate: genera un dataset .h5; fullsky: mapas de cielo completo (figuras 10 y 11)")
    parser.add_argument("--par", default="./sim_cfreq217.par", help="archivo de parámetros (JSON)")
    parser.add_argument("--out", default="../../data/output_sims_train",
                        help="ruta de salida sin la extensión .h5")
    parser.add_argument("--n_sims", type=int, default=None,
                        help="número de simulaciones (por defecto, el valor n_sims del .par)")
    args = parser.parse_args()

    if args.mode == "fullsky":
        plot_fullsky_from_parfile(args.par)
    else:
        make_parallel_simu4CMB(args.par, args.out, args.n_sims)


if __name__ == "__main__":
    main()
