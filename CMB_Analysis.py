#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PEP 8 Style

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Directorio de los datos
datasets_directory = "C:/Users/pablo/Desktop/TFG/Comprobacion Simulacion/PowerSpec/"

# Rutas de los archivos .h5 de salida de red neuronal
compare_file_list = [
    datasets_directory + 'Leaky.h5',
    datasets_directory + 'Leaky_LOWRES.h5',
    datasets_directory + 'Leaky_REDLOW.h5'
]

# Lista de validación (si es común, se repite el mismo archivo)
validation_file_list = [
    datasets_directory + 'Validation.h5'
] * len(compare_file_list)

# Directorio para guardar imágenes
Images_Directory = datasets_directory

class Read():

    def reading_the_data(flag):

        with h5py.File(validation_file_list[flag], "r") as f:
            a_group_key = list(f.keys())[1]
            Input_CMB = list(f[a_group_key])[5]

        with h5py.File(compare_file_list[flag], "r") as g:
            b_group_key = list(g.keys())[0]
            Output_Net = list(g[b_group_key])[5]

        return Input_CMB, Output_Net

class Analysis():

    def pkestimator(Imap, Area=1):
        from numpy import fft, indices, hypot, array, size, sqrt
        from numpy import argsort, mean, append, zeros, unique, pi

        # Imap = Imap / mean(Imap.flat) - 1
        Pmap = abs(fft.fftshift(fft.ifft2(Imap))) ** 2
        x, y = indices(Pmap.shape)
        center = array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])
        r = hypot(x - center[0], y - center[1])
        del x, y

        ind = argsort(r.flat)
        r = r.flat[ind]
        Pmap = Pmap.flat[ind]

        u, ind = unique(r, return_index=True)
        ind = append(ind, size(Pmap))
        Pk = zeros(u.shape, dtype=float)
        for i in range(size(ind) - 1):
            Pk[i] = mean(Pmap[ind[i]:ind[i + 1]])

        return u * 2.0 * pi / sqrt(Area), Pk

    def analysing_residual_map(flag):

        Input_CMB, Output_Net = Read.reading_the_data(flag)
        residual_map = np.zeros([256, 256, 1])

        residual_map = Input_CMB - Output_Net

        return residual_map

    def estimating_CMB_Pk(flag):

        Input_CMB, Output_Net = Read.reading_the_data(flag)
        residual_map = Analysis.analysing_residual_map(flag)

        # Patch_Size = 128-10
        Patch_Size = 256
        Pixsize = 90
        Area = (Patch_Size**2)*(Pixsize**2)*(((np.pi)/(180*3600))**2)

        k_Input_CMB, Pk_Input_CMB = Analysis.pkestimator(
            Input_CMB[:Patch_Size, :Patch_Size, 0], Area)
        k_Output_CMB, Pk_Output_CMB = Analysis.pkestimator(
            Output_Net[:Patch_Size, :Patch_Size, 0], Area)
        k_residual, Pk_residual_map = Analysis.pkestimator(
            residual_map[:Patch_Size, :Patch_Size, 0], Area)

        return Pk_Input_CMB, Pk_Output_CMB, Pk_residual_map, k_Input_CMB

    def estimating_CMB_Cl(flag):

        Pk_Input_CMB, Pk_Output_CMB, Pk_residual_map, k_Input_CMB = Analysis.estimating_CMB_Pk(
            flag)

        Patch_Size = 256
        Pixsize = 90
        Area = (Patch_Size**2)*(Pixsize**2)*(((np.pi)/(180*3600))**2)

        Cl_Input_CMB = Area*Pk_Input_CMB
        Cl_Output_CMB = Area*Pk_Output_CMB
        Cl_residual_map_CMB = Area*Pk_residual_map

        l_Input_CMB = [x + 0.5 for x in k_Input_CMB]

        return Cl_Input_CMB, Cl_Output_CMB, Cl_residual_map_CMB, l_Input_CMB

    def estimating_CMB_PowerSpectrum(flag):

        Cl_Input_CMB, Cl_Output_CMB, Cl_residual_map_CMB, l_Input_CMB = Analysis.estimating_CMB_Cl(
            flag)

        dT_Input_CMB = (
            ([x*(x+1) for x in l_Input_CMB]*Cl_Input_CMB)/(2*np.pi))
        dT_Output_CMB = (
            ([x*(x+1) for x in l_Input_CMB]*Cl_Output_CMB)/(2*np.pi))
        dT_Residual_CMB = (([x*(x+1) for x in l_Input_CMB]
                           * Cl_residual_map_CMB)/(2*np.pi))

        return dT_Input_CMB, dT_Output_CMB, dT_Residual_CMB, l_Input_CMB
    

    def bin_power_spectrum(ls, cls, step):
        """
        Bin the power spectrum using a fixed step size.

        Parameters:
        - ls (array): Multipole values (e.g., l = 2,3,4,...)
        - cls (array): Power spectrum values corresponding to ls
        - step (int): Bin step size (e.g., 50, 100, 200)

        Returns:
        - l_binned (array): Mean l in each bin
        - cl_binned (array): Mean Cl in each bin
        - std_cl (array): Std of Cl in each bin
        """

        l_min = np.min(ls)
        l_max = np.max(ls)

        # Define bin edges
        bin_edges = np.arange(l_min, l_max + step, step)

        # Prepare lists to collect results
        l_binned = []
        cl_binned = []
        std_cl = []

        for i in range(len(bin_edges) - 1):
            lmin = bin_edges[i]
            lmax = bin_edges[i + 1]

            # Mask: select ls and cls within this bin
            mask = (ls >= lmin) & (ls < lmax)
            l_in_bin = ls[mask]
            cl_in_bin = cls[mask]
        
            # If bin is not empty, compute mean and std
            if len(l_in_bin) > 0:
                l_binned.append(np.mean(l_in_bin))
                cl_binned.append(np.mean(cl_in_bin))
                std_cl.append(np.std(cl_in_bin))

        return np.array(l_binned), np.array(cl_binned), np.array(std_cl)

    def bin_spectrum(l, spectrum, step_LowMultipoles, step_HighMultipoles):
        """
        Bin the spectrum into low (l <= 1000) and high (l > 1000) multipoles with different step sizes.

        Parameters:
        - l (array): Multipole values
        - spectrum (array): Power spectrum values
        - step_LowMultipoles (int): Step size for l <= 1000
        - step_HighMultipoles (int): Step size for l > 1000

        Returns:
        - l_binned (array): Binned l values (low + high)
        - cl_binned (array): Binned spectrum values
        - std_cl_binned (array): Standard deviation of spectrum values in bins
        """

        # Split into low and high multipoles
        l_LowMultipoles = l[l <= 1000]
        l_HighMultipoles = l[l > 1000]
        spec_low = spectrum[l <= 1000]
        spec_high = spectrum[l > 1000]

        l_binned_low, cl_binned_low, std_cl_binned_low = Analysis.bin_power_spectrum(l_LowMultipoles, spec_low, step=step_LowMultipoles)
        l_binned_high, cl_binned_high, std_cl_binned_high = Analysis.bin_power_spectrum(l_HighMultipoles, spec_high, step=step_HighMultipoles)

        l_binned = np.concatenate([l_binned_low, l_binned_high])
        cl_binned = np.concatenate([cl_binned_low, cl_binned_high])
        std_cl_binned = np.concatenate([std_cl_binned_low, std_cl_binned_high])

        return l_binned, cl_binned, std_cl_binned

    def linear_rebinning_dT_By_Low_And_High_Multipoles_byLaura(flag):
        """
        Bins the input, output, and residual CMB power spectra with different binning for low and high multipoles.

        Parameters:
        - flag: Flag passed to the Analysis.estimating_CMB_PowerSpectrum() function (controls input data selection)

        Returns:
        - mean_Pk_input: Binned input spectrum
        - std_Pk_input: Std deviation of binned input spectrum
        - mean_Pk_output: Binned output spectrum
        - std_Pk_output: Std deviation of binned output spectrum
        - mean_Pk_residual: Binned residual spectrum
        - std_Pk_residual: Std deviation of binned residual spectrum
        - l_bin: Binned multipole values
        """

        dT_Input_CMB, dT_Output_CMB, dT_Residual_CMB, l_Input_CMB = Analysis.estimating_CMB_PowerSpectrum(flag)

        # Convert spectra from K^2 units to μK^2 
        dT_Input_CMB = dT_Input_CMB*1e12
        dT_Output_CMB = dT_Output_CMB*1e12
        dT_Residual_CMB = dT_Residual_CMB*1e12

        step_LowMultipoles = 25
        step_HighMultipoles = 200

        l = np.array(l_Input_CMB)

        # Bin Input
        l_bin, mean_Pk_input, std_Pk_input = Analysis.bin_spectrum(l, dT_Input_CMB, step_LowMultipoles, step_HighMultipoles)

        # Bin Output
        _, mean_Pk_output, std_Pk_output = Analysis.bin_spectrum(l, dT_Output_CMB, step_LowMultipoles, step_HighMultipoles)

        # Bin Residual
        _, mean_Pk_residual, std_Pk_residual = Analysis.bin_spectrum(l, dT_Residual_CMB, step_LowMultipoles, step_HighMultipoles)
    
        return mean_Pk_input, std_Pk_input, mean_Pk_output, std_Pk_output, mean_Pk_residual, std_Pk_residual, l_bin


    def linear_rebinning_dT_By_Low_And_High_Multipoles(flag):

        dT_Input_CMB, dT_Output_CMB, dT_Residual_CMB, l_Input_CMB = Analysis.estimating_CMB_PowerSpectrum(
            flag)

        dT_Input_CMB = dT_Input_CMB*10e11
        dT_Output_CMB = dT_Output_CMB*10e11
        dT_Residual_CMB = dT_Residual_CMB*10e11

        step_LowMultipoles = 50
        step_HighMultipoles = 200

        l = np.array(l_Input_CMB)

        dT_Input_lowMultipoles = dT_Input_CMB[l <= 1000]
        dT_Output_lowMultipoles = dT_Output_CMB[l <= 1000]
        dT_Residual_lowMultipoles = dT_Residual_CMB[l <= 1000]
        dT_Input_HighMultipoles = dT_Input_CMB[l > 1000]
        dT_Output_HighMultipoles = dT_Output_CMB[l > 1000]
        dT_Residual_HighMultipoles = dT_Residual_CMB[l > 1000]

        l_LowMultipoles = l[l <= 1000]
        l_HighMultipoles = l[l > 1000]

        k_bin = np.arange(50, 2500, 25)
        l_bin = k_bin+0.5

        mean_Pk_input_Low_Multipoles = np.zeros(0)
        std_Pk_input_Low_Multipoles = np.zeros(0)
        mean_Pk_output_Low_Multipoles = np.zeros(0)
        std_Pk_output_Low_Multipoles = np.zeros(0)
        mean_Pk_residual_Low_Multipoles = np.zeros(0)
        std_Pk_residual_Low_Multipoles = np.zeros(0)
        mean_Pk_input_High_Multipoles = np.zeros(0)
        std_Pk_input_High_Multipoles = np.zeros(0)
        mean_Pk_output_High_Multipoles = np.zeros(0)
        std_Pk_output_High_Multipoles = np.zeros(0)
        mean_Pk_residual_High_Multipoles = np.zeros(0)
        std_Pk_residual_High_Multipoles = np.zeros(0)

        for element in range(len(l_bin)):

            mean_dummy_input = np.mean(dT_Input_lowMultipoles[np.abs(
                l_LowMultipoles - l_bin[element]) < (step_LowMultipoles/2)])
            std_dummy_input = np.std(dT_Input_lowMultipoles[np.abs(
                l_LowMultipoles - l_bin[element]) < (step_LowMultipoles/2)])
            mean_Pk_input_Low_Multipoles = np.append(
                mean_Pk_input_Low_Multipoles, mean_dummy_input)
            std_Pk_input_Low_Multipoles = np.append(
                std_Pk_input_Low_Multipoles, std_dummy_input)

            mean_dummy_output = np.mean(dT_Output_lowMultipoles[np.abs(
                l_LowMultipoles - l_bin[element]) < (step_LowMultipoles/2)])
            std_dummy_output = np.std(dT_Output_lowMultipoles[np.abs(
                l_LowMultipoles - l_bin[element]) < (step_LowMultipoles/2)])
            mean_Pk_output_Low_Multipoles = np.append(
                mean_Pk_output_Low_Multipoles, mean_dummy_output)
            std_Pk_output_Low_Multipoles = np.append(
                std_Pk_output_Low_Multipoles, std_dummy_output)

            mean_dummy_residual = np.mean(dT_Residual_lowMultipoles[np.abs(
                l_LowMultipoles - l_bin[element]) < (step_LowMultipoles/2)])
            std_dummy_residual = np.std(dT_Residual_lowMultipoles[np.abs(
                l_LowMultipoles - l_bin[element]) < (step_LowMultipoles/2)])
            mean_Pk_residual_Low_Multipoles = np.append(
                mean_Pk_residual_Low_Multipoles, mean_dummy_residual)
            std_Pk_residual_Low_Multipoles = np.append(
                std_Pk_residual_Low_Multipoles, std_dummy_residual)

            mean_dummy_input = np.mean(dT_Input_HighMultipoles[np.abs(
                l_HighMultipoles - l_bin[element]) < (step_HighMultipoles/2)])
            std_dummy_input = np.std(dT_Input_HighMultipoles[np.abs(
                l_HighMultipoles - l_bin[element]) < (step_HighMultipoles/2)])
            mean_Pk_input_High_Multipoles = np.append(
                mean_Pk_input_High_Multipoles, mean_dummy_input)
            std_Pk_input_High_Multipoles = np.append(
                std_Pk_input_High_Multipoles, std_dummy_input)

            mean_dummy_output = np.mean(dT_Output_HighMultipoles[np.abs(
                l_HighMultipoles - l_bin[element]) < (step_HighMultipoles/2)])
            std_dummy_output = np.std(dT_Output_HighMultipoles[np.abs(
                l_HighMultipoles - l_bin[element]) < (step_HighMultipoles/2)])
            mean_Pk_output_High_Multipoles = np.append(
                mean_Pk_output_High_Multipoles, mean_dummy_output)
            std_Pk_output_High_Multipoles = np.append(
                std_Pk_output_High_Multipoles, std_dummy_output)

            mean_dummy_residual = np.mean(dT_Residual_HighMultipoles[np.abs(
                l_HighMultipoles - l_bin[element]) < (step_HighMultipoles/2)])
            std_dummy_residual = np.std(dT_Residual_HighMultipoles[np.abs(
                l_HighMultipoles - l_bin[element]) < (step_HighMultipoles/2)])
            mean_Pk_residual_High_Multipoles = np.append(
                mean_Pk_residual_High_Multipoles, mean_dummy_residual)
            std_Pk_residual_High_Multipoles = np.append(
                std_Pk_residual_High_Multipoles, std_dummy_residual)

        m1 = np.append(mean_Pk_input_Low_Multipoles,
                       mean_Pk_input_High_Multipoles)
        mean_Pk_input = [x for x in m1 if x == x]
        m2 = np.append(std_Pk_input_Low_Multipoles,
                       std_Pk_input_High_Multipoles)
        std_Pk_input = [x for x in m2 if x == x]
        m3 = np.append(mean_Pk_output_Low_Multipoles,
                       mean_Pk_output_High_Multipoles)
        mean_Pk_output = [x for x in m3 if x == x]
        m4 = np.append(std_Pk_output_Low_Multipoles,
                       std_Pk_output_High_Multipoles)
        std_Pk_output = [x for x in m4 if x == x]
        m5 = np.append(mean_Pk_residual_Low_Multipoles,
                       mean_Pk_residual_High_Multipoles)
        mean_Pk_residual = [x for x in m5 if x == x]
        m6 = np.append(std_Pk_residual_Low_Multipoles,
                       std_Pk_residual_High_Multipoles)
        std_Pk_residual = [x for x in m6 if x == x]

        mean_Pk_input = np.array(mean_Pk_input)
        std_Pk_input = np.array(std_Pk_input)
        mean_Pk_output = np.array(mean_Pk_output)
        std_Pk_output = np.array(std_Pk_output)
        mean_Pk_residual = np.array(mean_Pk_residual)
        std_Pk_residual = np.array(std_Pk_residual)

        return mean_Pk_input, std_Pk_input, mean_Pk_output, std_Pk_output, mean_Pk_residual, std_Pk_residual, l_bin

    def linear_rebinning_dT(flag):

        dT_Input_CMB, dT_Output_CMB, dT_Residual_CMB, l_Input_CMB = Analysis.estimating_CMB_PowerSpectrum(flag)

        dT_Input_CMB = dT_Input_CMB*10e11
        dT_Output_CMB = dT_Output_CMB*10e11
        dT_Residual_CMB = dT_Residual_CMB*10e11

        step = 250
        k_bin = np.arange(1000, 2500, 30)
        l_bin = k_bin+0.5

        mean_Pk_input = np.zeros(0)
        std_Pk_input = np.zeros(0)
        mean_Pk_output = np.zeros(0)
        std_Pk_output = np.zeros(0)
        mean_Pk_residual = np.zeros(0)
        std_Pk_residual = np.zeros(0)

        for element in range(len(l_bin)):

            mean_dummy_input = np.mean(
                dT_Input_CMB[np.abs(l_Input_CMB - l_bin[element]) < (step/2)])
            std_dummy_input = np.std(dT_Output_CMB[np.abs(
                l_Input_CMB - l_bin[element]) < (step/2)])
            mean_Pk_input = np.append(mean_Pk_input, mean_dummy_input)
            std_Pk_input = np.append(std_Pk_input, std_dummy_input)

            mean_dummy_output = np.mean(
                dT_Output_CMB[np.abs(l_Input_CMB - l_bin[element]) < (step/2)])
            std_dummy_output = np.std(
                dT_Output_CMB[np.abs(l_Input_CMB - l_bin[element]) < (step/2)])
            mean_Pk_output = np.append(mean_Pk_output, mean_dummy_output)
            std_Pk_output = np.append(std_Pk_output, std_dummy_output)

            mean_dummy_residual = np.mean(
                dT_Residual_CMB[np.abs(l_Input_CMB - l_bin[element]) < (step/2)])
            std_dummy_residual = np.std(
                dT_Residual_CMB[np.abs(l_Input_CMB - l_bin[element]) < (step/2)])
            mean_Pk_residual = np.append(mean_Pk_residual, mean_dummy_residual)
            std_Pk_residual = np.append(std_Pk_residual, std_dummy_residual)

        return mean_Pk_input, std_Pk_input, mean_Pk_output, std_Pk_output, mean_Pk_residual, std_Pk_residual, l_bin

class Plots():
    
    def plotting_CMB_PowerSpectrum_Individual_Patch():
        f, (a0, a1, a2) = plt.subplots(3, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [1, 1, 1]})

        colors = ['tab:blue', 'tab:orange', 'tab:green']
        labels = ['Salida Alta Resolución', 'Salida Baja Resolución', 'Salida Baja Resolución Re-entrenada']

        for flag in range(len(compare_file_list)):
            mean_Pk_input, std_Pk_input, mean_Pk_output, std_Pk_output, mean_Pk_residual, std_Pk_residual, l_bin = Analysis.linear_rebinning_dT_By_Low_And_High_Multipoles_byLaura(flag)

            a0.plot(l_bin, mean_Pk_output, color=colors[flag], linestyle='-', label={labels[flag]} )
            a0.fill_between(l_bin, mean_Pk_input - std_Pk_input, mean_Pk_input + std_Pk_input, color=colors[flag], alpha=0.1)
            a0.fill_between(l_bin, mean_Pk_output - std_Pk_output, mean_Pk_output + std_Pk_output, color=colors[flag], alpha=0.1)

            diff = mean_Pk_input - mean_Pk_output
            a1.plot(l_bin, diff, color=colors[flag], label=f'{labels[flag]}'  r'$\quad \Delta\mathcal{D}_{l}^{TT}$')
            a1.fill_between(l_bin, diff - std_Pk_output, diff + std_Pk_output, color=colors[flag], alpha=0.1)

            a2.plot(l_bin, mean_Pk_residual, color=colors[flag], label=f'{labels[flag]} Residual')
            a2.fill_between(l_bin, mean_Pk_residual - std_Pk_residual, mean_Pk_residual + std_Pk_residual, color=colors[flag], alpha=0.1)

        a0.plot(l_bin, mean_Pk_input, color='black', label='Entrada CMB')
        a0.set_ylabel(r'$\mathcal{D}_\ell^{TT}$ [$\mu K^2$]')
        a1.set_ylabel(r'$\Delta\mathcal{D}_\ell^{TT}$ [$\mu K^2$]')
        a2.set_ylabel(r'$\mathcal{D}_\ell^{TT}$ [$\mu K^2$]')
        a2.set_xlabel(r'$\ell$')
        a1.axhline(y=0, color='black', linestyle='dashed', linewidth=1)

        a0.set_ylim(top=3500)
        a1.set_ylim(-1150, 1150)
        a2.set_ylim(0, 1500)

        a0.set_xlim(500,2600)
        a1.set_xlim(500,2600)
        a2.set_xlim(500,2600)

        a0.legend()
        a1.legend()
        a2.legend()

        f.tight_layout()
        f.savefig(Images_Directory + 'Pk_patches_Comparación_LowHighMultipoles_25bin.pdf')
        plt.show()

Plots.plotting_CMB_PowerSpectrum_Individual_Patch()

