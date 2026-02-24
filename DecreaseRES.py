import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Parámetros
fwhm_new = 600.0  # arcsec
in_fwhm = np.array([433.2, 294.0, 295.2])  # arcsec por canal
pixsize = 90.0  # arcsec/píxel

# Calcular sigma en píxeles
sigma = np.sqrt(fwhm_new**2 - in_fwhm**2)
sigma = sigma / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # en arcsec
sigma = sigma / pixsize  # en píxeles

print("Sigma por canal:", sigma)

# Lectura de nuestro archivo .h5
with h5py.File("output_sims_train.h5", "r") as h5f:
    inputs = h5f["M"][:].astype(np.float32)
    labels = h5f["M0"][:].astype(np.float32)


# Aplicamos filtro gaussiano a cada canal
inputs_smoothed = np.copy(inputs)
for i in range(3):  
    for j in range(inputs.shape[0]):  
        inputs_smoothed[j, :, :, i] = gaussian_filter(inputs[j, :, :, i], sigma[i])

'''
# Pintamos los plots para comparar
idx = 0
fig, axs = plt.subplots(3, 2, figsize=(10, 8))
for i in range(3):
    axs[i, 0].imshow(inputs[idx, :, :, i], cmap='inferno')
    axs[i, 0].set_title(f'Canal {i} - Original')
    axs[i, 1].imshow(inputs_smoothed[idx, :, :, i], cmap='inferno')
    axs[i, 1].set_title(f'Canal {i} - Degradado')
plt.tight_layout()
plt.show()
'''

# Guardar nuevo archivo .h5 con los datos degradados
with h5py.File("output_sims_degraded_train.h5", "w") as h5f_out:
    h5f_out.create_dataset("M", data=inputs_smoothed)
    h5f_out.create_dataset("M0", data=labels)

print("Archivo guardado como output_sims_degraded.h5")
