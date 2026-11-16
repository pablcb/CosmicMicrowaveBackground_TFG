"""
DecreaseRES.py — Degradación de la resolución de los parches simulados.

Aplica un filtro gaussiano a cada canal del parche total ("M") para llevar su
resolución desde el FWHM original de cada frecuencia hasta un FWHM común de
600" (10 arcmin). La etiqueta ("M0", el CMB) se guarda sin modificar.

    sigma_canal = sqrt(FWHM_nuevo^2 - FWHM_original^2) / (2 sqrt(2 ln 2)) / pixsize

Se usó para los dos casos de resolución alterada del TFG:
  - modelo de alta resolución validado con entradas de baja resolución,
  - modelo re-entrenado (train/test/validación) a baja resolución.

Uso (desde esta carpeta), cambiando INPUT_FILE/OUTPUT_FILE para cada set
(train, test, validate):
    python DecreaseRES.py
"""

import os
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Archivos de entrada/salida (relativos a src/preprocessing/)
INPUT_FILE = "../../data/output_sims_train.h5"
OUTPUT_FILE = "../../data/DecreaseRES/output_sims_degraded_train.h5"

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
with h5py.File(INPUT_FILE, "r") as h5f:
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
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with h5py.File(OUTPUT_FILE, "w") as h5f_out:
    h5f_out.create_dataset("M", data=inputs_smoothed)
    h5f_out.create_dataset("M0", data=labels)

print("Archivo guardado como", OUTPUT_FILE)
