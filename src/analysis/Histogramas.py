"""
Histogramas.py — Estadística de los residuos sobre todo el set de validación.

Para cada una de las simulaciones de validación calcula la media y la desviación
estándar del mapa de residuos (salida de la red - CMB real) y dibuja sus
histogramas, anotando la media global y la dispersión (en µK).
Genera las figuras 25 y 30 de la memoria. La incertidumbre global combinada se
obtiene como sigma_total = sqrt(std(medias)^2 + mean(desviaciones)^2).

Uso (desde esta carpeta): elige el caso descomentando el par de rutas
archivo_total / archivo_limpio correspondiente y ejecuta
    python Histogramas.py
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

# Rutas de los archivos HDF5 (relativas a src/analysis/; ver data/README.md)
archivo_total = "../../data/output_sims_validate.h5"
#archivo_total = "../../data/DecreaseRES/output_sims_degraded_validate.h5"  # Para redlow y lowres

#archivo_limpio = "../../data/output_sims_validate_Outputs_CENN_I.h5"  # Contiene M (parche limpiado)
archivo_limpio = "../../data/output_sims_validate_Outputs_CENN_I_conv.h5"  # Contiene M (parche limpiado)
#archivo_limpio = "../../data/output_sims_validate_Outputs_CENN_I_Leaky.h5"  # Contiene M (parche limpiado)
#archivo_limpio = "../../data/DecreaseRES/outs/output_sims_degraded_Outputs_CENN_I_RedLOW.h5" #redlow
#archivo_limpio = "../../data/DecreaseRES/outs/output_sims_degraded_Outputs_CENN_I_LOWRES.h5" #lowres

# Abrir el archivo con M y M0
with h5py.File(archivo_total, "r") as f_total:
    M = f_total["M"][:]   # Parche total
    M0 = f_total["M0"][:] # Label

# Abrir el archivo con M (parche limpiado)
with h5py.File(archivo_limpio, "r") as f_clean:
    TestValidacion = f_clean["M"][:]  # Parche limpiado por la red

# Creamos los residuos (parche limpiado - label)
residuos = TestValidacion - M0

# Calculamos la media y desviación estándar de los residuos para cada simulación
media_residuos = np.mean(residuos, axis=(1, 2)) * 1e6  # Convertido a µK
desviacion_residuos = np.std(residuos, axis=(1, 2)) * 1e6  # Convertido a µK

# Calculamos medias y desviaciones globales
media_total = np.mean(media_residuos)
std_total = np.std(media_residuos)

desviacion_total = np.mean(desviacion_residuos)
std_desviacion = np.std(desviacion_residuos)

# Incertidumbre global combinada: sqrt(std(medias)^2 + mean(desviaciones)^2)
sigma_total = np.sqrt(std_total**2 + desviacion_total**2)

print(f"Media de las medias: {media_total:.2f} µK | Dispersión de las medias: {std_total:.2f} µK")
print(f"Media de las desviaciones: {desviacion_total:.2f} µK | Dispersión de las desviaciones: {std_desviacion:.2f} µK")
print(f"Incertidumbre global combinada: {sigma_total:.2f} µK")

# Pintamos los histogramas
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

hist1 = axes[0].hist(media_residuos, bins=125, color='blue', alpha=0.7, edgecolor='black')
axes[0].set_title("Histograma de Medias de Residuos")
axes[0].set_xlim([-200, 350])  
axes[0].set_xlabel("Media de Residuos (µK)")
axes[0].set_ylabel("Frecuencia")

max_height1 = max(hist1[0]) 

# Agregamos texto en el histograma
axes[0].annotate(f"Media: {media_total:.2f} µK\nDesviación: {std_total:.2f} µK", 
                 xy=(50, max_height1), xytext=(100, max_height1 * 0.9), 
                 fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.6))


hist2 = axes[1].hist(desviacion_residuos, bins=85, color='red', alpha=0.7, edgecolor='black')
axes[1].set_title("Histograma de Desviaciones Estándar de Residuos")
axes[1].set_xlim([0, 140])  
axes[1].set_xlabel("Desviación Estándar de Residuos (µK)")
axes[1].set_ylabel("Frecuencia")

max_height2 = max(hist2[0])

# Agregamos texto en el histograma
axes[1].annotate(f"Media: {desviacion_total:.2f} µK\nDesviación: {std_desviacion:.2f} µK", 
                 xy=(20, max_height2), xytext=(40, max_height2 * 0.9), 
                 fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.6))

# Guardamos la figura
plt.savefig("Histogramas_Residuos.png", bbox_inches='tight')
plt.show()
