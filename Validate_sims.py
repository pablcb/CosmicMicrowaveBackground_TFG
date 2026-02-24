import h5py
import numpy as np
import matplotlib.pyplot as plt

# Rutas de los archivos HDF5
# archivo_total = "C:/Users/pablo/Desktop/TFG/Comprobacion Simulacion/DecreaseRES/output_sims_degraded_validate.h5"  # Para redlow y lowres
archivo_total = "C:/Users/pablo/Desktop/TFG/Comprobacion Simulacion/output_sims_validate.h5"  

# archivo_limpio = "C:/Users/pablo/Desktop/TFG/Comprobacion Simulacion/output_sims_validate_Outputs_CENN_I.h5" # ReLU
# archivo_limpio = "C:/Users/pablo/Desktop/TFG/Comprobacion Simulacion/output_sims_validate_Outputs_CENN_I_conv.h5" # conv
# archivo_limpio = "C:/Users/pablo/Desktop/TFG/Comprobacion Simulacion/output_sims_validate_Outputs_CENN_I_Leaky.h5" # Leaky
# archivo_limpio = "C:/Users/pablo/Desktop/TFG/Comprobacion Simulacion/DecreaseRES/outs/output_sims_degraded_Outputs_CENN_I_RedLOW.h5" #redlow
archivo_limpio = "C:/Users/pablo/Desktop/TFG/Comprobacion Simulacion/DecreaseRES/outs/output_sims_degraded_Outputs_CENN_I_LOWRES.h5" #lowres

# Número de simulaciones aleatorias a extraer
num_simulaciones = 5

# Abrir el archivo con M y M0
with h5py.File(archivo_total, "r") as f_total:
    print("Datasets en archivo_total:", list(f_total.keys()))  

    # Obtenemos el total de simulaciones disponibles
    total_simulaciones = f_total["M"].shape[0]

    indices = np.sort(np.random.choice(total_simulaciones, num_simulaciones, replace=False))

    M = f_total["M"][indices]   # Parche total
    M0 = f_total["M0"][indices] # Label

# Abrir el archivo con M (parche limpiado)
with h5py.File(archivo_limpio, "r") as f_clean:
    print("Datasets en archivo_limpio:", list(f_clean.keys()))  

    # Extraer los mismos índices para el parche limpiado
    TestValidacion = f_clean["M"][indices]  # Parche limpiado por la red

# Creamos los residuos (parche limpiado - label)
residuos = TestValidacion - M0

'''
# Calcular la media y desviación estándar de los residuos
media_residuos = np.mean(residuos)
desviacion_residuos = np.std(residuos)

# Imprimir los resultados
print(f"Media de los residuos: {media_residuos:.6f}")
print(f"Desviación estándar de los residuos: {desviacion_residuos:.6f}")
'''

# Seleccionar canal central (input total en la frecuencia del CMB)
M_central = M[:, :, :, 1]

# Convertimos todo a microkelvin
M_uk = M_central * 1e6
M0_uk = M0 * 1e6
TestValidacion_uk = TestValidacion * 1e6
residuos_uk = (TestValidacion - M0) * 1e6

# Creamos la figura para hacer los plots
fig, axes = plt.subplots(num_simulaciones, 4, figsize=(18, 4 * num_simulaciones))

for i in range(num_simulaciones):
    # INPUT TOTAL
    im = axes[i, 0].imshow(M_uk[i], cmap='viridis')
    axes[i, 0].set_title("Entrada parche total")
    plt.colorbar(im, ax=axes[i, 0], shrink=0.9)

    # CMB REAL (LABEL)
    im = axes[i, 1].imshow(M0_uk[i], cmap='viridis')
    axes[i, 1].set_title("Entrada CMB")
    plt.colorbar(im, ax=axes[i, 1], shrink=0.9)

    # OUTPUT DE LA RED (CENN)
    im = axes[i, 2].imshow(TestValidacion_uk[i], cmap='viridis')
    axes[i, 2].set_title("Salida CENN")
    plt.colorbar(im, ax=axes[i, 2], shrink=0.9)

    # RESIDUOS
    im = axes[i, 3].imshow(residuos_uk[i], cmap='viridis')
    axes[i, 3].set_title("Residuos")
    plt.colorbar(im, ax=axes[i, 3], shrink=0.9)

plt.tight_layout()
plt.savefig("Residuos_LOWRES.png", dpi=300)
plt.show()

