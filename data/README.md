# Datos

Los datasets del TFG (miles de parches de 256×256 píxeles en formato HDF5) **no se incluyen en el repositorio** por su tamaño. Esta carpeta es el lugar donde los scripts los buscan por defecto. Las rutas son relativas a cada carpeta de `src/`.

## Formato de los archivos `.h5`

| Archivo | Dataset | Forma | Contenido |
|:--|:--|:--|:--|
| Simulaciones (`output_sims_*.h5`) | `M` | `(N, 256, 256, 3)` | Parche total en 143, 217 y 353 GHz |
| | `M0` | `(N, 256, 256, 1)` | Etiqueta: señal CMB |
| Salidas de la red (`*_Outputs_CENN_I*.h5`) | `M` | `(N, 256, 256, 1)` | CMB recuperado por la red (desnormalizado) |

Los scripts de análisis tratan los mapas como valores en K y los convierten a µK multiplicando por 10⁶.

## Organización esperada

```
data/
├── output_sims_train.h5                          # 10000 simulaciones
├── output_sims_test.h5                           # 3000 simulaciones
├── output_sims_validate.h5                       # 1000 simulaciones
├── output_sims_validate_Outputs_CENN_I.h5        # salida NN ReLU
├── output_sims_validate_Outputs_CENN_I_conv.h5   # salida variante con capas conv extra
├── output_sims_validate_Outputs_CENN_I_Leaky.h5  # salida NN LeakyReLU
├── DecreaseRES/
│   ├── output_sims_degraded_{train,test,validate}.h5   # sets suavizados a FWHM 10'
│   └── outs/
│       ├── output_sims_degraded_Outputs_CENN_I_LOWRES.h5   # modelo de alta resolución validado a baja resolución
│       └── output_sims_degraded_Outputs_CENN_I_RedLOW.h5   # red re-entrenada a baja resolución
└── PowerSpec/                                    # entradas de CMB_Analysis.py
    ├── Validation.h5                             # set de validación (etiquetas M0)
    ├── Leaky.h5                                  # salida de alta resolución
    ├── Leaky_LOWRES.h5                           # alta resolución validada a baja resolución
    └── Leaky_REDLOW.h5                           # re-entrenada a baja resolución
```

`CENN.py` guarda sus salidas en `src/model/Models_I/`. Después se copian aquí, renombradas con el sufijo del caso correspondiente (`_Leaky`, `_LOWRES`, `_RedLOW`, ...), para que los scripts de `src/analysis/` las encuentren.
