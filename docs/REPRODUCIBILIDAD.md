# Reproducibilidad del entrenamiento

Este documento fija la configuración exacta con la que se entrenan las cinco redes del artículo, para que cualquier persona pueda volver a obtener los mismos checkpoints.

Todos los comandos se ejecutan desde la raíz del repositorio.

## Condiciones comunes

| Elemento | Valor |
|---|---|
| Semilla | `42` (`--seed 42`) para las cinco redes |
| División de sujetos | `data_split.json` (20 train / 5 val / 5 test) |
| Dataset | `~/Documents/DataSet` (`--dataset`) |
| Filtro de la señal (redes) | Butterworth paso bajo, orden 6, 7 Hz |

`data_split.json`:

- Train: 25, 1, 27, 29, 12, 30, 10, 23, 9, 28, 19, 22, 5, 11, 17, 4, 13, 26, 7, 6
- Validación: 8, 3, 2, 16, 20
- Test: 24, 18, 14, 21, 15

## Entrenar todo

```bash
./train_all.sh
```

Para entrenar solo algunas redes:

```bash
./train_all.sh tcn ttl
```

Los logs quedan en `nn-results/train_logs/<red>.log`. El script sobrescribe los checkpoints de `nn-results/<red>/checkpoints/`, así que conviene hacer un backup antes.

Variables de entorno opcionales: `PYTHON` (por defecto `.venv/bin/python`), `SEED`, `SPLIT`, `DATASET`, `LOG_DIR`.

## Comando de cada red

### 1. Clasificador de pie de apoyo (`GaitStaticPivotNet`)

```bash
python -m models.nn_pivot_classifier --epochs 100 --batch-size 1024 --lr 1e-3 --patience 25 --seed 42 --data-split data_split.json
```

- Capas ocultas [16, 8, 8], dropout 0.15. Entrada por frame (sin ventanas).
- AdamW (fused en GPU), weight decay 1e-4, ReduceLROnPlateau (factor 0.5, paciencia 5), pérdida BCE.
- Checkpoint: `nn-results/pivot_classifier/checkpoints/best_pivot_classifier.pt`

### 2. Pose relativa al pie de apoyo (`GaitStaticNet`)

```bash
python -m models.nn_static_pivot --epochs 300 --batch-size 128 --lr 1e-3 --patience 30 --seed 42 --data-split data_split.json
```

- Capas ocultas [64, 64, 32], dropout [0.15, 0.15, 0.1]. Entrada por frame.
- AdamW (fused en GPU), weight decay 1e-4, ReduceLROnPlateau (factor 0.5, paciencia 5).
- Checkpoint: `nn-results/static_pivot/checkpoints/best_static_model.pt`

### 3. TCN Lite (`GaitMultiTaskNetTCN`, modelo propuesto)

```bash
python -m models.nn_tcn_lite --window-size 25 --window-stride 2 --batch-size 12 --lr 5e-4 --epochs 500 --patience 60 --seed 42 --data-split data_split.json
```

- Canales [8, 16, 32, 32], kernels [7, 7, 5, 5], dilataciones [1, 2, 4, 8], cabezas de 48 neuronas, dropout 0.2 (14,729 parámetros).
- AdamW, weight decay 1e-5, CosineAnnealingWarmRestarts (T_0 = 30, T_mult = 2), precisión mixta en GPU.
- Checkpoint: `nn-results/tcn_lite/checkpoints/best_tcn_lite.pt`

### 4. TTL Lite (`GaitMultiTaskNetTTLLite`)

```bash
python -m models.nn_ttl_lite --window-size 25 --window-stride 3 --batch-size 24 --lr 5e-4 --epochs 300 --patience 60 --seed 42 --data-split data_split.json
```

- TCN [16, 16, 32] (kernels [7, 7, 5], dilataciones [1, 2, 4]), BiLSTM 32 × 1 capa, atención con 2 cabezas, dropout 0.2.
- AdamW, weight decay 1e-5, CosineAnnealingWarmRestarts (T_0 = 30, T_mult = 2), precisión mixta en GPU.
- Checkpoint: `nn-results/ttl_lite/checkpoints/best_ttl_lite.pt`

### 5. Transformer multitarea (`GaitMultiTaskNet`)

```bash
python -m models.nn_transformer_multitask --window-size 25 --window-stride 3 --batch-size 24 --lr 5e-4 --epochs 300 --patience 60 --seed 42 --data-split data_split.json
```

- CNN [4, 8, 16], BiLSTM 12 × 3 capas, atención con 12 cabezas, dropout 0.2.
- AdamW, weight decay 1e-5, CosineAnnealingWarmRestarts (T_0 = 30, T_mult = 2), precisión mixta en GPU.
- Checkpoint: `nn-results/transformer_multitask/checkpoints/best_multitask_model_trans.pt`

## Qué guarda cada checkpoint

Además de los pesos, cada `.pt` guarda:

- `norm_stats`: media y desviación de la normalización z-score, calculadas solo con train.
- `config`: hiperparámetros reales del entrenamiento (semilla, batch, lr, ventana, paso, sujetos de cada conjunto, arquitectura).
- `epoch` y `val_loss` del mejor modelo.

Los scripts de evaluación leen la ventana de inferencia y la normalización desde el propio checkpoint, así que no dependen de los `norm_stats.json` sueltos.

## Determinismo

Con la misma semilla, el orden de los batches y el aumento de datos se repiten exactamente. Al inicio, cada script fija `random`, `numpy` y `torch`, y PyTorch siembra los workers del `DataLoader` a partir de esa semilla.

Se comprobó que dos entrenamientos de cada red con la misma semilla producen pesos idénticos bit a bit, en este entorno:

- GPU NVIDIA GeForce RTX 5080 (driver 595.99.02)
- PyTorch 2.14.0 (CUDA 13.0, cuDNN 9.24), NumPy 2.5.2

En otra GPU u otra versión de PyTorch/CUDA los kernels pueden cambiar, y el resultado puede diferir ligeramente aunque la semilla sea la misma.

Cambiar la arquitectura también cambia el orden de los batches, aunque la semilla sea la misma: la inicialización del modelo consume números aleatorios antes de crear los batches.

## Origen de cada configuración

- **TCN Lite, TTL Lite y Transformer:** al reentrenarlas con estos comandos se obtuvieron pesos idénticos bit a bit a los checkpoints originales del artículo (misma época y misma `val_loss`).
- **Pivote y estática:** los checkpoints originales (septiembre de 2026) salieron de una versión del código que no está en git y no se pudieron reproducir. La configuración actual se eligió con una búsqueda en rejilla de batch × lr, seleccionando por la `val_loss` (nunca por test) con semilla 42:

| Red | Batch | lr | Mejor `val_loss` |
|---|---|---|---|
| Estática | 64 / **128** / 256 | **1e-3** / 3e-4 | **0.00080** (128, 1e-3; con 300 épocas, 0.00075 en la época 201) |
| Pivote | 256 / **1024** | **1e-3** / 5e-3 | **0.3448** (1024, 1e-3, época 20) |

Los checkpoints originales se conservan en `backups/nn-results_antes_reentreno_2026-11-12/`.

## Resultados actuales (reentrenamiento del 2026-11-12)

Evaluación completa en `nn-results/full_neural_vs_analytical/results-v4.txt`; la anterior, con el pivote y la estática originales, está en `results-v3.txt`. TCN Lite, TTL Lite, Transformer y el pipeline analítico no cambian. Solo cambian las filas de la línea base estática:

| Métrica (test, filtro P95) | v3 (originales) | v4 (actuales) |
|---|---|---|
| Exactitud del pivote en el pipeline | 88.25 % | 89.64 % |
| MPJPE relativo, Estática (P+P) | 0.0444 m | 0.0338 m |
| MPJPE global, Estática (P+P) | 0.2669 m | 0.2670 m |
| MAE / RMSE global, Estática (P+P) | 0.0917 / 0.1470 m | 0.0927 / 0.1493 m |
| MPJPE global, Oráculo | 0.3012 m | 0.2938 m |
| MAE / RMSE global, Oráculo | 0.0981 / 0.1594 m | 0.1002 / 0.1633 m |
| P95 del error por frame, Estática (figuras) | ~65 cm | 63.0 cm |

Consecuencia para el artículo: la red estática ahora tiene el **mejor MPJPE relativo a la pelvis** (0.034 m frente a 0.036 m de TCN Lite), pero sigue siendo mucho peor en el marco global (0.27 m frente a 0.066 m). `v2_manuscript/template.tex` y las figuras de `v2_manuscript/figures/` ya están actualizados con estos valores.

## Evaluación

```bash
python evaluate_full_neural_vs_analytical.py
python generate_manuscript_figures.py
```

La evaluación usa los mismos sujetos de test de `data_split.json` y la ventana guardada en cada checkpoint (25 muestras).
