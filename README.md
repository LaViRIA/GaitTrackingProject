# GaitTrackingProject

Reconstrucción 3D de la marcha del tren inferior a partir de 4 IMUs, comparando un pipeline cinemático analítico contra varias redes neuronales, con marcadores ArUco como ground truth.

## Estructura

```
process_all.py                         Pipeline analítico (Madgwick/EKF + PCA + cinemática directa); librería base
evaluate_full_neural_vs_analytical.py  Comparativa final: analítico vs todas las redes
experiment_matrix.py                   Grid search de hiperparámetros del pipeline analítico
experiment_matrix_tcn.py               Grid search de hiperparámetros de TCN Lite
analyze_results.py                     Análisis/gráficas del CSV de experiment_matrix.py
generate_manuscript_figures.py         Figuras del manuscrito (v2_manuscript/figures/)
generate_split.py, data_split.json     División fija de sujetos train/val/test

models/                                Redes neuronales (entrenamiento + clases usadas en la evaluación)
  nn_tcn_lite.py                       TCN dilatada no causal (modelo propuesto)
  nn_ttl_lite.py                       Híbrido TCN + Transformer
  nn_transformer_multitask.py          Conv1D + BiLSTM + Transformer
  nn_pivot_classifier.py               Clasificador de pie de apoyo
  nn_static_pivot.py                   Pose relativa al pie pivote

nn-results/                            Checkpoints y resultados de las redes
v2_manuscript/                         Manuscrito (template.tex/.md/.pdf, secciones, figuras)
docs/                                  Notas de trabajo, guía de códigos, respuesta a revisores
```

`Results/` (salidas del pipeline analítico y de las matrices) no está versionado.

## Uso

Todos los comandos se ejecutan desde la raíz del repositorio.

```bash
pip install -r requirements.txt

python process_all.py                          # pipeline analítico
python experiment_matrix.py                    # grid search analítico
python analyze_results.py                      # análisis del grid search
python -m models.nn_tcn_lite                   # entrenar un modelo (igual para los demás en models/)
python experiment_matrix_tcn.py                # grid search TCN Lite
python evaluate_full_neural_vs_analytical.py   # comparativa final
python generate_manuscript_figures.py          # figuras del manuscrito
```

Ver [docs/GUIA_DE_CODIGOS.md](docs/GUIA_DE_CODIGOS.md) para el detalle de cada script.
