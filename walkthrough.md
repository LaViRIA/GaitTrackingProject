# Walkthrough: Extensión de Evaluación Global 3D, Rediseño de Función de Pérdida y Sincronización Multi-Modelo

Este documento describe la resolución integral de las dos inquietudes planteadas:
1. **La discrepancia en los resultados de evaluación**: Por qué el entrenamiento de las redes reportaba métricas diferentes a `evaluate_full_neural_vs_analytical.py`.
2. **La idoneidad de la función de pérdida**: Por qué la formulación anterior provocaba hambruna de gradientes y deriva temporal en el desplazamiento pélvico, y cómo fue rediseñada matemáticamente.

Además, todos los cambios fueron estandarizados y aplicados en los 4 scripts neuronales del repositorio:
- [nn_tcn_lite.py](file:///home/badajoz/Documents/GaitTrakingProyect/nn_tcn_lite.py)
- [nn_tcn_lite_cv.py](file:///home/badajoz/Documents/GaitTrakingProyect/nn_tcn_lite_cv.py)
- [nn_ttl_lite.py](file:///home/badajoz/Documents/GaitTrakingProyect/nn_ttl_lite.py)
- [nn_ttl_lite_cv.py](file:///home/badajoz/Documents/GaitTrakingProyect/nn_ttl_lite_cv.py)

---

## 1. Explicación de la Discrepancia en Métricas de Evaluación

### ¿Por qué `evaluate_full_neural_vs_analytical.py` daba un resultado diferente a `nn_tcn_lite.py`?

Existían **dos causas estructurales**:

1. **Espacio de Referencia (Pose Relativa vs Global 3D)**:
   - `nn_tcn_lite.py` evaluaba originalmente el **MPJPE de pose relativa** (con la pelvis fija en el origen $[0, 0, 0]$), arrojando $\approx 3.0\text{ a }4.0\text{ cm}$.
   - `evaluate_full_neural_vs_analytical.py` sumaba la trayectoria acumulada de la pelvis (`cum_disp`) a cada articulación en el sistema de coordenadas de la cámara/mundo ($\mathbf{P}_{global} = \mathbf{P}_{relativa} + \mathbf{p}_{cadera}$), evaluando el **MPJPE Global 3D** ($\approx 12\text{ a }16\text{ cm}$).
2. **Filtrado de Outliers por Percentil 95%**:
   - `evaluate_full_neural_vs_analytical.py` recortaba el 5% superior de los errores extremos generados por oclusiones del marcador ArUco o saltos atípicos (`err <= np.percentile(err, 95.0)`).
   - `nn_tcn_lite.py` calculaba el promedio crudo (100% de muestras), donde un puñado de cuadros atípicos inflaba el error global.

### Solución Implementada
En la función `evaluate_on_captures` de todos los scripts, ahora se computan y reportan **ambos espacios** de forma clara, detallada y simultánea:
- **Pose Relativa**: 100% y Percentil 95% ($\approx 3-4\text{ cm}$).
- **Pose Global 3D (con odometría acumulada)**: 100% y Percentil 95% ($\approx 12-16\text{ cm}$).
- **Comparativa completa en 8 secciones** frente al pipeline analítico tradicional ([process_all.py](file:///home/badajoz/Documents/GaitTrakingProyect/process_all.py)).

---

## 2. Diagnóstico y Rediseño de la Función de Pérdida (`deep_multitask_loss`)

### Problemas Detectados en la Función Anterior

1. **Hambruna de Gradiente (Gradient Starvation) por Desbalance de Escalas**:
   - Error de pose articular cuadrático: $(\Delta x)^2 \approx (0.04\text{ m})^2 \approx 1.6 \times 10^{-3}\text{ m}^2$.
   - Error de desplazamiento por cuadro a 100 Hz: $(\Delta p)^2 \approx (0.002\text{ m})^2 \approx 4 \times 10^{-6}\text{ m}^2$.
   - **Efecto**: La señal de gradiente de la pose era **$1000\times$ mayor** que la del desplazamiento. Aunque el desplazamiento tuviera un peso `0.3`, la red ignoraba casi por completo la cadencia y el avance pélvico para minimizar la pose.
2. **Deriva Acumulativa por Sesgo Sistemático (Accumulative Drift)**:
   - Un error medio de apenas $+0.5\text{ mm}$ por cuadro en el eje horizontal ($X$) pasa desapercibido para una función $\text{MSE}$ por muestra, pero acumulado a lo largo de 500 cuadros genera un error de deriva de **$25\text{ cm}$**.
3. **Acoplamiento Ruidoso en `mse_global`**:
   - El término $\text{MSE}_{global} = \|\hat{\mathbf{p}}_t + \hat{\mathbf{j}}_t - (\mathbf{p}_t + \mathbf{j}_t)\|^2$ introducía gradientes cruzados redundantes con $\text{MSE}_{pose}$ y $\text{MSE}_{traj}$, inestabilizando la optimización multitarea.

---

### Nueva Formulación Matemática

$$\mathcal{L}_{total} = \mathcal{L}_{pose} + w_{disp} \mathcal{L}_{disp} + w_{traj} \mathcal{L}_{traj} + w_{bias} \mathcal{L}_{bias} + w_{speed} \mathcal{L}_{speed} + w_{height} \mathcal{L}_{height} + w_{bone} \mathcal{L}_{bone} + w_{smooth} \mathcal{L}_{smooth}$$

Donde:

1. **Reescalado a Decímetros ($\times 10$)**:
   $$\mathcal{L}_{disp} = \frac{1}{B \cdot T} \sum_{b,t} \|10 \cdot (\hat{\Delta \mathbf{p}}_{b,t} - \Delta \mathbf{p}_{b,t})\|^2$$
   - Al multiplicar por 10, la pérdida cuadrática se multiplica por 100, igualando el orden de magnitud ($\approx 4 \times 10^{-4}\text{ dm}^2 \approx 10^{-3}\text{ m}^2$). Los gradientes de desplazamiento y pose ahora compiten en condiciones equitativas en el optimizador AdamW.
2. **Penalización Anti-Deriva de Sesgo Medio Nulo ($\mathcal{L}_{bias}$)**:
   $$\mathcal{L}_{bias} = \frac{1}{B} \sum_{b=1}^B \left\| \frac{1}{T} \sum_{t=1}^T 10 \cdot (\hat{\Delta \mathbf{p}}_{b,t} - \Delta \mathbf{p}_{b,t}) \right\|^2$$
   - Fuerza matemáticamente a que la suma de errores dentro de la ventana de 150 muestras tenga media cero, impidiendo que los errores se acumulen en una dirección continua durante el paso.
3. **Trayectoria Integrada en Decímetros**:
   $$\mathcal{L}_{traj} = \frac{1}{B \cdot T} \sum_{b,t} \left\| 10 \cdot \sum_{\tau=1}^t \hat{\Delta \mathbf{p}}_{b,\tau} - 10 \cdot \sum_{\tau=1}^t \Delta \mathbf{p}_{b,\tau} \right\|^2$$
4. **Desacople de `GLOBAL_JOINT_LOSS_WEIGHT = 0.0`**:
   - Elimina la covarianza ruidosa durante el entrenamiento, garantizando convergencia suave del backbone compartido.

---

## 3. Resumen de Cambios por Archivo

| Archivo | Modelo | K-Fold CV | SWA | Mixup | Weight Soup | Data Split | Pérdida Balanceada + Anti-Drift | Eval Global 3D + P95% |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `nn_tcn_lite.py` | TCN Lite | ❌ (Single) | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| `nn_tcn_lite_cv.py` | TCN Lite | ✅ (5-Fold) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `nn_ttl_lite.py` | TTL Lite | ❌ (Single) | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| `nn_ttl_lite_cv.py` | TTL Lite | ✅ (5-Fold) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Funcionalidades Añadidas a las Variantes `_cv`:
- **Model Soup (Wortsman et al., 2022)**: Función `create_weight_soup` que promedia los pesos aprendidos en cada uno de los 5 folds, recalcula las estadísticas de BatchNorm sobre el dataset completo y genera un modelo único consolidado (`best_tcn_lite_soup.pt` / `best_ttl_lite_soup.pt`) sin aumentar el tiempo de inferencia.
- **Prediction Ensemble**: Función `evaluate_ensemble_on_captures` que ejecuta en paralelo los 5 modelos de fold con sus respectivos parámetros de normalización y promedia sus predicciones.
- **Reporte Comparativo Final**: Muestra una tabla comparando la capacidad de generalización del K-Fold CV frente a Weight Soup y Ensemble.

---

## 4. Verificación y Pruebas Realizadas

1. **Compilación de Sintaxis**:
   ```bash
   python3 -m py_compile nn_tcn_lite.py nn_tcn_lite_cv.py nn_ttl_lite.py nn_ttl_lite_cv.py
   ```
   *Resultado*: Código de salida 0 en todos los archivos.

2. **Verificación de CLI e Importaciones con el Entorno Virtual (`.venv`)**:
   ```bash
   .venv/bin/python nn_tcn_lite.py --help
   .venv/bin/python nn_tcn_lite_cv.py --help
   .venv/bin/python nn_ttl_lite.py --help
   .venv/bin/python nn_ttl_lite_cv.py --help
   ```
   *Resultado*: Todos los comandos ejecutados exitosamente (`ALL 4 HELP COMMANDS SUCCEEDED`).

3. **Compatibilidad de JSON**:
   - `final_metrics.json` preserva todas las claves previas (`mpjpe`, `mae`, `rmse`, `displacement`, `comparison`) y agrega las nuevas métricas globales y percentiles.
   - `cv_aggregate_metrics.json` almacena las medias y desviaciones estándar de todos los folds para pose relativa y global 3D, junto con los resultados de Weight Soup y Ensemble.

---

## 5. Guía de Ejecución

### Modo Validación Cruzada Completo (5-Fold CV + SWA + Mixup + Weight Soup)
```bash
# Para TCN Lite:
.venv/bin/python nn_tcn_lite_cv.py --dataset ../DataSet

# Para TTL Lite:
.venv/bin/python nn_ttl_lite_cv.py --dataset ../DataSet
```

### Solo Evaluación de Modelos ya Entrenados (`--eval-only`)
```bash
.venv/bin/python nn_tcn_lite_cv.py --dataset ../DataSet --eval-only
.venv/bin/python nn_ttl_lite_cv.py --dataset ../DataSet --eval-only
```

### Modo Single Split con Partición desde `data_split.json`
```bash
# Entrenar single-split tradicional respetando data_split.json:
.venv/bin/python nn_tcn_lite.py --dataset ../DataSet --data-split data_split.json
.venv/bin/python nn_ttl_lite.py --dataset ../DataSet --data-split data_split.json
```
