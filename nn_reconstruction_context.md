# Contexto del Proyecto: `nn_reconstruction.py`

Este documento describe la arquitectura, los componentes y la lógica principal del script `nn_reconstruction.py`, el cual es el núcleo central del sistema de predicción de marcha en 3D basado en Deep Learning utilizando mediciones inerciales (IMUs) y medidas antropométricas.

## 1. Visión General de la Red (`GaitMultiTaskNet`)

El modelo es una red neuronal profunda diseñada para una tarea **multitarea** de series temporales (Secuencia a Secuencia). 

**Entradas (`input_dim`):**
- **Sensores (24 canales):** 4 IMUs x (3 Aceleración + 3 Giroscopio). 
- **Antropometría (3 canales):** Longitud del fémur (normalizada), longitud de la tibia (normalizada), y una bandera booleana de dirección de la marcha (0 o 1).

**Salidas (`output_dim = 15`):**
- **Pose Relativa 3D (12 valores):** Coordenadas `(x, y, z)` para 4 articulaciones (`Rodilla Der`, `Rodilla Izq`, `Tobillo Der`, `Tobillo Izq`) relativas al centro de la cadera.
- **Desplazamiento Global 3D (3 valores):** El desplazamiento absoluto `(dx, dy, dz)` de la cadera en cada *frame* respecto al *frame* anterior.

### Arquitectura Interna
1. **Backbone CNN 1D:** Una capa convolucional `Conv1d` extrae características espaciales locales a partir de los datos crudos del sensor, suavizando el ruido.
2. **BiLSTM (3 Capas):** Procesa la secuencia de manera bidireccional, dándole al modelo contexto histórico y futuro de la marcha (ej. fase de balanceo vs. fase de apoyo).
3. **Transformer Encoder:** Un mecanismo de Auto-Atención (Self-Attention) para capturar dependencias temporales a largo plazo que el LSTM puede olvidar.
4. **Cabezas de Predicción (Linear):** Dos proyecciones independientes para predecir la **pose relativa** y el **desplazamiento global**.

---

## 2. Dataset y Aumentación (`GaitDataset`)

El dataset procesa las capturas convirtiéndolas en tensores mediante **ventanas deslizantes** (`window_size`, `stride`). 

**Técnicas de Preprocesamiento implementadas:**
- **Normalización Z-Score:** Los canales de las IMUs se normalizan con la media y desviación estándar del conjunto de entrenamiento.
- **Antropometría (`BONE_NORM`):** Las medidas en metros de los huesos (ej. $0.41$) se dividen por un heurístico ($0.45$) para que su varianza esté cerca de $1.0$, emparejándolas con los canales IMU normalizados.
- **Data Augmentation (Entrenamiento):**
  - **Jittering Gaussiano:** Se inyecta ruido aleatorio para evitar el sobreajuste a firmas de ruido de sensores específicos.
  - **Channel Dropout (5%):** Apaga esporádicamente una IMU entera (6 canales) para forzar a la red a predecir el movimiento apoyándose en el resto del cuerpo, mejorando la robustez.

---

## 3. Función de Pérdida (`deep_multitask_loss`)

Es el motor de optimización que restringe físicamente las predicciones en el espacio 3D real usando el Ground Truth (GT) del sistema de captura óptico (ArUco). 

Componentes penalizados:
- **`mse_pose`:** Error Cuadrático Medio directo entre los puntos de las piernas predichos y el GT.
- **`mse_disp`:** Error de la velocidad instantánea global de la cadera.
- **`mse_speed` & `mse_traj`:** Se penaliza la velocidad 3D (derivada de la pose) y la trayectoria 3D integrada cumulativamente. Esto previene el *drift* (desvío a largo plazo).
- **`bone`:** Constricción esquelética. Penaliza si la distancia entre Cadera-Rodilla o Rodilla-Tobillo difiere de la medida original antropométrica. Evita que las piernas "se estiren" o "se encojan" de forma irreal.
- **`mse_height`:** Asegura que los puntos Z (altura) se mantengan lógicos respecto al suelo.
- **`smooth`:** Restricción de 2da Derivada. Penaliza movimientos bruscos o *jitter* irreal en las predicciones.

---

## 4. Bucle de Entrenamiento (Training Loop)

- **Optimizador:** `AdamW` (Adam con Weight Decay) para regularización y evitar sobreajuste.
- **Precisión Mixta (AMP):** Utiliza `GradScaler` para calcular los gradientes en `float16` y acelerar el entrenamiento en GPUs de NVIDIA.
- **LR Scheduler (`CosineAnnealingWarmRestarts`):** El *learning rate* decrece en forma de coseno para un ajuste fino, y cada 30 épocas "salta" de regreso a un valor alto. Esto permite escapar de valles o mínimos locales, un problema muy común en regresión cinemática.
- **Early Stopping & Checkpoint:** Guarda los pesos del modelo (`best_multitask_model.pt`) solo cuando el error total en Validación se reduce. Si el error no mejora en 30 épocas, detiene el entrenamiento. Incluye dentro del `.pt` los arrays de `norm_stats` (`mean`, `std`) para asegurar inferencias reproducibles sin problemas de conversión.

---

## 5. Funciones Auxiliares
- **`predict_full_sequence_multitask`:** Toma un modelo entrenado y una captura cruda, normaliza las entradas, infiere en batches (deslizantes), y superpone las ventanas (promediándolas) para regresar una serie temporal completa suavizada y la trayectoria global integrada.
- **`plot_trajectory_comparison`:** Genera un panel de visualización 2D (Sagital) comparando las oscilaciones sinusoidales predichas por la NN vs. el Ground Truth.
