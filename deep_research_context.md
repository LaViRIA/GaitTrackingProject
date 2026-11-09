# Contexto de Investigación: Reconstrucción de Marcha 3D basada en IMUs y Deep Learning

> 📌 **Documento Exhaustivo Actualizado con Literatura 2020–2026:** Consulta [**`DEEP_RESEARCH_NEURAL_NETWORKS.md`**](file:///home/badajoz/Documents/GaitTrakingProyect/DEEP_RESEARCH_NEURAL_NETWORKS.md) para el protocolo completo de Deep Research, taxonomía de modelos, tabla comparativa con el estado del arte y prompts listos para agentes de búsqueda científica.

Este documento resume la arquitectura, objetivos y técnicas implementadas en el proyecto `GaitTrackingProject`. Está diseñado para servir como un *prompt* de contexto o guía de búsqueda (Deep Research) para encontrar artículos científicos (papers), tesis o investigaciones relacionadas con este enfoque específico.

---

## 1. Objetivo General del Proyecto
Desarrollar un sistema de captura de movimiento (MoCap) y reconstrucción cinemática 3D del tren inferior (marcha humana) de bajo costo, utilizando únicamente sensores inerciales (IMUs) montados en el cuerpo, junto con medidas antropométricas básicas del sujeto. El sistema se apoya en arquitecturas de Deep Learning para estimar posiciones relativas (pose) y el desplazamiento global (trayectoria) en el espacio 3D, utilizando como *Ground Truth* (GT) un sistema óptico basado en marcadores ArUco.

## 2. Configuración de Hardware y Datos
- **Sensores:** 4 IMUs ubicados en los miembros inferiores (Muslo derecho/izquierdo, Pantorrilla derecha/izquierda).
- **Canales de Entrada:** 24 canales continuos de datos inerciales (Acelerómetro 3D + Giroscopio 3D por cada uno de los 4 IMUs).
- **Medidas Antropométricas (Input estático):** Longitud del fémur, longitud de la tibia, ancho de cadera.
- **Ground Truth:** Seguimiento óptico de marcadores fiduciarios (ArUco) para capturar las posiciones 3D reales de la cadera, rodillas y tobillos.

---

## 3. Enfoques Neuronales Implementados (El "Core" del Sistema)

El proyecto aborda el problema desde tres enfoques/modelos diferentes para la inferencia de la marcha:

### A. Modelo Secuencia-a-Secuencia Multitarea (`GaitMultiTaskNet`)
Una arquitectura profunda diseñada para capturar la naturaleza temporal del movimiento de la marcha.
- **Arquitectura:** CNN 1D (extracción local/filtrado de ruido) + BiLSTM (3 capas, contexto bidireccional histórico) + Transformer Encoder (Self-Attention para dependencias a largo plazo) + Cabezas Lineales de Salida.
- **Enfoque Multitarea:** El modelo predice simultáneamente:
  1. **Pose Relativa 3D:** Coordenadas (x,y,z) de 4 articulaciones (rodillas y tobillos) relativas a la cadera.
  2. **Desplazamiento Global (Odometría):** Diferencial de movimiento (dx, dy, dz) de la cadera frame a frame para reconstruir la trayectoria global sin *drift* (deriva).
- **Pérdida Restringida Físicamente (Physics-Informed Loss):** La función de pérdida penaliza errores de MSE en pose, desplazamiento, velocidad y trayectoria acumulada. Además, integra una **restricción ósea (Bone Length Constraint)** para asegurar que la distancia entre articulaciones predichas coincida con la longitud real de los huesos del sujeto.

### B. Clasificador Estático de Pie Pivote (`GaitStaticPivotNet`)
Un modelo que determina instante a instante cuál es el pie de apoyo (fase de soporte de la marcha).
- **Arquitectura:** Multi-Layer Perceptron (MLP) Feed-Forward estático (frame-by-frame, sin memoria secuencial).
- **Objetivo:** Clasificación binaria (Pie Derecho vs Pie Izquierdo) utilizando los 24 canales IMU y antropometría.
- **Ground Truth:** Calculado evaluando cuál tobillo ArUco tiene la menor altura en Z y la menor velocidad en el espacio 3D (similar al concepto ZUPT - Zero Velocity Update).

### C. Reconstrucción Estática Relativa al Pivote (`GaitStaticNet`)
Un modelo para reconstruir la cinemática usando como ancla el pie que está en contacto con el suelo.
- **Arquitectura:** MLP estático (instante a instante).
- **Entradas:** 24 canales IMU + Antropometría + *El resultado del clasificador de pie pivote (0 o 1)*.
- **Salida:** Posiciones relativas 3D de 6 articulaciones (cadera L/R, rodilla L/R, tobillo L/R) **centradas respecto al pie pivote**, eliminando la necesidad de calcular el desplazamiento global complejo (Odometría inercial clásica).

### D. Modelo Temporal Convolutional Network Ligero (`GaitMultiTaskNetTCN`)
Una arquitectura basada en convoluciones 1D dilatadas (TCN Lite) diseñada para capturar dependencias temporales a largo plazo de manera más eficiente en parámetros que los modelos recurrentes (BiLSTM) o de atención.
- **Arquitectura:** Capa de proyección inicial + Backbone TCN (4 bloques de convoluciones 1D dilatadas con conexiones residuales) + Cabezas Multitarea independientes (MLP/Linear).
- **Enfoque Multitarea:** 
  1. **Pose Relativa 3D:** 18 coordenadas (6 articulaciones en 3D) relativas al centro de la cadera.
  2. **Desplazamiento Pélvico 3D:** Vector de traslación frame a frame de la cadera.
- **Pérdida Profunda Multitarea con Anclaje Físico (Deep Multitask Loss):** Además de la pose y trayectoria, incorpora términos físicos avanzados como un **anclaje cinemático de altura Z** (fijando la altura de la cadera basada en la restricción de que los tobillos no deben penetrar el suelo), **rapidez escalar** (para guiar la cadencia de marcha), **constancia ósea** (restricción de longitud de fémur y tibia) y suavidad temporal (penalización de aceleración abrupta/*jitter*).

---

## 4. Conceptos Clave y Palabras de Búsqueda para "Deep Research"

Para encontrar artículos científicos que hagan exactamente (o de forma muy similar) lo que hace este proyecto, utiliza estas combinaciones de palabras clave en motores como Google Scholar, IEEE Xplore, o PubMed:

**Términos sobre el Hardware y el Problema:**
- *IMU-based 3D gait kinematics reconstruction*
- *Inertial motion capture (MoCap) lower body*
- *Wearable sensors human pose estimation*

**Términos sobre el Modelo Multitarea Secuencial (BiLSTM + Transformer):**
- *Deep learning for IMU gait tracking BiLSTM Transformer*
- *Sequence-to-sequence IMU pose estimation*
- *Multi-task deep learning for inertial pose and trajectory estimation* (o *displacement estimation*)
- *Physics-informed neural networks (PINN) for IMU motion capture* (por el uso de constricciones óseas).
- *Bone-length constrained loss function human pose*

**Términos sobre el Modelo TCN y Restricciones Físicas Complejas:**
- *Temporal Convolutional Networks (TCN) for IMU kinematics*
- *Dilated 1D CNN for time-series motion capture*
- *Kinematic zero-elevation anchoring loss foot contact*
- *Multi-objective loss displacement trajectory and pose tracking*

**Términos sobre el Enfoque Estático / Pie Pivote (Zero-Velocity/Support Phase):**
- *Support foot detection from IMU data deep learning*
- *Learning Zero-Velocity Update (ZUPT) intervals using neural networks*
- *Pivot foot-centric inertial pose estimation*
- *Gait phase classification using IMU and MLP*

## 5. Preguntas de Investigación para Contrastar con la Literatura
Al revisar los *papers* que encuentres, busca cómo resuelven los siguientes retos que tú ya has enfrentado:
1. **Deriva Espacial (Drift):** ¿Cómo mitigan otros autores la doble integración del ruido del acelerómetro? (¿Usan Transformers, RNNs, o filtros de Kalman acoplados a NNs?).
2. **Representación del Espacio:** ¿Predicen la pose relativa a un punto ancla (como la cadera central o el pie en el suelo) o intentan predecir coordenadas absolutas?
3. **Restricciones Físicas:** ¿Utilizan *loss functions* personalizadas que mantengan la rigidez del esqueleto (longitudes óseas fijas), o implementan cinemática inversa (IK) post-procesamiento?
4. **Generalización:** ¿Cómo normalizan las diferencias de altura/antropometría entre diferentes sujetos para que la red neuronal generalice correctamente?
