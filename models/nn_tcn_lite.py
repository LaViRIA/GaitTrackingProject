"""
Red Neuronal: TCN Lite (Temporal Convolutional Network)
=======================================================
Arquitectura:
  - Temporal Convolutional Network (TCN) causal con convoluciones 1D dilatadas.
  - Bloques residuales con Dilated Conv1D + BatchNorm + ReLU + Dropout.
  - Cabezal Multitarea:
      1. Rama de Pose Relativa: [batch, seq, 18] (6 articulaciones x 3 coords).
      2. Rama de Desplazamiento de Cadera: [batch, seq, 3] (dx, dy, dz frame a frame).
  - Función de pérdida compuesta (deep_multitask_loss) con 8 términos (pose, disp, speed, traj, height, bone, smooth).

Propósito:
  - Reconstrucción cinemática 3D completa y continua del tren inferior a partir de secuencias temporales
    de 4 sensores IMU (RF, RT, LF, LT).
  - Inferencia secuencial ligera y de baja latencia apta para procesamiento temporal en tiempo real.

Directorio de salida: nn-results/tcn_lite/
Checkpoints: nn-results/tcn_lite/checkpoints/best_tcn_lite.pt

Uso:
    python -m models.nn_tcn_lite                         # Entrenamiento + evaluación completa
    python -m models.nn_tcn_lite --eval-only             # Solo evaluación con modelo guardado
    python -m models.nn_tcn_lite --epochs 500 --lr 1e-4  # Hiperparámetros personalizados

Autor: GaitTracking Project
"""

import sys
import time
import json
import glob
import random
import argparse
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import pearsonr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

# Importar utilidades existentes de carga de datos
from process_all import (
    read_mcd_file,
    read_aruco_file,
    adjust_aruco_data,
    measures_calculation,
    process_files,
    low_pass_filter,
    split_char,
)

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════

DATASET_PATH = Path("/home/badajoz/Documents/DataSet")
DATA_SPLIT_PATH = Path("data_split.json")
RESULTS_DIR = Path("nn-results/tcn_lite")
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINT_NAME = "best_tcn_lite.pt"
CHECKPOINT_PATH = CHECKPOINT_DIR / CHECKPOINT_NAME
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Datos ─────────────────────────────────────────────────────────────
WINDOW_SIZE = 50  # muestras por ventana (1.5s a 100Hz)
WINDOW_STRIDE = 3  # desplazamiento entre ventanas (0.25s overlap para dataset denso)
CUTOFF_FREQ = 7  # Hz, filtro pasa-bajas del MCD
FILTER_ORDER = 6  # orden del filtro Butterworth

# ── Modelo (TCN Lite) ─────────────────────────────────────────────────
TCN_CHANNELS = [8, 16, 32, 32]  # canales por bloque TCN
TCN_KERNEL_SIZES = [7, 7, 5, 5]  # tamaño de kernel por bloque
TCN_DILATIONS = [1, 2, 4, 8]  # dilatación por bloque (RF total ≈ 150)
HEAD_HIDDEN_DIM = 48  # neuronas en la capa oculta de las cabezas multitarea
DROPOUT = 0.2  # mayor dropout para dataset pequeño


# TCN_CHANNELS = [16, 16, 16, 32]  # canales por bloque TCN
# TCN_KERNEL_SIZES = [7, 7, 5, 5]  # tamaño de kernel por bloque
# TCN_DILATIONS = [1, 2, 4, 8]  # dilatación por bloque (RF total ≈ 150)
# HEAD_HIDDEN_DIM = 48  # neuronas en la capa oculta de las cabezas multitarea
# DROPOUT = 0.3  # mayor dropout para dataset pequeño

# ── Entrenamiento ────────────────────────────────────────────────────
BATCH_SIZE = 24
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 300
PATIENCE = 60  # Early stopping
GRAD_CLIP = 1.0

# ── Pesos de loss ────────────────────────────────────────────────────
BONE_LOSS_WEIGHT = 0.1
SMOOTH_LOSS_WEIGHT = 0.01
HEIGHT_LOSS_WEIGHT = 0.02
GLOBAL_JOINT_LOSS_WEIGHT = (
    0.0  # Desactivado: redundante con mse_pose + mse_traj (evita ruido cruzado)
)
DISP_LOSS_WEIGHT = (
    0.5  # Recalibrado con escala dm (x10) para balance de gradientes pose/disp
)
TRAJ_LOSS_WEIGHT = 0.3  # Trayectoria integrada dentro de la ventana (odometría)
SPEED_LOSS_WEIGHT = 0.1  # Rapidez escalar instantánea (cadencia)
BIAS_LOSS_WEIGHT = 0.5  # Pérdida de sesgo medio nulo (anti-drift acumulativo)
DISP_SCALE_FACTOR = (
    10.0  # Factor x10 (decímetros) para equilibrar magnitudes cuadráticas con la pose
)

# ── Articulaciones ──────────────────────────────────────────────────
JOINT_NAMES = ["hip_r", "hip_l", "knee_r", "knee_l", "ankle_r", "ankle_l"]
N_JOINTS = 6
N_COORDS = 3
OUTPUT_DIM = N_JOINTS * N_COORDS  # 18

# ── Sensores: 4 IMUs × (3 accel + 3 gyro) ──────────────────────────
IMU_NAMES = ["RF", "LF", "RT", "LT"]
SENSOR_CHANNELS = len(IMU_NAMES) * 6  # 24

# ── Features antropométricas: femur_length, tibia_length, hip_width, direction ─
ANTHROPOMETRIC_DIM = 4

INPUT_DIM = SENSOR_CHANNELS + ANTHROPOMETRIC_DIM  # 28

W = 72  # Ancho de las cajas de texto


# ═══════════════════════════════════════════════════════════════════════
#  CARGA Y PREPARACIÓN DE DATOS
# ═══════════════════════════════════════════════════════════════════════


def discover_captures(dataset_path: Path) -> list[tuple[str, str, str]]:
    """
    Descubre todos los pares MCD/ArUco en el dataset.

    Returns:
        Lista de (mcd_path, aruco_path, subject_id).
    """
    subjects = sorted(glob.glob("*", root_dir=str(dataset_path)))
    # Filtrar sujetos numéricos, excluir sujeto 0 (calibración)
    subjects = sorted([s for s in subjects if s.isdigit() and s != "0"], key=int)

    captures = []
    for s in subjects:
        for i in range(4):
            mcd_file = dataset_path / s / f"mcd-{i}.csv"
            aruco_file = dataset_path / s / f"arUcos-{i}.csv"
            if mcd_file.exists() and aruco_file.exists():
                captures.append((str(mcd_file), str(aruco_file), s))

    return captures


def prepare_capture(mcd_path: str, aruco_path: str, crop: bool = True) -> dict | None:
    """
    Procesa un par de archivos MCD/ArUco y retorna datos alineados
    de sensores + posiciones ground truth.

    Args:
        mcd_path: Ruta al archivo del sensor
        aruco_path: Ruta al archivo ground truth
        crop: Si es True (por defecto), recorta los datos al periodo donde hay ArUco válido.
              Si es False, devuelve todos los datos del sensor y setea gt_mask para indicar validez.

    Returns:
        dict con:
            - 'sensor_data': np.ndarray [N, 24] (canales IMU filtrados)
            - 'timestamps': np.ndarray [N] (ms)
            - 'gt_relative': np.ndarray [N, 18] (6 articulaciones × 3, relativas a cadera)
            - 'gt_hip': np.ndarray [N, 3] (posición absoluta de la cadera ArUco)
            - 'gt_mask': np.ndarray [N] (bool, True donde GT es válido)
            - 'femur_length': float
            - 'tibia_length': float
            - 'direction': str ('L' o 'R')
        None si hay error o datos insuficientes.
    """
    try:
        # Cargar datos MCD (ya filtrados y en unidades correctas)
        mcd_data = read_mcd_file(
            mcd_path, cutoff_freq=CUTOFF_FREQ, filter_order=FILTER_ORDER
        )

        # Cargar datos ArUco
        aruco_data = read_aruco_file(aruco_path)
        direction = mcd_data["measures"]["direction"]

        # Ajustar ArUco al origen (restar posición inicial del tobillo)
        aruco_data = adjust_aruco_data(aruco_data, direction)

        # Mapear llaves ArUco al formato del pipeline
        aruco_processed = {
            "hip": aruco_data["hip_position"],
            "knee_r": aruco_data["R_knee_position"],
            "knee_l": aruco_data["L_knee_position"],
            "ankle_r": aruco_data["R_ankle_position"],
            "ankle_l": aruco_data["L_ankle_position"],
        }

        # Filtrado del ruido en la profundidad (Eje Y)
        t_aruco = aruco_processed["hip"]["time"]
        if len(t_aruco) > 1:
            # Calcular frecuencia de muestreo (fs) promedio en Hz
            fs_aruco = 1000.0 / max(np.mean(np.diff(t_aruco)), 1.0)
            for key in aruco_processed:
                y_data = aruco_processed[key]["data"][:, 1]
                # Filtro pasa-bajas para suavizar la profundidad ruidosa
                try:
                    y_filtered = low_pass_filter(
                        y_data, cutoff_freq=2.0, fs=fs_aruco, order=3
                    )
                    aruco_processed[key]["data"][:, 1] = y_filtered
                except Exception:
                    pass  # En caso de que haya muy pocos datos para el filtro

        # (Las longitudes óseas de fémur y tibia se calcularán a partir de las distancias 3D reales entre marcadores)

        # ── Apilar canales de sensores: [N, 24] ──────────────────────
        mcd_time = mcd_data["time"]
        N = len(mcd_time)

        sensor_data = np.zeros((N, SENSOR_CHANNELS), dtype=np.float64)
        for idx, imu in enumerate(IMU_NAMES):
            sensor_data[:, idx * 6 : idx * 6 + 3] = mcd_data[f"acceleration_{imu}"]
            sensor_data[:, idx * 6 + 3 : idx * 6 + 6] = mcd_data[f"gyro_{imu}"]

        # ── Interpolar posiciones ArUco a timestamps del MCD ─────────
        # Encontrar el rango temporal válido donde TODAS las articulaciones
        # ArUco (cadera central, rodillas y tobillos) tienen datos disponibles
        aruco_raw_keys = ["hip", "knee_r", "knee_l", "ankle_r", "ankle_l"]
        t_min_all = -np.inf
        t_max_all = np.inf

        for key in aruco_raw_keys:
            aruco_t = aruco_processed[key]["time"].astype(float)
            t_min_all = max(t_min_all, aruco_t.min())
            t_max_all = min(t_max_all, aruco_t.max())

        valid_time_mask = (mcd_time >= t_min_all) & (mcd_time <= t_max_all)
        n_valid = int(np.sum(valid_time_mask))

        if n_valid < WINDOW_SIZE:
            return None

        valid_mcd_time = mcd_time[valid_time_mask]

        # Interpolar cadera central ArUco
        hip_t = aruco_processed["hip"]["time"].astype(float)
        hip_pos = aruco_processed["hip"]["data"]
        hip_interp = [interp1d(hip_t, hip_pos[:, i], kind="linear") for i in range(3)]
        hip_at_mcd = np.column_stack([hip_interp[i](valid_mcd_time) for i in range(3)])

        # Interpolar rodillas y tobillos a timestamps del MCD
        joints_at_mcd = {}
        for key in ["knee_r", "knee_l", "ankle_r", "ankle_l"]:
            aruco_t = aruco_processed[key]["time"].astype(float)
            aruco_pos = aruco_processed[key]["data"]

            joint_interp = [
                interp1d(aruco_t, aruco_pos[:, i], kind="linear") for i in range(3)
            ]
            joints_at_mcd[key] = np.column_stack(
                [joint_interp[i](valid_mcd_time) for i in range(3)]
            )

        # ── Profundidad en Marcha Recta: Hip Center Y=0 y planos Sagitales +-1/2 hip_width ──
        # Obtener ancho de cadera anthropométrico de las medidas del MCD
        raw_hw = float(mcd_data["measures"].get("hip_width", 0.25))
        hip_width = raw_hw  # if raw_hw > 0.05 else 0.25
        half_hw = hip_width / 2.0

        # Signo de Y según dirección de marcha ('L' -> hip_r en +Y, 'R' -> hip_r en -Y)
        side_sign = +1.0 if direction == "L" else -1.0
        half_hw_r = +half_hw * side_sign
        half_hw_l = -half_hw * side_sign

        # Fijar centro pélvico Y = 0.0 (marcha en el plano sagital)
        hip_at_mcd[:, 1] = 0.0

        # ── Construir GT relativo con 6 articulaciones ───────────────
        # Orden: hip_r, hip_l, knee_r, knee_l, ankle_r, ankle_l
        gt_relative = np.zeros((N, N_JOINTS * N_COORDS))

        knee_r_at_mcd = joints_at_mcd["knee_r"]
        knee_l_at_mcd = joints_at_mcd["knee_l"]
        ankle_r_at_mcd = joints_at_mcd["ankle_r"]
        ankle_l_at_mcd = joints_at_mcd["ankle_l"]

        # Posiciones relativas al centro de la cadera [0, 0, 0]
        hip_r_rel = np.column_stack(
            [np.zeros(n_valid), np.full(n_valid, half_hw_r), np.zeros(n_valid)]
        )
        hip_l_rel = np.column_stack(
            [np.zeros(n_valid), np.full(n_valid, half_hw_l), np.zeros(n_valid)]
        )

        knee_r_rel = np.column_stack(
            [
                knee_r_at_mcd[:, 0] - hip_at_mcd[:, 0],
                np.full(n_valid, half_hw_r),
                knee_r_at_mcd[:, 2] - hip_at_mcd[:, 2],
            ]
        )
        knee_l_rel = np.column_stack(
            [
                knee_l_at_mcd[:, 0] - hip_at_mcd[:, 0],
                np.full(n_valid, half_hw_l),
                knee_l_at_mcd[:, 2] - hip_at_mcd[:, 2],
            ]
        )

        ankle_r_rel = np.column_stack(
            [
                ankle_r_at_mcd[:, 0] - hip_at_mcd[:, 0],
                np.full(n_valid, half_hw_r),
                ankle_r_at_mcd[:, 2] - hip_at_mcd[:, 2],
            ]
        )
        ankle_l_rel = np.column_stack(
            [
                ankle_l_at_mcd[:, 0] - hip_at_mcd[:, 0],
                np.full(n_valid, half_hw_l),
                ankle_l_at_mcd[:, 2] - hip_at_mcd[:, 2],
            ]
        )

        all_rel_joints = [
            hip_r_rel,
            hip_l_rel,
            knee_r_rel,
            knee_l_rel,
            ankle_r_rel,
            ankle_l_rel,
        ]

        for j_idx, rel_pos in enumerate(all_rel_joints):
            gt_relative[valid_time_mask, j_idx * 3 : (j_idx + 1) * 3] = rel_pos

        # ── Estimar longitudes óseas (fémur y tibia) directamente de las distancias 3D entre marcadores ──
        femur_r_dists = np.linalg.norm(knee_r_rel - hip_r_rel, axis=1)
        femur_l_dists = np.linalg.norm(knee_l_rel - hip_l_rel, axis=1)
        tibia_r_dists = np.linalg.norm(ankle_r_rel - knee_r_rel, axis=1)
        tibia_l_dists = np.linalg.norm(ankle_l_rel - knee_l_rel, axis=1)

        femur_length = float(np.mean(np.concatenate([femur_r_dists, femur_l_dists])))
        tibia_length = float(np.mean(np.concatenate([tibia_r_dists, tibia_l_dists])))

        # Almacenar posición absoluta de la cadera (centro pélvico Y=0)
        gt_hip = np.zeros((N, 3))
        gt_hip[valid_time_mask] = hip_at_mcd

        # Desplazamiento por cuadro de la cadera (velocidad discreta)
        gt_hip_disp = np.zeros((N, 3))
        hip_disp_valid = np.diff(hip_at_mcd, axis=0)
        hip_disp_valid[:, 1] = 0.0  # Sin desplazamiento lateral Y para la marcha recta
        valid_indices = np.where(valid_time_mask)[0]
        if len(valid_indices) > 1:
            gt_hip_disp[valid_indices[1:]] = hip_disp_valid

        # Recortar a la región válida para eficiencia en entrenamiento
        if crop:
            first_valid = int(np.argmax(valid_time_mask))
            last_valid = N - int(np.argmax(valid_time_mask[::-1])) - 1
        else:
            first_valid = 0
            last_valid = N - 1

        return {
            "sensor_data": sensor_data[first_valid : last_valid + 1],
            "timestamps": mcd_time[first_valid : last_valid + 1],
            "gt_relative": gt_relative[first_valid : last_valid + 1],
            "gt_hip": gt_hip[first_valid : last_valid + 1],
            "gt_hip_disp": gt_hip_disp[first_valid : last_valid + 1],
            "gt_mask": valid_time_mask[first_valid : last_valid + 1],
            "first_valid": first_valid,
            "last_valid": last_valid,
            "femur_length": femur_length,
            "tibia_length": tibia_length,
            "hip_width": hip_width,
            "direction": direction,
        }

    except Exception as e:
        print(f"  [ERROR] {Path(mcd_path).name}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════════


class GaitDataset(Dataset):
    """
    Dataset PyTorch con ventanas deslizantes sobre capturas de marcha.

    Cada muestra es:
        x: Tensor [window_size, INPUT_DIM]  (sensores + antropometría)
        y_pose: Tensor [window_size, OUTPUT_DIM] (posiciones relativas GT)
        y_disp: Tensor [window_size, 3] (desplazamiento por frame de cadera GT)
        anthro: Tensor [3] (femur_len, tibia_len, direction)
    """

    # Valor típico para normalizar longitudes óseas (~rango [0.6, 1.2])
    BONE_NORM = 0.45

    def __init__(
        self,
        captures: list[dict],
        window_size: int = WINDOW_SIZE,
        stride: int = WINDOW_STRIDE,
        normalize: bool = True,
        stats: dict | None = None,
        augment: bool = False,
    ):
        self.window_size = window_size
        self.augment = augment
        self.windows: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

        all_sensor_chunks: list[np.ndarray] = []

        for cap in captures:
            sensor = cap["sensor_data"]
            gt_pose = cap["gt_relative"]
            gt_disp = cap.get("gt_hip_disp", np.zeros((len(sensor), 3)))
            mask = cap["gt_mask"]

            anthropometric = np.array(
                [
                    cap["femur_length"] / self.BONE_NORM,  # Normalizado ~[0.6, 1.2]
                    cap["tibia_length"] / self.BONE_NORM,  # Normalizado ~[0.6, 1.2]
                    cap["hip_width"] / self.BONE_NORM,
                    1.0 if cap["direction"] == "R" else 0.0,
                ],
                dtype=np.float32,
            )

            n_samples = len(sensor)
            for start in range(0, n_samples - window_size + 1, stride):
                end = start + window_size
                if np.all(mask[start:end]):
                    self.windows.append(
                        (
                            sensor[start:end].astype(np.float32),
                            gt_pose[start:end].astype(np.float32),
                            gt_disp[start:end].astype(np.float32),
                            anthropometric,
                        )
                    )
                    all_sensor_chunks.append(sensor[start:end])

        # Estadísticas de normalización (z-score por canal)
        if stats is not None:
            self.mean = stats["mean"]
            self.std = stats["std"]
        elif len(all_sensor_chunks) > 0:
            all_sensor_arr = np.concatenate(all_sensor_chunks, axis=0)
            self.mean = all_sensor_arr.mean(axis=0).astype(np.float32)
            self.std = all_sensor_arr.std(axis=0).astype(np.float32)
            self.std[self.std < 1e-8] = 1.0
        else:
            self.mean = np.zeros(SENSOR_CHANNELS, dtype=np.float32)
            self.std = np.ones(SENSOR_CHANNELS, dtype=np.float32)

        self.normalize = normalize

    def get_stats(self) -> dict:
        """Retorna estadísticas de normalización para val/test."""
        return {"mean": self.mean, "std": self.std}

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sensor, gt_pose, gt_disp, anthropometric = self.windows[idx]
        sensor = sensor.copy()  # No modificar el original
        gt_pose = gt_pose.copy()
        gt_disp = gt_disp.copy()

        if self.normalize:
            sensor = (sensor - self.mean) / self.std

        # ── Data Augmentation (solo entrenamiento) ───────────────────
        if self.augment:
            # 1. Submuestreo temporal (Time Warping)
            if random.random() < 0.5:
                seq_len = len(sensor)
                crop_ratio = np.random.uniform(0.85, 0.95)
                crop_len = int(seq_len * crop_ratio)
                start_idx = np.random.randint(0, seq_len - crop_len + 1)

                sensor_crop = sensor[start_idx : start_idx + crop_len]
                gt_pose_crop = gt_pose[start_idx : start_idx + crop_len]
                gt_disp_crop = gt_disp[start_idx : start_idx + crop_len]

                x_old = np.linspace(0, 1, crop_len)
                x_new = np.linspace(0, 1, seq_len)

                sensor = interp1d(x_old, sensor_crop, axis=0, kind="linear")(
                    x_new
                ).astype(np.float32)
                gt_pose = interp1d(x_old, gt_pose_crop, axis=0, kind="linear")(
                    x_new
                ).astype(np.float32)
                gt_disp = interp1d(x_old, gt_disp_crop, axis=0, kind="linear")(
                    x_new
                ).astype(np.float32)
                # Escalar desplazamiento por el cambio de velocidad (menos frames originales = mayor velocidad)
                gt_disp = gt_disp * (crop_len / seq_len)

            # 2. Escalado de Amplitud
            if random.random() < 0.5:
                scale = np.random.uniform(0.9, 1.1)
                sensor = sensor * scale

            # 3. Frame Dropout (Pérdida de paquetes)
            if random.random() < 0.5:
                drop_indices = np.random.choice(
                    len(sensor), size=int(len(sensor) * 0.02), replace=False
                )
                for idx_drop in drop_indices:
                    if idx_drop > 0:
                        sensor[idx_drop] = sensor[idx_drop - 1]

            # 4. Jitter gaussiano (σ = 0.02 de la señal normalizada)
            sensor = sensor + np.random.randn(*sensor.shape).astype(np.float32) * 0.02

            # 5. Dropout de canal: enmascarar 1 IMU completa (6 canales) con probabilidad 5%
            if np.random.rand() < 0.05:
                imu_idx = np.random.randint(0, len(IMU_NAMES))
                sensor[:, imu_idx * 6 : (imu_idx + 1) * 6] = 0.0

        # Difundir medidas antropométricas a todos los timesteps
        anthro_broadcast = np.tile(anthropometric, (self.window_size, 1))
        x = np.concatenate([sensor, anthro_broadcast], axis=1)

        return (
            torch.from_numpy(x),
            torch.from_numpy(gt_pose),
            torch.from_numpy(gt_disp),
            torch.from_numpy(anthropometric),
        )


# ═══════════════════════════════════════════════════════════════════════
#  ARQUITECTURA DEL MODELO — TCN LITE MULTITAREA
# ═══════════════════════════════════════════════════════════════════════


class TCNBlock(nn.Module):
    """
    Bloque de Temporal Convolutional Network con convolución 1D dilatada,
    BatchNorm, GELU, Dropout y conexión residual.

    La convolución dilatada permite un receptive field exponencialmente
    creciente sin pooling ni recurrencia.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Padding "same" para dilatación: (k-1)*d // 2
        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Conexión residual con proyección 1×1 si cambian los canales
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        # x: [batch, channels, seq_len]
        residual = self.residual(x)
        x = self.conv(x)
        # Recortar si la salida es ligeramente más larga que la entrada
        if x.size(2) != residual.size(2):
            x = x[:, :, : residual.size(2)]
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        return x + residual


class GaitMultiTaskNetTCN(nn.Module):
    """
    Red Neuronal Multitarea Ligera (TCN Lite) para Reconstrucción de Marcha.

    Utiliza una Temporal Convolutional Network (TCN) con convoluciones dilatadas
    para capturar dependencias temporales a largo plazo con un receptive field
    configurable, manteniendo un número de parámetros adecuado para datasets pequeños.

    Arquitectura:
      - Input Projection: Linear → GELU → LayerNorm
      - Backbone TCN: 4 bloques Conv1D dilatados con residuales
      - Cabeza 1 (Pose Head): Pose articular 3D [batch, seq_len, 18]
      - Cabeza 2 (Displacement Head): Desplazamiento pélvico 3D [batch, seq_len, 3]
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        output_dim: int = OUTPUT_DIM,
        tcn_channels: list[int] = None,
        tcn_kernel_sizes: list[int] = None,
        tcn_dilations: list[int] = None,
        head_hidden_dim: int = HEAD_HIDDEN_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        if tcn_channels is None:
            tcn_channels = TCN_CHANNELS
        if tcn_kernel_sizes is None:
            tcn_kernel_sizes = TCN_KERNEL_SIZES
        if tcn_dilations is None:
            tcn_dilations = TCN_DILATIONS

        # ── 1. Proyección de entrada ─────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, tcn_channels[0]),
            nn.GELU(),
            nn.LayerNorm(tcn_channels[0]),
        )

        # ── 2. Backbone TCN (convoluciones dilatadas) ────────────────
        self.tcn_blocks = nn.ModuleList()
        channels_in = [tcn_channels[0]] + tcn_channels[:-1]
        for i, (c_in, c_out) in enumerate(zip(channels_in, tcn_channels)):
            self.tcn_blocks.append(
                TCNBlock(
                    c_in,
                    c_out,
                    kernel_size=tcn_kernel_sizes[i],
                    dilation=tcn_dilations[i],
                    dropout=dropout,
                )
            )

        backbone_dim = tcn_channels[-1]  # Ej. 128

        # ── 3. Cabezas de salida multitarea ──────────────────────────

        # Cabeza 1: Pose Relativa [18]
        self.pose_head = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Linear(backbone_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, output_dim),
        )

        # Cabeza 2: Desplazamiento pélvico [3]
        self.disp_head = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Linear(backbone_dim, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 3),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        h = self.input_proj(x)  # [batch, seq_len, tcn_ch[0]]
        h = h.transpose(1, 2)  # [batch, tcn_ch[0], seq_len]

        for block in self.tcn_blocks:
            h = block(h)  # [batch, tcn_ch[-1], seq_len]

        h = h.transpose(1, 2)  # [batch, seq_len, backbone_dim]

        pred_pose = self.pose_head(h)  # [batch, seq_len, 18]
        pred_disp = self.disp_head(h)  # [batch, seq_len, 3]
        return pred_pose, pred_disp

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module):
    """Imprime una tabla detallada de la arquitectura del modelo, capas y parámetros."""
    print()
    print("═" * 80)
    print(
        "  🧠 ARQUITECTURA DE LA RED NEURONAL MULTITAREA (GaitMultiTaskNetTCN — TCN Lite)"
    )
    print("═" * 80)
    print(f"  {'Módulo / Bloque':<32} │ {'Tipo de Capa':<24} │ {'Parámetros':>15}")
    print("  " + "─" * 76)

    total_params = 0
    trainable_params = 0

    for name, child in model.named_children():
        p_count = sum(p.numel() for p in child.parameters())
        p_train = sum(p.numel() for p in child.parameters() if p.requires_grad)
        total_params += p_count
        trainable_params += p_train
        layer_type = child.__class__.__name__
        print(f"  {name:<32} │ {layer_type:<24} │ {p_count:>15,}")

    print("  " + "─" * 76)
    print(f"  {'PARÁMETROS TOTALES ENTRENABLES':<59} │ {trainable_params:>15,}")
    print("═" * 80)
    print()


# ═══════════════════════════════════════════════════════════════════════
#  FUNCIONES DE PÉRDIDA DEDICADAS Y MULTITAREA PROFUNDA
# ═══════════════════════════════════════════════════════════════════════


def bone_length_loss(
    predictions: torch.Tensor, anthropometric: torch.Tensor
) -> torch.Tensor:
    """Penaliza desviaciones de las longitudes óseas esperadas."""
    pred_joints = predictions.view(
        predictions.shape[0], predictions.shape[1], N_JOINTS, N_COORDS
    )

    # Índices: hip_r=0, hip_l=1, knee_r=2, knee_l=3, ankle_r=4, ankle_l=5
    hip_r = pred_joints[:, :, 0, :]
    hip_l = pred_joints[:, :, 1, :]
    knee_r = pred_joints[:, :, 2, :]
    knee_l = pred_joints[:, :, 3, :]
    ankle_r = pred_joints[:, :, 4, :]
    ankle_l = pred_joints[:, :, 5, :]

    femur_r_len = torch.norm(knee_r - hip_r, dim=-1)
    femur_l_len = torch.norm(knee_l - hip_l, dim=-1)

    tibia_r_len = torch.norm(ankle_r - knee_r, dim=-1)
    tibia_l_len = torch.norm(ankle_l - knee_l, dim=-1)

    femur_target = (anthropometric[:, 0] * GaitDataset.BONE_NORM).unsqueeze(1)
    tibia_target = (anthropometric[:, 1] * GaitDataset.BONE_NORM).unsqueeze(1)

    loss = (
        F.mse_loss(femur_r_len, femur_target.expand_as(femur_r_len))
        + F.mse_loss(femur_l_len, femur_target.expand_as(femur_l_len))
        + F.mse_loss(tibia_r_len, tibia_target.expand_as(tibia_r_len))
        + F.mse_loss(tibia_l_len, tibia_target.expand_as(tibia_l_len))
    ) / 4.0

    return loss


def smoothness_loss(predictions: torch.Tensor) -> torch.Tensor:
    """Penaliza cambios bruscos (jitter) en las predicciones."""
    if predictions.shape[1] < 3:
        return torch.tensor(0.0, device=predictions.device)

    velocity = predictions[:, 1:, :] - predictions[:, :-1, :]
    acceleration = velocity[:, 1:, :] - velocity[:, :-1, :]
    return torch.mean(acceleration**2)


def deep_multitask_loss(
    pred_pose: torch.Tensor,
    pred_disp: torch.Tensor,
    target_pose: torch.Tensor,
    target_disp: torch.Tensor,
    anthropometric: torch.Tensor,
    bone_weight: float = BONE_LOSS_WEIGHT,
    smooth_weight: float = SMOOTH_LOSS_WEIGHT,
    disp_weight: float = DISP_LOSS_WEIGHT,
    traj_weight: float = TRAJ_LOSS_WEIGHT,
    height_weight: float = HEIGHT_LOSS_WEIGHT,
    global_weight: float = GLOBAL_JOINT_LOSS_WEIGHT,
    speed_weight: float = SPEED_LOSS_WEIGHT,
    bias_weight: float = BIAS_LOSS_WEIGHT,
    disp_scale: float = DISP_SCALE_FACTOR,
) -> tuple[torch.Tensor, dict]:
    """
    Pérdida profunda multitarea 3D equilibrada con penalización anti-drift.
    Optimiza simultáneamente:
      1. Pose articular 3D relativa (rodillas y tobillos)
      2. Desplazamiento pélvico instantáneo 3D reescalado a dm (X, Y, Z)
      3. Rapidez escalar instantánea ||Δp|| 3D (cadencia de zancada)
      4. Trayectoria 3D integrada (cumsum) con énfasis horizontal (X, Y)
      5. Sesgo medio nulo dentro de la ventana (Anti-Drift de largo plazo)
      6. Anclaje de altura Z al suelo derivado de los tobillos GT
      7. Consistencia ósea y suavidad temporal
    """
    # 1. Pérdida de pose articular 3D en metros [m^2]
    mse_pose = F.mse_loss(pred_pose, target_pose)

    # 2. Pérdida de desplazamiento pélvico 3D reescalado a decímetros (x10)
    # Equilibra numéricamente la señal de gradiente (~10^-3) frente a la pose articular
    p_disp_s = pred_disp * disp_scale
    t_disp_s = target_disp * disp_scale
    mse_disp = F.mse_loss(p_disp_s, t_disp_s)

    # 3. Rapidez escalar instantánea 3D (cadencia de zancada)
    pred_speed = torch.norm(p_disp_s, dim=-1)
    target_speed = torch.norm(t_disp_s, dim=-1)
    mse_speed = F.mse_loss(pred_speed, target_speed)

    # 4. Trayectoria 3D integrada dentro de la ventana (cumsum)
    pred_hip_cumsum = torch.cumsum(pred_disp, dim=1)
    target_hip_cumsum = torch.cumsum(target_disp, dim=1)
    mse_traj = F.mse_loss(pred_hip_cumsum * disp_scale, target_hip_cumsum * disp_scale)

    # 5. Pérdida de sesgo medio acumulado (Anti-Drift): fuerza que la media del error sea 0
    disp_mean_err = torch.mean(p_disp_s - t_disp_s, dim=1)  # [B, 3]
    mse_bias = torch.mean(disp_mean_err**2)

    # 6. Anclaje cinemático de altura Z al suelo usando GT de tobillos
    target_j = target_pose.view(
        target_pose.shape[0], target_pose.shape[1], N_JOINTS, N_COORDS
    )
    # Índices: hip_r=0, hip_l=1, knee_r=2, knee_l=3, ankle_r=4, ankle_l=5
    target_ankle_r_z = target_j[:, :, 4, 2]
    target_ankle_l_z = target_j[:, :, 5, 2]

    # Altura cinemática de la cadera relativa al pie de apoyo (desde GT)
    target_h_kin = -torch.minimum(target_ankle_r_z, target_ankle_l_z)
    target_h_kin_rel = target_h_kin - target_h_kin[:, 0:1]

    pred_disp_z_cumsum = pred_hip_cumsum[:, :, 2]
    mse_height = F.mse_loss(pred_disp_z_cumsum, target_h_kin_rel)

    # 7. Pérdida global 3D opcional (desactivada por defecto para desacoplar gradientes ruidosos)
    if global_weight > 0.0:
        pred_j = pred_pose.view(
            pred_pose.shape[0], pred_pose.shape[1], N_JOINTS, N_COORDS
        )
        pred_hip_exp = pred_hip_cumsum.unsqueeze(2)
        target_hip_exp = target_hip_cumsum.unsqueeze(2)
        mse_global = F.mse_loss(pred_j + pred_hip_exp, target_j + target_hip_exp)
    else:
        mse_global = torch.tensor(0.0, device=pred_pose.device)

    # 8. Restricciones físicas
    bone = bone_length_loss(pred_pose, anthropometric)
    smooth = smoothness_loss(pred_pose) + smoothness_loss(pred_disp)

    total = (
        mse_pose
        + disp_weight * mse_disp
        + speed_weight * mse_speed
        + traj_weight * mse_traj
        + bias_weight * mse_bias
        + height_weight * mse_height
        + global_weight * mse_global
        + bone_weight * bone
        + smooth_weight * smooth
    )

    return total, {
        "total": total.item(),
        "mse_pose": mse_pose.item(),
        "mse_disp": mse_disp.item(),
        "mse_speed": mse_speed.item(),
        "mse_traj": mse_traj.item(),
        "mse_bias": mse_bias.item(),
        "mse_height": mse_height.item(),
        "mse_global": mse_global.item(),
        "bone": bone.item(),
        "smooth": smooth.item(),
    }


# ═══════════════════════════════════════════════════════════════════════
#  ENTRENAMIENTO MULTITAREA UNIFICADO
# ═══════════════════════════════════════════════════════════════════════


def train_multitask_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    epoch: int = 0,
    total_epochs: int = 0,
) -> dict:
    """Ejecuta una época de entrenamiento multitarea."""
    model.train()
    total_loss = 0.0
    total_pose = 0.0
    total_disp = 0.0
    total_traj = 0.0
    total_height = 0.0
    total_global = 0.0
    total_bone = 0.0
    total_smooth = 0.0
    total_bias = 0.0
    n_batches = 0
    total_batches = len(loader)

    for x, y_pose, y_disp, anthro in loader:
        x = x.to(DEVICE)
        y_pose = y_pose.to(DEVICE)
        y_disp = y_disp.to(DEVICE)
        anthro = anthro.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=(DEVICE.type == "cuda")):
            pred_pose, pred_disp = model(x)
            loss, loss_dict = deep_multitask_loss(
                pred_pose, pred_disp, y_pose, y_disp, anthro
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss_dict["total"]
        total_pose += loss_dict["mse_pose"]
        total_disp += loss_dict["mse_disp"]
        total_traj += loss_dict["mse_traj"]
        total_height += loss_dict["mse_height"]
        total_global += loss_dict["mse_global"]
        total_bone += loss_dict["bone"]
        total_smooth += loss_dict["smooth"]
        total_bias += loss_dict["mse_bias"]
        n_batches += 1

        # Barra de progreso compacta durante la época
        if total_epochs > 0:
            bar_len = 15
            progress = n_batches / total_batches
            filled = int(bar_len * progress)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(
                f"\r  Época {epoch:3d}/{total_epochs} │{bar}│ Batch {n_batches}/{total_batches} │ Train: {total_loss/n_batches:.4f}",
                end="",
                flush=True,
            )

    return {
        "total": total_loss / n_batches,
        "pose": total_pose / n_batches,
        "disp": total_disp / n_batches,
        "traj": total_traj / n_batches,
        "height": total_height / n_batches,
        "global": total_global / n_batches,
        "bone": total_bone / n_batches,
        "smooth": total_smooth / n_batches,
        "bias": total_bias / n_batches,
    }


@torch.no_grad()
def validate_multitask(model: nn.Module, loader: DataLoader) -> dict:
    """Ejecuta validación multitarea."""
    model.eval()
    total_loss = 0.0
    total_pose = 0.0
    total_disp = 0.0
    total_traj = 0.0
    total_height = 0.0
    total_global = 0.0
    total_bone = 0.0
    total_smooth = 0.0
    total_bias = 0.0
    n_batches = 0

    for x, y_pose, y_disp, anthro in loader:
        x = x.to(DEVICE)
        y_pose = y_pose.to(DEVICE)
        y_disp = y_disp.to(DEVICE)
        anthro = anthro.to(DEVICE)

        with autocast("cuda", enabled=(DEVICE.type == "cuda")):
            pred_pose, pred_disp = model(x)
            _, loss_dict = deep_multitask_loss(
                pred_pose, pred_disp, y_pose, y_disp, anthro
            )

        total_loss += loss_dict["total"]
        total_pose += loss_dict["mse_pose"]
        total_disp += loss_dict["mse_disp"]
        total_traj += loss_dict["mse_traj"]
        total_height += loss_dict["mse_height"]
        total_global += loss_dict["mse_global"]
        total_bone += loss_dict["bone"]
        total_smooth += loss_dict["smooth"]
        total_bias += loss_dict["mse_bias"]
        n_batches += 1

    if n_batches == 0:
        return {
            "total": float("inf"),
            "pose": 0,
            "disp": 0,
            "traj": 0,
            "height": 0,
            "global": 0,
            "bone": 0,
            "smooth": 0,
            "bias": 0,
        }

    return {
        "total": total_loss / n_batches,
        "pose": total_pose / n_batches,
        "disp": total_disp / n_batches,
        "traj": total_traj / n_batches,
        "height": total_height / n_batches,
        "global": total_global / n_batches,
        "bone": total_bone / n_batches,
        "smooth": total_smooth / n_batches,
        "bias": total_bias / n_batches,
    }


def train_multitask_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    patience: int = PATIENCE,
    checkpoint_tag: str = "",
) -> dict:
    """
    Entrena la red multitarea con early stopping y gradientes combinados.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    # CosineAnnealing con Warm Restarts: reinicia LR cada T_0 épocas
    # para escapar de mínimos locales
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=lr * 0.01
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_pose": [],
        "val_pose": [],
        "train_height": [],
        "val_height": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_state = None

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("═" * W)
    print("  🧠 ENTRENAMIENTO MULTITAREA UNIFICADO (Opción C)")
    print("═" * W)
    print(f"  Parámetros:        {model.count_parameters():,}")
    print(f"  Train batches:     {len(train_loader)}")
    print(f"  Val batches:       {len(val_loader)}")
    print(f"  Épocas máx:        {epochs}")
    print(f"  Early stopping:    {patience} épocas")
    print("═" * W)
    print()

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        tr = train_multitask_epoch(
            model, train_loader, optimizer, scaler, epoch=epoch, total_epochs=epochs
        )
        va = validate_multitask(model, val_loader)
        scheduler.step()  # CosineAnnealingWarmRestarts: step per epoch

        history["train_loss"].append(tr["total"])
        history["val_loss"].append(va["total"])
        history["train_pose"].append(tr["pose"])
        history["val_pose"].append(va["pose"])
        history["train_height"].append(tr["height"])
        history["val_height"].append(va["height"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if va["total"] < best_val_loss:
            best_val_loss = va["total"]
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            ckpt_name = (
                f"best_tcn_lite_{checkpoint_tag}.pt"
                if checkpoint_tag
                else CHECKPOINT_NAME
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "val_loss": best_val_loss,
                },
                CHECKPOINT_DIR / ckpt_name,
            )
        else:
            patience_counter += 1

        elapsed = time.time() - t_epoch
        current_lr = optimizer.param_groups[0]["lr"]

        if patience_counter == 0:
            print(
                f"\r  Época {epoch:3d}/{epochs} │ Train: {tr['total']:.4f} │ Val: {va['total']:.4f} ★ │ LR: {current_lr:.1e} │ {elapsed:.1f}s"
            )
        elif epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"\r  Época {epoch:3d}/{epochs} │ Train: {tr['total']:.4f} │ Val: {va['total']:.4f}   │ LR: {current_lr:.1e} │ {elapsed:.1f}s"
            )

        if patience_counter >= patience:
            print(
                f"\n  ⏹ Early stopping multitarea en época {epoch} (mejor: {best_epoch})"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    total_time = time.time() - t_start
    print(f"\n  ✅ Entrenamiento multitarea completado en {total_time / 60:.1f} min")
    print(f"  Mejor val loss: {best_val_loss:.6f} (época {best_epoch})")

    return history


# ═══════════════════════════════════════════════════════════════════════
#  PREDICCIÓN EN SECUENCIA COMPLETA MULTITAREA
# ═══════════════════════════════════════════════════════════════════════


@torch.no_grad()
def predict_full_sequence_multitask(
    model: nn.Module,
    sensor_data: np.ndarray,
    anthropometric: np.ndarray,
    norm_stats: dict,
    window_size: int = WINDOW_SIZE,
    stride: int = None,
    initial_hip: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Predice posiciones relativas y desplazamiento pélvico con la red multitarea.

    Returns:
        (pred_pose [N, 18], pred_disp [N, 3], pred_hip_global [N, 3], valid_mask [N])
    """
    if stride is None:
        stride = window_size // 2

    model.eval()
    N = len(sensor_data)
    pred_pose_sum = np.zeros((N, OUTPUT_DIM), dtype=np.float64)
    pred_disp_sum = np.zeros((N, 3), dtype=np.float64)
    counts = np.zeros(N, dtype=np.float64)

    sensor_norm = (sensor_data.astype(np.float32) - norm_stats["mean"]) / norm_stats[
        "std"
    ]

    anthro_broadcast = np.tile(anthropometric.astype(np.float32), (window_size, 1))

    for start in range(0, N - window_size + 1, stride):
        end = start + window_size
        x = np.concatenate([sensor_norm[start:end], anthro_broadcast], axis=1).astype(
            np.float32
        )
        x_tensor = torch.from_numpy(x).unsqueeze(0).to(DEVICE)

        with autocast("cuda", enabled=(DEVICE.type == "cuda")):
            pred_pose_t, pred_disp_t = model(x_tensor)

        pred_pose_np = pred_pose_t.float().cpu().numpy()[0]
        pred_disp_np = pred_disp_t.float().cpu().numpy()[0]

        pred_pose_sum[start:end] += pred_pose_np
        pred_disp_sum[start:end] += pred_disp_np
        counts[start:end] += 1

    valid = counts > 0
    pred_pose_sum[valid] /= counts[valid, np.newaxis]
    pred_disp_sum[valid] /= counts[valid, np.newaxis]

    # Integración del desplazamiento pélvico con corrección de anclaje de altura Z al suelo
    if initial_hip is not None and len(initial_hip) > 0:
        base_hip = initial_hip[0].copy()
    else:
        base_hip = np.zeros(3)

    # Anclaje cinemático continuo en Z
    pred_j_all = pred_pose_sum.reshape(-1, N_JOINTS, N_COORDS)
    # Índices: hip_r=0, hip_l=1, knee_r=2, knee_l=3, ankle_r=4, ankle_l=5
    ankles_z = np.minimum(pred_j_all[:, 4, 2], pred_j_all[:, 5, 2])
    kinematic_h_z = -ankles_z  # Altura de cadera respecto al pie de apoyo

    pred_hip_cumsum = np.cumsum(pred_disp_sum, axis=0) + base_hip

    # Reemplazar Z integrada inercial por el anclaje cinemático de altura + offset de origen
    pred_hip_global = pred_hip_cumsum.copy()
    pred_hip_global[:, 2] = base_hip[2] + (kinematic_h_z - kinematic_h_z[0])

    return pred_pose_sum, pred_disp_sum, pred_hip_global, valid


# ═══════════════════════════════════════════════════════════════════════
#  EVALUACIÓN
# ═══════════════════════════════════════════════════════════════════════


def compute_mpjpe(
    predictions: np.ndarray, targets: np.ndarray, percentile: float = 100.0
) -> dict[str, float]:
    """
    Mean Per Joint Position Error (MPJPE) en metros.

    Args:
        predictions: [N, 18] posiciones predichas (relativas o globales)
        targets: [N, 18] posiciones GT
        percentile: Percentil superior a evaluar (default: 100.0 para todo el conjunto; 95.0 para recortar outliers)

    Returns:
        dict con MPJPE por articulación y promedio general.
    """
    pred = predictions.reshape(-1, N_JOINTS, N_COORDS)
    tgt = targets.reshape(-1, N_JOINTS, N_COORDS)

    # Error euclidiano por articulación: [N, N_JOINTS]
    per_joint_error = np.linalg.norm(pred - tgt, axis=-1)

    result = {}
    for j, name in enumerate(JOINT_NAMES):
        err = per_joint_error[:, j]
        if percentile is not None and percentile < 100.0:
            threshold = np.percentile(err, percentile)
            err = err[err <= threshold]
        result[name] = float(np.mean(err))
    result["mean"] = float(np.mean([result[name] for name in JOINT_NAMES]))

    return result


def compute_mae_per_joint(
    predictions: np.ndarray, targets: np.ndarray, percentile: float = 100.0
) -> dict[str, float]:
    """MAE (Mean Absolute Error) por articulación en metros."""
    pred = predictions.reshape(-1, N_JOINTS, N_COORDS)
    tgt = targets.reshape(-1, N_JOINTS, N_COORDS)

    result = {}
    for j, name in enumerate(JOINT_NAMES):
        abs_diff = np.abs(pred[:, j, :] - tgt[:, j, :]).flatten()
        if percentile is not None and percentile < 100.0:
            threshold = np.percentile(abs_diff, percentile)
            abs_diff = abs_diff[abs_diff <= threshold]
        mae = float(np.mean(abs_diff))
        result[name] = mae
    result["mean"] = float(np.mean([result[name] for name in JOINT_NAMES]))

    return result


def compute_rmse_per_joint(
    predictions: np.ndarray, targets: np.ndarray, percentile: float = 100.0
) -> dict[str, float]:
    """RMSE por articulación en metros."""
    pred = predictions.reshape(-1, N_JOINTS, N_COORDS)
    tgt = targets.reshape(-1, N_JOINTS, N_COORDS)

    result = {}
    for j, name in enumerate(JOINT_NAMES):
        diff_sq = ((pred[:, j, :] - tgt[:, j, :]) ** 2).flatten()
        if percentile is not None and percentile < 100.0:
            threshold = np.percentile(diff_sq, percentile)
            diff_sq = diff_sq[diff_sq <= threshold]
        rmse = float(np.sqrt(np.mean(diff_sq)))
        result[name] = rmse
    result["mean"] = float(np.mean([result[name] for name in JOINT_NAMES]))

    return result


def compute_pearson_per_joint(
    predictions: np.ndarray, targets: np.ndarray
) -> dict[str, dict[str, float]]:
    """Correlación de Pearson por articulación y eje."""
    pred = predictions.reshape(-1, N_JOINTS, N_COORDS)
    tgt = targets.reshape(-1, N_JOINTS, N_COORDS)
    axis_labels = ["X", "Y", "Z"]

    result = {}
    for j, name in enumerate(JOINT_NAMES):
        joint_corr = {}
        for ax_i, ax_name in enumerate(axis_labels):
            p = pred[:, j, ax_i]
            t = tgt[:, j, ax_i]
            if np.std(p) > 1e-10 and np.std(t) > 1e-10:
                r_val, p_val = pearsonr(p, t)
            else:
                r_val, p_val = 0.0, 1.0
            joint_corr[ax_name] = {"r": float(r_val), "p_value": float(p_val)}
        result[name] = joint_corr

    return result


def compute_bland_altman(
    predictions: np.ndarray, targets: np.ndarray
) -> dict[str, dict[str, float]]:
    """Análisis Bland-Altman por eje."""
    axis_labels = ["X", "Y", "Z"]
    result = {}

    for ax_i, ax_name in enumerate(axis_labels):
        pred_ax = predictions[:, ax_i::3]  # Todas las articulaciones, eje ax_i
        tgt_ax = targets[:, ax_i::3]

        pred_flat = pred_ax.flatten()
        tgt_flat = tgt_ax.flatten()

        mean_vals = (pred_flat + tgt_flat) / 2.0
        diff_vals = pred_flat - tgt_flat

        bias = float(np.mean(diff_vals))
        std_diff = float(np.std(diff_vals))
        result[ax_name] = {
            "bias": bias,
            "std_diff": std_diff,
            "upper_loa": bias + 1.96 * std_diff,
            "lower_loa": bias - 1.96 * std_diff,
        }

    return result


def compute_displacement_metrics(
    all_pred_hip: np.ndarray,
    all_gt_hip: np.ndarray,
    all_pred_disp: np.ndarray,
    all_gt_disp: np.ndarray,
    epe_list: list[float],
) -> dict:
    """
    Métricas de error para el desplazamiento pélvico autónomo (ATE & EPE).

    Returns:
        dict con ATE, EPE, Pearson r por eje y RMSE de velocidad.
    """
    hip_diff = all_pred_hip - all_gt_hip  # [N, 3]
    ate_per_frame = np.linalg.norm(hip_diff, axis=1)  # [N]

    ate_mean = float(np.mean(ate_per_frame))
    ate_std = float(np.std(ate_per_frame))
    ate_max = float(np.max(ate_per_frame))

    ate_x = float(np.mean(np.abs(hip_diff[:, 0])))
    ate_y = float(np.mean(np.abs(hip_diff[:, 1])))
    ate_z = float(np.mean(np.abs(hip_diff[:, 2])))

    epe_mean = float(np.mean(epe_list)) if len(epe_list) > 0 else 0.0
    epe_max = float(np.max(epe_list)) if len(epe_list) > 0 else 0.0

    disp_diff = all_pred_disp - all_gt_disp
    disp_rmse = float(np.sqrt(np.mean(disp_diff**2)))

    corr_dict = {}
    for i, ax in enumerate(["X", "Y", "Z"]):
        if np.std(all_pred_hip[:, i]) > 1e-6 and np.std(all_gt_hip[:, i]) > 1e-6:
            r_val, _ = pearsonr(all_pred_hip[:, i], all_gt_hip[:, i])
        else:
            r_val = 0.0
        corr_dict[ax] = float(r_val)

    return {
        "ate_mean": ate_mean,
        "ate_std": ate_std,
        "ate_max": ate_max,
        "ate_x": ate_x,
        "ate_y": ate_y,
        "ate_z": ate_z,
        "epe_mean": epe_mean,
        "epe_max": epe_max,
        "disp_rmse": disp_rmse,
        "pearson": corr_dict,
    }


def evaluate_on_captures(
    model: nn.Module,
    captures: list[dict],
    norm_stats: dict,
    label: str = "Test",
    file_paths: list[tuple[str, str]] | None = None,
    window_size: int = WINDOW_SIZE,
) -> dict:
    """
    Evalúa la red multitarea en una lista de capturas completas.
    Si se proporcionan file_paths, también genera la comparativa completa
    con el pipeline analítico para todas las métricas.

    Returns:
        dict con todas las métricas de evaluación (NN y Analítico).
    """
    all_predictions = []
    all_targets = []
    all_hip_preds = []
    all_hip_gts = []
    all_disp_preds = []
    all_disp_gts = []
    epe_list = []

    # Para pipeline analítico
    all_ana_predictions = []
    all_ana_hip_preds = []
    all_ana_disp_preds = []
    epe_list_ana = []

    for idx, cap in enumerate(captures):
        anthropometric = np.array(
            [
                cap["femur_length"] / GaitDataset.BONE_NORM,
                cap["tibia_length"] / GaitDataset.BONE_NORM,
                cap["hip_width"] / GaitDataset.BONE_NORM,
                1.0 if cap["direction"] == "R" else 0.0,
            ],
            dtype=np.float32,
        )

        pred_pose, pred_disp, pred_hip_global, valid = predict_full_sequence_multitask(
            model,
            cap["sensor_data"],
            anthropometric,
            norm_stats,
            window_size=window_size,
            initial_hip=cap["gt_hip"],
        )

        # Usar solo timesteps con GT y predicción válida
        mask = cap["gt_mask"] & valid
        if np.sum(mask) < 10:
            continue

        pred_valid = pred_pose[mask]
        gt_valid = cap["gt_relative"][mask]

        hip_pred_valid = pred_hip_global[mask]
        hip_gt_valid = cap["gt_hip"][mask]

        disp_pred_valid = pred_disp[mask]
        disp_gt_valid = cap["gt_hip_disp"][mask]

        all_predictions.append(pred_valid)
        all_targets.append(gt_valid)
        all_hip_preds.append(hip_pred_valid)
        all_hip_gts.append(hip_gt_valid)
        all_disp_preds.append(disp_pred_valid)
        all_disp_gts.append(disp_gt_valid)

        final_diff = float(np.linalg.norm(hip_pred_valid[-1] - hip_gt_valid[-1]))
        epe_list.append(final_diff)

        # Inferencia Analítica si están disponibles los archivos
        if file_paths is not None and idx < len(file_paths):
            mcd_path, aruco_path = file_paths[idx]
            try:
                ana_res = process_files((mcd_path, aruco_path), use_3d=False)
                if ana_res is not None:
                    rec = ana_res[2]
                    hip_center = (rec["hip_r"] + rec["hip_l"]) / 2.0
                    first_valid = cap.get("first_valid", 0)
                    cap_len = len(cap["sensor_data"])

                    ana_rel_cap = np.zeros((cap_len, N_JOINTS * N_COORDS))
                    for j_idx, j_name in enumerate(JOINT_NAMES):
                        ana_rel_cap[:, j_idx * 3 : (j_idx + 1) * 3] = (
                            rec[j_name] - hip_center
                        )[first_valid : first_valid + cap_len]

                    ana_hip_global_cap = hip_center[first_valid : first_valid + cap_len]
                    if len(ana_hip_global_cap) > 0 and len(cap["gt_hip"]) > 0:
                        ana_offset = cap["gt_hip"][0] - ana_hip_global_cap[0]
                        ana_hip_global_cap = ana_hip_global_cap + ana_offset

                    ana_disp_cap = np.zeros((cap_len, 3))
                    ana_disp_diff = np.diff(ana_hip_global_cap, axis=0)
                    if len(ana_disp_diff) > 0:
                        ana_disp_cap[1:] = ana_disp_diff

                    all_ana_predictions.append(ana_rel_cap[mask])
                    all_ana_hip_preds.append(ana_hip_global_cap[mask])
                    all_ana_disp_preds.append(ana_disp_cap[mask])
                    epe_list_ana.append(
                        float(
                            np.linalg.norm(
                                ana_hip_global_cap[mask][-1] - hip_gt_valid[-1]
                            )
                        )
                    )
            except Exception:
                pass

    if len(all_predictions) == 0:
        print(f"  [WARN] No hay predicciones válidas para {label}")
        return {}

    # Concatenar todas las predicciones NN
    all_pred = np.concatenate(all_predictions, axis=0)
    all_tgt = np.concatenate(all_targets, axis=0)
    all_hip_p = np.concatenate(all_hip_preds, axis=0)
    all_hip_g = np.concatenate(all_hip_gts, axis=0)
    all_disp_p = np.concatenate(all_disp_preds, axis=0)
    all_disp_g = np.concatenate(all_disp_gts, axis=0)

    # ── Reconstrucción Global 3D (Espacio del Mundo Real) ─────────────
    all_pred_global = np.zeros_like(all_pred)
    all_tgt_global = np.zeros_like(all_tgt)
    for j in range(N_JOINTS):
        all_pred_global[:, j * 3 : (j + 1) * 3] = (
            all_pred[:, j * 3 : (j + 1) * 3] + all_hip_p
        )
        all_tgt_global[:, j * 3 : (j + 1) * 3] = (
            all_tgt[:, j * 3 : (j + 1) * 3] + all_hip_g
        )

    # Métricas NN - Pose Relativa (Pelvis en [0, 0, 0])
    mpjpe = compute_mpjpe(all_pred, all_tgt, percentile=100.0)
    mpjpe_p95 = compute_mpjpe(all_pred, all_tgt, percentile=95.0)
    mae = compute_mae_per_joint(all_pred, all_tgt, percentile=100.0)
    mae_p95 = compute_mae_per_joint(all_pred, all_tgt, percentile=95.0)
    rmse = compute_rmse_per_joint(all_pred, all_tgt, percentile=100.0)
    rmse_p95 = compute_rmse_per_joint(all_pred, all_tgt, percentile=95.0)
    pearson = compute_pearson_per_joint(all_pred, all_tgt)
    bland_altman = compute_bland_altman(all_pred, all_tgt)
    displacement_metrics = compute_displacement_metrics(
        all_hip_p, all_hip_g, all_disp_p, all_disp_g, epe_list
    )

    # Métricas NN - Pose Global 3D (Espacio de Cámara / Mundo Real con Odometría)
    global_mpjpe_100 = compute_mpjpe(all_pred_global, all_tgt_global, percentile=100.0)
    global_mpjpe_p95 = compute_mpjpe(all_pred_global, all_tgt_global, percentile=95.0)
    global_mae_100 = compute_mae_per_joint(
        all_pred_global, all_tgt_global, percentile=100.0
    )
    global_mae_p95 = compute_mae_per_joint(
        all_pred_global, all_tgt_global, percentile=95.0
    )
    global_rmse_100 = compute_rmse_per_joint(
        all_pred_global, all_tgt_global, percentile=100.0
    )
    global_rmse_p95 = compute_rmse_per_joint(
        all_pred_global, all_tgt_global, percentile=95.0
    )

    # Métricas Analítico (si están disponibles)
    has_ana = (
        len(all_ana_predictions) == len(all_predictions)
        and len(all_ana_predictions) > 0
    )
    if has_ana:
        all_ana_pred = np.concatenate(all_ana_predictions, axis=0)
        all_ana_hip_p = np.concatenate(all_ana_hip_preds, axis=0)
        all_ana_disp_p = np.concatenate(all_ana_disp_preds, axis=0)

        all_ana_global = np.zeros_like(all_ana_pred)
        for j in range(N_JOINTS):
            all_ana_global[:, j * 3 : (j + 1) * 3] = (
                all_ana_pred[:, j * 3 : (j + 1) * 3] + all_ana_hip_p
            )

        mpjpe_ana = compute_mpjpe(all_ana_pred, all_tgt, percentile=100.0)
        mpjpe_ana_p95 = compute_mpjpe(all_ana_pred, all_tgt, percentile=95.0)
        mae_ana = compute_mae_per_joint(all_ana_pred, all_tgt, percentile=100.0)
        mae_ana_p95 = compute_mae_per_joint(all_ana_pred, all_tgt, percentile=95.0)
        rmse_ana = compute_rmse_per_joint(all_ana_pred, all_tgt, percentile=100.0)
        rmse_ana_p95 = compute_rmse_per_joint(all_ana_pred, all_tgt, percentile=95.0)
        pearson_ana = compute_pearson_per_joint(all_ana_pred, all_tgt)
        bland_altman_ana = compute_bland_altman(all_ana_pred, all_tgt)
        displacement_metrics_ana = compute_displacement_metrics(
            all_ana_hip_p, all_hip_g, all_ana_disp_p, all_disp_g, epe_list_ana
        )

        global_mpjpe_ana_100 = compute_mpjpe(
            all_ana_global, all_tgt_global, percentile=100.0
        )
        global_mpjpe_ana_p95 = compute_mpjpe(
            all_ana_global, all_tgt_global, percentile=95.0
        )
        global_mae_ana_100 = compute_mae_per_joint(
            all_ana_global, all_tgt_global, percentile=100.0
        )
        global_mae_ana_p95 = compute_mae_per_joint(
            all_ana_global, all_tgt_global, percentile=95.0
        )
        global_rmse_ana_100 = compute_rmse_per_joint(
            all_ana_global, all_tgt_global, percentile=100.0
        )
        global_rmse_ana_p95 = compute_rmse_per_joint(
            all_ana_global, all_tgt_global, percentile=95.0
        )
    else:
        mpjpe_ana = mpjpe_ana_p95 = mae_ana = mae_ana_p95 = rmse_ana = rmse_ana_p95 = (
            pearson_ana
        ) = bland_altman_ana = displacement_metrics_ana = None
        global_mpjpe_ana_100 = global_mpjpe_ana_p95 = global_mae_ana_100 = (
            global_mae_ana_p95
        ) = global_rmse_ana_100 = global_rmse_ana_p95 = None

    # Imprimir resultados
    print()
    print("═" * W)
    title_label = (
        f"  📊 EVALUACIÓN COMPARATIVA – {label} (NN vs Pipeline Analítico)"
        if has_ana
        else f"  📊 EVALUACIÓN – {label}"
    )
    print(title_label)
    print("═" * W)

    print(f"\n  1) MPJPE Pose Relativa (Pelvis fija en [0, 0, 0])")
    print(f"  {'─' * 76}")
    if has_ana:
        print(
            f"     {'Articulación':<14}  {'NN (100%)':>10}  {'NN (P95%)':>10}  {'Analítico(100%)':>16}  {'Analítico(P95%)':>16}  {'Ganador':>9}"
        )
        print(
            f"     {'─' * 14}  {'─' * 10}  {'─' * 10}  {'─' * 16}  {'─' * 16}  {'─' * 9}"
        )
        win_m = "✅ NN" if mpjpe_p95["mean"] < mpjpe_ana_p95["mean"] else "📐 Analítico"
        print(
            f"     {'General':<14}  {mpjpe['mean']:>8.4f} m  {mpjpe_p95['mean']:>8.4f} m  {mpjpe_ana['mean']:>14.4f} m  {mpjpe_ana_p95['mean']:>14.4f} m  {win_m:>9}"
        )
        for name in JOINT_NAMES:
            win_j = "✅ NN" if mpjpe_p95[name] < mpjpe_ana_p95[name] else "📐 Analítico"
            print(
                f"     {name:<14}  {mpjpe[name]:>8.4f} m  {mpjpe_p95[name]:>8.4f} m  {mpjpe_ana[name]:>14.4f} m  {mpjpe_ana_p95[name]:>14.4f} m  {win_j:>9}"
            )
    else:
        print(
            f"     MPJPE general:  {mpjpe['mean']:.4f} m  (Percentil 95%: {mpjpe_p95['mean']:.4f} m)"
        )
        for name in JOINT_NAMES:
            print(
                f"     {name:<14}  {mpjpe[name]:.4f} m  (P95: {mpjpe_p95[name]:.4f} m)"
            )

    print(f"\n  2) MAE Pose Relativa por articulación")
    print(f"  {'─' * 76}")
    if has_ana:
        print(
            f"     {'Articulación':<14}  {'NN (100%)':>10}  {'NN (P95%)':>10}  {'Analítico(100%)':>16}  {'Analítico(P95%)':>16}  {'Ganador':>9}"
        )
        print(
            f"     {'─' * 14}  {'─' * 10}  {'─' * 10}  {'─' * 16}  {'─' * 16}  {'─' * 9}"
        )
        win_mae = "✅ NN" if mae_p95["mean"] < mae_ana_p95["mean"] else "📐 Analítico"
        print(
            f"     {'General':<14}  {mae['mean']:>8.4f} m  {mae_p95['mean']:>8.4f} m  {mae_ana['mean']:>14.4f} m  {mae_ana_p95['mean']:>14.4f} m  {win_mae:>9}"
        )
        for name in JOINT_NAMES:
            win_j = "✅ NN" if mae_p95[name] < mae_ana_p95[name] else "📐 Analítico"
            print(
                f"     {name:<14}  {mae[name]:>8.4f} m  {mae_p95[name]:>8.4f} m  {mae_ana[name]:>14.4f} m  {mae_ana_p95[name]:>14.4f} m  {win_j:>9}"
            )
    else:
        print(
            f"     MAE general:    {mae['mean']:.4f} m  (Percentil 95%: {mae_p95['mean']:.4f} m)"
        )
        for name in JOINT_NAMES:
            print(f"     {name:<14}  {mae[name]:.4f} m  (P95: {mae_p95[name]:.4f} m)")

    print(f"\n  3) RMSE Pose Relativa por articulación")
    print(f"  {'─' * 76}")
    if has_ana:
        print(
            f"     {'Articulación':<14}  {'NN (100%)':>10}  {'NN (P95%)':>10}  {'Analítico(100%)':>16}  {'Analítico(P95%)':>16}  {'Ganador':>9}"
        )
        print(
            f"     {'─' * 14}  {'─' * 10}  {'─' * 10}  {'─' * 16}  {'─' * 16}  {'─' * 9}"
        )
        win_r = "✅ NN" if rmse_p95["mean"] < rmse_ana_p95["mean"] else "📐 Analítico"
        print(
            f"     {'General':<14}  {rmse['mean']:>8.4f} m  {rmse_p95['mean']:>8.4f} m  {rmse_ana['mean']:>14.4f} m  {rmse_ana_p95['mean']:>14.4f} m  {win_r:>9}"
        )
        for name in JOINT_NAMES:
            win_j = "✅ NN" if rmse_p95[name] < rmse_ana[name] else "📐 Analítico"
            print(
                f"     {name:<14}  {rmse[name]:>8.4f} m  {rmse_p95[name]:>8.4f} m  {rmse_ana[name]:>14.4f} m  {rmse_ana_p95[name]:>14.4f} m  {win_j:>9}"
            )
    else:
        print(
            f"     RMSE general:   {rmse['mean']:.4f} m  (Percentil 95%: {rmse_p95['mean']:.4f} m)"
        )
        for name in JOINT_NAMES:
            print(f"     {name:<14}  {rmse[name]:.4f} m  (P95: {rmse_p95[name]:.4f} m)")

    print(f"\n  4) Correlación de Pearson r (Pose por eje)")
    print(f"  {'─' * 76}")
    if has_ana:
        print(
            f"     {'Articulación':<14}  {'NN (r_X / r_Y / r_Z)':>26}  {'Analítico (r_X / r_Y / r_Z)':>26}"
        )
        print(f"     {'─' * 14}  {'─' * 26}  {'─' * 26}")
        for name in JOINT_NAMES:
            rx_nn, ry_nn, rz_nn = (
                pearson[name]["X"]["r"],
                pearson[name]["Y"]["r"],
                pearson[name]["Z"]["r"],
            )
            rx_an, ry_an, rz_an = (
                pearson_ana[name]["X"]["r"],
                pearson_ana[name]["Y"]["r"],
                pearson_ana[name]["Z"]["r"],
            )
            print(
                f"     {name:<14}  {rx_nn:>7.4f} / {ry_nn:>6.4f} / {rz_nn:>6.4f}  {rx_an:>7.4f} / {ry_an:>6.4f} / {rz_an:>6.4f}"
            )
    else:
        print(f"     {'Articulación':<12}  {'r_X':>8}  {'r_Y':>8}  {'r_Z':>8}")
        print(f"     {'─' * 12}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
        for name in JOINT_NAMES:
            rx = pearson[name]["X"]["r"]
            ry = pearson[name]["Y"]["r"]
            rz = pearson[name]["Z"]["r"]
            print(f"     {name:<12}  {rx:>8.4f}  {ry:>8.4f}  {rz:>8.4f}")

    print(f"\n  5) Bland-Altman (Sesgo ± LoA por eje)")
    print(f"  {'─' * 76}")
    for ax_name in ["X", "Y", "Z"]:
        ba_nn = bland_altman[ax_name]
        if has_ana:
            ba_an = bland_altman_ana[ax_name]
            print(
                f"     Eje {ax_name}:  NN = {ba_nn['bias']:+.4f}m (LoA [{ba_nn['lower_loa']:+.4f}, {ba_nn['upper_loa']:+.4f}]) │ "
                f"Analítico = {ba_an['bias']:+.4f}m (LoA [{ba_an['lower_loa']:+.4f}, {ba_an['upper_loa']:+.4f}])"
            )
        else:
            print(
                f"     Eje {ax_name}:  Sesgo = {ba_nn['bias']:.4f} m,  "
                f"LoA = [{ba_nn['lower_loa']:.4f}, {ba_nn['upper_loa']:.4f}] m"
            )

    print(f"\n  6) Desplazamiento Pélvico Autónomo (Cadera Global)")
    print(f"  {'─' * 76}")
    if has_ana:
        d_nn = displacement_metrics
        d_an = displacement_metrics_ana
        print(
            f"     {'Métrica':<30}  {'NN (m)':>12}  {'Analítico (m)':>15}  {'Ganador':>12}"
        )
        print(f"     {'─' * 30}  {'─' * 12}  {'─' * 15}  {'─' * 12}")
        w_ate = "✅ NN" if d_nn["ate_mean"] < d_an["ate_mean"] else "📐 Analítico"
        print(
            f"     {'ATE Global':<30}  {d_nn['ate_mean']:>12.4f}  {d_an['ate_mean']:>15.4f}  {w_ate:>12}"
        )
        w_x = "✅ NN" if d_nn["ate_x"] < d_an["ate_x"] else "📐 Analítico"
        print(
            f"     {'ATE Eje X (Avance frontal)':<30}  {d_nn['ate_x']:>12.4f}  {d_an['ate_x']:>15.4f}  {w_x:>12}"
        )
        w_y = "✅ NN" if d_nn["ate_y"] < d_an["ate_y"] else "📐 Analítico"
        print(
            f"     {'ATE Eje Y (Desplazamiento lat.)':<30}  {d_nn['ate_y']:>12.4f}  {d_an['ate_y']:>15.4f}  {w_y:>12}"
        )
        w_z = "✅ NN" if d_nn["ate_z"] < d_an["ate_z"] else "📐 Analítico"
        print(
            f"     {'ATE Eje Z (Oscilación vert.)':<30}  {d_nn['ate_z']:>12.4f}  {d_an['ate_z']:>15.4f}  {w_z:>12}"
        )
        w_epe = "✅ NN" if d_nn["epe_mean"] < d_an["epe_mean"] else "📐 Analítico"
        print(
            f"     {'EPE (Deriva Final)':<30}  {d_nn['epe_mean']:>12.4f}  {d_an['epe_mean']:>15.4f}  {w_epe:>12}"
        )
        w_rx = (
            "✅ NN" if d_nn["pearson"]["X"] > d_an["pearson"]["X"] else "📐 Analítico"
        )
        print(
            f"     {'Correlación r (Avance X)':<30}  {d_nn['pearson']['X']:>12.4f}  {d_an['pearson']['X']:>15.4f}  {w_rx:>12}"
        )
        w_rz = (
            "✅ NN" if d_nn["pearson"]["Z"] > d_an["pearson"]["Z"] else "📐 Analítico"
        )
        print(
            f"     {'Correlación r (Oscilación Z)':<30}  {d_nn['pearson']['Z']:>12.4f}  {d_an['pearson']['Z']:>15.4f}  {w_rz:>12}"
        )
    else:
        print(
            f"     ATE (Absolute Trajectory Error): {displacement_metrics['ate_mean'] * 100:.2f} cm ± {displacement_metrics['ate_std'] * 100:.2f} cm"
        )
        print(
            f"     ATE Eje X (Avance frontal):      {displacement_metrics['ate_x'] * 100:.2f} cm"
        )
        print(
            f"     ATE Eje Y (Desplazamiento lat.): {displacement_metrics['ate_y'] * 100:.2f} cm"
        )
        print(
            f"     ATE Eje Z (Oscilación vertical): {displacement_metrics['ate_z'] * 100:.2f} cm"
        )
        print(
            f"     EPE (Deriva Final al terminar):   {displacement_metrics['epe_mean'] * 100:.2f} cm (Máx: {displacement_metrics['epe_max'] * 100:.2f} cm)"
        )
        print(
            f"     Correlación r (Avance Eje X):    {displacement_metrics['pearson']['X']:.4f}"
        )
        print(
            f"     Correlación r (Oscilación Eje Z): {displacement_metrics['pearson']['Z']:.4f}"
        )

    # ── 7) MPJPE Global 3D (Espacio del Mundo Real) ───────────────────
    print(
        f"\n  7) MPJPE Global 3D (Espacio del Mundo Real / Con Desplazamiento Acumulado)"
    )
    print(f"  {'─' * 76}")
    if has_ana:
        print(
            f"     {'Articulación':<14}  {'NN (100%)':>10}  {'NN (P95%)':>10}  {'Analítico(100%)':>16}  {'Analítico(P95%)':>16}  {'Ganador':>9}"
        )
        print(
            f"     {'─' * 14}  {'─' * 10}  {'─' * 10}  {'─' * 16}  {'─' * 16}  {'─' * 9}"
        )
        win_gm = (
            "✅ NN"
            if global_mpjpe_p95["mean"] < global_mpjpe_ana_p95["mean"]
            else "📐 Analítico"
        )
        print(
            f"     {'General':<14}  {global_mpjpe_100['mean']:>8.4f} m  {global_mpjpe_p95['mean']:>8.4f} m  {global_mpjpe_ana_100['mean']:>14.4f} m  {global_mpjpe_ana_p95['mean']:>14.4f} m  {win_gm:>9}"
        )
        for name in JOINT_NAMES:
            win_gj = (
                "✅ NN"
                if global_mpjpe_p95[name] < global_mpjpe_ana_p95[name]
                else "📐 Analítico"
            )
            print(
                f"     {name:<14}  {global_mpjpe_100[name]:>8.4f} m  {global_mpjpe_p95[name]:>8.4f} m  {global_mpjpe_ana_100[name]:>14.4f} m  {global_mpjpe_ana_p95[name]:>14.4f} m  {win_gj:>9}"
            )
    else:
        print(
            f"     MPJPE Global general:  {global_mpjpe_100['mean']:.4f} m  (Percentil 95%: {global_mpjpe_p95['mean']:.4f} m)"
        )
        for name in JOINT_NAMES:
            print(
                f"     {name:<14}  {global_mpjpe_100[name]:.4f} m  (P95: {global_mpjpe_p95[name]:.4f} m)"
            )

    # ── 8) MAE y RMSE Global 3D ───────────────────────────────────────
    print(f"\n  8) MAE y RMSE Global 3D (Espacio del Mundo Real)")
    print(f"  {'─' * 76}")
    if has_ana:
        print(
            f"     MAE Global General:   NN = {global_mae_100['mean']:.4f} m (P95: {global_mae_p95['mean']:.4f} m) │ "
            f"Analítico = {global_mae_ana_100['mean']:.4f} m (P95: {global_mae_ana_p95['mean']:.4f} m)"
        )
        print(
            f"     RMSE Global General:  NN = {global_rmse_100['mean']:.4f} m (P95: {global_rmse_p95['mean']:.4f} m) │ "
            f"Analítico = {global_rmse_ana_100['mean']:.4f} m (P95: {global_rmse_ana_p95['mean']:.4f} m)"
        )
    else:
        print(
            f"     MAE Global general:   {global_mae_100['mean']:.4f} m  (Percentil 95%: {global_mae_p95['mean']:.4f} m)"
        )
        print(
            f"     RMSE Global general:  {global_rmse_100['mean']:.4f} m  (Percentil 95%: {global_rmse_p95['mean']:.4f} m)"
        )

    print("═" * W)

    return {
        "mpjpe": mpjpe,
        "mpjpe_p95": mpjpe_p95,
        "mae": mae,
        "mae_p95": mae_p95,
        "rmse": rmse,
        "rmse_p95": rmse_p95,
        "global_mpjpe": global_mpjpe_100,
        "global_mpjpe_p95": global_mpjpe_p95,
        "global_mae": global_mae_100,
        "global_mae_p95": global_mae_p95,
        "global_rmse": global_rmse_100,
        "global_rmse_p95": global_rmse_p95,
        "pearson": pearson,
        "bland_altman": bland_altman,
        "displacement": displacement_metrics,
        "mpjpe_ana": mpjpe_ana,
        "mpjpe_ana_p95": mpjpe_ana_p95 if has_ana else None,
        "mae_ana": mae_ana,
        "mae_ana_p95": mae_ana_p95 if has_ana else None,
        "rmse_ana": rmse_ana,
        "rmse_ana_p95": rmse_ana_p95 if has_ana else None,
        "global_mpjpe_ana": global_mpjpe_ana_100 if has_ana else None,
        "global_mpjpe_ana_p95": global_mpjpe_ana_p95 if has_ana else None,
        "global_mae_ana": global_mae_ana_100 if has_ana else None,
        "global_mae_ana_p95": global_mae_ana_p95 if has_ana else None,
        "global_rmse_ana": global_rmse_ana_100 if has_ana else None,
        "global_rmse_ana_p95": global_rmse_ana_p95 if has_ana else None,
        "pearson_ana": pearson_ana,
        "bland_altman_ana": bland_altman_ana,
        "displacement_ana": displacement_metrics_ana,
        "all_predictions": all_pred,
        "all_targets": all_tgt,
        "all_pred_global": all_pred_global,
        "all_tgt_global": all_tgt_global,
        "all_hip_preds": all_hip_preds,
        "all_hip_gts": all_hip_gts,
    }


# ═══════════════════════════════════════════════════════════════════════
#  COMPARACIÓN CON PIPELINE ANALÍTICO
# ═══════════════════════════════════════════════════════════════════════


def compare_with_analytical(
    model: nn.Module,
    test_captures: list[dict],
    test_file_paths: list[tuple[str, str]],
    norm_stats: dict,
    window_size: int = WINDOW_SIZE,
) -> dict:
    """
    Compara las predicciones de la red multitarea con la reconstrucción
    analítica existente (process_all.py) usando ArUco como ground truth.

    Returns:
        dict con métricas comparativas.
    """
    print()
    print("═" * W)
    print("  🔀 COMPARACIÓN: Red Neuronal Multitarea vs Pipeline Analítico")
    print("═" * W)

    nn_errors = []
    analytical_errors = []
    capture_labels = []

    for cap, (mcd_path, aruco_path) in zip(test_captures, test_file_paths):
        label = f"{Path(mcd_path).parent.name}/{Path(mcd_path).name}"

        # ── Predicción NN Multitarea ─────────────────────────────────
        anthropometric = np.array(
            [
                cap["femur_length"] / GaitDataset.BONE_NORM,
                cap["tibia_length"] / GaitDataset.BONE_NORM,
                cap["hip_width"] / GaitDataset.BONE_NORM,
                1.0 if cap["direction"] == "R" else 0.0,
            ],
            dtype=np.float32,
        )

        nn_pred_pose, nn_pred_disp, nn_hip_global, nn_valid = (
            predict_full_sequence_multitask(
                model,
                cap["sensor_data"],
                anthropometric,
                norm_stats,
                window_size=window_size,
                initial_hip=cap["gt_hip"],
            )
        )
        nn_pred = nn_pred_pose

        # ── Reconstrucción analítica ─────────────────────────────────
        try:
            analytical_result = process_files((mcd_path, aruco_path), use_3d=False)
            if analytical_result is None:
                continue
        except Exception as e:
            print(f"  [SKIP] Analítico falló para {label}: {e}")
            continue

        _, _, reconstruction, aruco_processed, homogeneous_data, _, _ = (
            analytical_result
        )

        # Convertir reconstrucción analítica a posiciones relativas a cadera
        hip_center = (reconstruction["hip_r"] + reconstruction["hip_l"]) / 2.0

        analytical_relative = {}
        for j_name in JOINT_NAMES:
            analytical_relative[j_name] = reconstruction[j_name] - hip_center

        # Comparar en los timestamps válidos
        mask = cap["gt_mask"] & nn_valid
        n_valid = int(np.sum(mask))
        if n_valid < 10:
            continue

        # GT relativo
        gt_rel = cap["gt_relative"][mask]

        # NN relativo
        nn_rel = nn_pred[mask]

        # Analítico relativo (sincronizado con offset first_valid)
        first_valid = cap.get("first_valid", 0)
        cap_len = len(cap["sensor_data"])
        analytical_rel = np.zeros_like(nn_rel)

        for j_idx, j_name in enumerate(JOINT_NAMES):
            analytic_slice = analytical_relative[j_name][
                first_valid : first_valid + cap_len
            ]
            if len(analytic_slice) == cap_len:
                analytical_rel[:, j_idx * 3 : (j_idx + 1) * 3] = analytic_slice[mask]

        # MPJPE
        nn_mpjpe = compute_mpjpe(nn_rel, gt_rel)["mean"]
        analytical_mpjpe = compute_mpjpe(analytical_rel, gt_rel)["mean"]

        nn_errors.append(nn_mpjpe)
        analytical_errors.append(analytical_mpjpe)
        capture_labels.append(label)

        print(
            f"  {label:<30}  "
            f"NN: {nn_mpjpe:.4f} m  │  "
            f"Analítico: {analytical_mpjpe:.4f} m  │  "
            f"{'✅ NN' if nn_mpjpe < analytical_mpjpe else '📐 Analítico'}"
        )

    if len(nn_errors) == 0:
        print("  [WARN] No se pudieron comparar capturas")
        return {}

    nn_errors = np.array(nn_errors)
    analytical_errors = np.array(analytical_errors)

    print()
    print(f"  {'─' * 50}")
    print(
        f"  MPJPE promedio NN:         {np.mean(nn_errors):.4f} "
        f"± {np.std(nn_errors):.4f} m"
    )
    print(
        f"  MPJPE promedio Analítico:  {np.mean(analytical_errors):.4f} "
        f"± {np.std(analytical_errors):.4f} m"
    )
    nn_wins = int(np.sum(nn_errors < analytical_errors))
    print(f"  NN gana en:                {nn_wins}/{len(nn_errors)} capturas")
    improvement = (
        (np.mean(analytical_errors) - np.mean(nn_errors))
        / np.mean(analytical_errors)
        * 100
    )
    print(f"  Mejora relativa:           {improvement:+.1f}%")
    print("═" * W)

    return {
        "nn_errors": nn_errors,
        "analytical_errors": analytical_errors,
        "labels": capture_labels,
        "nn_mean": float(np.mean(nn_errors)),
        "analytical_mean": float(np.mean(analytical_errors)),
        "improvement_pct": float(improvement),
    }


# ═══════════════════════════════════════════════════════════════════════
#  VISUALIZACIÓN
# ═══════════════════════════════════════════════════════════════════════


def plot_training_curves(history: dict, save_dir: Path):
    """Genera gráficas de las curvas de entrenamiento."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss total
    axes[0].plot(history["train_loss"], label="Train", color="#4472C4", alpha=0.8)
    axes[0].plot(history["val_loss"], label="Validation", color="#ED7D31", alpha=0.8)
    axes[0].set_title("Loss Total", fontweight="bold")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale("log")

    # Pose Loss / MSE
    pose_tr_key = "train_pose" if "train_pose" in history else "train_mse"
    pose_va_key = "val_pose" if "val_pose" in history else "val_mse"
    if pose_tr_key in history and len(history[pose_tr_key]) > 0:
        axes[1].plot(
            history[pose_tr_key], label="Train Pose", color="#4472C4", alpha=0.8
        )
        axes[1].plot(history[pose_va_key], label="Val Pose", color="#ED7D31", alpha=0.8)
    axes[1].set_title("MSE Pose Articular", fontweight="bold")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("MSE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale("log")

    # Learning rate
    axes[2].plot(history["lr"], color="#70AD47")
    axes[2].set_title("Learning Rate", fontweight="bold")
    axes[2].set_xlabel("Época")
    axes[2].set_ylabel("LR")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale("log")

    plt.suptitle(
        "Curvas de Entrenamiento – Red Neuronal de Reconstrucción",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(save_dir / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {save_dir / 'training_curves.png'}")


def plot_mpjpe_comparison(eval_metrics: dict, save_dir: Path):
    """Boxplot de MPJPE por articulación."""
    if "per_capture_mpjpe" not in eval_metrics:
        return

    per_capture = eval_metrics["per_capture_mpjpe"]

    fig, ax = plt.subplots(figsize=(10, 6))

    data_per_joint = {}
    for name in JOINT_NAMES:
        data_per_joint[name] = [cap[name] for cap in per_capture]

    positions = list(range(len(JOINT_NAMES)))
    bp = ax.boxplot(
        [data_per_joint[name] for name in JOINT_NAMES],
        positions=positions,
        patch_artist=True,
        widths=0.6,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([n.replace("_", " ").title() for n in JOINT_NAMES])

    colors = ["#4472C4", "#ED7D31", "#A5A5A5", "#FFC000"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title("MPJPE por Articulación (Test Set)", fontsize=14, fontweight="bold")
    ax.set_ylabel("MPJPE (m)")
    ax.grid(True, alpha=0.3, axis="y")

    # Línea de media global
    global_mean = eval_metrics["mpjpe"]["mean"]
    ax.axhline(
        global_mean,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Media global: {global_mean:.4f} m",
    )
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_dir / "mpjpe_boxplot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {save_dir / 'mpjpe_boxplot.png'}")


def plot_trajectory_comparison(
    capture: dict,
    nn_pred: np.ndarray,
    nn_valid: np.ndarray,
    save_dir: Path,
    capture_label: str = "sample",
):
    """Gráfica de trayectorias comparativas (NN vs GT) en el plano sagital."""
    mask = capture["gt_mask"] & nn_valid
    if np.sum(mask) < 10:
        return

    gt = capture["gt_relative"][mask]
    pred = nn_pred[mask]
    time_s = (capture["timestamps"][mask] - capture["timestamps"][mask][0]) / 1000.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Trayectorias Relativas – {capture_label}",
        fontsize=14,
        fontweight="bold",
    )

    for j_idx, (j_name, ax) in enumerate(zip(JOINT_NAMES, axes.flatten())):
        gt_x = gt[:, j_idx * 3]
        gt_z = gt[:, j_idx * 3 + 2]
        pred_x = pred[:, j_idx * 3]
        pred_z = pred[:, j_idx * 3 + 2]

        ax.plot(time_s, gt_x, label="GT X", color="#4472C4", alpha=0.8)
        ax.plot(
            time_s, pred_x, label="NN X", color="#4472C4", linestyle="--", alpha=0.6
        )
        ax.plot(time_s, gt_z, label="GT Z", color="#ED7D31", alpha=0.8)
        ax.plot(
            time_s, pred_z, label="NN Z", color="#ED7D31", linestyle="--", alpha=0.6
        )

        ax.set_title(j_name.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Posición relativa (m)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(
        save_dir / f"trajectory_{capture_label}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"  ✓ {save_dir / f'trajectory_{capture_label}.png'}")


def plot_nn_vs_analytical(comparison: dict, save_dir: Path):
    """Gráfica de barras comparando NN vs Analítico."""
    if not comparison or len(comparison.get("nn_errors", [])) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Barras por captura
    n = len(comparison["nn_errors"])
    x = np.arange(n)
    width = 0.35

    axes[0].bar(
        x - width / 2,
        comparison["nn_errors"],
        width,
        label="Red Neuronal",
        color="#4472C4",
        alpha=0.8,
    )
    axes[0].bar(
        x + width / 2,
        comparison["analytical_errors"],
        width,
        label="Analítico",
        color="#ED7D31",
        alpha=0.8,
    )
    axes[0].set_xlabel("Captura")
    axes[0].set_ylabel("MPJPE (m)")
    axes[0].set_title("MPJPE por Captura", fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")
    if n <= 20:
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(
            [label.split("/")[-1] for label in comparison["labels"]],
            rotation=45,
            ha="right",
            fontsize=7,
        )

    # Resumen
    methods = ["Red Neuronal", "Analítico"]
    means = [comparison["nn_mean"], comparison["analytical_mean"]]
    colors = ["#4472C4", "#ED7D31"]
    bars = axes[1].bar(methods, means, color=colors, alpha=0.8, width=0.5)
    axes[1].set_ylabel("MPJPE medio (m)")
    axes[1].set_title("Comparación Global", fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, means):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.4f}",
            ha="center",
            fontweight="bold",
        )

    plt.suptitle(
        "Comparación: Red Neuronal vs Pipeline Analítico",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(save_dir / "nn_vs_analytical.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {save_dir / 'nn_vs_analytical.png'}")


def plot_bland_altman_nn(eval_metrics: dict, save_dir: Path):
    """Gráficas Bland-Altman para la red neuronal."""
    if "bland_altman" not in eval_metrics:
        return

    ba = eval_metrics["bland_altman"]
    pred = eval_metrics["all_predictions"]
    tgt = eval_metrics["all_targets"]

    axis_labels = ["X", "Y", "Z"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Bland-Altman: Red Neuronal vs Ground Truth",
        fontsize=14,
        fontweight="bold",
    )

    for ax_i, ax_name in enumerate(axis_labels):
        pred_ax = pred[:, ax_i::3].flatten()
        tgt_ax = tgt[:, ax_i::3].flatten()

        mean_vals = (pred_ax + tgt_ax) / 2.0
        diff_vals = pred_ax - tgt_ax
        ba_info = ba[ax_name]

        axes[ax_i].scatter(mean_vals, diff_vals, alpha=0.05, s=2, color="steelblue")
        axes[ax_i].axhline(
            ba_info["bias"],
            color="red",
            linestyle="-",
            linewidth=1.5,
            label=f"Sesgo: {ba_info['bias']:.4f} m",
        )
        axes[ax_i].axhline(
            ba_info["upper_loa"],
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"+1.96σ: {ba_info['upper_loa']:.4f} m",
        )
        axes[ax_i].axhline(
            ba_info["lower_loa"],
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"-1.96σ: {ba_info['lower_loa']:.4f} m",
        )
        axes[ax_i].set_xlabel(f"Media {ax_name} (m)")
        axes[ax_i].set_ylabel(f"Diferencia {ax_name} (m)")
        axes[ax_i].set_title(f"Eje {ax_name}")
        axes[ax_i].legend(fontsize=7)
        axes[ax_i].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_dir / "bland_altman_nn.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {save_dir / 'bland_altman_nn.png'}")


def plot_error_heatmap(eval_metrics: dict, save_dir: Path):
    """Heatmap de RMSE por articulación y eje."""
    if "all_predictions" not in eval_metrics:
        return

    pred = eval_metrics["all_predictions"].reshape(-1, N_JOINTS, N_COORDS)
    tgt = eval_metrics["all_targets"].reshape(-1, N_JOINTS, N_COORDS)
    axis_labels = ["X", "Y", "Z"]

    # Matriz de RMSE: [n_joints, 3]
    rmse_matrix = np.zeros((N_JOINTS, N_COORDS))
    for j in range(N_JOINTS):
        for ax in range(N_COORDS):
            diff = pred[:, j, ax] - tgt[:, j, ax]
            rmse_matrix[j, ax] = np.sqrt(np.mean(diff**2))

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(rmse_matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(N_COORDS))
    ax.set_xticklabels(axis_labels)
    ax.set_yticks(range(N_JOINTS))
    ax.set_yticklabels([n.replace("_", " ").title() for n in JOINT_NAMES])
    ax.set_xlabel("Eje")
    ax.set_ylabel("Articulación")
    ax.set_title("RMSE por Articulación y Eje (m)", fontweight="bold")

    for i in range(N_JOINTS):
        for j in range(N_COORDS):
            ax.text(
                j,
                i,
                f"{rmse_matrix[i, j]:.4f}",
                ha="center",
                va="center",
                color=(
                    "white" if rmse_matrix[i, j] > np.median(rmse_matrix) else "black"
                ),
                fontweight="bold",
            )

    plt.colorbar(im, ax=ax, label="RMSE (m)")
    plt.tight_layout()
    fig.savefig(save_dir / "rmse_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {save_dir / 'rmse_heatmap.png'}")


def plot_displacement_analysis(eval_metrics: dict, save_dir: Path):
    """Grafica el análisis de error del desplazamiento pélvico autónomo."""
    if (
        "displacement" not in eval_metrics
        or len(eval_metrics.get("all_hip_preds", [])) == 0
    ):
        return

    disp = eval_metrics["displacement"]
    all_hip_p = eval_metrics["all_hip_preds"]
    all_hip_g = eval_metrics["all_hip_gts"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Ejemplo: Primera captura del conjunto de test
    hip_p_cap = all_hip_p[0]
    hip_g_cap = all_hip_g[0]
    time_axis = np.arange(len(hip_p_cap)) * 0.01  # 100 Hz

    # Subplot 1: Eje X (Avance frontal)
    axes[0].plot(
        time_axis, hip_g_cap[:, 0], "k-", linewidth=2, label="ArUco Ground Truth"
    )
    axes[0].plot(
        time_axis, hip_p_cap[:, 0], "r--", linewidth=2, label="Red Neuronal (Autónoma)"
    )
    axes[0].set_xlabel("Tiempo (s)")
    axes[0].set_ylabel("Posición X (m)")
    axes[0].set_title(
        f"Desplazamiento Frontal (Eje X) — ATE: {disp['ate_x']*100:.2f} cm",
        fontweight="bold",
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Eje Z (Oscilación vertical)
    axes[1].plot(
        time_axis, hip_g_cap[:, 2], "k-", linewidth=2, label="ArUco Ground Truth"
    )
    axes[1].plot(
        time_axis, hip_p_cap[:, 2], "b--", linewidth=2, label="Red Neuronal (Autónoma)"
    )
    axes[1].set_xlabel("Tiempo (s)")
    axes[1].set_ylabel("Posición Z (m)")
    axes[1].set_title(
        f"Oscilación Vertical (Eje Z) — ATE: {disp['ate_z']*100:.2f} cm",
        fontweight="bold",
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f"Análisis de Desplazamiento Pélvico — ATE Global: {disp['ate_mean']*100:.2f} cm │ EPE Deriva Final: {disp['epe_mean']*100:.2f} cm",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(save_dir / "displacement_analysis.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {save_dir / 'displacement_analysis.png'}")


def create_3d_animation(
    capture: dict,
    nn_pred: np.ndarray,
    nn_valid: np.ndarray,
    save_path: Path,
    nn_hip_global: np.ndarray = None,
    title: str = "Reconstrucción 3D: Red Neuronal vs ArUco Ground Truth",
    step_subsample: int = 2,
    interactive: bool = False,
):
    """
    Crea y guarda/muestra una animación 3D comparativa cuadro por cuadro
    del esqueleto tren inferior reconstruido por la Red Neuronal vs ArUco (GT).

    Args:
        capture: dict de la captura con gt_relative, gt_hip, gt_mask, timestamps
        nn_pred: np.ndarray [N, 18] posiciones relativas predichas por la red
        nn_valid: np.ndarray [N] bool con validez de predicción
        save_path: Path de salida para el archivo de animación (.gif)
        nn_hip_global: np.ndarray [N, 3] cadera global autónoma predicha por la red
        title: Título de la gráfica
        step_subsample: Submuestreo de cuadros (default: 2, 50fps si fs=100Hz)
        interactive: Si es True, muestra la animación interactiva con plt.show()
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    mask = capture["gt_mask"] & nn_valid
    if np.sum(mask) < 10:
        print("  [WARN] Insuficientes muestras válidas para animación 3D")
        return

    valid_indices = np.where(mask)[0][::step_subsample]
    n_frames = len(valid_indices)

    gt_rel = capture["gt_relative"][valid_indices]  # [n_frames, 18]
    pred_rel = nn_pred[valid_indices]  # [n_frames, 18]
    gt_hip = capture["gt_hip"][valid_indices]  # [n_frames, 3]

    pred_j = pred_rel.reshape(n_frames, N_JOINTS, N_COORDS)
    gt_j = gt_rel.reshape(n_frames, N_JOINTS, N_COORDS)

    # 6 articulaciones: ankle_r, knee_r, hip_r, hip_l, knee_l, ankle_l
    # Índices: hip_r=0, hip_l=1, knee_r=2, knee_l=3, ankle_r=4, ankle_l=5
    nn_skeleton = np.zeros((n_frames, 6, 3))
    gt_skeleton = np.zeros((n_frames, 6, 3))

    for f in range(n_frames):
        hip_gt = gt_hip[f]
        hip_nn = (
            nn_hip_global[valid_indices[f]] if nn_hip_global is not None else hip_gt
        )

        # NN — usar hip_r/hip_l predichos directamente
        nn_hip_r = pred_j[f, 0] + hip_nn
        nn_hip_l = pred_j[f, 1] + hip_nn
        nn_knee_r = pred_j[f, 2] + hip_nn
        nn_knee_l = pred_j[f, 3] + hip_nn
        nn_ankle_r = pred_j[f, 4] + hip_nn
        nn_ankle_l = pred_j[f, 5] + hip_nn

        nn_skeleton[f] = np.array(
            [nn_ankle_r, nn_knee_r, nn_hip_r, nn_hip_l, nn_knee_l, nn_ankle_l]
        )

        # GT (ArUco) — usar hip_r/hip_l estimados del GT
        gt_hip_r = gt_j[f, 0] + hip_gt
        gt_hip_l = gt_j[f, 1] + hip_gt
        gt_knee_r = gt_j[f, 2] + hip_gt
        gt_knee_l = gt_j[f, 3] + hip_gt
        gt_ankle_r = gt_j[f, 4] + hip_gt
        gt_ankle_l = gt_j[f, 5] + hip_gt

        gt_skeleton[f] = np.array(
            [gt_ankle_r, gt_knee_r, gt_hip_r, gt_hip_l, gt_knee_l, gt_ankle_l]
        )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    (line_nn,) = ax.plot(
        [],
        [],
        [],
        "o-",
        label="Red Neuronal (NN)",
        color="#1F77B4",
        linewidth=3,
        markersize=6,
    )
    (line_gt,) = ax.plot(
        [],
        [],
        [],
        "o-",
        label="ArUco (Ground Truth)",
        color="#D62728",
        linewidth=3,
        markersize=6,
        alpha=0.7,
    )

    time_text = ax.text2D(
        0.05, 0.95, "", transform=ax.transAxes, fontsize=11, fontweight="bold"
    )

    ax.set_xlabel("X - Desplazamiento (m)", labelpad=10)
    ax.set_ylabel("Y - Lateral (m)", labelpad=10)
    ax.set_zlabel("Z - Altura (m)", labelpad=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    def init():
        line_nn.set_data_3d([], [], [])
        line_gt.set_data_3d([], [], [])
        time_text.set_text("")
        return line_nn, line_gt, time_text

    def update(frame):
        nn_pose = nn_skeleton[frame]
        gt_pose = gt_skeleton[frame]

        line_nn.set_data_3d(nn_pose[:, 0], nn_pose[:, 1], nn_pose[:, 2])
        line_gt.set_data_3d(gt_pose[:, 0], gt_pose[:, 1], gt_pose[:, 2])

        hip_x = gt_hip[frame, 0]
        ax.set_xlim(hip_x - 0.6, hip_x + 0.6)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.1, 1.1)

        t_sec = frame * step_subsample * 0.01
        time_text.set_text(f"Cuadro: {frame+1}/{n_frames}  │  Tiempo: {t_sec:.2f} s")
        return line_nn, line_gt, time_text

    anim = FuncAnimation(
        fig, update, init_func=init, frames=n_frames, interval=30, blit=False
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  🎬 Renderizando y guardando animación 3D GIF en {save_path}...")
    try:
        writer = PillowWriter(fps=25)
        anim.save(str(save_path), writer=writer)
        print(f"  ✓ Animación 3D guardada exitosamente: {save_path}")
    except Exception as e:
        print(f"  [ERROR] No se pudo guardar el archivo GIF: {e}")

    if interactive:
        try:
            plt.show()
        except Exception as e:
            print(f"  [INFO] No se pudo abrir ventana interactiva: {e}")

    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Reconstrucción de marcha con redes neuronales"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Solo evaluación con modelo guardado",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Generar animación 3D comparativa (NN vs ArUco)",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--window-stride", type=int, default=WINDOW_STRIDE)
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DATASET_PATH),
        help="Ruta al dataset",
    )
    parser.add_argument(
        "--n-val-subjects",
        type=int,
        default=3,
        help="Número de sujetos para el conjunto de validación (default: 3)",
    )
    parser.add_argument(
        "--n-test-subjects",
        type=int,
        default=5,
        help="Número de sujetos para el conjunto de test (default: 5)",
    )
    parser.add_argument(
        "--val-subjects",
        nargs="+",
        type=str,
        default=None,
        help="Lista de IDs de sujetos específicos para validación (ej: 25 26 27)",
    )
    parser.add_argument(
        "--test-subjects",
        nargs="+",
        type=str,
        default=None,
        help="Lista de IDs de sujetos específicos para test (ej: 28 29 30)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=PATIENCE,
        help="Paciencia del entrenamiento (default: 10)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Ruta al checkpoint del modelo",
    )
    parser.add_argument(
        "--shuffle-subjects",
        action="store_true",
        default=True,
        help="Mezclar aleatoriamente los sujetos antes de dividir en train/val/test (default: True)",
    )
    parser.add_argument(
        "--no-shuffle-subjects",
        dest="shuffle_subjects",
        action="store_false",
        help="Desactivar la mezcla aleatoria de sujetos (usar orden numérico por ID)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para la mezcla de sujetos y reproducibilidad (default: 42)",
    )
    parser.add_argument(
        "--checkpoint-tag",
        type=str,
        default="",
        help="Tag para identificar el checkpoint (default: '')",
    )
    parser.add_argument(
        "--data-split",
        type=str,
        default=str(DATA_SPLIT_PATH),
        help="Ruta al archivo JSON con la división de sujetos (default: data_split.json)",
    )
    args = parser.parse_args()

    # Fijar semillas para reproducibilidad
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    dataset_path = Path(args.dataset)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("═" * W)
    print("  🦿 RECONSTRUCCIÓN DE MARCHA CON REDES NEURONALES")
    print("═" * W)
    print(f"  Dataset:          {dataset_path}")
    print(f"  Device:           {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU:              {torch.cuda.get_device_name(0)}")
    print(f"  Window size:      {args.window_size}")
    print(f"  Window stride:    {args.window_stride}")
    print("═" * W)

    # ── 1. Descubrir capturas ────────────────────────────────────────
    print("\n  📂 Descubriendo capturas...")
    captures_info = discover_captures(dataset_path)
    print(f"  Encontradas: {len(captures_info)} pares de archivos")

    if len(captures_info) == 0:
        print("  [ERROR] No se encontraron archivos en el dataset")
        sys.exit(1)

    # ── 2. Preparar datos ────────────────────────────────────────────
    print("\n  ⚙️  Cargando y preparando capturas...")
    subjects_data: dict[str, list[dict]] = {}
    file_paths_by_subject: dict[str, list[tuple[str, str]]] = {}

    total_captures = len(captures_info)
    success_count = 0
    failed_count = 0
    bar_len = 30

    for idx, (mcd_path, aruco_path, subject_id) in enumerate(captures_info, start=1):
        cap = prepare_capture(mcd_path, aruco_path)
        if cap is not None:
            if subject_id not in subjects_data:
                subjects_data[subject_id] = []
                file_paths_by_subject[subject_id] = []
            subjects_data[subject_id].append(cap)
            file_paths_by_subject[subject_id].append((mcd_path, aruco_path))
            success_count += 1
        else:
            failed_count += 1

        progress = idx / total_captures
        filled = int(bar_len * progress)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(
            f"\r  Cargando capturas │{bar}│ {idx}/{total_captures} ({progress*100:.1f}%) │ ✓ {success_count} │ ✗ {failed_count}",
            end="",
            flush=True,
        )

    print()  # Nueva línea al terminar la barra de progreso
    print(f"\n  ✓ Capturas cargadas correctamente: {success_count}/{total_captures}")
    print(f"  ✗ Capturas fallidas/omitidas:     {failed_count}/{total_captures}")

    all_subjects = sorted(subjects_data.keys(), key=lambda x: int(x))

    if len(all_subjects) < 3:
        print("  [ERROR] Se necesitan al menos 3 sujetos para train/val/test")
        sys.exit(1)

    # ── 3. Dividir por sujeto (Configurable + data_split.json + Shuffle) ─
    split_path = Path(args.data_split) if args.data_split else None
    if (
        split_path
        and split_path.exists()
        and args.test_subjects is None
        and args.val_subjects is None
    ):
        with open(split_path, "r") as f:
            split_info = json.load(f)
        train_subjects = [
            str(s)
            for s in split_info.get("train_subjects", [])
            if str(s) in subjects_data
        ]
        val_subjects = [
            str(s)
            for s in split_info.get("val_subjects", [])
            if str(s) in subjects_data
        ]
        test_subjects = [
            str(s)
            for s in split_info.get("test_subjects", [])
            if str(s) in subjects_data
        ]
        print(f"\n  📊 División de datos cargada desde {split_path}:")
        print(f"    Train ({len(train_subjects)} sujetos):      {train_subjects}")
        print(f"    Validación ({len(val_subjects)} sujetos): {val_subjects}")
        print(f"    Test ({len(test_subjects)} sujetos):       {test_subjects}")
    else:
        subjects_pool = list(all_subjects)
        if args.shuffle_subjects:
            rng = random.Random(args.seed)
            rng.shuffle(subjects_pool)

        if args.test_subjects is not None:
            test_subjects = [s for s in args.test_subjects if s in subjects_data]
        else:
            n_test = min(args.n_test_subjects, max(1, len(subjects_pool) - 2))
            test_subjects = subjects_pool[-n_test:]

        remaining_subjects = [s for s in subjects_pool if s not in test_subjects]

        if args.val_subjects is not None:
            val_subjects = [s for s in args.val_subjects if s in remaining_subjects]
        else:
            n_val = min(args.n_val_subjects, max(1, len(remaining_subjects) - 1))
            val_subjects = remaining_subjects[-n_val:]

        train_subjects = [s for s in remaining_subjects if s not in val_subjects]

        print(
            f"\n  📊 División de datos (Shuffle={args.shuffle_subjects}, Seed={args.seed}):"
        )
        print(f"    Train ({len(train_subjects)} sujetos):      {train_subjects}")
        print(f"    Validación ({len(val_subjects)} sujetos): {val_subjects}")
        print(f"    Test ({len(test_subjects)} sujetos):       {test_subjects}")

    train_captures = []
    for s in train_subjects:
        train_captures.extend(subjects_data[s])

    val_captures = []
    for s in val_subjects:
        val_captures.extend(subjects_data[s])

    test_captures = []
    test_file_paths = []
    for s in test_subjects:
        test_captures.extend(subjects_data[s])
        test_file_paths.extend(file_paths_by_subject[s])

    print(f"    Train capturas:  {len(train_captures)}")
    print(f"    Val capturas:    {len(val_captures)}")
    print(f"    Test capturas:   {len(test_captures)}")

    # ── 4. Crear Datasets ────────────────────────────────────────────
    print("\n  📦 Creando datasets con ventanas deslizantes...")

    train_dataset = GaitDataset(
        train_captures,
        window_size=args.window_size,
        stride=args.window_stride,
        augment=True,  # Data augmentation: jitter + channel dropout
    )
    norm_stats = train_dataset.get_stats()

    val_dataset = GaitDataset(
        val_captures,
        window_size=args.window_size,
        stride=args.window_size,  # Sin solapamiento para validación
        stats=norm_stats,
    )

    print(f"    Train ventanas:  {len(train_dataset)}")
    print(f"    Val ventanas:    {len(val_dataset)}")

    if len(train_dataset) == 0:
        print("  [ERROR] No se generaron ventanas de entrenamiento")
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ── 5. Crear modelo multitarea unificado ─────────────────────────
    model = GaitMultiTaskNetTCN(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
    ).to(DEVICE)

    print_model_summary(model)

    # ── 6. Entrenar o cargar checkpoint ──────────────────────────────
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else (
            CHECKPOINT_DIR / f"best_tcn_lite_{args.checkpoint_tag}.pt"
            if args.checkpoint_tag
            else CHECKPOINT_PATH
        )
    )

    history = None
    if args.eval_only:
        if not checkpoint_path.exists():
            print(f"  [ERROR] No se encontró checkpoint en {checkpoint_path}")
            sys.exit(1)
        print(f"\n  📥 Cargando GaitMultiTaskNetTCN (TCN Lite) desde {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=DEVICE, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        history = train_multitask_model(
            model,
            train_loader,
            val_loader,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            checkpoint_tag=args.checkpoint_tag,
        )

        stats_to_save = {
            "mean": norm_stats["mean"].tolist(),
            "std": norm_stats["std"].tolist(),
        }
        with open(CHECKPOINT_DIR / "norm_stats.json", "w") as f:
            json.dump(stats_to_save, f)

        # Guardar norm_stats también dentro del checkpoint .pt (evita bugs de float64)
        ckpt_path = (
            CHECKPOINT_DIR / f"best_tcn_lite_{args.checkpoint_tag}.pt"
            if args.checkpoint_tag
            else CHECKPOINT_PATH
        )
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, weights_only=False)
            ckpt["norm_stats"] = {"mean": norm_stats["mean"], "std": norm_stats["std"]}
            torch.save(ckpt, ckpt_path)

        # Guardar historial
        history_to_save = {k: [float(v) for v in vals] for k, vals in history.items()}
        with open(RESULTS_DIR / "training_history.json", "w") as f:
            json.dump(history_to_save, f, indent=2)

    # ── 7. Evaluación en test set ────────────────────────────────────
    print("\n  📊 Evaluando en test set...")
    eval_metrics = evaluate_on_captures(
        model,
        test_captures,
        norm_stats,
        label="Test Set",
        file_paths=test_file_paths,
        window_size=args.window_size,
    )

    # ── 8. Comparación con pipeline analítico ────────────────────────
    print("\n  🔀 Comparando con pipeline analítico...")
    comparison = compare_with_analytical(
        model,
        test_captures,
        test_file_paths,
        norm_stats,
        window_size=args.window_size,
    )

    # ── 9. Gráficas de evaluación ────────────────────────────────────
    print("\n  🎨 Generando gráficas de evaluación...")
    if not args.eval_only and history:
        plot_training_curves(history, RESULTS_DIR)
    if eval_metrics:
        plot_mpjpe_comparison(eval_metrics, RESULTS_DIR)
        plot_bland_altman_nn(eval_metrics, RESULTS_DIR)
        plot_error_heatmap(eval_metrics, RESULTS_DIR)
        plot_displacement_analysis(eval_metrics, RESULTS_DIR)

    if comparison:
        plot_nn_vs_analytical(comparison, RESULTS_DIR)

    # Trayectorias y animación 3D de ejemplo (primera captura del test set)
    if len(test_captures) > 0:
        sample_subject = test_subjects[0]
        cap = test_captures[0]
        anthropometric = np.array(
            [
                cap["femur_length"] / GaitDataset.BONE_NORM,
                cap["tibia_length"] / GaitDataset.BONE_NORM,
                cap["hip_width"] / GaitDataset.BONE_NORM,
                1.0 if cap["direction"] == "R" else 0.0,
            ],
            dtype=np.float32,
        )
        pred_pose, pred_disp, pred_hip_global, nn_valid = (
            predict_full_sequence_multitask(
                model,
                cap["sensor_data"],
                anthropometric,
                norm_stats,
                window_size=args.window_size,
                initial_hip=cap["gt_hip"],
            )
        )
        plot_trajectory_comparison(
            cap,
            pred_pose,
            nn_valid,
            RESULTS_DIR,
            capture_label=f"test_{sample_subject}_0",
        )

        anim_gif_path = (
            RESULTS_DIR / f"3d_animation_nn_vs_aruco_subject_{sample_subject}.gif"
        )
        create_3d_animation(
            cap,
            pred_pose,
            nn_valid,
            anim_gif_path,
            nn_hip_global=pred_hip_global,
            title=f"Reconstrucción 3D Autónoma (TCN Lite vs ArUco) – Sujeto {sample_subject}",
            interactive=args.animate,
        )

    # ── 10. Guardar métricas finales ─────────────────────────────────
    final_metrics = {}
    if eval_metrics:
        final_metrics["mpjpe"] = eval_metrics.get("mpjpe", {})
        final_metrics["mae"] = eval_metrics.get("mae", {})
        final_metrics["rmse"] = eval_metrics.get("rmse", {})
        final_metrics["bland_altman"] = eval_metrics.get("bland_altman", {})
        final_metrics["displacement"] = eval_metrics.get("displacement", {})
        final_metrics["global_mpjpe"] = eval_metrics.get("global_mpjpe", {})
        final_metrics["global_mpjpe_p95"] = eval_metrics.get("global_mpjpe_p95", {})
    if comparison:
        final_metrics["comparison"] = {
            "nn_mean_mpjpe": comparison.get("nn_mean", 0),
            "analytical_mean_mpjpe": comparison.get("analytical_mean", 0),
            "improvement_pct": comparison.get("improvement_pct", 0),
        }

    with open(RESULTS_DIR / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2, default=str)

    # ── Resumen final ────────────────────────────────────────────────
    print()
    print("═" * W)
    print("  ✅ PROCESAMIENTO COMPLETO")
    print("═" * W)
    print(f"  Resultados guardados en:   {RESULTS_DIR.resolve()}")
    print(f"  Modelo guardado en:        {CHECKPOINT_DIR.resolve()}")
    if eval_metrics:
        print(f"  MPJPE (test):              {eval_metrics['mpjpe']['mean']:.4f} m")
    if comparison:
        print(f"  MPJPE NN:                  {comparison['nn_mean']:.4f} m")
        print(f"  MPJPE Analítico:           {comparison['analytical_mean']:.4f} m")
        print(f"  Mejora:                    {comparison['improvement_pct']:+.1f}%")
    print("═" * W)


if __name__ == "__main__":
    main()
