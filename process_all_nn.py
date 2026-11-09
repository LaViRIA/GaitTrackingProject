"""
Procesamiento y Evaluación de Marcha con Redes Neuronales Modulares
===================================================================

Replica la funcionalidad de process_all.py pero usando dos redes neuronales:
  1. OrientationNet: Reemplaza el filtro Madgwick + PCA + Slerp
  2. PivotNet: Reemplaza la lógica heurística de detección de tobillo pivote

El resto del pipeline (cinemática directa, métricas, gráficas) se reutiliza
directamente de process_all.py.

Uso:
    python process_all_nn.py                            # Procesamiento completo
    python process_all_nn.py --checkpoint-dir Results/nn_modular/checkpoints

Autor: GaitTracking Project
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from nn_modular import (
    BONE_NORM,
    IMU_NAMES,
    ORIENT_INPUT_DIM,
    ORIENT_OUTPUT_DIM,
    PIVOT_INPUT_DIM,
    SENSOR_CHANNELS,
    OrientationNet,
    PivotNet,
)
from nn_modular import (
    WINDOW_SIZE as DEFAULT_WINDOW_SIZE,
)
from scipy.spatial.transform import Rotation as R
from scipy.stats import pearsonr
from torch.amp import autocast

from process_all import (
    adjust_aruco_data,
    apply_transformation,
    compute_displacement_velocity,
    compute_knee_angles_series,
    detect_stride_lengths,
    euclidean_error_3Dpoint,
    figsize,
    get_all_poses,
    get_dataset,
    homogenize_3dpoint_data,
    low_pass_filter,
    mean_absolute_error,
    measures_calculation,
    read_aruco_file,
    read_mcd_file,
    split_char,
    transformation_matrix,
)

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_CHECKPOINT_DIR = Path("nn-results/modular/checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W = 72  # Ancho de las cajas de texto


# ═══════════════════════════════════════════════════════════════════════
#  CARGA DE MODELOS
# ═══════════════════════════════════════════════════════════════════════


def load_models(checkpoint_dir: Path) -> tuple:
    """
    Carga PivotNet y OrientationNet desde checkpoints.

    Returns:
        (pivot_model, orient_model, norm_stats)
    """
    # ── PivotNet ──────────────────────────────────────────────────────
    pivot_path = checkpoint_dir / "best_pivot_model.pt"
    if not pivot_path.exists():
        print(f"  [ERROR] No se encontró PivotNet en {pivot_path}")
        sys.exit(1)

    pivot_ckpt = torch.load(pivot_path, map_location=DEVICE, weights_only=False)
    pivot_model = PivotNet().to(DEVICE)
    pivot_model.load_state_dict(pivot_ckpt["model_state_dict"])
    pivot_model.eval()
    print(f"  ✓ PivotNet cargado (época {pivot_ckpt.get('epoch', '?')})")

    # ── OrientationNet ────────────────────────────────────────────────
    orient_path = checkpoint_dir / "best_orientation_model.pt"
    if not orient_path.exists():
        print(f"  [ERROR] No se encontró OrientationNet en {orient_path}")
        sys.exit(1)

    orient_ckpt = torch.load(orient_path, map_location=DEVICE, weights_only=False)
    orient_model = OrientationNet().to(DEVICE)
    orient_model.load_state_dict(orient_ckpt["model_state_dict"])
    orient_model.eval()
    print(f"  ✓ OrientationNet cargado (época {orient_ckpt.get('epoch', '?')})")

    # ── Estadísticas de normalización ─────────────────────────────────
    # Intentar cargar desde checkpoint primero, luego desde JSON
    norm_stats = None
    if "norm_stats" in orient_ckpt:
        norm_stats = orient_ckpt["norm_stats"]
    elif "norm_stats" in pivot_ckpt:
        norm_stats = pivot_ckpt["norm_stats"]
    else:
        stats_path = checkpoint_dir / "norm_stats.json"
        if stats_path.exists():
            with open(stats_path, "r") as f:
                stats_json = json.load(f)
            norm_stats = {
                "mean": np.array(stats_json["mean"], dtype=np.float32),
                "std": np.array(stats_json["std"], dtype=np.float32),
            }

    if norm_stats is None:
        print("  [ERROR] No se encontraron estadísticas de normalización")
        sys.exit(1)

    # Asegurar que sean numpy arrays float32
    if isinstance(norm_stats["mean"], np.ndarray):
        norm_stats["mean"] = norm_stats["mean"].astype(np.float32)
        norm_stats["std"] = norm_stats["std"].astype(np.float32)
    else:
        norm_stats["mean"] = np.array(norm_stats["mean"], dtype=np.float32)
        norm_stats["std"] = np.array(norm_stats["std"], dtype=np.float32)

    print(f"  ✓ Estadísticas de normalización cargadas")

    return pivot_model, orient_model, norm_stats


# ═══════════════════════════════════════════════════════════════════════
#  INFERENCIA CON REDES NEURONALES
# ═══════════════════════════════════════════════════════════════════════


@torch.no_grad()
def predict_orientations_nn(
    model: OrientationNet,
    sensor_data: np.ndarray,
    anthropometric: np.ndarray,
    norm_stats: dict,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> dict:
    """
    Predice cuaterniones de orientación para todos los segmentos.

    Args:
        model: OrientationNet entrenado
        sensor_data: [N, 24] datos de sensores (ya filtrados, en unidades SI)
        anthropometric: [3] (femur/BONE_NORM, tibia/BONE_NORM, direction)
        norm_stats: dict con 'mean' y 'std' de normalización
        window_size: tamaño de la ventana

    Returns:
        dict con 'rf', 'lf', 'rt', 'lt', 'hip' → scipy Rotation arrays
    """
    model.eval()
    N = len(sensor_data)

    # Normalizar sensores
    sensor_norm = (sensor_data.astype(np.float32) - norm_stats["mean"]) / norm_stats[
        "std"
    ]

    # Broadcast antropométrico
    anthro_broadcast = np.tile(anthropometric.astype(np.float32), (window_size, 1))

    # Inferencia con ventanas solapadas + promedio
    pred_sum = np.zeros((N, ORIENT_OUTPUT_DIM), dtype=np.float64)
    counts = np.zeros(N, dtype=np.float64)
    stride = window_size // 2

    for start in range(0, N - window_size + 1, stride):
        end = start + window_size
        x_input = np.concatenate([sensor_norm[start:end], anthro_broadcast], axis=1)
        x = torch.from_numpy(x_input).unsqueeze(0).to(DEVICE)

        with autocast("cuda", enabled=(DEVICE.type == "cuda")):
            pred = model(x)

        pred_np = pred.float().cpu().numpy()[0]
        pred_sum[start:end] += pred_np
        counts[start:end] += 1

    # Promediar predicciones solapadas
    valid = counts > 0
    pred_sum[valid] /= counts[valid, np.newaxis]

    # Extender predicciones a frames no cubiertos (inicio/final)
    if not np.all(valid):
        first_valid = np.argmax(valid)
        last_valid = N - np.argmax(valid[::-1]) - 1
        pred_sum[:first_valid] = pred_sum[first_valid]
        pred_sum[last_valid + 1 :] = pred_sum[last_valid]

    # Reshapear y normalizar cuaterniones
    pred_quats = pred_sum.reshape(N, 5, 4)
    norms = np.linalg.norm(pred_quats, axis=-1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    pred_quats = pred_quats / norms

    # Convertir a dict de scipy Rotation (scalar-last, que es default de scipy)
    return {
        "rf": R.from_quat(pred_quats[:, 0]),
        "lf": R.from_quat(pred_quats[:, 1]),
        "rt": R.from_quat(pred_quats[:, 2]),
        "lt": R.from_quat(pred_quats[:, 3]),
        "hip": R.from_quat(pred_quats[:, 4]),
    }


@torch.no_grad()
def predict_pivots_nn(
    model: PivotNet,
    sensor_data: np.ndarray,
    norm_stats: dict,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> np.ndarray:
    """
    Predice el tobillo pivote frame-a-frame.

    Args:
        model: PivotNet entrenado
        sensor_data: [N, 24] datos de sensores
        norm_stats: dict con 'mean' y 'std'
        window_size: tamaño de la ventana

    Returns:
        [N] probabilidad de pivote derecho (>0.5 = derecho, <0.5 = izquierdo)
    """
    model.eval()
    N = len(sensor_data)

    sensor_norm = (sensor_data.astype(np.float32) - norm_stats["mean"]) / norm_stats[
        "std"
    ]

    pred_sum = np.zeros(N, dtype=np.float64)
    counts = np.zeros(N, dtype=np.float64)
    stride = window_size // 2

    for start in range(0, N - window_size + 1, stride):
        end = start + window_size
        x = torch.from_numpy(sensor_norm[start:end]).unsqueeze(0).to(DEVICE)

        with autocast("cuda", enabled=(DEVICE.type == "cuda")):
            pred = model(x)

        pred_np = pred.float().cpu().numpy()[0]
        pred_sum[start:end] += pred_np
        counts[start:end] += 1

    valid = counts > 0
    pred_sum[valid] /= counts[valid]

    # Extender a frames no cubiertos
    if not np.all(valid):
        first_valid = np.argmax(valid)
        last_valid = N - np.argmax(valid[::-1]) - 1
        pred_sum[:first_valid] = pred_sum[first_valid]
        pred_sum[last_valid + 1 :] = pred_sum[last_valid]

    return pred_sum


# ═══════════════════════════════════════════════════════════════════════
#  RECONSTRUCCIÓN CON REDES NEURONALES
# ═══════════════════════════════════════════════════════════════════════


def mcd_reconstruction_nn(
    mcd_data: dict,
    pivot_model: PivotNet,
    orient_model: OrientationNet,
    norm_stats: dict,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> dict:
    """
    Reconstrucción del esqueleto usando ambas redes neuronales.

    Reemplaza:
      - get_orientation() → OrientationNet
      - Lógica de pivote en mcd_reconstruction() → PivotNet

    Mantiene:
      - get_all_poses() (cinemática directa)
      - Lógica de transformación de tobillo pivote

    Returns:
        dict con hip_r, hip_l, knee_r, knee_l, ankle_r, ankle_l → [N, 3]
    """
    direction = mcd_data["measures"]["direction"]

    points = {
        "hip_r": np.array([0, mcd_data["measures"]["hip_width"] / 2, 0]),
        "knee_r": np.array([0, 0, -mcd_data["measures"]["femur_length"]]),
        "ankle_r": np.array([0, 0, -mcd_data["measures"]["tibia_length"]]),
        "hip_l": np.array([0, -mcd_data["measures"]["hip_width"] / 2, 0]),
        "knee_l": np.array([0, 0, -mcd_data["measures"]["femur_length"]]),
        "ankle_l": np.array([0, 0, -mcd_data["measures"]["tibia_length"]]),
    }

    # ── Apilar canales de sensores [N, 24] ────────────────────────────
    N = len(mcd_data["time"])
    sensor_data = np.zeros((N, SENSOR_CHANNELS), dtype=np.float64)
    for idx, imu in enumerate(IMU_NAMES):
        sensor_data[:, idx * 6 : idx * 6 + 3] = mcd_data[f"acceleration_{imu}"]
        sensor_data[:, idx * 6 + 3 : idx * 6 + 6] = mcd_data[f"gyro_{imu}"]

    # ── Preparar vector antropométrico ────────────────────────────────
    anthropometric = np.array(
        [
            mcd_data["measures"]["femur_length"] / BONE_NORM,
            mcd_data["measures"]["tibia_length"] / BONE_NORM,
            1.0 if direction == "R" else 0.0,
        ],
        dtype=np.float32,
    )

    # ══════════════════════════════════════════════════════════════════
    #  1. OrientationNet → Cuaterniones de orientación por segmento
    # ══════════════════════════════════════════════════════════════════
    quats = predict_orientations_nn(
        orient_model, sensor_data, anthropometric, norm_stats, window_size
    )

    # ══════════════════════════════════════════════════════════════════
    #  2. Cinemática directa (misma que process_all.py)
    # ══════════════════════════════════════════════════════════════════
    pose = get_all_poses(quats, points)

    # ══════════════════════════════════════════════════════════════════
    #  3. PivotNet → Estimación del tobillo pivote
    # ══════════════════════════════════════════════════════════════════
    pivot_pred = predict_pivots_nn(pivot_model, sensor_data, norm_stats, window_size)

    # ══════════════════════════════════════════════════════════════════
    #  4. Reconstrucción con tobillo pivote (misma lógica que original)
    # ══════════════════════════════════════════════════════════════════
    reconstruction = {
        "hip_r": np.empty((N, 3)),
        "hip_l": np.empty((N, 3)),
        "knee_r": np.empty((N, 3)),
        "knee_l": np.empty((N, 3)),
        "ankle_r": np.empty((N, 3)),
        "ankle_l": np.empty((N, 3)),
    }

    last_T_al = np.eye(4)
    last_T_ar = np.eye(4)

    for i in range(N):
        if i == 0:
            is_right_pivot = direction == "R"
        else:
            is_right_pivot = pivot_pred[i] > 0.5

        if is_right_pivot:
            T_a_2_o = transformation_matrix(
                pose["ankle_r"][i],
                R.from_quat([1, 0, 0, 0], scalar_first=True),
            )
            T_pivot = last_T_ar
        else:
            T_a_2_o = transformation_matrix(
                pose["ankle_l"][i],
                R.from_quat([1, 0, 0, 0], scalar_first=True),
            )
            T_pivot = last_T_al

        T_acu = np.linalg.inv(T_a_2_o) @ T_pivot

        for key in reconstruction:
            reconstruction[key][i] = apply_transformation(T_acu, pose[key][i])

        last_T_al = transformation_matrix(
            reconstruction["ankle_l"][i],
            R.from_quat([1, 0, 0, 0], scalar_first=True),
        )
        last_T_ar = transformation_matrix(
            reconstruction["ankle_r"][i],
            R.from_quat([1, 0, 0, 0], scalar_first=True),
        )

    return reconstruction


# ═══════════════════════════════════════════════════════════════════════
#  PROCESAMIENTO DE ARCHIVOS (equivalente a process_files)
# ═══════════════════════════════════════════════════════════════════════


def process_files_nn(
    files: tuple[str, str],
    pivot_model: PivotNet,
    orient_model: OrientationNet,
    norm_stats: dict,
    use_3d: bool = False,
    cutoff_freq: int = 7,
    filter_order: int = 6,
    window_size: int = DEFAULT_WINDOW_SIZE,
):
    """
    Procesa un par de archivos MCD/ArUco usando las redes neuronales.

    Equivalente a process_files() de process_all.py pero usa:
      - OrientationNet en lugar de Madgwick
      - PivotNet en lugar de lógica heurística de pivote

    Returns:
        Misma estructura que process_files():
        (gen_error, common_time, reconstruction, aruco_processed,
         homogeneous_data, velocity_data, extra_metrics)
    """
    # ── Leer archivos ─────────────────────────────────────────────────
    mcd_data = read_mcd_file(
        files[0], cutoff_freq=cutoff_freq, filter_order=filter_order
    )
    aruco_data = read_aruco_file(files[1])

    direction = mcd_data["measures"]["direction"]
    hidden_leg = "L" if direction == "R" else "R"

    aruco_data = adjust_aruco_data(aruco_data, direction)

    aruco_processed = {
        "hip_r": aruco_data["hip_position"],
        "hip_l": aruco_data["hip_position"],
        "knee_r": aruco_data["R_knee_position"],
        "knee_l": aruco_data["L_knee_position"],
        "ankle_r": aruco_data["R_ankle_position"],
        "ankle_l": aruco_data["L_ankle_position"],
    }
    for key in aruco_processed:
        aruco_processed[key]["data"][:, 1] = 0

    # ── Longitudes óseas ──────────────────────────────────────────────
    femur_length, tibia_length = measures_calculation(aruco_processed, direction)
    mcd_data["measures"]["femur_length"] = femur_length
    mcd_data["measures"]["tibia_length"] = tibia_length
    if (
        "hip_width" not in mcd_data["measures"]
        or mcd_data["measures"]["hip_width"] <= 0.05
    ):
        mcd_data["measures"]["hip_width"] = 0.25

    # ══════════════════════════════════════════════════════════════════
    #  RECONSTRUCCIÓN CON REDES NEURONALES
    # ══════════════════════════════════════════════════════════════════
    reconstruction = mcd_reconstruction_nn(
        mcd_data, pivot_model, orient_model, norm_stats, window_size
    )

    # ── Corrección de rotación (misma que process_all.py) ─────────────
    z_target = 0 if direction == "L" else np.pi
    T_fix = transformation_matrix([0, 0, 0], R.from_euler("ZXY", [z_target, 0, 0]))
    for i in range(reconstruction["hip_r"].shape[0]):
        for key in reconstruction:
            reconstruction[key][i] = apply_transformation(T_fix, reconstruction[key][i])

    # ══════════════════════════════════════════════════════════════════
    #  EVALUACIÓN (idéntica a process_all.py)
    # ══════════════════════════════════════════════════════════════════

    # Homogenizar datos
    homogeneous_data = {}
    for key in reconstruction:
        if key == f"hip_{hidden_leg.lower()}":
            continue
        t, a_out, b_out = homogenize_3dpoint_data(
            reconstruction[key],
            aruco_processed[key]["data"],
            mcd_data["time"],
            aruco_processed[key]["time"],
        )
        trim_min = t > 1000
        trim_max = t < t.max() - 500
        trim = trim_min & trim_max
        homogeneous_data[key] = {
            "time": t[trim],
            "mcd": a_out[trim],
            "aruco": b_out[trim],
        }

    common_time = set()
    for key in homogeneous_data:
        homogeneous_data[key]["euclidean"] = np.linalg.norm(
            homogeneous_data[key]["mcd"] - homogeneous_data[key]["aruco"], axis=1
        )
        homogeneous_data[key]["euclidean_mean"] = np.mean(
            homogeneous_data[key]["euclidean"]
        )
        homogeneous_data[key]["mae"] = mean_absolute_error(
            homogeneous_data[key]["mcd"], homogeneous_data[key]["aruco"]
        )
        homogeneous_data[key]["mae_mean"] = np.mean(homogeneous_data[key]["mae"])
        homogeneous_data[key]["eu_var"] = np.var(homogeneous_data[key]["euclidean"])
        homogeneous_data[key]["eu_std"] = np.std(homogeneous_data[key]["euclidean"])

        if len(common_time) == 0:
            common_time = set(homogeneous_data[key]["time"])
        else:
            common_time = common_time.intersection(set(homogeneous_data[key]["time"]))

    # Error euclidiano general
    common_time = sorted(list(common_time))
    five_points = np.empty((0, len(common_time)))
    for key in homogeneous_data:
        isin = np.isin(homogeneous_data[key]["time"], list(common_time))
        five_points = np.append(
            five_points, [homogeneous_data[key]["euclidean"][isin]], axis=0
        )

    mcd_common = np.empty((0, len(common_time), 3))
    aruco_common = np.empty((0, len(common_time), 3))
    for key in homogeneous_data:
        isin = np.isin(homogeneous_data[key]["time"], common_time)
        mcd_common = np.append(mcd_common, [homogeneous_data[key]["mcd"][isin]], axis=0)
        aruco_common = np.append(
            aruco_common, [homogeneous_data[key]["aruco"][isin]], axis=0
        )

    gen_eucliden_error = np.empty(0)
    d_all_position = np.empty((0, 3))
    m_all_position = np.empty((0, 3))
    for i in range(len(common_time)):
        d = mcd_common[:, i, :]
        m = aruco_common[:, i, :]
        d_position = np.mean(d, axis=0)
        m_position = np.mean(m, axis=0)
        d_all_position = np.append(d_all_position, [d_position], axis=0)
        m_all_position = np.append(m_all_position, [m_position], axis=0)
        err = euclidean_error_3Dpoint(d_position, m_position)
        gen_eucliden_error = np.append(gen_eucliden_error, err)

    # ── Velocidad de desplazamiento ───────────────────────────────────
    velocity_data = {}
    aruco_hip_key = "hip_r"
    if aruco_hip_key in aruco_processed:
        aruco_hip_t = aruco_processed[aruco_hip_key]["time"].astype(float)
        aruco_hip_pos = aruco_processed[aruco_hip_key]["data"]
        velocity_data["aruco_hip"] = compute_displacement_velocity(
            aruco_hip_pos, aruco_hip_t, trim_ms=1000
        )

    rec_hip_pos = (reconstruction["hip_r"] + reconstruction["hip_l"]) / 2.0
    velocity_data["reconstruction_hip"] = compute_displacement_velocity(
        rec_hip_pos, mcd_data["time"], trim_ms=1000
    )

    for side in ["r", "l"]:
        ankle_key = f"ankle_{side}"
        if ankle_key in aruco_processed:
            a_t = aruco_processed[ankle_key]["time"].astype(float)
            a_pos = aruco_processed[ankle_key]["data"]
            velocity_data[f"aruco_{ankle_key}"] = compute_displacement_velocity(
                a_pos, a_t, trim_ms=1000
            )
        velocity_data[f"reconstruction_{ankle_key}"] = compute_displacement_velocity(
            reconstruction[ankle_key], mcd_data["time"], trim_ms=1000
        )

    # ══════════════════════════════════════════════════════════════════
    #  MÉTRICAS CIENTÍFICAS (idénticas a process_all.py)
    # ══════════════════════════════════════════════════════════════════
    extra_metrics = {}
    axis_labels = ["X", "Y", "Z"]

    # RMSE
    rmse_per_joint = {}
    for key in homogeneous_data:
        diff = homogeneous_data[key]["mcd"] - homogeneous_data[key]["aruco"]
        rmse_per_joint[key] = np.sqrt(np.mean(diff**2))
    extra_metrics["rmse_per_joint"] = rmse_per_joint
    extra_metrics["rmse_mean"] = np.mean(list(rmse_per_joint.values()))

    # Pearson
    pearson_per_joint = {}
    for key in homogeneous_data:
        joint_corr = {}
        for ax_i, ax_name in enumerate(axis_labels):
            mcd_ax = homogeneous_data[key]["mcd"][:, ax_i]
            aruco_ax = homogeneous_data[key]["aruco"][:, ax_i]
            if np.std(mcd_ax) > 1e-10 and np.std(aruco_ax) > 1e-10:
                r_val, p_val = pearsonr(mcd_ax, aruco_ax)
            else:
                r_val, p_val = 0.0, 1.0
            joint_corr[ax_name] = {"r": r_val, "p_value": p_val}
        pearson_per_joint[key] = joint_corr
    extra_metrics["pearson_per_joint"] = pearson_per_joint

    # Bland-Altman
    bland_altman = {}
    for ax_i, ax_name in enumerate(axis_labels):
        all_mcd_ax = np.concatenate(
            [homogeneous_data[key]["mcd"][:, ax_i] for key in homogeneous_data]
        )
        all_aruco_ax = np.concatenate(
            [homogeneous_data[key]["aruco"][:, ax_i] for key in homogeneous_data]
        )
        diff_vals = all_mcd_ax - all_aruco_ax
        bland_altman[ax_name] = {
            "mean": (all_mcd_ax + all_aruco_ax) / 2.0,
            "diff": diff_vals,
            "bias": np.mean(diff_vals),
            "std_diff": np.std(diff_vals),
            "upper_loa": np.mean(diff_vals) + 1.96 * np.std(diff_vals),
            "lower_loa": np.mean(diff_vals) - 1.96 * np.std(diff_vals),
        }
    extra_metrics["bland_altman"] = bland_altman

    # Longitud de zancada
    stride_data = {}
    for side in ["r", "l"]:
        ankle_key = f"ankle_{side}"
        rec_strides, rec_stride_t = detect_stride_lengths(
            reconstruction[ankle_key], mcd_data["time"]
        )
        stride_data[f"reconstruction_{ankle_key}"] = {
            "lengths": rec_strides,
            "times": rec_stride_t,
            "mean": np.mean(rec_strides) if len(rec_strides) > 0 else 0.0,
            "std": np.std(rec_strides) if len(rec_strides) > 0 else 0.0,
        }
        if ankle_key in aruco_processed:
            aruco_ankle_pos = aruco_processed[ankle_key]["data"]
            aruco_ankle_t = aruco_processed[ankle_key]["time"].astype(float)
            aruco_strides, aruco_stride_t = detect_stride_lengths(
                aruco_ankle_pos, aruco_ankle_t
            )
            stride_data[f"aruco_{ankle_key}"] = {
                "lengths": aruco_strides,
                "times": aruco_stride_t,
                "mean": np.mean(aruco_strides) if len(aruco_strides) > 0 else 0.0,
                "std": np.std(aruco_strides) if len(aruco_strides) > 0 else 0.0,
            }
    extra_metrics["stride_data"] = stride_data

    # ROM de rodilla
    knee_rom = {}
    for side in ["r", "l"]:
        rec_knee_angles = compute_knee_angles_series(
            reconstruction[f"hip_{side}"],
            reconstruction[f"knee_{side}"],
            reconstruction[f"ankle_{side}"],
        )
        knee_rom[f"reconstruction_{side}"] = {
            "angles": rec_knee_angles,
            "rom": np.max(rec_knee_angles) - np.min(rec_knee_angles),
            "max": np.max(rec_knee_angles),
            "min": np.min(rec_knee_angles),
            "mean": np.mean(rec_knee_angles),
        }
        knee_key = f"knee_{side}"
        ankle_key = f"ankle_{side}"
        hip_key = f"hip_{side}"
        if knee_key in homogeneous_data:
            aruco_knee_angles = compute_knee_angles_series(
                (
                    homogeneous_data[hip_key]["aruco"]
                    if hip_key in homogeneous_data
                    else homogeneous_data[f"hip_{direction.lower()}"]["aruco"]
                ),
                homogeneous_data[knee_key]["aruco"],
                homogeneous_data[ankle_key]["aruco"],
            )
            mcd_knee_angles = compute_knee_angles_series(
                (
                    homogeneous_data[hip_key]["mcd"]
                    if hip_key in homogeneous_data
                    else homogeneous_data[f"hip_{direction.lower()}"]["mcd"]
                ),
                homogeneous_data[knee_key]["mcd"],
                homogeneous_data[ankle_key]["mcd"],
            )
            knee_rom[f"aruco_{side}"] = {
                "angles": aruco_knee_angles,
                "rom": np.max(aruco_knee_angles) - np.min(aruco_knee_angles),
                "max": np.max(aruco_knee_angles),
                "min": np.min(aruco_knee_angles),
                "mean": np.mean(aruco_knee_angles),
            }
            knee_rom[f"homogeneous_mcd_{side}"] = {
                "angles": mcd_knee_angles,
                "rom": np.max(mcd_knee_angles) - np.min(mcd_knee_angles),
                "max": np.max(mcd_knee_angles),
                "min": np.min(mcd_knee_angles),
                "mean": np.mean(mcd_knee_angles),
            }
    extra_metrics["knee_rom"] = knee_rom

    return (
        gen_eucliden_error,
        sorted(list(common_time)),
        reconstruction,
        aruco_processed,
        homogeneous_data,
        velocity_data,
        extra_metrics,
    )


# ═══════════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Procesamiento de marcha con redes neuronales modulares"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Directorio con los checkpoints de PivotNet y OrientationNet",
    )
    parser.add_argument(
        "--cutoff-freq",
        type=int,
        default=7,
        help="Frecuencia de corte del filtro pasa-bajas (Hz)",
    )
    parser.add_argument(
        "--filter-order",
        type=int,
        default=6,
        help="Orden del filtro Butterworth",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Tamaño de ventana para inferencia",
    )
    args = parser.parse_args()

    print()
    print("═" * W)
    print("  🦿 RECONSTRUCCIÓN DE MARCHA CON REDES NEURONALES MODULARES")
    print("═" * W)
    print(f"  Device:         {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU:            {torch.cuda.get_device_name(0)}")
    print(f"  Checkpoints:    {args.checkpoint_dir}")
    print("═" * W)

    # ── 1. Cargar modelos ─────────────────────────────────────────────
    print("\n  📥 Cargando modelos...")
    pivot_model, orient_model, norm_stats = load_models(Path(args.checkpoint_dir))

    # ── 2. Seleccionar dataset ────────────────────────────────────────
    dataset_dir = getattr(args, "dataset", "/home/moybadajoz/Documents/DataSet")
    print(f"\n  📂 Cargando capturas de: {dataset_dir}")
    import glob

    subjects = sorted(glob.glob("*", root_dir=dataset_dir))
    subjects = sorted([s for s in subjects if s.isdigit()], key=int)
    files = []
    for s in subjects:
        for j in range(4):
            mcd = Path(dataset_dir) / s / f"mcd-{j}.csv"
            aruco = Path(dataset_dir) / s / f"arUcos-{j}.csv"
            if mcd.exists() and aruco.exists():
                files.append((str(mcd), str(aruco)))

    if len(files) == 0:
        print("No files found")
        sys.exit(1)

    target_files = [f for f in files if Path(f[0]).parent.name != "0"]

    print(f"\n  📂 Procesando {len(target_files)} capturas...")
    t_start = time.time()

    # ── 3. Procesar cada captura ──────────────────────────────────────
    # (secuencial, ya que los modelos necesitan estar en GPU/CPU)
    raw_results = []
    for i, f in enumerate(target_files, 1):
        try:
            result = process_files_nn(
                f,
                pivot_model,
                orient_model,
                norm_stats,
                cutoff_freq=args.cutoff_freq,
                filter_order=args.filter_order,
                window_size=args.window_size,
            )
            raw_results.append(result)
        except Exception as e:
            print(f"  [ERROR] {f[0]}: {e}")
            raw_results.append(None)

        bar_len = 30
        progress = i / len(target_files)
        filled = int(bar_len * progress)
        bar = "█" * filled + "░" * (bar_len - filled)
        valid_count = sum(1 for r in raw_results if r is not None)
        print(
            f"\r  │{bar}│ {i}/{len(target_files)} ({progress * 100:.1f}%) │ ✓ {valid_count}",
            end="",
            flush=True,
        )

    elapsed = time.time() - t_start
    print()

    valid_data = [
        (f, res) for f, res in zip(target_files, raw_results) if res is not None
    ]
    target_files_valid = [f for f, _ in valid_data]
    gen_errors = [res for _, res in valid_data]

    # ══════════════════════════════════════════════════════════════════
    #  IMPRESIÓN DE RESULTADOS (misma estructura que process_all.py)
    # ══════════════════════════════════════════════════════════════════

    mean_error_per_point = np.empty(0)
    for file in gen_errors:
        homogeneous = file[4]
        mean_error_per_point = np.append(
            mean_error_per_point,
            np.mean([homogeneous[key]["euclidean_mean"] for key in homogeneous]),
        )

    gen_mean_error = np.array([np.mean(err[0]) for err in gen_errors])
    best_idx = np.argmin(gen_mean_error)
    worst_idx = np.argmax(gen_mean_error)
    mean_idx = np.argmin(np.abs(gen_mean_error - np.mean(gen_mean_error)))

    print()
    print("═" * W)
    print("  RESUMEN DEL DATASET (Red Neuronal Modular)")
    print("═" * W)
    n_failed = len(raw_results) - len(gen_errors)
    print(
        f"  Capturas procesadas:       {len(gen_errors)} / {len(gen_errors) + n_failed}"
    )
    print(f"  Capturas válidas:          {len(gen_errors)}")
    print(f"  Tiempo de procesamiento:   {elapsed:.1f} s")
    print("═" * W)

    # Error Euclidiano General
    print()
    print("═" * W)
    print("  📊 ERROR EUCLIDIANO GENERAL (centroide de 5 articulaciones)")
    print("═" * W)
    print(f"  Error medio:               {np.mean(gen_mean_error):.6f} m")
    print(f"  Desviación estándar:       {np.std(gen_mean_error):.6f} m")
    print(f"  Varianza:                  {np.var(gen_mean_error):.6f} m²")
    print(f"  Mínimo:                    {np.min(gen_mean_error):.6f} m")
    print(f"  Máximo:                    {np.max(gen_mean_error):.6f} m")
    print(
        f"  Rango:                     {np.max(gen_mean_error) - np.min(gen_mean_error):.6f} m"
    )
    print("─" * W)
    print(f"  📐 Error medio por punto:")
    print(f"  Media:                     {np.mean(mean_error_per_point):.6f} m")
    print(f"  Desviación estándar:       {np.std(mean_error_per_point):.6f} m")
    print(f"  Mínimo:                    {np.min(mean_error_per_point):.6f} m")
    print(f"  Máximo:                    {np.max(mean_error_per_point):.6f} m")
    print("═" * W)

    # Capturas destacadas
    print()
    print("═" * W)
    print("  🏆 CAPTURAS DESTACADAS")
    print("═" * W)
    for label, idx, icon in [
        ("Mejor", best_idx, "🟢"),
        ("Peor", worst_idx, "🔴"),
        ("Promedio", mean_idx, "🔵"),
    ]:
        file_label = "/".join(target_files_valid[idx][0].split(split_char)[-2:])
        print(
            f"  {icon} {label + ':':<12} {file_label:<30} error = {gen_mean_error[idx]:.6f} m"
        )
    print("═" * W)

    # Error por sujeto
    errors_by_subject = {}
    for f, err in zip(target_files_valid, gen_errors):
        subject = f[0].split(split_char)[-2]
        if subject not in errors_by_subject:
            errors_by_subject[subject] = []
        errors_by_subject[subject].extend(err[0])

    subjects = sorted(
        errors_by_subject.keys(), key=lambda x: int(x) if x.isdigit() else x
    )

    print()
    print("═" * W)
    print("  📋 ERROR POR SUJETO")
    print("═" * W)
    print(
        f"  {'Sujeto':>8}  {'Media (m)':>12}  {'Std (m)':>12}  "
        f"{'Mín (m)':>12}  {'Máx (m)':>12}"
    )
    print(f"  {'─' * 8}  {'─' * 12}  {'─' * 12}  {'─' * 12}  {'─' * 12}")
    for s in subjects:
        vals = np.array(errors_by_subject[s])
        print(
            f"  {s:>8}  {np.mean(vals):>12.6f}  {np.std(vals):>12.6f}  "
            f"{np.min(vals):>12.6f}  {np.max(vals):>12.6f}"
        )
    print("═" * W)

    # Velocidad de desplazamiento
    rec_hip_velocities = []
    aruco_hip_velocities = []
    for err in gen_errors:
        vel_data = err[5]
        rec_hip_velocities.append(
            vel_data.get("reconstruction_hip", {}).get("mean_velocity", 0.0)
        )
        aruco_hip_velocities.append(
            vel_data.get("aruco_hip", {}).get("mean_velocity", 0.0)
        )
    rec_hip_velocities = np.array(rec_hip_velocities)
    aruco_hip_velocities = np.array(aruco_hip_velocities)

    print()
    print("═" * W)
    print("  🚶 VELOCIDAD DE DESPLAZAMIENTO (cadera)")
    print("═" * W)
    print(f"  {'':30}  {'NN Modular':>16}  {'ArUco (GT)':>16}")
    print(f"  {'─' * 30}  {'─' * 16}  {'─' * 16}")
    print(
        f"  {'Velocidad media (m/s)':30}  {np.mean(rec_hip_velocities):>16.4f}  "
        f"{np.mean(aruco_hip_velocities):>16.4f}"
    )
    print(
        f"  {'± Desv. estándar (m/s)':30}  {np.std(rec_hip_velocities):>16.4f}  "
        f"{np.std(aruco_hip_velocities):>16.4f}"
    )
    print("═" * W)

    # Métricas científicas
    all_rmse_means = [err[6]["rmse_mean"] for err in gen_errors]
    all_rmse_per_joint = {}
    for err in gen_errors:
        for joint, rmse_val in err[6]["rmse_per_joint"].items():
            if joint not in all_rmse_per_joint:
                all_rmse_per_joint[joint] = []
            all_rmse_per_joint[joint].append(rmse_val)

    axis_labels = ["X", "Y", "Z"]
    pearson_by_joint = {}
    for err in gen_errors:
        for joint, corr_data in err[6]["pearson_per_joint"].items():
            if joint not in pearson_by_joint:
                pearson_by_joint[joint] = {ax: [] for ax in axis_labels}
            for ax in axis_labels:
                pearson_by_joint[joint][ax].append(corr_data[ax]["r"])

    print()
    print("═" * W)
    print("  🔬 MÉTRICAS CIENTÍFICAS (NN Modular)")
    print("═" * W)

    print(f"\n  1) RMSE")
    print(f"  {'─' * 40}")
    print(
        f"     RMSE general: {np.mean(all_rmse_means):.4f} ± {np.std(all_rmse_means):.4f} m"
    )
    print(f"\n     {'Articulación':<12}  {'RMSE (m)':>12}")
    print(f"     {'─' * 12}  {'─' * 12}")
    for joint in sorted(all_rmse_per_joint.keys()):
        vals = np.array(all_rmse_per_joint[joint])
        print(f"     {joint:<12}  {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print(f"\n  2) Correlación de Pearson")
    print(f"  {'─' * 40}")
    print(f"     {'Articulación':<12}  {'r_X':>8}  {'r_Y':>8}  {'r_Z':>8}")
    print(f"     {'─' * 12}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
    for joint in sorted(pearson_by_joint.keys()):
        rx = np.mean(pearson_by_joint[joint]["X"])
        ry = np.mean(pearson_by_joint[joint]["Y"])
        rz = np.mean(pearson_by_joint[joint]["Z"])
        print(f"     {joint:<12}  {rx:>8.4f}  {ry:>8.4f}  {rz:>8.4f}")

    # ROM de rodilla
    rom_rec_all = []
    rom_aruco_all = []
    for err in gen_errors:
        kr = err[6]["knee_rom"]
        for side in ["r", "l"]:
            if f"reconstruction_{side}" in kr:
                rom_rec_all.append(kr[f"reconstruction_{side}"]["rom"])
            if f"aruco_{side}" in kr:
                rom_aruco_all.append(kr[f"aruco_{side}"]["rom"])
    rom_rec_all = np.array(rom_rec_all) if rom_rec_all else np.array([0.0])
    rom_aruco_all = np.array(rom_aruco_all) if rom_aruco_all else np.array([0.0])

    print(f"\n  3) ROM de rodilla")
    print(f"  {'─' * 40}")
    print(f"     NN Modular:  {np.mean(rom_rec_all):.2f} ± {np.std(rom_rec_all):.2f}°")
    print(
        f"     ArUco (GT):  {np.mean(rom_aruco_all):.2f} ± {np.std(rom_aruco_all):.2f}°"
    )
    print("═" * W)

    # ══════════════════════════════════════════════════════════════════
    #  GRÁFICAS
    # ══════════════════════════════════════════════════════════════════

    results_dir = Path("nn-results/modular")
    results_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("  🎨 Generando gráficas...")

    # Boxplot por sujeto
    data_to_plot = [errors_by_subject[s] for s in subjects]
    fig_box, ax_box = plt.subplots(figsize=figsize)
    ax_box.boxplot(data_to_plot, tick_labels=subjects)
    ax_box.set_title("Error Euclidiano por Sujeto (NN Modular)")
    ax_box.set_xlabel("Sujeto")
    ax_box.set_ylabel("Error Euclidiano (m)")
    ax_box.grid(True)
    ax_box.set_ylim(0, 0.8)
    fig_box.savefig(results_dir / "boxplot_error_nn.png", dpi=300, bbox_inches="tight")
    print(f"  ✓ {results_dir / 'boxplot_error_nn.png'}")

    # Errores temporales: mejor, peor, promedio
    for tag, idx, color in [
        ("best", best_idx, "green"),
        ("worst", worst_idx, "red"),
        ("mean", mean_idx, "blue"),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(gen_errors[idx][1], gen_errors[idx][0], color=color)
        ax.set_title(f"Error {tag.capitalize()} (NN Modular)")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Euclidean error (m)")
        ax.grid(True)
        ax.set_ylim(0, 1)
        fig.savefig(results_dir / f"error_{tag}_nn.png", dpi=300, bbox_inches="tight")
        print(f"  ✓ {results_dir / f'error_{tag}_nn.png'}")
        plt.close(fig)

    # Bland-Altman
    ba_data = gen_errors[mean_idx][6]["bland_altman"]
    fig_ba, axs_ba = plt.subplots(1, 3, figsize=(15, 5))
    fig_ba.suptitle("Bland-Altman: NN Modular vs Ground Truth")
    for ax_i, ax_name in enumerate(axis_labels):
        ba = ba_data[ax_name]
        axs_ba[ax_i].scatter(ba["mean"], ba["diff"], alpha=0.15, s=3, color="steelblue")
        axs_ba[ax_i].axhline(
            ba["bias"],
            color="red",
            linestyle="-",
            linewidth=1.5,
            label=f"Sesgo: {ba['bias']:.4f} m",
        )
        axs_ba[ax_i].axhline(
            ba["upper_loa"],
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"+1.96σ: {ba['upper_loa']:.4f} m",
        )
        axs_ba[ax_i].axhline(
            ba["lower_loa"],
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"-1.96σ: {ba['lower_loa']:.4f} m",
        )
        axs_ba[ax_i].set_xlabel(f"Media {ax_name} (m)")
        axs_ba[ax_i].set_ylabel(f"Diferencia {ax_name} (m)")
        axs_ba[ax_i].set_title(f"Eje {ax_name}")
        axs_ba[ax_i].legend(fontsize=7)
        axs_ba[ax_i].grid(True, alpha=0.3)
    fig_ba.tight_layout()
    fig_ba.savefig(results_dir / "bland_altman_nn.png", dpi=300, bbox_inches="tight")
    print(f"  ✓ {results_dir / 'bland_altman_nn.png'}")
    plt.close(fig_ba)

    plt.close("all")

    print()
    print("═" * W)
    print("  ✅ PROCESAMIENTO CON NN MODULAR COMPLETADO")
    print("═" * W)
