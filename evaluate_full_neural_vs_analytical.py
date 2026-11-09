"""
Evaluación y Comparativa: Pipeline Neuronal Completo vs Método Analítico
(INCLUYENDO DESPLAZAMIENTO GLOBAL EN EL ESPACIO 3D)
========================================================================

Este script realiza la reconstrucción cinemática del tren inferior combinando:
  1. Red Clasificadora de Pie Pivote (GaitStaticPivotNet) -> Predice el pie en apoyo
  2. Red de Reconstrucción Estática (GaitStaticNet)       -> Predice poses 3D relativas
  3. Acumulación de Pasos / Desplazamiento Global en el Espacio
y compara cuantitativa y visualmente los resultados contra:
  - El Método Analítico Tradicional (Madgwick + Heurística de Pivote + Cadena Cinemática)
  - El Ground Truth Óptico (ArUco con Desplazamiento Global)

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
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import interp1d

import torch
import torch.nn as nn
import torch.nn.functional as F

# Importar modelos y utilidades existentes
from nn_pivot_classifier import GaitStaticPivotNet
from nn_static_pivot import (
    GaitStaticNet,
    GaitStaticDataset,
    discover_captures,
    prepare_static_capture,
    JOINT_NAMES,
    N_JOINTS,
    N_COORDS,
)
from process_all import (
    process_files,
    read_mcd_file,
    read_aruco_file,
    adjust_aruco_data,
    low_pass_filter,
)
from nn_transformer_multitask import (
    GaitMultiTaskNet,
    predict_full_sequence_multitask as predict_trans,
)
from nn_tcn_lite import (
    GaitMultiTaskNetTCN,
    predict_full_sequence_multitask as predict_tcn,
)
from nn_ttl_lite import (
    GaitMultiTaskNetTTLLite,
    predict_full_sequence_multitask as predict_ttl,
)
from multiprocessing import Pool
from functools import partial

BONE_NORM = GaitStaticDataset.BONE_NORM

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN Y RUTAS
# ═══════════════════════════════════════════════════════════════════════

DATASET_PATH = Path("/home/badajoz/Documents/DataSet")
RESULTS_DIR = Path("nn-results/full_neural_vs_analytical")
PIVOT_CKPT_PATH = Path(
    "nn-results/pivot_classifier/checkpoints/best_pivot_classifier.pt"
)
POSE_CKPT_PATH = Path("nn-results/static_pivot/checkpoints/best_static_model.pt")
TCN_CKPT_PATH = Path("nn-results/tcn_lite/checkpoints/best_tcn_lite.pt")
TTL_CKPT_PATH = Path("nn-results/ttl_lite/checkpoints/best_ttl_lite.pt")
TRANS_CKPT_PATH = Path(
    "nn-results/transformer_multitask/checkpoints/best_multitask_model_trans.pt"
)
TCN_STATS_PATH = Path("nn-results/tcn_lite/checkpoints/norm_stats.json")
TTL_STATS_PATH = Path("nn-results/ttl_lite/checkpoints/norm_stats.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════
#  ACUMULACIÓN DE DESPLAZAMIENTO GLOBAL
# ═══════════════════════════════════════════════════════════════════════


def accumulate_global_trajectory(
    rel_poses: np.ndarray, pivot_flags: np.ndarray, init_pivot_pos: np.ndarray
) -> np.ndarray:
    """
    Acumula las poses relativas a lo largo del espacio 3D según los cambios de pie pivote.
    rel_poses: [N, 18] (6 articulaciones x 3 coords relativas al pivote)
    pivot_flags: [N] (1.0 = Derecho, 0.0 = Izquierdo)
    init_pivot_pos: [3] Posición inicial en el espacio global del pie pivote inicial.
    """
    N = len(rel_poses)
    global_poses = np.zeros((N, N_JOINTS * N_COORDS), dtype=np.float64)

    # Orden de articulaciones: 0:hip_r, 1:hip_l, 2:knee_r, 3:knee_l, 4:ankle_r, 5:ankle_l
    curr_pivot_world = init_pivot_pos.copy()

    for i in range(N):
        curr_pivot_is_right = pivot_flags[i] >= 0.5
        active_pivot_idx = 4 if curr_pivot_is_right else 5  # 4=ankle_r, 5=ankle_l

        if i > 0:
            prev_pivot_is_right = pivot_flags[i - 1] >= 0.5
            if curr_pivot_is_right != prev_pivot_is_right:
                # El pivote cambió en el frame i.
                # La nueva posición global del pivote es la posición donde cayó en el frame i-1
                new_pivot_joint_idx = 4 if curr_pivot_is_right else 5
                curr_pivot_world = global_poses[
                    i - 1, new_pivot_joint_idx * 3 : (new_pivot_joint_idx + 1) * 3
                ].copy()

        active_pivot_rel = rel_poses[
            i, active_pivot_idx * 3 : (active_pivot_idx + 1) * 3
        ]

        for j in range(N_JOINTS):
            j_rel = rel_poses[i, j * 3 : (j + 1) * 3]
            j_global = j_rel - active_pivot_rel + curr_pivot_world
            global_poses[i, j * 3 : (j + 1) * 3] = j_global

    return global_poses


def get_ground_truth_global_trajectory(
    mcd_path: str, aruco_path: str, cap: dict
) -> np.ndarray | None:
    """
    Extrae la trayectoria 3D global real obtenida por el sistema óptico ArUco.
    """
    try:
        mcd_data = read_mcd_file(mcd_path, cutoff_freq=9, filter_order=6)
        aruco_data = read_aruco_file(aruco_path)
        direction = mcd_data["measures"]["direction"]

        aruco_data = adjust_aruco_data(aruco_data, direction)
        aruco_processed = {
            "hip": aruco_data["hip_position"],
            "knee_r": aruco_data["R_knee_position"],
            "knee_l": aruco_data["L_knee_position"],
            "ankle_r": aruco_data["R_ankle_position"],
            "ankle_l": aruco_data["L_ankle_position"],
        }

        mcd_time = mcd_data["time"]
        aruco_raw_keys = ["hip", "knee_r", "knee_l", "ankle_r", "ankle_l"]
        t_min_all = -np.inf
        t_max_all = np.inf
        for key in aruco_raw_keys:
            aruco_t = aruco_processed[key]["time"].astype(float)
            t_min_all = max(t_min_all, aruco_t.min())
            t_max_all = min(t_max_all, aruco_t.max())

        valid_time_mask = (mcd_time >= t_min_all) & (mcd_time <= t_max_all)
        valid_mcd_time = mcd_time[valid_time_mask]
        n_valid = len(valid_mcd_time)

        hip_t = aruco_processed["hip"]["time"].astype(float)
        hip_pos = aruco_processed["hip"]["data"]
        hip_interp = [interp1d(hip_t, hip_pos[:, i], kind="linear") for i in range(3)]
        hip_at_mcd = np.column_stack([hip_interp[i](valid_mcd_time) for i in range(3)])

        # Fijar centro pélvico Y = 0.0 (marcha en plano sagital, consistente con entrenamiento)
        hip_at_mcd[:, 1] = 0.0

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

        hip_width = float(mcd_data["measures"].get("hip_width", 0.25))
        half_hw = hip_width / 2.0
        side_sign = +1.0 if direction == "L" else -1.0
        half_hw_r = +half_hw * side_sign
        half_hw_l = -half_hw * side_sign

        gt_global = np.zeros((n_valid, N_JOINTS * N_COORDS), dtype=np.float64)

        # Construir GT global con modelo sagital (Y=0 para centro pélvico,
        # joints en planos laterales fijos ±half_hw) — consistente con
        # el entrenamiento de las redes neuronales en nn_tcn_lite.py
        # 0: hip_r, 1: hip_l, 2: knee_r, 3: knee_l, 4: ankle_r, 5: ankle_l
        gt_global[:, 0] = hip_at_mcd[:, 0]  # hip_r X = hip X
        gt_global[:, 1] = half_hw_r  # hip_r Y = offset lateral fijo
        gt_global[:, 2] = hip_at_mcd[:, 2]  # hip_r Z = hip Z
        gt_global[:, 3] = hip_at_mcd[:, 0]  # hip_l X = hip X
        gt_global[:, 4] = half_hw_l  # hip_l Y = offset lateral fijo
        gt_global[:, 5] = hip_at_mcd[:, 2]  # hip_l Z = hip Z
        # Rodillas y tobillos: X y Z del ArUco real, Y en plano sagital fijo
        gt_global[:, 6] = joints_at_mcd["knee_r"][:, 0]  # knee_r X
        gt_global[:, 7] = half_hw_r  # knee_r Y fijo
        gt_global[:, 8] = joints_at_mcd["knee_r"][:, 2]  # knee_r Z
        gt_global[:, 9] = joints_at_mcd["knee_l"][:, 0]  # knee_l X
        gt_global[:, 10] = half_hw_l  # knee_l Y fijo
        gt_global[:, 11] = joints_at_mcd["knee_l"][:, 2]  # knee_l Z
        gt_global[:, 12] = joints_at_mcd["ankle_r"][:, 0]  # ankle_r X
        gt_global[:, 13] = half_hw_r  # ankle_r Y fijo
        gt_global[:, 14] = joints_at_mcd["ankle_r"][:, 2]  # ankle_r Z
        gt_global[:, 15] = joints_at_mcd["ankle_l"][:, 0]  # ankle_l X
        gt_global[:, 16] = half_hw_l  # ankle_l Y fijo
        gt_global[:, 17] = joints_at_mcd["ankle_l"][:, 2]  # ankle_l Z

        return gt_global
    except Exception as e:
        print(
            f"    [WARN Ground Truth] Error calculando GT global en {Path(mcd_path).name}: {e}"
        )
        return None


# ═══════════════════════════════════════════════════════════════════════
#  MÉTRICAS Y CÁLCULOS DE ERROR
# ═══════════════════════════════════════════════════════════════════════


def compute_mpjpe(
    predictions: np.ndarray, targets: np.ndarray, percentile: float = 95.0
) -> dict[str, float]:
    """Mean Per Joint Position Error en metros (evaluado sobre el percentil dado, e.g. 95%)."""
    pred = predictions.reshape(-1, N_JOINTS, N_COORDS)
    tgt = targets.reshape(-1, N_JOINTS, N_COORDS)
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


def compute_mae(
    predictions: np.ndarray, targets: np.ndarray, percentile: float = 95.0
) -> dict[str, float]:
    """
    Mean Absolute Error por articulación evaluado por componente cartesiana (X, Y, Z)
    en metros sobre el percentil dado (e.g. 95%).
    """
    pred = predictions.reshape(-1, N_JOINTS, N_COORDS)
    tgt = targets.reshape(-1, N_JOINTS, N_COORDS)
    abs_diff = np.abs(pred - tgt)
    result = {}
    for j, name in enumerate(JOINT_NAMES):
        err = abs_diff[:, j, :].flatten()
        if percentile is not None and percentile < 100.0:
            threshold = np.percentile(err, percentile)
            err = err[err <= threshold]
        result[name] = float(np.mean(err))
    result["mean"] = float(np.mean([result[name] for name in JOINT_NAMES]))
    return result


def compute_rmse(
    predictions: np.ndarray, targets: np.ndarray, percentile: float = 95.0
) -> dict[str, float]:
    """
    Root Mean Squared Error por articulación evaluado por componente cartesiana (X, Y, Z)
    en metros sobre el percentil dado (e.g. 95%).
    """
    pred = predictions.reshape(-1, N_JOINTS, N_COORDS)
    tgt = targets.reshape(-1, N_JOINTS, N_COORDS)
    sq_diff = (pred - tgt) ** 2
    result = {}
    for j, name in enumerate(JOINT_NAMES):
        err = sq_diff[:, j, :].flatten()
        if percentile is not None and percentile < 100.0:
            threshold = np.percentile(err, percentile)
            err = err[err <= threshold]
        result[name] = float(np.sqrt(np.mean(err)))
    result["mean"] = float(np.mean([result[name] for name in JOINT_NAMES]))
    return result


# ═══════════════════════════════════════════════════════════════════════
#  CARGA DE MODELOS NEURONALES
# ═══════════════════════════════════════════════════════════════════════


def load_pivot_model(ckpt_path: Path):
    """Carga el modelo PivotNet y sus estadísticas de normalización."""
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No se encontró el checkpoint de PivotNet en: {ckpt_path}"
        )
    checkpoint = torch.load(ckpt_path, weights_only=False)

    state_dict = checkpoint["model_state_dict"]
    w0_shape = state_dict["net.0.weight"].shape
    w4_shape = state_dict["net.4.weight"].shape
    w8_shape = state_dict["net.8.weight"].shape

    hidden_layers = [w0_shape[0], w4_shape[0], w8_shape[0]]
    print(f"  ✓ Cargando PivotNet con capas ocultas: {hidden_layers}")

    model = GaitStaticPivotNet(hidden_layers=hidden_layers).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    norm_stats = checkpoint["norm_stats"]
    return model, norm_stats


def load_pose_model(ckpt_path: Path):
    """Carga el modelo GaitStaticNet (Pose Reconstruction)."""
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No se encontró el checkpoint de GaitStaticNet en: {ckpt_path}"
        )
    checkpoint = torch.load(ckpt_path, weights_only=False)

    state_dict = checkpoint["model_state_dict"]
    w0_shape = state_dict["net.0.weight"].shape
    w4_shape = state_dict["net.4.weight"].shape
    w8_shape = state_dict["net.8.weight"].shape

    hidden_layers = [w0_shape[0], w4_shape[0], w8_shape[0]]
    print(f"  ✓ Cargando PoseNet con capas ocultas: {hidden_layers}")

    model = GaitStaticNet(hidden_layers=hidden_layers).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    norm_stats = checkpoint["norm_stats"]
    return model, norm_stats


# ═══════════════════════════════════════════════════════════════════════
#  GRAFICACIÓN Y VISUALIZACIÓN
# ═══════════════════════════════════════════════════════════════════════


def plot_trajectory_2d_and_bars(
    mpjpe_nn,
    mpjpe_tcn,
    mpjpe_ttl,
    mpjpe_trans,
    mpjpe_ana,
    gt_sample,
    nn_sample,
    tcn_sample,
    ttl_sample,
    trans_sample,
    ana_sample,
    save_dir,
    percentile: float = 95.0,
):
    """Genera gráficas de trayectoria 2D en el suelo (X vs Z) y de error MPJPE por articulación."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # --- Gráfica 1: Trayectoria 2D de la Cadera (Desplazamiento Global X vs Z) ---
    gt_hip_x = (gt_sample[:, 0] + gt_sample[:, 3]) / 2.0
    gt_hip_z = (gt_sample[:, 2] + gt_sample[:, 5]) / 2.0

    nn_hip_x = (nn_sample[:, 0] + nn_sample[:, 3]) / 2.0
    nn_hip_z = (nn_sample[:, 2] + nn_sample[:, 5]) / 2.0

    axes[0].plot(
        gt_hip_x, gt_hip_z, label="ArUco (Ground Truth)", color="#2CA02C", linewidth=2.5
    )
    axes[0].plot(
        nn_hip_x,
        nn_hip_z,
        label="Red Estática",
        color="#1F77B4",
        linestyle="--",
        linewidth=2,
    )

    if tcn_sample is not None:
        tcn_hip_x = (tcn_sample[:, 0] + tcn_sample[:, 3]) / 2.0
        tcn_hip_z = (tcn_sample[:, 2] + tcn_sample[:, 5]) / 2.0
        axes[0].plot(
            tcn_hip_x,
            tcn_hip_z,
            label="TCN Lite",
            color="#9467BD",
            linestyle="-.",
            linewidth=2,
        )

    if ttl_sample is not None:
        ttl_hip_x = (ttl_sample[:, 0] + ttl_sample[:, 3]) / 2.0
        ttl_hip_z = (ttl_sample[:, 2] + ttl_sample[:, 5]) / 2.0
        axes[0].plot(
            ttl_hip_x,
            ttl_hip_z,
            label="TTL Lite",
            color="#8C564B",
            linestyle="-.",
            linewidth=2,
        )

    if trans_sample is not None:
        trans_hip_x = (trans_sample[:, 0] + trans_sample[:, 3]) / 2.0
        trans_hip_z = (trans_sample[:, 2] + trans_sample[:, 5]) / 2.0
        axes[0].plot(
            trans_hip_x,
            trans_hip_z,
            label="Transformer",
            color="#E377C2",
            linestyle="-.",
            linewidth=2,
        )

    if ana_sample is not None:
        ana_hip_x = (ana_sample[:, 0] + ana_sample[:, 3]) / 2.0
        ana_hip_z = (ana_sample[:, 2] + ana_sample[:, 5]) / 2.0
        axes[0].plot(
            ana_hip_x,
            ana_hip_z,
            label="Método Analítico",
            color="#FF7F0E",
            linestyle=":",
            linewidth=2,
        )

    axes[0].set_title(
        "Trayectoria 2D del Desplazamiento Global de la Cadera", fontweight="bold"
    )
    axes[0].set_xlabel("Posición X (m)")
    axes[0].set_ylabel("Posición Z (m)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    # --- Gráfica 2: Barras de Error MPJPE Global ---
    labels = JOINT_NAMES + ["Promedio General"]
    nn_vals = [mpjpe_nn[name] * 100 for name in JOINT_NAMES] + [mpjpe_nn["mean"] * 100]
    tcn_vals = (
        [mpjpe_tcn[name] * 100 for name in JOINT_NAMES] + [mpjpe_tcn["mean"] * 100]
        if mpjpe_tcn
        else [0] * len(labels)
    )
    ttl_vals = (
        [mpjpe_ttl[name] * 100 for name in JOINT_NAMES] + [mpjpe_ttl["mean"] * 100]
        if mpjpe_ttl
        else [0] * len(labels)
    )
    trans_vals = (
        [mpjpe_trans[name] * 100 for name in JOINT_NAMES] + [mpjpe_trans["mean"] * 100]
        if mpjpe_trans
        else [0] * len(labels)
    )
    ana_vals = (
        [mpjpe_ana[name] * 100 for name in JOINT_NAMES] + [mpjpe_ana["mean"] * 100]
        if mpjpe_ana
        else [0] * len(labels)
    )

    x = np.arange(len(labels))
    width = 0.15

    rects1 = axes[1].bar(
        x - 1.5 * width,
        nn_vals,
        width,
        label="Red Estática",
        color="#1F77B4",
        alpha=0.9,
    )
    rects3 = axes[1].bar(
        x - 0.5 * width, tcn_vals, width, label="TCN Lite", color="#9467BD", alpha=0.9
    )
    rects_ttl = axes[1].bar(
        x + 0.5 * width, ttl_vals, width, label="TTL Lite", color="#8C564B", alpha=0.9
    )
    rects4 = axes[1].bar(
        x + 1.5 * width,
        trans_vals,
        width,
        label="Transformer",
        color="#E377C2",
        alpha=0.9,
    )
    if mpjpe_ana:
        rects2 = axes[1].bar(
            x + 2.5 * width,
            ana_vals,
            width,
            label="Método Analítico",
            color="#FF7F0E",
            alpha=0.9,
        )

    pct_label = f" (P{percentile:.0f}%)" if percentile < 100.0 else ""
    axes[1].set_ylabel(f"Error MPJPE Global{pct_label} (cm)", fontweight="bold")
    axes[1].set_title(
        f"Comparativa de Error Global por Articulación{pct_label}", fontweight="bold"
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right", fontweight="bold")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, axis="y", alpha=0.3)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                axes[1].annotate(
                    f"{height:.1f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                    rotation=90,
                )

    autolabel(rects1)
    autolabel(rects3)
    autolabel(rects_ttl)
    autolabel(rects4)
    if mpjpe_ana:
        autolabel(rects2)

    plt.tight_layout()
    save_path = save_dir / "global_trajectory_and_error.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Gráficas de trayectoria y error global guardadas en: {save_path}")


def save_global_comparison_animation(
    gt_pose: np.ndarray,
    pred_nn_pose: np.ndarray,
    tcn_pose: np.ndarray,
    ttl_pose: np.ndarray,
    trans_pose: np.ndarray,
    ana_pose: np.ndarray,
    save_path: str,
    max_frames: int = 400,
):
    """Guarda una animación 3D del esqueleto DESPLAZÁNDOSE a lo largo de la caminata en el espacio real."""
    n_frames = min(max_frames, len(gt_pose))
    gt = gt_pose[:n_frames].reshape(-1, N_JOINTS, N_COORDS)
    pred = pred_nn_pose[:n_frames].reshape(-1, N_JOINTS, N_COORDS)
    tcn = (
        tcn_pose[:n_frames].reshape(-1, N_JOINTS, N_COORDS)
        if tcn_pose is not None
        else None
    )
    ttl = (
        ttl_pose[:n_frames].reshape(-1, N_JOINTS, N_COORDS)
        if ttl_pose is not None
        else None
    )
    trans = (
        trans_pose[:n_frames].reshape(-1, N_JOINTS, N_COORDS)
        if trans_pose is not None
        else None
    )
    ana = (
        ana_pose[:n_frames].reshape(-1, N_JOINTS, N_COORDS)
        if ana_pose is not None
        else None
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    connections = [("pelvis", 0), ("pelvis", 1), (0, 2), (2, 4), (1, 3), (3, 5)]

    def get_skeleton_lines(pose_frame):
        pelvis = (pose_frame[0] + pose_frame[1]) / 2.0
        pts = np.vstack([pose_frame, pelvis])
        lines = []
        for p1, p2 in connections:
            i1 = 6 if p1 == "pelvis" else p1
            i2 = 6 if p2 == "pelvis" else p2
            lines.append(np.vstack([pts[i1], pts[i2]]))
        return lines

    x_all = np.concatenate([gt[:, :, 0].flatten(), pred[:, :, 0].flatten()])
    z_all = np.concatenate([gt[:, :, 2].flatten(), pred[:, :, 2].flatten()])
    x_min, x_max = x_all.min() - 0.3, x_all.max() + 0.3
    z_min, z_max = z_all.min() - 0.3, z_all.max() + 0.3

    def update(frame):
        ax.clear()
        ax.set_title(
            f"Reconstrucción 3D con Desplazamiento Global - Frame {frame}",
            fontweight="bold",
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        ax.view_init(elev=20, azim=-45)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([z_min, z_max])

        lines_gt = get_skeleton_lines(gt[frame])
        for line in lines_gt:
            ax.plot(
                line[:, 0],
                line[:, 1] + 0.7,
                line[:, 2],
                color="#2CA02C",
                linewidth=3,
                label="ArUco" if line is lines_gt[0] else "",
            )

        lines_pred = get_skeleton_lines(pred[frame])
        for line in lines_pred:
            ax.plot(
                line[:, 0],
                line[:, 1] + 0.35,
                line[:, 2],
                color="#1F77B4",
                linewidth=2,
                linestyle="--",
                label="Estática" if line is lines_pred[0] else "",
            )

        if tcn is not None and frame < len(tcn):
            lines_tcn = get_skeleton_lines(tcn[frame])
            for line in lines_tcn:
                ax.plot(
                    line[:, 0],
                    line[:, 1] + 0.0,
                    line[:, 2],
                    color="#9467BD",
                    linewidth=2,
                    linestyle="-.",
                    label="TCN Lite" if line is lines_tcn[0] else "",
                )

        if ttl is not None and frame < len(ttl):
            lines_ttl = get_skeleton_lines(ttl[frame])
            for line in lines_ttl:
                ax.plot(
                    line[:, 0],
                    line[:, 1] - 0.175,
                    line[:, 2],
                    color="#8C564B",
                    linewidth=2,
                    linestyle="-.",
                    label="TTL Lite" if line is lines_ttl[0] else "",
                )

        if trans is not None and frame < len(trans):
            lines_trans = get_skeleton_lines(trans[frame])
            for line in lines_trans:
                ax.plot(
                    line[:, 0],
                    line[:, 1] - 0.35,
                    line[:, 2],
                    color="#E377C2",
                    linewidth=2,
                    linestyle="-.",
                    label="Transformer" if line is lines_trans[0] else "",
                )

        if ana is not None and frame < len(ana):
            lines_ana = get_skeleton_lines(ana[frame])
            for line in lines_ana:
                x_offset = gt[0, 0, 0] - ana[0, 0, 0]
                z_offset = gt[0, 0, 2] - ana[0, 0, 2]
                ax.plot(
                    line[:, 0] + x_offset,
                    line[:, 1] - 0.7,
                    line[:, 2] + z_offset,
                    color="#FF7F0E",
                    linewidth=2,
                    linestyle=":",
                    label="Analítico" if line is lines_ana[0] else "",
                )

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper left")

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50)
    print(
        f"\n  🎥 Generando animación GIF 3D con Desplazamiento Global ({n_frames} frames)..."
    )
    ani.save(save_path, writer="pillow", fps=20)
    plt.close(fig)
    print(f"  ✓ Animación con desplazamiento guardada en: {save_path}")


# ═══════════════════════════════════════════════════════════════════════
#  PROGRAMA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Evaluación de Reconstrucción con Desplazamiento Global"
    )
    parser.add_argument(
        "--dataset", type=str, default=str(DATASET_PATH), help="Ruta al dataset"
    )
    parser.add_argument(
        "--data-split",
        type=str,
        default="data_split.json",
        help="Ruta al archivo JSON con la división de sujetos (default: data_split.json)",
    )
    parser.add_argument(
        "--n-test-subjects", type=int, default=5, help="Número de sujetos para test"
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Percentil de corte para cálculo limpio de métricas (default: 95.0)",
    )
    parser.add_argument(
        "--window-size-tcn",
        type=int,
        default=50,
        help="Tamaño de ventana para TCN Lite (default: 50)",
    )
    parser.add_argument(
        "--window-size-ttl",
        type=int,
        default=150,
        help="Tamaño de ventana para TTL Lite (default: 150)",
    )
    parser.add_argument(
        "--window-size-trans",
        type=int,
        default=150,
        help="Tamaño de ventana para Transformer (default: 150)",
    )
    parser.add_argument(
        "--anim-frames",
        type=int,
        default=400,
        help="Número máximo de frames para la animación GIF 3D (default: 400)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("═" * 80)
    print("  🏃 RECONSTRUCCIÓN Y COMPARATIVA CON DESPLAZAMIENTO GLOBAL 3D")
    print("  (PivotNet + PoseNet + Acumulación de Paso vs Analítico vs ArUco Real)")
    print("═" * 80)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Cargar Modelos Entrenados
    print("\n  📦 Cargando Redes Neuronales Entrenadas...")
    pivot_model, norm_stats_pivot = load_pivot_model(PIVOT_CKPT_PATH)
    pose_model, norm_stats_pose = load_pose_model(POSE_CKPT_PATH)

    # TCN Lite — cargar checkpoint y estadísticas de normalización
    tcn_ckpt_path = TCN_CKPT_PATH
    if not tcn_ckpt_path.exists():
        if (TCN_CKPT_PATH.parent / "best_tcn_lite_soup.pt").exists():
            tcn_ckpt_path = TCN_CKPT_PATH.parent / "best_tcn_lite_soup.pt"
        elif Path("nn-results/tcn_lite/checkpoints/best_tcn_lite.pt").exists():
            tcn_ckpt_path = Path("nn-results/tcn_lite/checkpoints/best_tcn_lite.pt")

    ckpt_tcn = torch.load(tcn_ckpt_path, map_location=DEVICE, weights_only=False)
    state_tcn = (
        ckpt_tcn["model_state_dict"] if "model_state_dict" in ckpt_tcn else ckpt_tcn
    )

    if "norm_stats" in ckpt_tcn and ckpt_tcn["norm_stats"] is not None:
        norm_stats_tcn = ckpt_tcn["norm_stats"]
    elif (tcn_ckpt_path.parent / "norm_stats.json").exists():
        with open(tcn_ckpt_path.parent / "norm_stats.json", "r") as f:
            norm_stats_tcn = json.load(f)
    elif TCN_STATS_PATH.exists():
        with open(TCN_STATS_PATH, "r") as f:
            norm_stats_tcn = json.load(f)
    else:
        raise FileNotFoundError(
            f"No se encontraron estadísticas de normalización para TCN"
        )

    for k in ["mean", "std"]:
        if isinstance(norm_stats_tcn.get(k), list):
            norm_stats_tcn[k] = np.array(norm_stats_tcn[k], dtype=np.float32)
        elif isinstance(norm_stats_tcn.get(k), torch.Tensor):
            norm_stats_tcn[k] = norm_stats_tcn[k].cpu().numpy().astype(np.float32)

    in_dim_tcn = (
        state_tcn["input_proj.0.weight"].shape[1]
        if "input_proj.0.weight" in state_tcn
        else 28
    )
    num_tcn = sum(
        1
        for k in state_tcn
        if k.startswith("tcn_blocks.") and k.endswith(".conv.weight")
    )
    tcn_ch = [state_tcn[f"tcn_blocks.{i}.conv.weight"].shape[0] for i in range(num_tcn)]
    tcn_k = [state_tcn[f"tcn_blocks.{i}.conv.weight"].shape[2] for i in range(num_tcn)]
    tcn_d = [2**i for i in range(num_tcn)]
    head_h_tcn = (
        state_tcn["pose_head.1.weight"].shape[0]
        if "pose_head.1.weight" in state_tcn
        else 48
    )
    print(
        f"  ✓ Cargando TCN Lite desde {tcn_ckpt_path.name} con canales={tcn_ch}, kernels={tcn_k}"
    )
    tcn_model = GaitMultiTaskNetTCN(
        input_dim=in_dim_tcn,
        tcn_channels=tcn_ch,
        tcn_kernel_sizes=tcn_k,
        tcn_dilations=tcn_d,
        head_hidden_dim=head_h_tcn,
    ).to(DEVICE)
    tcn_model.load_state_dict(state_tcn)
    tcn_model.eval()

    # TTL Lite — inferir arquitectura desde el checkpoint
    ckpt_ttl = torch.load(TTL_CKPT_PATH, map_location=DEVICE, weights_only=False)
    state_ttl = (
        ckpt_ttl["model_state_dict"] if "model_state_dict" in ckpt_ttl else ckpt_ttl
    )
    in_dim_ttl = (
        state_ttl["input_proj.0.weight"].shape[1]
        if "input_proj.0.weight" in state_ttl
        else 28
    )
    num_ttl = sum(
        1
        for k in state_ttl
        if k.startswith("tcn_blocks.") and k.endswith(".conv.weight")
    )
    ttl_ch = [state_ttl[f"tcn_blocks.{i}.conv.weight"].shape[0] for i in range(num_ttl)]
    ttl_k = [state_ttl[f"tcn_blocks.{i}.conv.weight"].shape[2] for i in range(num_ttl)]
    ttl_d = [2**i for i in range(num_ttl)]
    lstm_h_ttl = state_ttl["lstm.weight_hh_l0"].shape[1]
    lstm_l_ttl = sum(
        1 for k in state_ttl if k.startswith("lstm.weight_ih_l") and "reverse" not in k
    )
    attn_l_ttl = sum(
        1
        for k in state_ttl
        if k.startswith("attention.layers.") and k.endswith(".norm1.weight")
    )
    head_h_ttl = state_ttl["pose_head.1.weight"].shape[0]
    print(
        f"  ✓ Cargando TTL Lite con TCN={ttl_ch}, LSTM_H={lstm_h_ttl}, "
        f"LSTM_L={lstm_l_ttl}, Attn_L={attn_l_ttl}"
    )
    ttl_model = GaitMultiTaskNetTTLLite(
        input_dim=in_dim_ttl,
        tcn_channels=ttl_ch,
        tcn_kernel_sizes=ttl_k,
        tcn_dilations=ttl_d,
        lstm_hidden=lstm_h_ttl,
        lstm_layers=lstm_l_ttl,
        n_attn_layers=attn_l_ttl,
        head_hidden_dim=head_h_ttl,
    ).to(DEVICE)
    ttl_model.load_state_dict(state_ttl)
    ttl_model.eval()

    if "norm_stats" in ckpt_ttl:
        norm_stats_ttl = ckpt_ttl["norm_stats"]
    elif TTL_STATS_PATH.exists():
        with open(TTL_STATS_PATH, "r") as f:
            norm_stats_ttl = json.load(f)
    else:
        norm_stats_ttl = norm_stats_tcn

    # Transformer Multitask — inferir arquitectura completa del checkpoint
    ckpt_trans = torch.load(TRANS_CKPT_PATH, map_location=DEVICE, weights_only=False)
    state_trans = (
        ckpt_trans["model_state_dict"]
        if "model_state_dict" in ckpt_trans
        else ckpt_trans
    )
    in_dim_trans = (
        state_trans["input_proj.0.weight"].shape[1]
        if "input_proj.0.weight" in state_trans
        else 28
    )
    # Inferir hiperparámetros de arquitectura desde los pesos del checkpoint
    cnn_ch_trans = []
    for i in range(10):
        key = f"cnn_blocks.{i}.conv.weight"
        if key in state_trans:
            cnn_ch_trans.append(state_trans[key].shape[0])
        else:
            break
    lstm_hidden_trans = state_trans["lstm.weight_hh_l0"].shape[1]
    lstm_layers_trans = sum(
        1
        for k in state_trans
        if k.startswith("lstm.weight_ih_l") and "reverse" not in k
    )
    n_attn_layers_trans = sum(
        1
        for k in state_trans
        if k.startswith("attention.layers.") and k.endswith(".norm1.weight")
    )
    print(
        f"  ✓ Cargando Transformer con CNN={cnn_ch_trans}, LSTM_H={lstm_hidden_trans}, "
        f"LSTM_L={lstm_layers_trans}, Attn_L={n_attn_layers_trans}"
    )
    trans_model = GaitMultiTaskNet(
        input_dim=in_dim_trans,
        cnn_channels=cnn_ch_trans,
        lstm_hidden=lstm_hidden_trans,
        lstm_layers=lstm_layers_trans,
        n_attn_layers=n_attn_layers_trans,
    ).to(DEVICE)
    trans_model.load_state_dict(state_trans)
    trans_model.eval()

    if "norm_stats" in ckpt_trans:
        norm_stats_trans = ckpt_trans["norm_stats"]
    else:
        norm_stats_trans = norm_stats_tcn

    # 2. Descubrir y dividir Dataset por Sujetos
    print("\n  📂 Descubriendo capturas y preparando conjunto de Test...")
    dataset_path = Path(args.dataset)
    captures_info = discover_captures(dataset_path)

    subjects_data = {}
    file_paths_by_subject = {}

    for mcd_path, aruco_path, subj in captures_info:
        cap = prepare_static_capture(mcd_path, aruco_path)
        if cap is not None:
            if subj not in subjects_data:
                subjects_data[subj] = []
                file_paths_by_subject[subj] = []
            subjects_data[subj].append(cap)
            file_paths_by_subject[subj].append((mcd_path, aruco_path))

    # Cargar split desde archivo
    split_path = (
        Path(args.data_split)
        if getattr(args, "data_split", None)
        else Path("data_split.json")
    )
    if split_path.exists():
        with open(split_path, "r") as f:
            split = json.load(f)
        test_subjs = [
            str(s) for s in split.get("test_subjects", []) if str(s) in subjects_data
        ]
        print(f"    Sujetos de Test (desde {split_path}): {test_subjs}")
    else:
        subjects = list(subjects_data.keys())
        random.shuffle(subjects)
        n_test = min(args.n_test_subjects, len(subjects))
        test_subjs = subjects[-n_test:]
        print(f"    Sujetos de Test (aleatorios): {test_subjs}")

    print(f"    Sujetos de Test ({len(test_subjs)}): {test_subjs}")

    # Pre-calcular Método Analítico en paralelo para todos los archivos de test
    all_test_files = []
    for subj in test_subjs:
        all_test_files.extend(file_paths_by_subject[subj])

    print(
        f"\n  ⚙️ Ejecutando Método Analítico en paralelo ({len(all_test_files)} capturas)..."
    )
    process_func = partial(
        process_files,
        use_3d=False,
        cutoff_freq=9,
        filter_order=6,
        pca_percentile=20,
        madgwick_gain=0.02,
        slerp_clip_min=0.3,
        slerp_clip_max=0.7,
        calibration_samples=100,
        fusion_filter="madgwick",
        pivot_height_threshold=0.005,
        swing_velocity_threshold=0.3,
    )
    with Pool() as pool:
        ana_results_list = pool.map(process_func, all_test_files)

    ana_results_dict = {f: res for f, res in zip(all_test_files, ana_results_list)}

    # 3. Inferencia de Reconstrucción Global con Desplazamiento
    all_nn_global_preds = []
    all_oracle_global_preds = []
    all_tcn_global_preds = []
    all_ttl_global_preds = []
    all_trans_global_preds = []
    all_ana_global_preds = []
    all_gt_global_targets = []  # GT para P+P, Oracle (usan todas las capturas)
    all_gt_tcn_targets = []  # GT alineado con máscara TCN
    all_gt_ttl_targets = []  # GT alineado con máscara TTL
    all_gt_trans_targets = []  # GT alineado con máscara Transformer
    all_gt_ana_targets = []  # GT alineado con analítico
    all_pivot_acc_nn = []

    # Listas para métricas en espacio relativo (pelvis=[0,0,0])
    all_tcn_rel_preds = []
    all_ttl_rel_preds = []
    all_trans_rel_preds = []
    all_nn_rel_preds = []
    all_oracle_rel_preds = []
    all_ana_rel_preds = []
    all_gt_rel_targets = []  # GT relativo para P+P/Oracle
    all_gt_tcn_rel_targets = []
    all_gt_ttl_rel_targets = []
    all_gt_trans_rel_targets = []
    all_gt_ana_rel_targets = []

    test_sample_for_anim = None

    print("\n  🔬 Ejecutando Inferencia con Acumulación de Desplazamiento...")

    for subj in test_subjs:
        for cap, (mcd_path, aruco_path) in zip(
            subjects_data[subj], file_paths_by_subject[subj]
        ):
            sensor_raw = cap["sensor_data"].astype(np.float32)
            gt_rel_pose = cap["gt_relative"]
            gt_pivot = cap["pivot_flag"]
            cap_len = len(gt_rel_pose)

            # --- Paso 1: Ground Truth Global con Desplazamiento Real ---
            gt_global = get_ground_truth_global_trajectory(mcd_path, aruco_path, cap)
            if gt_global is None or len(gt_global) != cap_len:
                continue

            # --- Paso 2: Normalización e Inferencia de PivotNet ---
            sensor_norm_pivot = (
                sensor_raw - norm_stats_pivot["mean"]
            ) / norm_stats_pivot["std"]
            sensor_norm_pose = (sensor_raw - norm_stats_pose["mean"]) / norm_stats_pose[
                "std"
            ]

            anthro = np.array(
                [
                    cap["femur_length"] / BONE_NORM,
                    cap["tibia_length"] / BONE_NORM,
                    cap["hip_width"] / BONE_NORM,
                    1.0 if cap["direction"] == "R" else 0.0,
                ],
                dtype=np.float32,
            )
            anthro_broadcast = np.tile(anthro, (cap_len, 1))

            x_pivot = np.concatenate([sensor_norm_pivot, anthro_broadcast], axis=1)
            x_pivot_tensor = torch.from_numpy(x_pivot).to(DEVICE)
            with torch.no_grad():
                pivot_logits = pivot_model(x_pivot_tensor)
                pivot_probs = torch.sigmoid(pivot_logits).cpu().numpy().flatten()

            pred_pivot_flag = (pivot_probs >= 0.5).astype(np.float32).reshape(-1, 1)

            acc_pivot = np.mean(
                (pred_pivot_flag.flatten() == gt_pivot.flatten()).astype(np.float32)
            )
            all_pivot_acc_nn.append(acc_pivot)

            # --- Paso 3: Inferencia de Poses Relativas (PoseNet) ---
            x_pose_nn = np.concatenate(
                [sensor_norm_pose, anthro_broadcast, pred_pivot_flag], axis=1
            )
            x_pose_nn_tensor = torch.from_numpy(x_pose_nn).to(DEVICE)
            with torch.no_grad():
                pred_rel_pose_nn = pose_model(x_pose_nn_tensor).cpu().numpy()

            # NOTA: Estático y Oracle usan accumulate_global_trajectory (acumulación
            # paso-a-paso del pivote), lo cual es propenso a drift acumulativo.
            # En contraste, TCN/TTL/Trans predicen directamente el desplazamiento
            # global de la cadera, evitando drift. Esto explica por qué el Oracle
            # puede ser peor que TCN aunque use el pivote perfecto del GT.

            # Oracle con GT Pivot
            gt_pivot_broadcast = gt_pivot.reshape(-1, 1).astype(np.float32)
            x_pose_oracle = np.concatenate(
                [sensor_norm_pose, anthro_broadcast, gt_pivot_broadcast], axis=1
            )
            x_pose_oracle_tensor = torch.from_numpy(x_pose_oracle).to(DEVICE)
            with torch.no_grad():
                pred_rel_pose_oracle = pose_model(x_pose_oracle_tensor).cpu().numpy()

            # --- Paso 4: Acumulación del Desplazamiento Global para la Red Neuronal ---
            # Determinar posición inicial del pie pivote desde el GT
            init_pivot_is_right = pred_pivot_flag[0, 0] == 1.0
            init_pivot_idx = 4 if init_pivot_is_right else 5
            init_pivot_pos = gt_global[0, init_pivot_idx * 3 : (init_pivot_idx + 1) * 3]

            pred_global_nn = accumulate_global_trajectory(
                pred_rel_pose_nn, pred_pivot_flag.flatten(), init_pivot_pos
            )

            init_oracle_is_right = gt_pivot[0] == 1.0
            init_oracle_idx = 4 if init_oracle_is_right else 5
            init_oracle_pos = gt_global[
                0, init_oracle_idx * 3 : (init_oracle_idx + 1) * 3
            ]
            pred_global_oracle = accumulate_global_trajectory(
                pred_rel_pose_oracle, gt_pivot.flatten(), init_oracle_pos
            )

            # --- Paso 5: Reconstrucción Global del Método Analítico ---
            first_valid = cap.get("first_valid", 0)
            ana_res = ana_results_dict.get((mcd_path, aruco_path))
            ana_global = None
            ana_rel_pose = None
            if ana_res is not None:
                _, _, rec, _, _, _, _ = ana_res
                if len(rec["hip_r"]) >= (first_valid + cap_len):
                    hip_center = (rec["hip_r"] + rec["hip_l"]) / 2.0
                    ana_hip = hip_center[first_valid : first_valid + cap_len]
                    gt_hip_all = (gt_global[:, 0:3] + gt_global[:, 3:6]) / 2.0
                    ana_offset = (
                        (gt_hip_all[0] - ana_hip[0])
                        if len(ana_hip) > 0
                        else np.zeros(3)
                    )
                    ana_global = np.zeros((cap_len, N_JOINTS * N_COORDS))
                    ana_rel_pose = np.zeros((cap_len, N_JOINTS * N_COORDS))
                    for j_idx, j_name in enumerate(JOINT_NAMES):
                        ana_global[:, j_idx * 3 : (j_idx + 1) * 3] = (
                            rec[j_name][first_valid : first_valid + cap_len]
                            + ana_offset
                        )
                        ana_rel_pose[:, j_idx * 3 : (j_idx + 1) * 3] = (
                            rec[j_name] - hip_center
                        )[first_valid : first_valid + cap_len]

            # --- Paso 6: Reconstrucción Secuencial (TCN, TTL y Transformer) ---
            anthro_4d = np.array(
                [
                    cap["femur_length"] / BONE_NORM,
                    cap["tibia_length"] / BONE_NORM,
                    cap["hip_width"] / BONE_NORM,
                    1.0 if cap["direction"] == "R" else 0.0,
                ],
                dtype=np.float32,
            )
            anthro_3d = np.array(
                [
                    cap["femur_length"] / BONE_NORM,
                    cap["tibia_length"] / BONE_NORM,
                    1.0 if cap["direction"] == "R" else 0.0,
                ],
                dtype=np.float32,
            )

            anthro_tcn = anthro_4d if in_dim_tcn == 28 else anthro_3d
            anthro_ttl = anthro_4d if in_dim_ttl == 28 else anthro_3d
            anthro_trans = anthro_4d if in_dim_trans == 28 else anthro_3d

            # Centro pélvico inicial en el espacio de la cámara (promedio de ambas caderas)
            gt_pelvis_0 = np.expand_dims(
                (gt_global[0, 0:3] + gt_global[0, 3:6]) / 2.0, axis=0
            )

            pred_pose_tcn, _, pred_hip_global_tcn, valid_tcn = predict_tcn(
                tcn_model,
                sensor_raw,
                anthro_tcn,
                norm_stats_tcn,
                window_size=args.window_size_tcn,
                initial_hip=gt_pelvis_0,
            )
            pred_pose_ttl, _, pred_hip_global_ttl, valid_ttl = predict_ttl(
                ttl_model,
                sensor_raw,
                anthro_ttl,
                norm_stats_ttl,
                window_size=args.window_size_ttl,
                initial_hip=gt_pelvis_0,
            )

            pred_pose_trans, _, pred_hip_global_trans, valid_trans = predict_trans(
                trans_model,
                sensor_raw,
                anthro_trans,
                norm_stats_trans,
                window_size=args.window_size_trans,
                initial_hip=gt_pelvis_0,
            )

            tcn_global = np.zeros_like(pred_pose_tcn)
            ttl_global = np.zeros_like(pred_pose_ttl)
            trans_global = np.zeros_like(pred_pose_trans)
            for j in range(N_JOINTS):
                tcn_global[:, j * 3 : (j + 1) * 3] = (
                    pred_pose_tcn[:, j * 3 : (j + 1) * 3] + pred_hip_global_tcn
                )
                ttl_global[:, j * 3 : (j + 1) * 3] = (
                    pred_pose_ttl[:, j * 3 : (j + 1) * 3] + pred_hip_global_ttl
                )
                trans_global[:, j * 3 : (j + 1) * 3] = (
                    pred_pose_trans[:, j * 3 : (j + 1) * 3] + pred_hip_global_trans
                )

            # --- Acumulación con máscaras independientes por modelo ---
            # P+P y Oracle usan todos los frames (sin ventana sliding)
            all_nn_global_preds.append(pred_global_nn)
            all_oracle_global_preds.append(pred_global_oracle)
            all_gt_global_targets.append(gt_global)

            # TCN: máscara propia
            if np.any(valid_tcn):
                all_tcn_global_preds.append(tcn_global[valid_tcn])
                all_gt_tcn_targets.append(gt_global[valid_tcn])

            # TTL: máscara propia
            if np.any(valid_ttl):
                all_ttl_global_preds.append(ttl_global[valid_ttl])
                all_gt_ttl_targets.append(gt_global[valid_ttl])

            # Transformer: máscara propia
            if np.any(valid_trans):
                all_trans_global_preds.append(trans_global[valid_trans])
                all_gt_trans_targets.append(gt_global[valid_trans])

            # Analítico: acumula GT solo cuando el analítico está disponible
            if ana_global is not None:
                all_ana_global_preds.append(ana_global)
                all_gt_ana_targets.append(gt_global)

            # --- Acumulación de pose relativa (pelvis en [0,0,0]) ---
            # Ground Truth relativo al centro pélvico (caderas)
            gt_pelvis = (gt_global[:, 0:3] + gt_global[:, 3:6]) / 2.0
            gt_rel_pelvis = np.zeros_like(gt_global)
            for j in range(N_JOINTS):
                gt_rel_pelvis[:, j * 3 : (j + 1) * 3] = (
                    gt_global[:, j * 3 : (j + 1) * 3] - gt_pelvis
                )

            # Red Estática y Oracle: transformar de relativas al pivote a relativas a pelvis
            nn_hip_center = (pred_rel_pose_nn[:, 0:3] + pred_rel_pose_nn[:, 3:6]) / 2.0
            pred_rel_pelvis_nn = np.zeros_like(pred_rel_pose_nn)
            for j in range(N_JOINTS):
                pred_rel_pelvis_nn[:, j * 3 : (j + 1) * 3] = (
                    pred_rel_pose_nn[:, j * 3 : (j + 1) * 3] - nn_hip_center
                )

            oracle_hip_center = (
                pred_rel_pose_oracle[:, 0:3] + pred_rel_pose_oracle[:, 3:6]
            ) / 2.0
            pred_rel_pelvis_oracle = np.zeros_like(pred_rel_pose_oracle)
            for j in range(N_JOINTS):
                pred_rel_pelvis_oracle[:, j * 3 : (j + 1) * 3] = (
                    pred_rel_pose_oracle[:, j * 3 : (j + 1) * 3] - oracle_hip_center
                )

            all_nn_rel_preds.append(pred_rel_pelvis_nn)
            all_oracle_rel_preds.append(pred_rel_pelvis_oracle)
            all_gt_rel_targets.append(gt_rel_pelvis)

            # Modelos secuenciales (TCN, TTL, Transformer) ya están centrados en pelvis
            if np.any(valid_tcn):
                all_tcn_rel_preds.append(pred_pose_tcn[valid_tcn])
                all_gt_tcn_rel_targets.append(gt_rel_pelvis[valid_tcn])
            if np.any(valid_ttl):
                all_ttl_rel_preds.append(pred_pose_ttl[valid_ttl])
                all_gt_ttl_rel_targets.append(gt_rel_pelvis[valid_ttl])
            if np.any(valid_trans):
                all_trans_rel_preds.append(pred_pose_trans[valid_trans])
                all_gt_trans_rel_targets.append(gt_rel_pelvis[valid_trans])

            # Método analítico ya está centrado en pelvis (rec[j_name] - hip_center)
            if ana_global is not None and ana_rel_pose is not None:
                all_ana_rel_preds.append(ana_rel_pose)
                all_gt_ana_rel_targets.append(gt_rel_pelvis)

            # Muestra para animación (usa intersección solo para visualización)
            valid_anim = valid_tcn & valid_ttl & valid_trans
            n_valid_anim = int(np.sum(valid_anim))
            if ana_global is not None and n_valid_anim > 0:
                if test_sample_for_anim is None or n_valid_anim > len(
                    test_sample_for_anim[0]
                ):
                    test_sample_for_anim = (
                        gt_global[valid_anim],
                        pred_global_nn[valid_anim],
                        tcn_global[valid_anim],
                        ttl_global[valid_anim],
                        trans_global[valid_anim],
                        ana_global[valid_anim],
                    )

    if len(all_nn_global_preds) == 0:
        print("  [ERROR] No se completó la inferencia global.")
        return

    # 4. Consolidar y Calcular Métricas
    nn_global_all = np.concatenate(all_nn_global_preds, axis=0)
    oracle_global_all = np.concatenate(all_oracle_global_preds, axis=0)
    gt_global_all = np.concatenate(all_gt_global_targets, axis=0)

    # Cada modelo secuencial usa su propio GT alineado
    tcn_global_all = (
        np.concatenate(all_tcn_global_preds, axis=0)
        if all_tcn_global_preds
        else np.array([])
    )
    gt_tcn_all = (
        np.concatenate(all_gt_tcn_targets, axis=0)
        if all_gt_tcn_targets
        else np.array([])
    )
    ttl_global_all = (
        np.concatenate(all_ttl_global_preds, axis=0)
        if all_ttl_global_preds
        else np.array([])
    )
    gt_ttl_all = (
        np.concatenate(all_gt_ttl_targets, axis=0)
        if all_gt_ttl_targets
        else np.array([])
    )
    trans_global_all = (
        np.concatenate(all_trans_global_preds, axis=0)
        if all_trans_global_preds
        else np.array([])
    )
    gt_trans_all = (
        np.concatenate(all_gt_trans_targets, axis=0)
        if all_gt_trans_targets
        else np.array([])
    )

    pct = args.percentile
    pct_str = f" (Percentil {pct:.0f}%)" if pct < 100.0 else ""

    print(
        f"\n  Frames evaluados: P+P/Oracle={len(nn_global_all)}, TCN={len(tcn_global_all)}, TTL={len(ttl_global_all)}, Trans={len(trans_global_all)}, Ana={len(np.concatenate(all_ana_global_preds, axis=0)) if len(all_ana_global_preds) > 0 else 0}"
    )

    mpjpe_nn_global = compute_mpjpe(nn_global_all, gt_global_all, percentile=pct)
    mae_nn_global = compute_mae(nn_global_all, gt_global_all, percentile=pct)
    rmse_nn_global = compute_rmse(nn_global_all, gt_global_all, percentile=pct)

    mpjpe_oracle_global = compute_mpjpe(
        oracle_global_all, gt_global_all, percentile=pct
    )
    mae_oracle_global = compute_mae(oracle_global_all, gt_global_all, percentile=pct)
    rmse_oracle_global = compute_rmse(oracle_global_all, gt_global_all, percentile=pct)

    mpjpe_tcn_global = (
        compute_mpjpe(tcn_global_all, gt_tcn_all, percentile=pct)
        if len(tcn_global_all) > 0
        else {n: 0.0 for n in JOINT_NAMES + ["mean"]}
    )
    mae_tcn_global = (
        compute_mae(tcn_global_all, gt_tcn_all, percentile=pct)
        if len(tcn_global_all) > 0
        else {n: 0.0 for n in JOINT_NAMES + ["mean"]}
    )
    rmse_tcn_global = (
        compute_rmse(tcn_global_all, gt_tcn_all, percentile=pct)
        if len(tcn_global_all) > 0
        else {n: 0.0 for n in JOINT_NAMES + ["mean"]}
    )

    mpjpe_ttl_global = (
        compute_mpjpe(ttl_global_all, gt_ttl_all, percentile=pct)
        if len(ttl_global_all) > 0
        else {n: 0.0 for n in JOINT_NAMES + ["mean"]}
    )
    mae_ttl_global = (
        compute_mae(ttl_global_all, gt_ttl_all, percentile=pct)
        if len(ttl_global_all) > 0
        else {n: 0.0 for n in JOINT_NAMES + ["mean"]}
    )
    rmse_ttl_global = (
        compute_rmse(ttl_global_all, gt_ttl_all, percentile=pct)
        if len(ttl_global_all) > 0
        else {n: 0.0 for n in JOINT_NAMES + ["mean"]}
    )

    mpjpe_trans_global = (
        compute_mpjpe(trans_global_all, gt_trans_all, percentile=pct)
        if len(trans_global_all) > 0
        else {n: 0.0 for n in JOINT_NAMES + ["mean"]}
    )
    mae_trans_global = (
        compute_mae(trans_global_all, gt_trans_all, percentile=pct)
        if len(trans_global_all) > 0
        else {n: 0.0 for n in JOINT_NAMES + ["mean"]}
    )
    rmse_trans_global = (
        compute_rmse(trans_global_all, gt_trans_all, percentile=pct)
        if len(trans_global_all) > 0
        else {n: 0.0 for n in JOINT_NAMES + ["mean"]}
    )

    has_ana = len(all_ana_global_preds) > 0
    if has_ana:
        ana_global_all = np.concatenate(all_ana_global_preds, axis=0)
        gt_ana_all = np.concatenate(all_gt_ana_targets, axis=0)
        mpjpe_ana_global = compute_mpjpe(ana_global_all, gt_ana_all, percentile=pct)
        mae_ana_global = compute_mae(ana_global_all, gt_ana_all, percentile=pct)
        rmse_ana_global = compute_rmse(ana_global_all, gt_ana_all, percentile=pct)

    # --- Métricas en Espacio Relativo (Pelvis en [0, 0, 0]) ---
    nn_rel_all = np.concatenate(all_nn_rel_preds, axis=0)
    oracle_rel_all = np.concatenate(all_oracle_rel_preds, axis=0)
    gt_rel_all = np.concatenate(all_gt_rel_targets, axis=0)

    tcn_rel_all = (
        np.concatenate(all_tcn_rel_preds, axis=0) if all_tcn_rel_preds else np.array([])
    )
    gt_tcn_rel_all = (
        np.concatenate(all_gt_tcn_rel_targets, axis=0)
        if all_gt_tcn_rel_targets
        else np.array([])
    )
    ttl_rel_all = (
        np.concatenate(all_ttl_rel_preds, axis=0) if all_ttl_rel_preds else np.array([])
    )
    gt_ttl_rel_all = (
        np.concatenate(all_gt_ttl_rel_targets, axis=0)
        if all_gt_ttl_rel_targets
        else np.array([])
    )
    trans_rel_all = (
        np.concatenate(all_trans_rel_preds, axis=0)
        if all_trans_rel_preds
        else np.array([])
    )
    gt_trans_rel_all = (
        np.concatenate(all_gt_trans_rel_targets, axis=0)
        if all_gt_trans_rel_targets
        else np.array([])
    )

    mpjpe_nn_rel = compute_mpjpe(nn_rel_all, gt_rel_all, percentile=pct)
    mpjpe_oracle_rel = compute_mpjpe(oracle_rel_all, gt_rel_all, percentile=pct)
    mpjpe_tcn_rel = (
        compute_mpjpe(tcn_rel_all, gt_tcn_rel_all, percentile=pct)
        if len(tcn_rel_all) > 0
        else {n: 0.0 for n in JOINT_NAMES + ["mean"]}
    )
    mpjpe_ttl_rel = (
        compute_mpjpe(ttl_rel_all, gt_ttl_rel_all, percentile=pct)
        if len(ttl_rel_all) > 0
        else {n: 0.0 for n in JOINT_NAMES + ["mean"]}
    )
    mpjpe_trans_rel = (
        compute_mpjpe(trans_rel_all, gt_trans_rel_all, percentile=pct)
        if len(trans_rel_all) > 0
        else {n: 0.0 for n in JOINT_NAMES + ["mean"]}
    )

    has_ana_rel = len(all_ana_rel_preds) > 0
    if has_ana_rel:
        ana_rel_all = np.concatenate(all_ana_rel_preds, axis=0)
        gt_ana_rel_all = np.concatenate(all_gt_ana_rel_targets, axis=0)
        mpjpe_ana_rel = compute_mpjpe(ana_rel_all, gt_ana_rel_all, percentile=pct)

    # --- Tabla de Métricas Relativas ---
    print(f"\n  📐 MÉTRICAS EN ESPACIO RELATIVO (Pelvis en [0, 0, 0]){pct_str}")
    print("  " + "─" * 110)
    print(
        f"     {'Articulación':<14}  {'Estático(P+P)':>15} {'Oracle(GT Piv)':>15}  {'TCN Lite':>15}  {'TTL Lite':>15}  {'Transformer':>15}  {'Analítico':>12}  {'Ganador':>10}"
    )
    print(
        f"     {'─' * 14}  {'─' * 15}  {'─' * 15}  {'─' * 15}  {'─' * 15}  {'─' * 15}  {'─' * 12}  {'─' * 10}"
    )
    for name in JOINT_NAMES:
        nn_e = mpjpe_nn_rel[name]
        orc_e = mpjpe_oracle_rel[name]
        tcn_e = mpjpe_tcn_rel[name]
        ttl_e = mpjpe_ttl_rel[name]
        trans_e = mpjpe_trans_rel[name]
        ana_e = mpjpe_ana_rel[name] if has_ana_rel else float("nan")
        errors = [
            ("Estático", nn_e),
            ("TCN", tcn_e),
            ("TTL", ttl_e),
            ("Trans", trans_e),
        ]
        if has_ana_rel:
            errors.append(("Analítico", ana_e))
        win = sorted(errors, key=lambda x: x[1])[0][0]
        print(
            f"     {name:<14}  {nn_e:>13.4f} m  {orc_e:>13.4f} m {tcn_e:>13.4f} m  {ttl_e:>13.4f} m  {trans_e:>13.4f} m  {ana_e:>10.4f} m  {win:>10}"
        )
    print(
        f"     {'─' * 14}  {'─' * 15}  {'─' * 15}  {'─' * 15}  {'─' * 15}  {'─' * 15}  {'─' * 12}  {'─' * 10}"
    )
    ana_mean_rel = mpjpe_ana_rel["mean"] if has_ana_rel else float("nan")
    mean_errors_rel = [
        ("Estático", mpjpe_nn_rel["mean"]),
        ("TCN", mpjpe_tcn_rel["mean"]),
        ("TTL", mpjpe_ttl_rel["mean"]),
        ("Trans", mpjpe_trans_rel["mean"]),
    ]
    if has_ana_rel:
        mean_errors_rel.append(("Analítico", mpjpe_ana_rel["mean"]))
    win_gen_rel = sorted(mean_errors_rel, key=lambda x: x[1])[0][0]
    print(
        f"     {'GENERAL':<14}  {mpjpe_nn_rel['mean']:>13.4f} m  {mpjpe_oracle_rel['mean']:>13.4f} m  {mpjpe_tcn_rel['mean']:>13.4f} m  {mpjpe_ttl_rel['mean']:>13.4f} m  {mpjpe_trans_rel['mean']:>13.4f} m  {ana_mean_rel:>10.4f} m  {win_gen_rel:>10}"
    )

    if has_ana_rel:
        print("\n  🏆 MEJORAS RELATIVAS VS ANALÍTICO — ESPACIO RELATIVO:")
        for model_name, mpjpe_dict in [
            ("Estático (P+P)", mpjpe_nn_rel),
            ("TCN Lite", mpjpe_tcn_rel),
            ("TTL Lite", mpjpe_ttl_rel),
            ("Transformer", mpjpe_trans_rel),
        ]:
            imp = (
                (mpjpe_ana_rel["mean"] - mpjpe_dict["mean"]) / mpjpe_ana_rel["mean"]
            ) * 100
            print(f"     [{model_name}] Reducción MPJPE Relativo: {imp:+.2f}%")

    # 5. Imprimir Resultados
    print()
    print("═" * 85)
    print(f"  📊 RESULTADOS CON DESPLAZAMIENTO GLOBAL 3D (Test Set{pct_str})")
    print("═" * 85)
    print(
        f"  • Exactitud de Clasificación PivotNet: {np.mean(all_pivot_acc_nn)*100:.2f}%"
    )
    print("  " + "─" * 80)

    def print_metric_table(
        metric_name, dict_nn, dict_tcn, dict_ttl, dict_trans, dict_orc, dict_ana
    ):
        print(f"\n  {metric_name} Global")
        print("  " + "─" * 110)
        print(
            f"     {'Articulación':<14}  {'Estático(P+P)':>15} {'Oracle(GT Piv)':>15}  {'TCN Lite':>15}  {'TTL Lite':>15}  {'Transformer':>15}  {'Analítico':>12}  {'Ganador':>10}"
        )
        print(
            f"     {'─' * 14}  {'─' * 15}  {'─' * 15}  {'─' * 15}  {'─' * 15}  {'─' * 15}  {'─' * 12}  {'─' * 10}"
        )

        for name in JOINT_NAMES:
            nn_e = dict_nn[name]
            orc_e = dict_orc[name]
            tcn_e = dict_tcn[name]
            ttl_e = dict_ttl[name]
            trans_e = dict_trans[name]
            ana_e = dict_ana[name] if has_ana else float("nan")

            errors = [
                ("Estático", nn_e),
                ("TCN", tcn_e),
                ("TTL", ttl_e),
                ("Trans", trans_e),
            ]
            if has_ana:
                errors.append(("Analítico", ana_e))
            win = sorted(errors, key=lambda x: x[1])[0][0]

            print(
                f"     {name:<14}  {nn_e:>13.4f} m  {orc_e:>13.4f} m {tcn_e:>13.4f} m  {ttl_e:>13.4f} m  {trans_e:>13.4f} m  {ana_e:>10.4f} m  {win:>10}"
            )

        print(
            f"     {'─' * 14}  {'─' * 15}  {'─' * 15}  {'─' * 15}  {'─' * 15}  {'─' * 15}  {'─' * 12}  {'─' * 10}"
        )

        mean_errors = [
            ("Estático", dict_nn["mean"]),
            ("TCN", dict_tcn["mean"]),
            ("TTL", dict_ttl["mean"]),
            ("Trans", dict_trans["mean"]),
        ]
        if has_ana:
            mean_errors.append(("Analítico", dict_ana["mean"]))
        win_gen = sorted(mean_errors, key=lambda x: x[1])[0][0]

        ana_mean = dict_ana["mean"] if has_ana else float("nan")

        print(
            f"     {'GENERAL':<14}  {dict_nn['mean']:>13.4f} m  {dict_orc['mean']:>13.4f} m  {dict_tcn['mean']:>13.4f} m  {dict_ttl['mean']:>13.4f} m  {dict_trans['mean']:>13.4f} m  {ana_mean:>10.4f} m  {win_gen:>10}"
        )

    print_metric_table(
        f"1) MPJPE{pct_str}",
        mpjpe_nn_global,
        mpjpe_tcn_global,
        mpjpe_ttl_global,
        mpjpe_trans_global,
        mpjpe_oracle_global,
        mpjpe_ana_global if has_ana else None,
    )
    print_metric_table(
        f"2) MAE{pct_str}",
        mae_nn_global,
        mae_tcn_global,
        mae_ttl_global,
        mae_trans_global,
        mae_oracle_global,
        mae_ana_global if has_ana else None,
    )
    print_metric_table(
        f"3) RMSE{pct_str}",
        rmse_nn_global,
        rmse_tcn_global,
        rmse_ttl_global,
        rmse_trans_global,
        rmse_oracle_global,
        rmse_ana_global if has_ana else None,
    )

    if has_ana:
        print(
            "\n  🏆 MEJORAS RELATIVAS VS MÉTODO ANALÍTICO (Valores positivos indican mejora):"
        )

        def print_improvement(model_name, mpjpe_dict, mae_dict):
            imp_mpjpe = (
                (mpjpe_ana_global["mean"] - mpjpe_dict["mean"])
                / mpjpe_ana_global["mean"]
            ) * 100
            imp_mae = (
                (mae_ana_global["mean"] - mae_dict["mean"]) / mae_ana_global["mean"]
            ) * 100
            print(f"     [{model_name}]")
            print(f"       • Reducción de Error MPJPE: {imp_mpjpe:+.2f}%")
            print(f"       • Reducción de Error MAE:   {imp_mae:+.2f}%\n")

        print_improvement("Red Estática (P+P)", mpjpe_nn_global, mae_nn_global)
        print_improvement("TCN Lite", mpjpe_tcn_global, mae_tcn_global)
        print_improvement("TTL Lite", mpjpe_ttl_global, mae_ttl_global)
        print_improvement("Transformer", mpjpe_trans_global, mae_trans_global)

    print("═" * 85)

    # 6. Generar Gráficas y Animaciones
    print("═" * 95)

    if test_sample_for_anim is not None:
        gt_s, nn_s, tcn_s, ttl_s, trans_s, ana_s = test_sample_for_anim
        plot_trajectory_2d_and_bars(
            mpjpe_nn_global,
            mpjpe_tcn_global,
            mpjpe_ttl_global,
            mpjpe_trans_global,
            mpjpe_ana_global if has_ana else None,
            gt_s,
            nn_s,
            tcn_s,
            ttl_s,
            trans_s,
            ana_s,
            RESULTS_DIR,
            percentile=pct,
        )
        anim_path = RESULTS_DIR / "global_pipeline_comparison.gif"
        save_global_comparison_animation(
            gt_s,
            nn_s,
            tcn_s,
            ttl_s,
            trans_s,
            ana_s,
            str(anim_path),
            max_frames=args.anim_frames,
        )

    print()


if __name__ == "__main__":
    main()
