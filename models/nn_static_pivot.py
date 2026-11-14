"""
Neural network: static reconstruction relative to the pivot (Static Pivot)
========================================================================
Architecture:
  - Deep feed-forward neural network (MLP) with dense residual blocks (Linear + BatchNorm + LeakyReLU + Dropout).
  - Input: instantaneous vector (24 IMU values: accelerometers and gyroscopes of 4 sensors) + 3 normalized anthropometric measurements.
  - Output: 18-dimensional 3D pose tensor (6 joints x 3 coordinates) referenced to the current pivot foot.
  - Loss function: pose MSE + static_bone_length_loss (strict bone-length consistency).

Purpose:
  - Estimate the joint pose decoupled from spatial progression, using the stance (pivot) foot as the
    anchor of the local reference frame, which later allows a cumulative global reconstruction.

Output directory: nn-results/static_pivot/
Checkpoints: nn-results/static_pivot/checkpoints/best_static_model.pt

Usage:
    python -m models.nn_static_pivot                         # Training + evaluation
    python -m models.nn_static_pivot --eval-only             # Evaluation only

Author: GaitTracking Project
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
from scipy.interpolate import interp1d

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import the existing data-loading utilities
from process_all import (
    read_mcd_file,
    read_aruco_file,
    adjust_aruco_data,
    low_pass_filter,
    process_files,
)

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

DATASET_PATH = Path("DataSet")  # default: DataSet/ in the repository root (override with --dataset)
DATA_SPLIT_PATH = (
    Path(__file__).resolve().parent / "data_split.json"
    if (Path(__file__).resolve().parent / "data_split.json").exists()
    else Path("data_split.json")
)
RESULTS_DIR = Path("nn-results/static_pivot")
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINT_NAME = "best_static_model.pt"
CHECKPOINT_PATH = CHECKPOINT_DIR / CHECKPOINT_NAME
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Data ──────────────────────────────────────────────────────────────
CUTOFF_FREQ = 7  # Hz, low-pass filter of the MCD
FILTER_ORDER = 6  # Butterworth filter order

# ── Training ─────────────────────────────────────────────────────────
BATCH_SIZE = 1024  # Larger batch size to make better use of the GPU (previously 256)
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 150
PATIENCE = 30
GRAD_CLIP = 1.0

# ── Joints and input/output ──────────────────────────────────────────
JOINT_NAMES = ["hip_r", "hip_l", "knee_r", "knee_l", "ankle_r", "ankle_l"]
N_JOINTS = 6
N_COORDS = 3
OUTPUT_DIM = N_JOINTS * N_COORDS  # 18

IMU_NAMES = ["RF", "LF", "RT", "LT"]
SENSOR_CHANNELS = len(IMU_NAMES) * 6  # 24
ANTHROPOMETRIC_DIM = 4  # femur, tibia, direction
PIVOT_FLAG_DIM = 1  # 1 if the right foot is the pivot, 0 if it is the left one
INPUT_DIM = SENSOR_CHANNELS + ANTHROPOMETRIC_DIM + PIVOT_FLAG_DIM  # 28

# ── Model architecture (MLP) ─────────────────────────────────────────
HIDDEN_LAYERS = [64, 64, 32]
DROPOUT_RATES = [0.15, 0.15, 0.1]


# ═══════════════════════════════════════════════════════════════════════
#  DATA PREPARATION (NO SEQUENCE, HIP-RELATIVE)
# ═══════════════════════════════════════════════════════════════════════


def discover_captures(dataset_path: Path) -> list[tuple[str, str, str]]:
    subjects = sorted(glob.glob("*", root_dir=str(dataset_path)))
    subjects = sorted([s for s in subjects if s.isdigit() and s != "0"], key=int)
    captures = []
    for s in subjects:
        for i in range(4):
            mcd_file = dataset_path / s / f"mcd-{i}.csv"
            aruco_file = dataset_path / s / f"arUcos-{i}.csv"
            if mcd_file.exists() and aruco_file.exists():
                captures.append((str(mcd_file), str(aruco_file), s))
    return captures


def prepare_static_capture(mcd_path: str, aruco_path: str) -> dict | None:
    """
    Process an MCD/ArUco pair to produce static (frame-by-frame) data.
    Removes the displacement by centring every joint
    on the hip at each instant.
    """
    try:
        mcd_data = read_mcd_file(
            mcd_path, cutoff_freq=CUTOFF_FREQ, filter_order=FILTER_ORDER
        )
        aruco_data = read_aruco_file(aruco_path)
        direction = mcd_data["measures"]["direction"]

        # Shift ArUco to the origin
        aruco_data = adjust_aruco_data(aruco_data, direction)

        aruco_processed = {
            "hip": aruco_data["hip_position"],
            "knee_r": aruco_data["R_knee_position"],
            "knee_l": aruco_data["L_knee_position"],
            "ankle_r": aruco_data["R_ankle_position"],
            "ankle_l": aruco_data["L_ankle_position"],
        }

        # Filter the Y noise
        t_aruco = aruco_processed["hip"]["time"]
        if len(t_aruco) > 1:
            fs_aruco = 1000.0 / max(np.mean(np.diff(t_aruco)), 1.0)
            for key in aruco_processed:
                y_data = aruco_processed[key]["data"][:, 1]
                try:
                    y_filtered = low_pass_filter(
                        y_data, cutoff_freq=2.0, fs=fs_aruco, order=3
                    )
                    aruco_processed[key]["data"][:, 1] = y_filtered
                except Exception:
                    pass

        mcd_time = mcd_data["time"]
        N = len(mcd_time)
        sensor_data = np.zeros((N, SENSOR_CHANNELS), dtype=np.float64)
        for idx, imu in enumerate(IMU_NAMES):
            sensor_data[:, idx * 6 : idx * 6 + 3] = mcd_data[f"acceleration_{imu}"]
            sensor_data[:, idx * 6 + 3 : idx * 6 + 6] = mcd_data[f"gyro_{imu}"]

        # Interpolate ArUco
        aruco_raw_keys = ["hip", "knee_r", "knee_l", "ankle_r", "ankle_l"]
        t_min_all = -np.inf
        t_max_all = np.inf
        for key in aruco_raw_keys:
            aruco_t = aruco_processed[key]["time"].astype(float)
            t_min_all = max(t_min_all, aruco_t.min())
            t_max_all = min(t_max_all, aruco_t.max())

        valid_time_mask = (mcd_time >= t_min_all) & (mcd_time <= t_max_all)
        n_valid = int(np.sum(valid_time_mask))
        if n_valid < 10:
            return None
        valid_mcd_time = mcd_time[valid_time_mask]

        hip_t = aruco_processed["hip"]["time"].astype(float)
        hip_pos = aruco_processed["hip"]["data"]
        hip_interp = [interp1d(hip_t, hip_pos[:, i], kind="linear") for i in range(3)]
        hip_at_mcd = np.column_stack([hip_interp[i](valid_mcd_time) for i in range(3)])

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

        raw_hw = float(mcd_data["measures"].get("hip_width", 0.25))
        hip_width = raw_hw  # if raw_hw > 0.05 else 0.25
        half_hw = hip_width / 2.0
        side_sign = +1.0 if direction == "L" else -1.0
        half_hw_r = +half_hw * side_sign
        half_hw_l = -half_hw * side_sign

        # Remove the displacement by fixing Y=0
        hip_at_mcd[:, 1] = 0.0

        gt_relative = np.zeros((N, N_JOINTS * N_COORDS))

        # Positions relative to the pivot foot (taking the lowest Z and lowest velocity)
        z_r = joints_at_mcd["ankle_r"][:, 2]
        z_l = joints_at_mcd["ankle_l"][:, 2]

        # Compute the approximate velocity of each ankle
        dt = np.gradient(valid_mcd_time)
        dt[dt == 0] = 1e-6  # Avoid division by zero

        v_r = np.linalg.norm(
            np.gradient(joints_at_mcd["ankle_r"], axis=0) / dt[:, np.newaxis], axis=1
        )
        v_l = np.linalg.norm(
            np.gradient(joints_at_mcd["ankle_l"], axis=0) / dt[:, np.newaxis], axis=1
        )

        # Combine height (Z) and velocity to determine the pivot
        # Z is usually in the range [0, 0.2] m, V in [0, 2] m/s. A weight balances them.
        weight_v = 0.1
        score_r = z_r + weight_v * v_r
        score_l = z_l + weight_v * v_l

        pivot_x = np.where(
            score_r < score_l,
            joints_at_mcd["ankle_r"][:, 0],
            joints_at_mcd["ankle_l"][:, 0],
        )
        pivot_z = np.where(score_r < score_l, z_r, z_l)
        pivot_is_right = (score_r < score_l).astype(np.float32)

        hip_r_rel = np.column_stack(
            [
                hip_at_mcd[:, 0] - pivot_x,
                np.full(n_valid, half_hw_r),
                hip_at_mcd[:, 2] - pivot_z,
            ]
        )
        hip_l_rel = np.column_stack(
            [
                hip_at_mcd[:, 0] - pivot_x,
                np.full(n_valid, half_hw_l),
                hip_at_mcd[:, 2] - pivot_z,
            ]
        )

        knee_r_rel = np.column_stack(
            [
                joints_at_mcd["knee_r"][:, 0] - pivot_x,
                np.full(n_valid, half_hw_r),
                joints_at_mcd["knee_r"][:, 2] - pivot_z,
            ]
        )
        knee_l_rel = np.column_stack(
            [
                joints_at_mcd["knee_l"][:, 0] - pivot_x,
                np.full(n_valid, half_hw_l),
                joints_at_mcd["knee_l"][:, 2] - pivot_z,
            ]
        )

        ankle_r_rel = np.column_stack(
            [
                joints_at_mcd["ankle_r"][:, 0] - pivot_x,
                np.full(n_valid, half_hw_r),
                joints_at_mcd["ankle_r"][:, 2] - pivot_z,
            ]
        )
        ankle_l_rel = np.column_stack(
            [
                joints_at_mcd["ankle_l"][:, 0] - pivot_x,
                np.full(n_valid, half_hw_l),
                joints_at_mcd["ankle_l"][:, 2] - pivot_z,
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

        femur_r_dists = np.linalg.norm(knee_r_rel - hip_r_rel, axis=1)
        femur_l_dists = np.linalg.norm(knee_l_rel - hip_l_rel, axis=1)
        tibia_r_dists = np.linalg.norm(ankle_r_rel - knee_r_rel, axis=1)
        tibia_l_dists = np.linalg.norm(ankle_l_rel - knee_l_rel, axis=1)

        femur_length = float(np.mean(np.concatenate([femur_r_dists, femur_l_dists])))
        tibia_length = float(np.mean(np.concatenate([tibia_r_dists, tibia_l_dists])))

        first_valid = int(np.argmax(valid_time_mask))
        last_valid = N - int(np.argmax(valid_time_mask[::-1])) - 1

        return {
            "sensor_data": sensor_data[first_valid : last_valid + 1],
            "gt_relative": gt_relative[first_valid : last_valid + 1],
            "pivot_flag": pivot_is_right,
            "femur_length": femur_length,
            "tibia_length": tibia_length,
            "hip_width": hip_width,
            "direction": direction,
            "first_valid": first_valid,
        }
    except Exception as e:
        print(f"  [ERROR] {Path(mcd_path).name}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
#  STATIC DATASET
# ═══════════════════════════════════════════════════════════════════════


class GaitStaticDataset(Dataset):
    BONE_NORM = 0.45

    def __init__(
        self, captures: list[dict], normalize: bool = True, stats: dict | None = None
    ):
        self.inputs = []
        self.targets = []

        all_sensors = []

        for cap in captures:
            sensor = cap["sensor_data"].astype(np.float32)
            gt_pose = cap["gt_relative"].astype(np.float32)

            anthro = np.array(
                [
                    cap["femur_length"] / self.BONE_NORM,
                    cap["tibia_length"] / self.BONE_NORM,
                    cap["hip_width"] / self.BONE_NORM,
                    1.0 if cap["direction"] == "R" else 0.0,
                ],
                dtype=np.float32,
            )

            # Repeat the anthropometry for each frame
            anthro_broadcast = np.tile(anthro, (len(sensor), 1))

            # Pivot flag (1 if right, 0 if left)
            pivot_flag = cap["pivot_flag"].reshape(-1, 1)

            # Concatenate the extra features
            extra_features = np.concatenate([anthro_broadcast, pivot_flag], axis=1)

            self.inputs.append((sensor, extra_features))
            self.targets.append(gt_pose)
            all_sensors.append(sensor)

        if len(self.inputs) > 0:
            self.sensor_data = np.concatenate([inp[0] for inp in self.inputs], axis=0)
            self.anthro_data = np.concatenate([inp[1] for inp in self.inputs], axis=0)
            self.target_data = np.concatenate(self.targets, axis=0)
        else:
            self.sensor_data = np.array([])
            self.anthro_data = np.array([])
            self.target_data = np.array([])

        if stats is not None:
            self.mean = stats["mean"]
            self.std = stats["std"]
        elif len(all_sensors) > 0:
            all_sensor_arr = np.concatenate(all_sensors, axis=0)
            self.mean = all_sensor_arr.mean(axis=0).astype(np.float32)
            self.std = all_sensor_arr.std(axis=0).astype(np.float32)
            self.std[self.std < 1e-8] = 1.0
        else:
            self.mean = np.zeros(SENSOR_CHANNELS, dtype=np.float32)
            self.std = np.ones(SENSOR_CHANNELS, dtype=np.float32)

        self.normalize = normalize

        # --- OPTIMIZATION: pre-compute tensors ---
        if len(self.sensor_data) > 0:
            if self.normalize:
                self.sensor_data = (self.sensor_data - self.mean) / self.std

            x_np = np.concatenate([self.sensor_data, self.anthro_data], axis=1)
            self.x = torch.from_numpy(x_np).float()
            self.y = torch.from_numpy(self.target_data).float()
            self.anthro = torch.from_numpy(self.anthro_data).float()
        else:
            self.x = torch.tensor([])
            self.y = torch.tensor([])
            self.anthro = torch.tensor([])

    def get_stats(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.anthro[idx]


# ═══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD MODEL
# ═══════════════════════════════════════════════════════════════════════


class GaitStaticNet(nn.Module):
    """
    MLP that takes the data of one instant (sensors + anthropometry) and
    directly predicts the relative pose at that instant.
    """

    def __init__(
        self,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_layers=None,
        dropout_rates=None,
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = HIDDEN_LAYERS
        if dropout_rates is None:
            dropout_rates = DROPOUT_RATES

        layers = []
        current_dim = input_dim

        for h_dim, drop_rate in zip(hidden_layers, dropout_rates):
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop_rate))
            current_dim = h_dim

        layers.append(nn.Linear(current_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════
#  LOSS
# ═══════════════════════════════════════════════════════════════════════


def static_bone_length_loss(
    predictions: torch.Tensor, anthropometric: torch.Tensor
) -> torch.Tensor:
    pred_joints = predictions.view(-1, N_JOINTS, N_COORDS)

    hip_r = pred_joints[:, 0, :]
    hip_l = pred_joints[:, 1, :]
    knee_r = pred_joints[:, 2, :]
    knee_l = pred_joints[:, 3, :]
    ankle_r = pred_joints[:, 4, :]
    ankle_l = pred_joints[:, 5, :]

    femur_r_len = torch.norm(knee_r - hip_r, dim=-1)
    femur_l_len = torch.norm(knee_l - hip_l, dim=-1)
    tibia_r_len = torch.norm(ankle_r - knee_r, dim=-1)
    tibia_l_len = torch.norm(ankle_l - knee_l, dim=-1)

    femur_target = anthropometric[:, 0] * GaitStaticDataset.BONE_NORM
    tibia_target = anthropometric[:, 1] * GaitStaticDataset.BONE_NORM

    loss = (
        F.mse_loss(femur_r_len, femur_target)
        + F.mse_loss(femur_l_len, femur_target)
        + F.mse_loss(tibia_r_len, tibia_target)
        + F.mse_loss(tibia_l_len, tibia_target)
    ) / 4.0
    return loss


def static_loss(pred_pose, target_pose, anthropometric, bone_weight=0.1):
    mse = F.mse_loss(pred_pose, target_pose)
    bone = static_bone_length_loss(pred_pose, anthropometric)
    total = mse + bone_weight * bone
    return total, mse.item(), bone.item()


# ═══════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════


def print_model_summary(model: nn.Module):
    """Print a summary of the MLP model architecture."""
    print()
    print("═" * 80)
    print("  🧠 STATIC NEURAL NETWORK ARCHITECTURE (GaitStaticNet)")
    print("═" * 80)
    print(f"  {'Module / Block':<32} │ {'Layer type':<24} │ {'Parameters':>15}")
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
    print(f"  {'TOTAL TRAINABLE PARAMETERS':<59} │ {trainable_params:>15,}")
    print("═" * 80)
    print()


def plot_training_curves(history: dict, save_dir: Path):
    """Plot and save the training loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss total
    axes[0].plot(history["train_loss"], label="Train Loss", color="#4472C4", alpha=0.8)
    axes[0].plot(
        history["val_loss"], label="Validation Loss", color="#ED7D31", alpha=0.8
    )
    axes[0].set_title("Total Loss (MSE + Bone)", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale("log")

    # Learning rate
    if "lr" in history and len(history["lr"]) > 0:
        axes[1].plot(history["lr"], color="#70AD47")
        axes[1].set_title("Learning Rate", fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("LR")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale("log")

    plt.suptitle(
        "Training curves – static neural network",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    save_path = save_dir / "training_curves.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ Training curves saved to: {save_path}")


def train_epoch(model, loader, optimizer, epoch, total_epochs):
    model.train()
    total_loss, total_mse, total_bone = 0.0, 0.0, 0.0
    n_batches = len(loader)

    for i, (x, y, anthro) in enumerate(loader):
        x, y, anthro = x.to(DEVICE), y.to(DEVICE), anthro.to(DEVICE)

        optimizer.zero_grad()
        pred = model(x)
        loss, mse, bone = static_loss(pred, y, anthro)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse
        total_bone += bone

        if total_epochs > 0:
            bar_len = 15
            progress = (i + 1) / n_batches
            filled = int(bar_len * progress)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(
                f"\r  Epoch {epoch:3d}/{total_epochs} │{bar}│ Batch {i+1}/{n_batches} │ Train Loss: {total_loss/(i+1):.4f}",
                end="",
                flush=True,
            )

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader):
    model.eval()
    total_loss, total_mse, total_bone = 0.0, 0.0, 0.0
    n_batches = len(loader)

    if n_batches == 0:
        return float("inf")

    for x, y, anthro in loader:
        x, y, anthro = x.to(DEVICE), y.to(DEVICE), anthro.to(DEVICE)
        pred = model(x)
        loss, mse, bone = static_loss(pred, y, anthro)

        total_loss += loss.item()
        total_mse += mse
        total_bone += bone

    return total_loss / n_batches


# ═══════════════════════════════════════════════════════════════════════
#  EVALUATION AND ANALYTICAL COMPARISON
# ═══════════════════════════════════════════════════════════════════════


def compute_mpjpe(
    predictions: np.ndarray, targets: np.ndarray, percentile: float = 100.0
) -> dict[str, float]:
    pred = predictions.reshape(-1, N_JOINTS, N_COORDS)
    tgt = targets.reshape(-1, N_JOINTS, N_COORDS)
    per_joint_error = np.linalg.norm(pred - tgt, axis=-1)  # [N, 6]
    if percentile < 100.0:
        frame_errors = np.mean(per_joint_error, axis=1)
        threshold = np.percentile(frame_errors, percentile)
        mask = frame_errors <= threshold
        per_joint_error = per_joint_error[mask]
    result = {
        name: float(np.mean(per_joint_error[:, j]))
        for j, name in enumerate(JOINT_NAMES)
    }
    result["mean"] = float(np.mean(per_joint_error))
    return result


def compute_mae_per_joint(
    predictions: np.ndarray, targets: np.ndarray, percentile: float = 100.0
) -> dict[str, float]:
    pred = predictions.reshape(-1, N_JOINTS, N_COORDS)
    tgt = targets.reshape(-1, N_JOINTS, N_COORDS)
    diff = np.abs(pred - tgt)
    if percentile < 100.0:
        frame_errors = np.mean(diff, axis=(1, 2))
        threshold = np.percentile(frame_errors, percentile)
        mask = frame_errors <= threshold
        diff = diff[mask]
    result = {name: float(np.mean(diff[:, j, :])) for j, name in enumerate(JOINT_NAMES)}
    result["mean"] = float(np.mean(list(result.values())))
    return result


def compute_rmse_per_joint(
    predictions: np.ndarray, targets: np.ndarray, percentile: float = 100.0
) -> dict[str, float]:
    pred = predictions.reshape(-1, N_JOINTS, N_COORDS)
    tgt = targets.reshape(-1, N_JOINTS, N_COORDS)
    sq_diff = (pred - tgt) ** 2
    if percentile < 100.0:
        frame_errors = np.mean(np.sqrt(np.sum(sq_diff, axis=-1)), axis=1)
        threshold = np.percentile(frame_errors, percentile)
        mask = frame_errors <= threshold
        sq_diff = sq_diff[mask]
    result = {
        name: float(np.sqrt(np.mean(sq_diff[:, j, :])))
        for j, name in enumerate(JOINT_NAMES)
    }
    result["mean"] = float(np.mean(list(result.values())))
    return result


def save_comparison_animation(
    gt_pose: np.ndarray,
    pred_pose: np.ndarray,
    ana_pose: np.ndarray,
    save_path: str,
    max_frames: int = 400,
):
    """Save a GIF animation comparing ArUco, analytical and neural network."""
    import matplotlib.animation as animation

    gt = gt_pose.reshape(-1, N_JOINTS, N_COORDS)
    pred = pred_pose.reshape(-1, N_JOINTS, N_COORDS)
    ana = ana_pose.reshape(-1, N_JOINTS, N_COORDS) if ana_pose is not None else None

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    connections = [("pelvis", 0), ("pelvis", 1), (0, 2), (2, 4), (1, 3), (3, 5)]

    def get_skeleton_lines(pose_frame):
        # The pelvis is no longer 0,0,0; it is the midpoint between hip_r and hip_l
        pelvis = (pose_frame[0] + pose_frame[1]) / 2.0
        pts = np.vstack([pose_frame, pelvis])
        lines = []
        for p1, p2 in connections:
            i1 = 6 if p1 == "pelvis" else p1
            i2 = 6 if p2 == "pelvis" else p2
            lines.append(np.vstack([pts[i1], pts[i2]]))
        return lines

    def update(frame):
        ax.clear()
        ax.set_title(f"Static comparison - Frame {frame}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        # Isometric view
        ax.view_init(elev=20, azim=-45)

        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([-0.2, 1.0])

        lines_gt = get_skeleton_lines(gt[frame])
        for line in lines_gt:
            ax.plot(
                line[:, 0],
                line[:, 1],
                line[:, 2],
                color="#2CA02C",
                linewidth=3,
                label="ArUco (GT)" if line is lines_gt[0] else "",
            )

        if ana is not None:
            lines_ana = get_skeleton_lines(ana[frame])
            for line in lines_ana:
                ax.plot(
                    line[:, 0],
                    line[:, 1],
                    line[:, 2],
                    color="#FF7F0E",
                    linewidth=2,
                    linestyle=":",
                    label="Analytical" if line is lines_ana[0] else "",
                )

        lines_pred = get_skeleton_lines(pred[frame])
        for line in lines_pred:
            ax.plot(
                line[:, 0],
                line[:, 1],
                line[:, 2],
                color="#1F77B4",
                linewidth=2,
                linestyle="--",
                label="Neural Net" if line is lines_pred[0] else "",
            )

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper left")

    n_frames = min(max_frames, len(gt))
    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50)
    print(f"\n  🎥 Generating animation ({n_frames} frames)...")
    ani.save(save_path, writer="pillow", fps=20)
    plt.close(fig)
    print(f"  ✓ Animation saved to: {save_path}")


@torch.no_grad()
def evaluate_static_model(
    model, test_captures, test_file_paths, norm_stats, anim_frames: int = 400
):
    model.eval()

    all_predictions = []
    all_targets = []
    all_ana_predictions = []
    all_ana_targets = []

    print("\n  🔬 Evaluating on the test set and comparing with the analytical method...")

    for cap, (mcd_path, aruco_path) in zip(test_captures, test_file_paths):
        # Infer NN
        sensor_norm = (
            cap["sensor_data"].astype(np.float32) - norm_stats["mean"]
        ) / norm_stats["std"]
        anthro = np.array(
            [
                cap["femur_length"] / GaitStaticDataset.BONE_NORM,
                cap["tibia_length"] / GaitStaticDataset.BONE_NORM,
                cap["hip_width"] / GaitStaticDataset.BONE_NORM,
                1.0 if cap["direction"] == "R" else 0.0,
            ],
            dtype=np.float32,
        )
        anthro_broadcast = np.tile(anthro, (len(sensor_norm), 1))
        pivot_flag = cap["pivot_flag"].reshape(-1, 1)

        x = np.concatenate([sensor_norm, anthro_broadcast, pivot_flag], axis=1)
        x_tensor = torch.from_numpy(x).to(DEVICE)

        pred_pose = model(x_tensor).cpu().numpy()
        gt_pose = cap["gt_relative"]

        all_predictions.append(pred_pose)
        all_targets.append(gt_pose)

        # Analytical pipeline
        try:
            ana_res = process_files((mcd_path, aruco_path), use_3d=False)
            if ana_res is not None:
                _, _, rec, _, _, _, _ = ana_res
                hip_center = (rec["hip_r"] + rec["hip_l"]) / 2.0

                cap_len = len(gt_pose)
                if len(hip_center) >= cap_len:
                    first_valid = cap.get("first_valid", 0)
                    ana_rel = np.zeros((cap_len, N_JOINTS * N_COORDS))

                    pivot_is_right = cap["pivot_flag"]
                    valid_hip_center = hip_center[first_valid : first_valid + cap_len]
                    valid_ankle_r = rec["ankle_r"][first_valid : first_valid + cap_len]
                    valid_ankle_l = rec["ankle_l"][first_valid : first_valid + cap_len]

                    pivot = np.zeros_like(valid_hip_center)
                    pivot[:, 0] = np.where(
                        pivot_is_right == 1.0, valid_ankle_r[:, 0], valid_ankle_l[:, 0]
                    )
                    pivot[:, 1] = valid_hip_center[:, 1]
                    pivot[:, 2] = np.where(
                        pivot_is_right == 1.0, valid_ankle_r[:, 2], valid_ankle_l[:, 2]
                    )

                    for j_idx, j_name in enumerate(JOINT_NAMES):
                        valid_j = rec[j_name][first_valid : first_valid + cap_len]
                        ana_rel[:, j_idx * 3 : (j_idx + 1) * 3] = valid_j - pivot

                    # Avoid exact-length errors if the filter shortened the analytical output too much
                    if len(ana_rel) == cap_len:
                        all_ana_predictions.append(ana_rel)
                        all_ana_targets.append(gt_pose)
        except Exception:
            pass

    if len(all_predictions) == 0:
        return

    all_pred = np.concatenate(all_predictions, axis=0)
    all_tgt = np.concatenate(all_targets, axis=0)

    mpjpe = compute_mpjpe(all_pred, all_tgt)
    mpjpe_p95 = compute_mpjpe(all_pred, all_tgt, percentile=95.0)
    mae = compute_mae_per_joint(all_pred, all_tgt)
    mae_p95 = compute_mae_per_joint(all_pred, all_tgt, percentile=95.0)
    rmse = compute_rmse_per_joint(all_pred, all_tgt)
    rmse_p95 = compute_rmse_per_joint(all_pred, all_tgt, percentile=95.0)

    has_ana = len(all_ana_predictions) > 0
    if has_ana:
        all_ana_pred = np.concatenate(all_ana_predictions, axis=0)
        all_ana_tgt = np.concatenate(all_ana_targets, axis=0)
        mpjpe_ana = compute_mpjpe(all_ana_pred, all_ana_tgt)
        mae_ana = compute_mae_per_joint(all_ana_pred, all_ana_tgt)
        rmse_ana = compute_rmse_per_joint(all_ana_pred, all_ana_tgt)

    print()
    print("═" * 80)
    title = (
        "  📊 COMPARATIVE EVALUATION – Test Set (NN vs analytical pipeline)"
        if has_ana
        else "  📊 EVALUATION – Test Set"
    )
    print(title)
    print("═" * 80)

    print(f"\n  1) MPJPE (Mean Per Joint Position Error)")
    print(f"  {'─' * 70}")
    if has_ana:
        print(
            f"     {'Joint':<14}  {'Neural network (NN)':>20}  {'Analytical pipeline':>20}  {'Winner':>12}"
        )
        print(f"     {'─' * 14}  {'─' * 20}  {'─' * 20}  {'─' * 12}")
        win_m = "✅ NN" if mpjpe["mean"] < mpjpe_ana["mean"] else "📐 Analytical"
        print(
            f"     {'General':<14}  {mpjpe['mean']:>18.4f} m  {mpjpe_ana['mean']:>18.4f} m  {win_m:>12}"
        )
        print(
            f"     {'General (P95)':<14}  {mpjpe_p95['mean']:>18.4f} m  {'--':>18}    {'--':>12}"
        )
        for name in JOINT_NAMES:
            win_j = "✅ NN" if mpjpe[name] < mpjpe_ana[name] else "📐 Analytical"
            print(
                f"     {name:<14}  {mpjpe[name]:>18.4f} m  {mpjpe_ana[name]:>18.4f} m  {win_j:>12}"
            )
    else:
        print(f"     MPJPE general:        {mpjpe['mean']:.4f} m")
        print(f"     MPJPE general (P95):  {mpjpe_p95['mean']:.4f} m")
        for name in JOINT_NAMES:
            print(f"     {name:<14}  {mpjpe[name]:.4f} m")

    print(f"\n  2) MAE (Mean Absolute Error per joint)")
    print(f"  {'─' * 70}")
    if has_ana:
        print(
            f"     {'Joint':<14}  {'Neural network (NN)':>20}  {'Analytical pipeline':>20}  {'Winner':>12}"
        )
        print(f"     {'─' * 14}  {'─' * 20}  {'─' * 20}  {'─' * 12}")
        win_mae = "✅ NN" if mae["mean"] < mae_ana["mean"] else "📐 Analytical"
        print(
            f"     {'General':<14}  {mae['mean']:>18.4f} m  {mae_ana['mean']:>18.4f} m  {win_mae:>12}"
        )
        print(
            f"     {'General (P95)':<14}  {mae_p95['mean']:>18.4f} m  {'--':>18}    {'--':>12}"
        )
        for name in JOINT_NAMES:
            win_j = "✅ NN" if mae[name] < mae_ana[name] else "📐 Analytical"
            print(
                f"     {name:<14}  {mae[name]:>18.4f} m  {mae_ana[name]:>18.4f} m  {win_j:>12}"
            )
    else:
        print(f"     MAE general:        {mae['mean']:.4f} m")
        print(f"     MAE general (P95):  {mae_p95['mean']:.4f} m")
        for name in JOINT_NAMES:
            print(f"     {name:<14}  {mae[name]:.4f} m")

    print(f"\n  3) RMSE per joint")
    print(f"  {'─' * 70}")
    if has_ana:
        print(
            f"     {'Joint':<14}  {'Neural network (NN)':>20}  {'Analytical pipeline':>20}  {'Winner':>12}"
        )
        print(f"     {'─' * 14}  {'─' * 20}  {'─' * 20}  {'─' * 12}")
        win_rmse = "✅ NN" if rmse["mean"] < rmse_ana["mean"] else "📐 Analytical"
        print(
            f"     {'General':<14}  {rmse['mean']:>18.4f} m  {rmse_ana['mean']:>18.4f} m  {win_rmse:>12}"
        )
        print(
            f"     {'General (P95)':<14}  {rmse_p95['mean']:>18.4f} m  {'--':>18}    {'--':>12}"
        )
        for name in JOINT_NAMES:
            win_j = "✅ NN" if rmse[name] < rmse_ana[name] else "📐 Analytical"
            print(
                f"     {name:<14}  {rmse[name]:>18.4f} m  {rmse_ana[name]:>18.4f} m  {win_j:>12}"
            )
    else:
        print(f"     RMSE general:        {rmse['mean']:.4f} m")
        print(f"     RMSE general (P95):  {rmse_p95['mean']:.4f} m")
        for name in JOINT_NAMES:
            print(f"     {name:<14}  {rmse[name]:.4f} m")

    if has_ana:
        imp_mpjpe = ((mpjpe_ana["mean"] - mpjpe["mean"]) / mpjpe_ana["mean"]) * 100
        imp_mae = ((mae_ana["mean"] - mae["mean"]) / mae_ana["mean"]) * 100
        imp_rmse = ((rmse_ana["mean"] - rmse["mean"]) / rmse_ana["mean"]) * 100

        print("\n  🏆 Global relative improvements (NN vs analytical):")
        print(f"     • MPJPE: {imp_mpjpe:+.2f}%")
        print(f"     • MAE:   {imp_mae:+.2f}%")
        print(f"     • RMSE:  {imp_rmse:+.2f}%")

    print("═" * 80)

    # Save the consolidated metrics
    final_metrics = {
        "mpjpe": mpjpe,
        "mpjpe_p95": mpjpe_p95,
        "mae": mae,
        "mae_p95": mae_p95,
        "rmse": rmse,
        "rmse_p95": rmse_p95,
    }
    if has_ana:
        final_metrics["mpjpe_ana"] = mpjpe_ana
        final_metrics["mae_ana"] = mae_ana
        final_metrics["rmse_ana"] = rmse_ana
        final_metrics["improvements"] = {
            "mpjpe": imp_mpjpe,
            "mae": imp_mae,
            "rmse": imp_rmse,
        }
    with open(RESULTS_DIR / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"  ✅ Final metrics saved to: {RESULTS_DIR / 'final_metrics.json'}")

    # Generate the animation for the longest capture of the test set
    if len(all_predictions) > 0:
        best_idx = int(np.argmax([len(t) for t in all_targets]))
        gt_sample = all_targets[best_idx]
        pred_sample = all_predictions[best_idx]
        ana_sample = (
            all_ana_predictions[best_idx]
            if (has_ana and best_idx < len(all_ana_predictions))
            else None
        )

        anim_path = RESULTS_DIR / "skeleton_comparison.gif"
        save_comparison_animation(
            gt_sample, pred_sample, ana_sample, str(anim_path), max_frames=anim_frames
        )

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Static gait reconstruction with an MLP"
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE, help="Initial learning rate"
    )
    parser.add_argument(
        "--patience", type=int, default=PATIENCE, help="Early-stopping patience"
    )
    parser.add_argument(
        "--dataset", type=str, default=str(DATASET_PATH), help="Path to the dataset"
    )
    parser.add_argument(
        "--n-val-subjects",
        type=int,
        default=3,
        help="Number of validation subjects",
    )
    parser.add_argument(
        "--n-test-subjects", type=int, default=5, help="Number of test subjects"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--eval-only", action="store_true", help="Evaluation only, with the saved model"
    )
    parser.add_argument(
        "--checkpoint-tag",
        type=str,
        default="",
        help="Tag to identify the checkpoint (default: '')",
    )
    parser.add_argument(
        "--anim-frames",
        type=int,
        default=400,
        help="Maximum number of frames for the GIF animation (default: 400)",
    )
    parser.add_argument(
        "--data-split",
        type=str,
        default=str(DATA_SPLIT_PATH),
        help="Path to the JSON file with the subject split (default: data_split.json)",
    )
    args = parser.parse_args()

    # Set the seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("═" * 80)
    print("  🏃 STATIC RECONSTRUCTION (NO SEQUENCE)")
    print("═" * 80)
    print(f"  Dataset:       {args.dataset}")
    print(f"  Data split:    {args.data_split}")
    print(f"  Device:        {DEVICE}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    if args.checkpoint_tag:
        print(f"  Checkpoint tag:{args.checkpoint_tag}")
    print("═" * 80)

    dataset_path = Path(args.dataset)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n  📂 Discovering captures...")
    captures_info = discover_captures(dataset_path)

    if len(captures_info) == 0:
        print("  [ERROR] No files found")
        return

    print("\n  ⚙️ Loading static captures...")
    subjects_data = {}
    file_paths_by_subject = {}

    total_captures = len(captures_info)
    success_count = 0
    failed_count = 0
    bar_len = 30

    for idx, (mcd_path, aruco_path, subj) in enumerate(captures_info, start=1):
        cap = prepare_static_capture(mcd_path, aruco_path)
        if cap is not None:
            if subj not in subjects_data:
                subjects_data[subj] = []
                file_paths_by_subject[subj] = []
            subjects_data[subj].append(cap)
            file_paths_by_subject[subj].append((mcd_path, aruco_path))
            success_count += 1
        else:
            failed_count += 1

        progress = idx / total_captures
        filled = int(bar_len * progress)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(
            f"\r  Loading captures │{bar}│ {idx}/{total_captures} ({progress*100:.1f}%) │ ✓ {success_count} │ ✗ {failed_count}",
            end="",
            flush=True,
        )

    print(f"\n  ✓ Captures loaded successfully: {success_count}")

    split_path = Path(args.data_split) if getattr(args, "data_split", None) else None
    if split_path and not split_path.exists():
        alt_path = Path(__file__).resolve().parent / split_path
        if alt_path.exists():
            split_path = alt_path

    loaded_from_split = False
    if split_path and split_path.exists():
        try:
            with open(split_path, "r") as f:
                split_info = json.load(f)
            cand_train = [
                str(s)
                for s in split_info.get("train_subjects", [])
                if str(s) in subjects_data
            ]
            cand_val = [
                str(s)
                for s in split_info.get("val_subjects", [])
                if str(s) in subjects_data
            ]
            cand_test = [
                str(s)
                for s in split_info.get("test_subjects", [])
                if str(s) in subjects_data
            ]
            if len(cand_train) > 0 and len(cand_val) > 0 and len(cand_test) > 0:
                train_subjs = cand_train
                val_subjs = cand_val
                test_subjs = cand_test
                loaded_from_split = True
                print(f"\n  📊 Dataset split by subject (from {split_path}):")
            else:
                print(
                    f"\n  ⚠️ [WARN] The file {split_path} does not contain enough valid subjects. An automatic split will be used."
                )
        except Exception as e:
            print(
                f"\n  ⚠️ [WARN] Error reading {split_path}: {e}. An automatic split will be used."
            )

    if not loaded_from_split:
        subjects = list(subjects_data.keys())
        random.shuffle(subjects)

        n_test = min(args.n_test_subjects, max(1, len(subjects) - 2))
        n_val = min(args.n_val_subjects, max(1, len(subjects) - n_test - 1))

        test_subjs = subjects[-n_test:]
        val_subjs = subjects[-(n_test + n_val) : -n_test]
        train_subjs = subjects[: -(n_test + n_val)]
        print("\n  📊 Dataset split:")

    # Save the split used
    split_record = {
        "source": str(split_path) if loaded_from_split else "auto_generated",
        "train_subjects": [int(s) if s.isdigit() else s for s in train_subjs],
        "val_subjects": [int(s) if s.isdigit() else s for s in val_subjs],
        "test_subjects": [int(s) if s.isdigit() else s for s in test_subjs],
    }
    with open(RESULTS_DIR / "data_split_used.json", "w") as f:
        json.dump(split_record, f, indent=2)

    train_caps = [c for s in train_subjs for c in subjects_data[s]]
    val_caps = [c for s in val_subjs for c in subjects_data[s]]

    test_caps = []
    test_file_paths = []
    for s in test_subjs:
        for c, fpath in zip(subjects_data[s], file_paths_by_subject[s]):
            test_caps.append(c)
            test_file_paths.append(fpath)

    print(
        f"    Train ({len(train_subjs)} subjects): {train_subjs} -> {len(train_caps)} captures"
    )
    print(
        f"    Val   ({len(val_subjs)} subjects): {val_subjs} -> {len(val_caps)} captures"
    )
    print(f"    Test  ({len(test_subjs)} subjects): {test_subjs}")

    print("\n  📦 Preparing tensors (Dataset)...")
    train_dataset = GaitStaticDataset(train_caps)
    norm_stats = train_dataset.get_stats()
    val_dataset = GaitStaticDataset(val_caps, stats=norm_stats)

    # Configure the DataLoader with multi-threading and pin_memory
    num_workers = 4 if torch.cuda.is_available() else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    model = GaitStaticNet().to(DEVICE)

    # Actual training configuration, saved inside the .pt checkpoint
    train_config = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
        "cutoff_freq": CUTOFF_FREQ,
        "filter_order": FILTER_ORDER,
        "train_subjects": train_subjs,
        "val_subjects": val_subjs,
        "test_subjects": test_subjs,
        "hidden_layers": HIDDEN_LAYERS,
        "dropout_rates": DROPOUT_RATES,
    }
    print_model_summary(model)

    ckpt_name = (
        f"best_static_model_{args.checkpoint_tag}.pt"
        if args.checkpoint_tag
        else CHECKPOINT_NAME
    )
    ckpt_path = CHECKPOINT_DIR / ckpt_name

    if args.eval_only:
        if not ckpt_path.exists():
            print(f"  [ERROR] No checkpoint found at {ckpt_path}")
            return
        print(f"\n  📥 Loading GaitStaticNet from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "norm_stats" in checkpoint:
            norm_stats = checkpoint["norm_stats"]
    else:
        # Enable the fused kernel if available (PyTorch 2.0+)
        fused_opt = torch.cuda.is_available()
        try:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=WEIGHT_DECAY,
                fused=fused_opt,
            )
        except TypeError:
            # Fallback if PyTorch is old and does not support 'fused'
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY
            )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0

        history = {"train_loss": [], "val_loss": [], "lr": []}

        print("\n  🚀 Starting training...")
        t_start = time.time()
        for epoch in range(1, args.epochs + 1):
            t_epoch = time.time()
            tr_loss = train_epoch(model, train_loader, optimizer, epoch, args.epochs)
            val_loss = validate(model, val_loader)
            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(val_loss)
            history["lr"].append(current_lr)

            elapsed = time.time() - t_epoch

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "norm_stats": norm_stats,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "config": train_config,
                    },
                    ckpt_path,
                )
                print(
                    f"\r  Epoch {epoch:3d}/{args.epochs} │ Train: {tr_loss:.4f} │ Val: {val_loss:.4f} ★ │ LR: {current_lr:.1e} │ {elapsed:.1f}s"
                )
            else:
                patience_counter += 1
                print(
                    f"\r  Epoch {epoch:3d}/{args.epochs} │ Train: {tr_loss:.4f} │ Val: {val_loss:.4f}   │ LR: {current_lr:.1e} │ {elapsed:.1f}s",
                    end="",
                )
                if epoch % 10 == 0:
                    print("", flush=True)
                # Warm restart (LR warm-up) at half the patience
                if patience_counter == args.patience // 2:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = args.lr
                    scheduler._reset()  # Reset the ReduceLROnPlateau counter

                if patience_counter >= args.patience:
                    print(
                        f"\n  ⏹ Early stopping at epoch {epoch} (best epoch: {best_epoch})"
                    )
                    break

        total_time = time.time() - t_start
        print(f"\n  ✅ Training completed in {total_time / 60:.1f} min")
        print(f"  Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")

        # Save the history and plot
        with open(RESULTS_DIR / "training_history.json", "w") as f:
            json.dump(history, f, indent=4)

        plot_training_curves(history, RESULTS_DIR)

        # Load the best saved model for evaluation
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate the best model on the test set
    evaluate_static_model(
        model, test_caps, test_file_paths, norm_stats, anim_frames=args.anim_frames
    )


if __name__ == "__main__":
    main()
