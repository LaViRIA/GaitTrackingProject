"""
Neural network: pivot-foot classifier (Pivot Classifier)
===========================================================
Architecture:
  - Multi-layer feed-forward network (MLP) with BatchNorm + Dropout + LeakyReLU.
  - Optional alternative model: non-causal 1D TCN for temporal classification (GaitTCNPivotNet).
  - Input: concatenated vector of instantaneous IMU readings (24 features: accel, gyro of 4 sensors) + 4 anthropometric measurements.
  - Output: binary sigmoid probability (1.0 = right foot in stance, 0.0 = left foot in stance).
  - Loss function: Binary Cross Entropy with L2 regularization.

Purpose:
  - Determine the stance (pivot) foot with high accuracy at every time instant, to allow
    accumulating strides and reconstructing the global displacement without optical sensors.

Output directory: nn-results/pivot_classifier/
Checkpoints: nn-results/pivot_classifier/checkpoints/best_pivot_classifier.pt

Usage:
    python -m models.nn_pivot_classifier                         # Training + evaluation
    python -m models.nn_pivot_classifier --eval-only             # Evaluation only

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

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

# Import the existing data-loading utilities
from process_all import (
    read_mcd_file,
    read_aruco_file,
    adjust_aruco_data,
    low_pass_filter,
)

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION AND CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

DATASET_PATH = Path("DataSet")  # default: DataSet/ in the repository root (override with --dataset)
DATA_SPLIT_PATH = (
    Path(__file__).resolve().parent / "data_split.json"
    if (Path(__file__).resolve().parent / "data_split.json").exists()
    else Path("data_split.json")
)
RESULTS_DIR = Path("nn-results/pivot_classifier")
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINT_NAME = "best_pivot_classifier.pt"
CHECKPOINT_PATH = CHECKPOINT_DIR / CHECKPOINT_NAME
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Filtering and data parameters ─────────────────────────────────────
CUTOFF_FREQ = 7  # Hz, low-pass filter of the MCD
FILTER_ORDER = 6  # Butterworth filter order

# ── Training ─────────────────────────────────────────────────────────
BATCH_SIZE = 1024
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
PATIENCE = 25
GRAD_CLIP = 1.0
BONE_NORM = 0.45

# ── Channels and dimensions ───────────────────────────────────────────
IMU_NAMES = ["RF", "LF", "RT", "LT"]
SENSOR_CHANNELS = len(IMU_NAMES) * 6  # 24 channels (Acc X,Y,Z + Gyro X,Y,Z for 4 IMUs)
ANTHROPOMETRIC_DIM = 4  # Femur, Tibia, Hip Width, Direction
INPUT_DIM = SENSOR_CHANNELS + ANTHROPOMETRIC_DIM  # 28
OUTPUT_DIM = 1  # Binary logit (1.0 = right foot, 0.0 = left foot)

# ── MLP architecture (static feed-forward) ────────────────────────────
HIDDEN_LAYERS = [16, 8, 8]
DROPOUT_RATES = [0.15, 0.15, 0.15]


# ═══════════════════════════════════════════════════════════════════════
#  DATA PREPARATION AND GROUND-TRUTH EXTRACTION
# ═══════════════════════════════════════════════════════════════════════


def discover_captures(dataset_path: Path) -> list[tuple[str, str, str]]:
    """Find the mcd-i.csv and arUcos-i.csv file pairs of each subject."""
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


def prepare_pivot_capture(mcd_path: str, aruco_path: str) -> dict | None:
    """
    Process an MCD/ArUco pair to produce pivot-foot classification data.
    Ground Truth:
        score_r = z_ankle_r + 0.1 * v_ankle_r
        score_l = z_ankle_l + 0.1 * v_ankle_l
        pivot_flag = 1.0 if score_r < score_l (right) else 0.0 (left)
    """
    try:
        mcd_data = read_mcd_file(
            mcd_path, cutoff_freq=CUTOFF_FREQ, filter_order=FILTER_ORDER
        )
        aruco_data = read_aruco_file(aruco_path)
        direction = mcd_data["measures"]["direction"]

        # Shift ArUco to the hip origin
        aruco_data = adjust_aruco_data(aruco_data, direction)

        aruco_processed = {
            "hip": aruco_data["hip_position"],
            "ankle_r": aruco_data["R_ankle_position"],
            "ankle_l": aruco_data["L_ankle_position"],
        }

        # Filter the noise in Y
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

        # Interpolate ArUco to the MCD time base
        aruco_raw_keys = ["hip", "ankle_r", "ankle_l"]
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

        joints_at_mcd = {}
        for key in ["ankle_r", "ankle_l"]:
            aruco_t = aruco_processed[key]["time"].astype(float)
            aruco_pos = aruco_processed[key]["data"]
            joint_interp = [
                interp1d(aruco_t, aruco_pos[:, i], kind="linear") for i in range(3)
            ]
            joints_at_mcd[key] = np.column_stack(
                [joint_interp[i](valid_mcd_time) for i in range(3)]
            )

        # Ground-truth extraction for the pivot (Z and velocity)
        z_r = joints_at_mcd["ankle_r"][:, 2]
        z_l = joints_at_mcd["ankle_l"][:, 2]

        dt = np.gradient(valid_mcd_time)
        dt[dt == 0] = 1e-6

        v_r = np.linalg.norm(
            np.gradient(joints_at_mcd["ankle_r"], axis=0) / dt[:, np.newaxis], axis=1
        )
        v_l = np.linalg.norm(
            np.gradient(joints_at_mcd["ankle_l"], axis=0) / dt[:, np.newaxis], axis=1
        )

        weight_v = 0.1
        score_r = z_r + weight_v * v_r
        score_l = z_l + weight_v * v_l

        # 1.0 if the right foot is the pivot, 0.0 if it is the left one
        pivot_is_right = (score_r < score_l).astype(np.float32)

        # Anthropometry
        femur_length = float(mcd_data["measures"].get("femur_length", 0.4))
        tibia_length = float(mcd_data["measures"].get("tibia_length", 0.4))
        hip_width = float(mcd_data["measures"].get("hip_width", 0.25))

        first_valid = int(np.argmax(valid_time_mask))
        last_valid = N - int(np.argmax(valid_time_mask[::-1])) - 1

        return {
            "sensor_data": sensor_data[first_valid : last_valid + 1],
            "pivot_flag": pivot_is_right,
            "femur_length": femur_length,
            "tibia_length": tibia_length,
            "hip_width": hip_width,
            "direction": direction,
            "mcd_time": valid_mcd_time,
            "first_valid": first_valid,
        }
    except Exception as e:
        print(f"  [ERROR] {Path(mcd_path).name}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
#  DATASET PYTORCH
# ═══════════════════════════════════════════════════════════════════════


class GaitPivotDataset(Dataset):
    """
    Static dataset for binary pivot-foot classification.
    Input: [normalized_sensors (24), normalized_anthropometry (4)] -> 28 features
    Label: [pivot_flag (1)] (1.0 = right, 0.0 = left)
    """

    def __init__(
        self, captures: list[dict], normalize: bool = True, stats: dict | None = None
    ):
        self.inputs = []
        self.targets = []
        all_sensors = []

        for cap in captures:
            sensor = cap["sensor_data"].astype(np.float32)
            pivot_flag = cap["pivot_flag"].astype(np.float32).reshape(-1, 1)

            anthro = np.array(
                [
                    cap["femur_length"] / BONE_NORM,
                    cap["tibia_length"] / BONE_NORM,
                    cap["hip_width"] / BONE_NORM,
                    1.0 if cap["direction"] == "R" else 0.0,
                ],
                dtype=np.float32,
            )

            anthro_broadcast = np.tile(anthro, (len(sensor), 1))

            self.inputs.append((sensor, anthro_broadcast))
            self.targets.append(pivot_flag)
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

        if len(self.sensor_data) > 0:
            if self.normalize:
                self.sensor_data = (self.sensor_data - self.mean) / self.std

            x_np = np.concatenate([self.sensor_data, self.anthro_data], axis=1)
            self.x = torch.from_numpy(x_np).float()
            self.y = torch.from_numpy(self.target_data).float()
        else:
            self.x = torch.tensor([])
            self.y = torch.tensor([])

    def get_stats(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ═══════════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════


class GaitStaticPivotNet(nn.Module):
    """
    Feed-forward MLP that predicts the pivot foot instant by instant.
    Input: 28 features (24 IMU + 4 anthropometry).
    Output: scalar logit (1 value without sigmoid, used with BCEWithLogitsLoss).
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


class GaitTCNPivotNet(nn.Module):
    """
    Optional temporal TCN model to process sequences if needed.
    """

    def __init__(self, input_dim=INPUT_DIM, hidden_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.conv1 = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=5, padding=2, dilation=1
        )
        self.conv2 = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=5, padding=4, dilation=2
        )
        self.conv3 = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=5, padding=8, dilation=4
        )

        self.out = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        h = self.proj(x)
        h = h.transpose(1, 2)
        h = F.gelu(self.conv1(h))
        h = F.gelu(self.conv2(h))
        h = F.gelu(self.conv3(h))
        h = h.transpose(1, 2)
        return self.out(h).squeeze(-1)


def print_model_summary(model: nn.Module):
    """Print a summary of the neural model architecture and its size in parameters and KB/MB."""
    print()
    print("═" * 80)
    print("  🧠 CLASSIFIER NETWORK ARCHITECTURE (GaitStaticPivotNet)")
    print("═" * 80)
    print(
        f"  {'#':<4} │ {'Module / Layer':<28} │ {'Layer type':<20} │ {'Parameters':>15}"
    )
    print("  " + "─" * 76)

    total_params = 0
    trainable_params = 0

    layers_to_show = (
        list(model.net.named_children())
        if hasattr(model, "net") and isinstance(model.net, nn.Sequential)
        else list(model.named_children())
    )

    for idx, (name, child) in enumerate(layers_to_show, start=1):
        p_count = sum(p.numel() for p in child.parameters())
        p_train = sum(p.numel() for p in child.parameters() if p.requires_grad)
        total_params += p_count
        trainable_params += p_train
        layer_type = child.__class__.__name__
        print(f"  {idx:<4} │ {name:<28} │ {layer_type:<20} │ {p_count:>15,}")

    param_bytes = total_params * 4  # float32 = 4 bytes
    param_kb = param_bytes / 1024
    param_mb = param_kb / 1024

    print("  " + "─" * 76)
    print(f"  {'TOTAL TRAINABLE PARAMETERS':<56} │ {trainable_params:>15,}")
    print(
        f"  {'APPROXIMATE MEMORY SIZE (Float32)':<56} │ {param_kb:>12.2f} KB ({param_mb:.4f} MB)"
    )
    print("═" * 80)
    print()


# ═══════════════════════════════════════════════════════════════════════
#  TRAINING AND VALIDATION
# ═══════════════════════════════════════════════════════════════════════


def train_epoch(model, loader, optimizer, criterion, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    correct_preds = 0
    total_samples = 0
    n_batches = len(loader)

    for i, (x, y) in enumerate(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item() * len(y)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct_preds += (preds == y).sum().item()
        total_samples += len(y)

        if total_epochs > 0 and (i + 1) % max(1, n_batches // 5) == 0:
            progress = (i + 1) / n_batches
            bar_len = 15
            filled = int(bar_len * progress)
            bar = "█" * filled + "░" * (bar_len - filled)
            acc = correct_preds / total_samples
            print(
                f"\r  Epoch {epoch:3d}/{total_epochs} │{bar}│ Loss: {total_loss/total_samples:.4f} │ Acc: {acc*100:.2f}%",
                end="",
                flush=True,
            )

    avg_loss = total_loss / total_samples
    avg_acc = correct_preds / total_samples
    return avg_loss, avg_acc


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct_preds = 0
    total_samples = 0

    if len(loader) == 0:
        return float("inf"), 0.0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * len(y)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct_preds += (preds == y).sum().item()
        total_samples += len(y)

    avg_loss = total_loss / total_samples
    avg_acc = correct_preds / total_samples
    return avg_loss, avg_acc


# ═══════════════════════════════════════════════════════════════════════
#  EVALUATION AND PLOTTING
# ═══════════════════════════════════════════════════════════════════════


def plot_training_curves(history: dict, save_dir: Path):
    """Plot and save the training loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(
        history["train_loss"], label="Train Loss", color="#4472C4", linewidth=2
    )
    axes[0].plot(history["val_loss"], label="Val Loss", color="#ED7D31", linewidth=2)
    axes[0].set_title("Binary loss (BCE loss)", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(
        np.array(history["train_acc"]) * 100,
        label="Train Accuracy",
        color="#70AD47",
        linewidth=2,
    )
    axes[1].plot(
        np.array(history["val_acc"]) * 100,
        label="Val Accuracy",
        color="#FFC000",
        linewidth=2,
    )
    axes[1].set_title("Classification accuracy (%)", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        "Training curves – pivot-foot neural classifier",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    save_path = save_dir / "training_curves.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ Training curves saved to: {save_path}")


def evaluate_and_plot_metrics(model, test_captures, norm_stats, save_dir: Path):
    """Evaluate the network on the test set and produce the confusion matrix, ROC curve and time series."""
    model.eval()

    all_targets = []
    all_probs = []
    subject_results = {}

    for cap in test_captures:
        sensor_norm = (
            cap["sensor_data"].astype(np.float32) - norm_stats["mean"]
        ) / norm_stats["std"]
        anthro = np.array(
            [
                cap["femur_length"] / BONE_NORM,
                cap["tibia_length"] / BONE_NORM,
                cap["hip_width"] / BONE_NORM,
                1.0 if cap["direction"] == "R" else 0.0,
            ],
            dtype=np.float32,
        )
        anthro_broadcast = np.tile(anthro, (len(sensor_norm), 1))

        x = np.concatenate([sensor_norm, anthro_broadcast], axis=1)
        x_tensor = torch.from_numpy(x).to(DEVICE)

        with torch.no_grad():
            logits = model(x_tensor).cpu()
            probs = torch.sigmoid(logits).numpy().flatten()

        tgt = cap["pivot_flag"].flatten()
        all_targets.append(tgt)
        all_probs.append(probs)

    if len(all_targets) == 0:
        print("  [WARNING] No test captures to evaluate.")
        return

    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs)
    y_pred = (y_prob >= 0.5).astype(np.float32)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    print()
    print("═" * 80)
    print("  📊 TEST SET EVALUATION – PIVOT-FOOT CLASSIFIER")
    print("═" * 80)
    print(f"  • Accuracy:              {acc * 100:.2f}%")
    print(f"  • Precision:             {prec * 100:.2f}%")
    print(f"  • Recall (sensitivity):  {rec * 100:.2f}%")
    print(f"  • F1-Score:             {f1 * 100:.2f}%")
    print(f"  • ROC area (ROC-AUC):    {auc:.4f}")
    print("  " + "─" * 50)
    print("  Confusion matrix:")
    print(f"    TN: {cm[0,0]:>7d} | FP: {cm[0,1]:>7d}")
    print(f"    FN: {cm[1,0]:>7d} | TP: {cm[1,1]:>7d}")
    print("═" * 80)

    # ── Plot: confusion matrix and ROC curve ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion matrix heatmap
    im = axes[0].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    axes[0].set_title("Confusion matrix", fontweight="bold")
    fig.colorbar(im, ax=axes[0])
    tick_marks = [0, 1]
    classes = ["Left (0)", "Right (1)"]
    axes[0].set_xticks(tick_marks)
    axes[0].set_xticklabels(classes)
    axes[0].set_yticks(tick_marks)
    axes[0].set_yticklabels(classes)
    axes[0].set_ylabel("Ground Truth")
    axes[0].set_xlabel("Prediction")

    for i in range(2):
        for j in range(2):
            axes[0].text(
                j,
                i,
                f"{cm[i, j]:,}",
                horizontalalignment="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontweight="bold",
            )

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[1].plot(fpr, tpr, color="#4472C4", linewidth=2, label=f"ROC (AUC = {auc:.4f})")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--")
    axes[1].set_title("ROC curve", fontweight="bold")
    axes[1].set_xlabel("False positive rate (FPR)")
    axes[1].set_ylabel("True positive rate (TPR)")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    metrics_path = save_dir / "evaluation_metrics.png"
    fig.savefig(metrics_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Confusion matrix and ROC curve saved to: {metrics_path}")

    # ── Plot: timeline of the first test capture ──────────────────────
    if len(test_captures) > 0:
        cap_sample = test_captures[0]
        tgt_sample = cap_sample["pivot_flag"]
        prob_sample = all_probs[0][: len(tgt_sample)]
        pred_sample = (prob_sample >= 0.5).astype(np.float32)

        fig, ax = plt.subplots(figsize=(14, 4))
        frames = np.arange(len(tgt_sample))
        ax.plot(
            frames,
            tgt_sample,
            label="Ground Truth ArUco",
            color="#2CA02C",
            linewidth=2,
            alpha=0.8,
        )
        ax.plot(
            frames,
            prob_sample,
            label="Predicted probability (right foot)",
            color="#1F77B4",
            linestyle="--",
            linewidth=1.5,
        )
        ax.step(
            frames,
            pred_sample,
            label="Predicted binary classification",
            color="#D62728",
            linestyle=":",
            linewidth=2,
            where="mid",
        )

        ax.set_title(
            "Pivot-foot prediction vs ground truth (time series)",
            fontweight="bold",
        )
        ax.set_xlabel("Frame (instant)")
        ax.set_ylabel("Stance state (1 = right, 0 = left)")
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["Left", "Prob 0.5", "Right"])
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        timeline_path = save_dir / "pivot_timeline_test.png"
        plt.tight_layout()
        fig.savefig(timeline_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Example time series saved to: {timeline_path}")

    # Save the consolidated metrics
    final_metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist(),
    }
    with open(save_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"  ✅ Final metrics saved to: {save_dir / 'final_metrics.json'}")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN PROGRAM
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Training of the pivot-foot neural classifier"
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
        "--data-split",
        type=str,
        default=str(DATA_SPLIT_PATH),
        help="Path to the JSON file with the subject split (default: data_split.json)",
    )
    args = parser.parse_args()

    # Set the random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("═" * 80)
    print("  🦶 PIVOT-FOOT NEURAL CLASSIFIER (STATIC, INSTANT BY INSTANT)")
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
        print("  [ERROR] No valid captures found.")
        return

    print("\n  ⚙️ Processing captures and generating the ArUco ground truth...")
    subjects_data = {}
    total_captures = len(captures_info)
    success_count = 0
    failed_count = 0
    bar_len = 30

    for idx, (mcd_path, aruco_path, subj) in enumerate(captures_info, start=1):
        cap = prepare_pivot_capture(mcd_path, aruco_path)
        if cap is not None:
            if subj not in subjects_data:
                subjects_data[subj] = []
            subjects_data[subj].append(cap)
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

    print(f"\n  ✓ Captures processed successfully: {success_count}")

    # Split by subject (avoids data leakage)
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
        print("\n  📊 Dataset split by subject:")

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
    test_caps = [c for s in test_subjs for c in subjects_data[s]]

    print(
        f"    Train ({len(train_subjs)} subjects): {train_subjs} -> {len(train_caps)} captures"
    )
    print(
        f"    Val   ({len(val_subjs)} subjects): {val_subjs} -> {len(val_caps)} captures"
    )
    print(
        f"    Test  ({len(test_subjs)} subjects): {test_subjs} -> {len(test_caps)} captures"
    )

    print("\n  📦 Preparing tensors (Dataset)...")
    train_dataset = GaitPivotDataset(train_caps)
    norm_stats = train_dataset.get_stats()
    val_dataset = GaitPivotDataset(val_caps, stats=norm_stats)

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

    model = GaitStaticPivotNet().to(DEVICE)

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
        f"best_pivot_classifier_{args.checkpoint_tag}.pt"
        if args.checkpoint_tag
        else CHECKPOINT_NAME
    )
    ckpt_path = CHECKPOINT_DIR / ckpt_name

    if args.eval_only:
        if not ckpt_path.exists():
            print(f"  [ERROR] No checkpoint found at {ckpt_path}")
            return
        print(f"\n  📥 Loading GaitStaticPivotNet from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "norm_stats" in checkpoint:
            norm_stats = checkpoint["norm_stats"]
    else:
        fused_opt = torch.cuda.is_available()
        try:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=WEIGHT_DECAY,
                fused=fused_opt,
            )
        except TypeError:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY
            )

        criterion = nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }

        print("\n  🚀 Starting training...")
        t_start = time.time()

        for epoch in range(1, args.epochs + 1):
            t_epoch = time.time()
            tr_loss, tr_acc = train_epoch(
                model, train_loader, optimizer, criterion, epoch, args.epochs
            )
            val_loss, val_acc = validate(model, val_loader, criterion)
            scheduler.step(val_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(val_acc)
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
                        "val_acc": val_acc,
                        "config": train_config,
                    },
                    ckpt_path,
                )
                print(
                    f"\r  Epoch {epoch:3d}/{args.epochs} │ Train L: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% │ Val L: {val_loss:.4f} Acc: {val_acc*100:.2f}% ★ │ {elapsed:.1f}s"
                )
            else:
                patience_counter += 1
                print(
                    f"\r  Epoch {epoch:3d}/{args.epochs} │ Train L: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% │ Val L: {val_loss:.4f} Acc: {val_acc*100:.2f}%   │ {elapsed:.1f}s",
                    end="",
                )
                if epoch % 10 == 0:
                    print("", flush=True)

                if patience_counter >= args.patience:
                    print(
                        f"\n  ⏹ Early stopping at epoch {epoch} (best epoch: {best_epoch})"
                    )
                    break

        total_time = time.time() - t_start
        print(f"\n  ✅ Training completed in {total_time / 60:.1f} min")
        print(
            f"  Best validation loss: {best_val_loss:.6f} (Acc: {history['val_acc'][best_epoch-1]*100:.2f}%) at epoch {best_epoch}"
        )

        # Save the JSON history
        with open(RESULTS_DIR / "training_history.json", "w") as f:
            json.dump(history, f, indent=4)

        plot_training_curves(history, RESULTS_DIR)

        # Load the best checkpoint for evaluation
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate the model on the test set
    print("\n  🔍 Evaluating the checkpoint on the test set...")
    evaluate_and_plot_metrics(model, test_caps, norm_stats, RESULTS_DIR)


if __name__ == "__main__":
    main()
