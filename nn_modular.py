"""
Red Neuronal: Pipeline Modular de Marcha (Modular PivotNet)
===========================================================
Arquitectura:
  - PivotNet: Red convolucional 1D causal con dilataciones progresivas (d=1, 2, 4) y residual connections.
  - Proyección inicial densa con LayerNorm + GELU.
  - Clasificación binaria temporal de apoyo de zancada.
  - OrientationNet (Opcional): Estimación de cuaterniones de orientación de los 5 segmentos anatómicos.

Propósito:
  - Entrenar submódulos neuronales especializados y desacoplados para integrarse en pipelines híbridos
    (ej. detección neuronal de pivote + cinemática analítica de orientación).

Directorio de salida: nn-results/modular/
Checkpoints: nn-results/modular/checkpoints/best_pivot_model.pt

Uso:
    python nn_modular.py                         # Entrenamiento del modelo PivotNet

Autor: GaitTracking Project
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler

# Re-use process_all.py functions
from process_all import (
    adjust_aruco_data,
    read_aruco_file,
    read_mcd_file,
    get_orientation,
    get_all_poses,
)

# ═══════════════════════════════════════════════════════════════════════
#  CONSTANTS REQUIRED BY process_all_nn.py
# ═══════════════════════════════════════════════════════════════════════
BONE_NORM = 0.45
IMU_NAMES = ["RF", "LF", "RT", "LT"]
SENSOR_CHANNELS = len(IMU_NAMES) * 6  # 24 channels
PIVOT_INPUT_DIM = SENSOR_CHANNELS
ORIENT_INPUT_DIM = SENSOR_CHANNELS + 3  # sensors + anthropometric
ORIENT_OUTPUT_DIM = 5 * 4  # 5 segments x 4 quat values
WINDOW_SIZE = 150
WINDOW_STRIDE = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════════════
#  MODELS
# ═══════════════════════════════════════════════════════════════════════


class PivotNet(nn.Module):
    """
    Red Neuronal para predecir qué pie está en apoyo (pivote).
    Entrada: [batch, seq_len, 24] (acelerómetro y giroscopio de 4 IMUs)
    Salida: [batch, seq_len] (probabilidad de que el pie derecho sea pivote)
    """

    def __init__(self, input_dim=PIVOT_INPUT_DIM, hidden_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )

        # Temporal Convolutional Network simple
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
        h = h.transpose(1, 2)  # [batch, hidden, seq_len]

        h = F.gelu(self.conv1(h))
        h = F.gelu(self.conv2(h))
        h = F.gelu(self.conv3(h))

        h = h.transpose(1, 2)  # [batch, seq_len, hidden]

        logits = self.out(h).squeeze(-1)  # [batch, seq_len]
        return torch.sigmoid(logits)


class OrientationNet(nn.Module):
    """
    Placeholder para OrientationNet requerido por process_all_nn.py
    """

    def __init__(self):
        super().__init__()
        self.dummy = nn.Linear(ORIENT_INPUT_DIM, ORIENT_OUTPUT_DIM)

    def forward(self, x):
        return self.dummy(x)


# ═══════════════════════════════════════════════════════════════════════
#  DATASET & PREPARATION
# ═══════════════════════════════════════════════════════════════════════


def compute_pivot_labels(mcd_data, aruco_data, points, direction):
    """
    Calcula los labels del pivote usando el ArUco si es posible,
    o usando la misma heurística de process_all.py si no hay ArUco completo.
    Label 1 = Pie derecho en apoyo, Label 0 = Pie izquierdo en apoyo.
    """
    N = len(mcd_data["time"])
    labels = np.zeros(N, dtype=np.float32)

    # Obtener orientacion y pose de madgwick (heurística de process_all)
    quats = get_orientation(mcd_data)
    pose = get_all_poses(quats, points)

    for i in range(N):
        if i == 0:
            z_compare = 1 if direction == "R" else 0
        else:
            diff_z = pose["ankle_l"][i][2] - pose["ankle_r"][i][2]
            mag_gyro_rt = np.linalg.norm(mcd_data["gyro_RT"][i])
            mag_gyro_lt = np.linalg.norm(mcd_data["gyro_LT"][i])

            if abs(diff_z) > 0.01:  # pivot_height_threshold
                z_compare = diff_z
            else:
                z_compare = mag_gyro_lt - mag_gyro_rt

        v_rt = mcd_data["gyro_RT"][i, 1]
        v_lt = mcd_data["gyro_LT"][i, 1]

        if z_compare <= 0:
            if v_lt > 0.5 and v_lt > v_rt:  # swing_velocity_threshold
                z_compare = 1
        else:
            if v_rt > 0.5 and v_rt > v_lt:
                z_compare = -1

        labels[i] = 1.0 if z_compare > 0 else 0.0

    return labels


class PivotDataset(Dataset):
    def __init__(self, dataset_path, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE):
        self.window_size = window_size
        self.windows = []

        # Discover files
        subjects = sorted(
            [
                s
                for s in glob.glob("*", root_dir=str(dataset_path))
                if s.isdigit() and s != "0"
            ]
        )

        all_sensors = []

        file_pairs = []
        for s in subjects:
            for i in range(4):
                mcd_path = Path(dataset_path) / s / f"mcd-{i}.csv"
                aruco_path = Path(dataset_path) / s / f"arUcos-{i}.csv"
                if mcd_path.exists() and aruco_path.exists():
                    file_pairs.append((mcd_path, aruco_path))

        total_files = len(file_pairs)
        for idx_f, (mcd_path, aruco_path) in enumerate(file_pairs):
            progress = (idx_f + 1) / total_files
            bar_len = 30
            filled = int(bar_len * progress)
            bar = "=" * filled + "-" * (bar_len - filled)
            sys.stdout.write(
                f"\rCargando dataset: [{bar}] {idx_f+1}/{total_files} archivos"
            )
            sys.stdout.flush()

            mcd_data = read_mcd_file(str(mcd_path))
            aruco_data = read_aruco_file(str(aruco_path))
            direction = mcd_data["measures"]["direction"]

            # Points for heuristic
            points = {
                "hip_r": np.array(
                    [0, mcd_data["measures"].get("hip_width", 0.25) / 2, 0]
                ),
                "knee_r": np.array(
                    [0, 0, -mcd_data["measures"].get("femur_length", 0.4)]
                ),
                "ankle_r": np.array(
                    [0, 0, -mcd_data["measures"].get("tibia_length", 0.4)]
                ),
                "hip_l": np.array(
                    [0, -mcd_data["measures"].get("hip_width", 0.25) / 2, 0]
                ),
                "knee_l": np.array(
                    [0, 0, -mcd_data["measures"].get("femur_length", 0.4)]
                ),
                "ankle_l": np.array(
                    [0, 0, -mcd_data["measures"].get("tibia_length", 0.4)]
                ),
            }

            # Compute ground truth labels
            labels = compute_pivot_labels(mcd_data, aruco_data, points, direction)

            # Stack sensor data
            N = len(mcd_data["time"])
            sensor_data = np.zeros((N, SENSOR_CHANNELS), dtype=np.float32)
            for idx, imu in enumerate(IMU_NAMES):
                sensor_data[:, idx * 6 : idx * 6 + 3] = mcd_data[f"acceleration_{imu}"]
                sensor_data[:, idx * 6 + 3 : idx * 6 + 6] = mcd_data[f"gyro_{imu}"]

            all_sensors.append(sensor_data)

            # Create windows
            for start in range(0, N - window_size + 1, stride):
                end = start + window_size
                self.windows.append((sensor_data[start:end], labels[start:end]))
        sys.stdout.write("\n")

        # Compute normalization stats
        if all_sensors:
            concat_sensors = np.concatenate(all_sensors, axis=0)
            self.mean = concat_sensors.mean(axis=0, dtype=np.float32)
            self.std = concat_sensors.std(axis=0, dtype=np.float32)
            self.std[self.std < 1e-8] = 1.0
        else:
            self.mean = np.zeros(SENSOR_CHANNELS, dtype=np.float32)
            self.std = np.ones(SENSOR_CHANNELS, dtype=np.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sensor, label = self.windows[idx]
        sensor = (sensor - self.mean) / self.std
        return torch.from_numpy(sensor), torch.from_numpy(label)


# ═══════════════════════════════════════════════════════════════════════
#  TRAINING SCRIPT
# ═══════════════════════════════════════════════════════════════════════


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="/home/moybadajoz/Documents/DataSet"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Épocas sin mejora para detener el entrenamiento",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=150,
        help="Tamaño de la ventana (frames) para la red neuronal",
    )
    parser.add_argument("--out-dir", type=str, default="nn-results/modular/checkpoints")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando dataset...")
    dataset = PivotDataset(args.dataset, window_size=args.window_size)
    if len(dataset) == 0:
        print("Error: No se encontraron datos.")
        return

    # Split 70/15/15 train/val/test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    print("\n" + "=" * 40)
    print("📊 INFORMACIÓN DEL DATASET")
    print("=" * 40)
    print(f"Total de ventanas (muestras): {len(dataset)}")
    print(f"  - Entrenamiento: {train_size} muestras")
    print(f"  - Validación:    {val_size} muestras")
    print(f"  - Test:          {test_size} muestras")
    print("=" * 40 + "\n")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = PivotNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    print("=" * 40)
    print("🧠 INFORMACIÓN DE LA RED NEURONAL")
    print("=" * 40)
    print(f"Modelo: PivotNet (TCN)")
    print(
        f"Entrada (Batch, Seq_len, Canales): (batch_size, {args.window_size}, {PIVOT_INPUT_DIM})"
    )
    print(f"Salida (Batch, Seq_len):           (batch_size, {args.window_size})")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables:            {total_params:,}")
    print("=" * 40 + "\n")

    best_loss = float("inf")
    epochs_without_improvement = 0

    print(f"Iniciando entrenamiento por {args.epochs} epocas...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        num_batches = len(train_loader)
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()

            preds = model(x)
            loss = criterion(preds, y)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

            # Mostrar avance de los batches
            progress = (batch_idx + 1) / num_batches
            bar_len = 20
            filled = int(bar_len * progress)
            bar = "=" * filled + "-" * (bar_len - filled)
            sys.stdout.write(
                f"\rEpoch {epoch:03d}/{args.epochs} [{bar}] Batch {batch_idx+1}/{num_batches}"
            )
            sys.stdout.flush()

        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x)
                loss = criterion(preds, y)
                val_loss += loss.item() * x.size(0)

                preds_bin = (preds > 0.5).float()
                correct += (preds_bin == y).sum().item()
                total += y.numel()

        val_loss /= len(val_ds)
        val_acc = correct / total * 100

        improved = False
        if val_loss < best_loss:
            best_loss = val_loss
            improved = True
            epochs_without_improvement = 0
            # Guardar el modelo en el formato esperado por process_all_nn.py
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "norm_stats": {
                        "mean": dataset.mean.tolist(),
                        "std": dataset.std.tolist(),
                    },
                },
                out_dir / "best_pivot_model.pt",
            )
        else:
            epochs_without_improvement += 1

        mark = "*\n" if improved else " "

        progress = epoch / args.epochs
        bar_len = 20
        filled = int(bar_len * progress)
        bar = "=" * filled + "-" * (bar_len - filled)

        sys.stdout.write(
            f"\r[{bar}] Epoch {epoch:03d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% {mark}"
        )
        sys.stdout.flush()

        if epochs_without_improvement >= args.patience:
            sys.stdout.write(
                f"\n\n[!] Deteniendo entrenamiento: {args.patience} épocas sin mejora (Early Stopping).\n"
            )
            sys.stdout.flush()
            break

    sys.stdout.write("\n\nEntrenamiento finalizado.\n")


if __name__ == "__main__":
    train()
