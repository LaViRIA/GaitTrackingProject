"""
Experiment matrix for the TCN Lite neural network (nn_tcn_lite.py)
====================================================================

Architecture / purpose:
  - Runs an exhaustive or targeted search (grid / random search) over the key training
    and data-split hyperparameters of the GaitMultiTaskNetTCN network:
      1. Learning Rate (lr)
      2. Batch Size
      3. Window Size (time length of the window in frames)
      4. Window Stride (shift / overlap between windows)
      5. Number of test subjects (n_test_subjects)
      6. Number of validation subjects (n_val_subjects)

  - Outputs:
      - CSV file with the detailed record of each run:
        `nn-results/experiment_matrix_tcn/experiment_matrix_tcn.csv`
      - Analysis plots and interaction heatmaps:
        `nn-results/experiment_matrix_tcn/heatmaps/`
      - Comparison table of the best models (top 10):
        `nn-results/experiment_matrix_tcn/summary_table.png`
      - Checkpoint of the best global model:
        `nn-results/experiment_matrix_tcn/checkpoints/best_grid_tcn_lite.pt`

Usage:
    # Run the targeted grid search (recommended):
    python experiment_matrix_tcn.py

    # Random search over N combinations:
    python experiment_matrix_tcn.py --mode random --n-trials 20

    # Set the maximum epochs and the early-stopping patience:
    python experiment_matrix_tcn.py --epochs 60 --patience 10

    # Only generate plots and analysis from an existing CSV:
    python experiment_matrix_tcn.py --analyze-only

Author: GaitTracking Project
"""

import sys
import os
import time
import json
import csv
import random
import itertools
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for fast plot saving
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import the TCN Lite network components and the loading utilities
from models.nn_tcn_lite import (
    GaitMultiTaskNetTCN,
    GaitDataset,
    discover_captures,
    prepare_capture,
    train_multitask_epoch,
    validate_multitask,
    evaluate_on_captures,
    predict_full_sequence_multitask,
    INPUT_DIM,
    OUTPUT_DIM,
    N_JOINTS,
    N_COORDS,
    JOINT_NAMES,
    DATASET_PATH as DEFAULT_DATASET_PATH,
    split_char,
)

# ═══════════════════════════════════════════════════════════════════════
#  HYPERPARAMETER MATRIX CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_OUTPUT_DIR = Path("nn-results/experiment_matrix_tcn")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
W = 74  # Text format width

# Hyperparameter grid
GRID_HYPERPARAMS = {
    "lr": [5e-4],  # Learning rate
    "batch_size": [24],  # Batch size
    "window_size": [50],  # Frames per window
    "window_stride": [3],  # Overlap (shift in frames)
    "n_test_subjects": [3],  # Subjects in the test set
    "n_val_subjects": [3],  # Subjects in the validation set
}

# Base / reference configuration for marginal comparisons
BASELINE_CONFIG = {
    "lr": 5e-4,
    "batch_size": 32,
    "window_size": 150,
    "window_stride": 25,
    "n_test_subjects": 2,
    "n_val_subjects": 2,
}


# ═══════════════════════════════════════════════════════════════════════
#  SEARCH-SPACE GENERATION
# ═══════════════════════════════════════════════════════════════════════


def generate_targeted_grid(grid_dict: dict, baseline: dict) -> list[dict]:
    """
    Generate a targeted grid that varies one hyperparameter at a time
    around the baseline, plus cross combinations of the most sensitive parameters.
    Reduces the combinatorial space from hundreds to a few dozen representative runs.
    """
    combinations = []
    seen = set()

    def add_combo(c: dict):
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            combinations.append(c)

    # 1. Base case
    add_combo(baseline.copy())

    # 2. Individual variations of each hyperparameter
    for param, values in grid_dict.items():
        for val in values:
            combo = baseline.copy()
            combo[param] = val
            add_combo(combo)

    # 3. Key interactions: LR x Batch Size
    for lr in grid_dict["lr"]:
        for bs in grid_dict["batch_size"]:
            combo = baseline.copy()
            combo["lr"] = lr
            combo["batch_size"] = bs
            add_combo(combo)

    # 4. Key interactions: Window Size x Stride
    for ws in grid_dict["window_size"]:
        for st in grid_dict["window_stride"]:
            if st < ws:  # The stride must be smaller than the window
                combo = baseline.copy()
                combo["window_size"] = ws
                combo["window_stride"] = st
                add_combo(combo)

    # 5. Key interactions: n_test x n_val
    for nt in grid_dict["n_test_subjects"]:
        for nv in grid_dict["n_val_subjects"]:
            combo = baseline.copy()
            combo["n_test_subjects"] = nt
            combo["n_val_subjects"] = nv
            add_combo(combo)

    return combinations


def generate_full_grid(grid_dict: dict) -> list[dict]:
    """Generate the full Cartesian product of all the hyperparameters."""
    keys = list(grid_dict.keys())
    values = list(grid_dict.values())
    combinations = []
    for product in itertools.product(*values):
        combo = dict(zip(keys, product))
        if combo["window_stride"] < combo["window_size"]:
            combinations.append(combo)
    return combinations


def generate_random_grid(grid_dict: dict, n_trials: int, seed: int = 42) -> list[dict]:
    """Generate N random combinations of the search space."""
    random.seed(seed)
    full = generate_full_grid(grid_dict)
    random.shuffle(full)
    return full[: min(n_trials, len(full))]


# ═══════════════════════════════════════════════════════════════════════
#  DATASET LOADING AND IN-MEMORY CACHE
# ═══════════════════════════════════════════════════════════════════════


def load_and_cache_dataset(
    dataset_path: Path,
) -> tuple[dict[str, list], dict[str, list], list[str]]:
    """
    Read and process every capture of the dataset once, in RAM.
    Avoids disk reads and repeated filtering in every iteration of the matrix.
    """
    print()
    print("═" * W)
    print("  📂 LOADING AND PRE-PROCESSING THE FULL DATASET IN MEMORY")
    print("═" * W)
    t0 = time.time()

    captures_info = discover_captures(dataset_path)
    if len(captures_info) == 0:
        print(f"  [ERROR] No captures found in {dataset_path}")
        sys.exit(1)

    subjects_data = defaultdict(list)
    file_paths_by_subject = defaultdict(list)

    for mcd_path, aruco_path, subject_id in captures_info:
        cap = prepare_capture(mcd_path, aruco_path)
        if cap is not None:
            subjects_data[subject_id].append(cap)
            file_paths_by_subject[subject_id].append((mcd_path, aruco_path))

    available_subjects = sorted(
        subjects_data.keys(), key=lambda s: int(s) if s.isdigit() else s
    )
    total_caps = sum(len(caps) for caps in subjects_data.values())
    elapsed = time.time() - t0

    print(
        f"  ✓ {total_caps} captures processed from {len(available_subjects)} subjects in {elapsed:.2f}s"
    )
    print(f"  Available subjects: {available_subjects}")
    print("═" * W)

    return subjects_data, file_paths_by_subject, available_subjects


# ═══════════════════════════════════════════════════════════════════════
#  TRAINING AND EVALUATION OF ONE COMBINATION
# ═══════════════════════════════════════════════════════════════════════


def run_single_experiment(
    exp_id: int,
    config: dict,
    subjects_data: dict,
    file_paths_by_subject: dict,
    available_subjects: list,
    epochs: int = 60,
    patience: int = 10,
    seed: int = 42,
    checkpoints_dir: Path | None = None,
) -> dict:
    """
    Run a complete experiment for one hyperparameter combination:
      1. Split Train/Val/Test according to n_test and n_val.
      2. Build the datasets and data loaders with window_size and window_stride.
      3. Train GaitMultiTaskNetTCN with early stopping.
      4. Evaluate thoroughly on the test set (MPJPE, MAE, Pearson, ATE).
    """
    lr = config["lr"]
    batch_size = config["batch_size"]
    window_size = config["window_size"]
    window_stride = config["window_stride"]
    n_test = config["n_test_subjects"]
    n_val = config["n_val_subjects"]

    # Set the seeds for a reproducible split
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Subject split
    shuffled_subjects = available_subjects.copy()
    random.shuffle(shuffled_subjects)

    n_test = min(n_test, max(1, len(shuffled_subjects) - 2))
    test_subjects = shuffled_subjects[:n_test]
    remaining = shuffled_subjects[n_test:]

    n_val = min(n_val, max(1, len(remaining) - 1))
    val_subjects = remaining[:n_val]
    train_subjects = remaining[n_val:]

    train_captures = [c for s in train_subjects for c in subjects_data[s]]
    val_captures = [c for s in val_subjects for c in subjects_data[s]]
    test_captures = [c for s in test_subjects for c in subjects_data[s]]
    test_file_paths = [p for s in test_subjects for p in file_paths_by_subject[s]]

    # 2. Dataset construction
    train_dataset = GaitDataset(
        train_captures,
        window_size=window_size,
        stride=window_stride,
        augment=True,
    )
    norm_stats = train_dataset.get_stats()

    val_dataset = GaitDataset(
        val_captures,
        window_size=window_size,
        stride=window_size,  # No overlap in validation
        stats=norm_stats,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print(f"  [NOTICE Exp {exp_id}] Not enough windows for this configuration.")
        return None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Instantiate the TCN model
    model = GaitMultiTaskNetTCN(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=25, T_mult=2, eta_min=lr * 0.01
    )

    # 4. Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    patience_cnt = 0
    best_state = None

    t_train_start = time.time()
    for ep in range(1, epochs + 1):
        tr = train_multitask_epoch(
            model, train_loader, optimizer, scaler, epoch=ep, total_epochs=epochs
        )
        va = validate_multitask(model, val_loader)
        scheduler.step()

        if va["total"] < best_val_loss:
            best_val_loss = va["total"]
            best_epoch = ep
            patience_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1

        if patience_cnt >= patience:
            break

    train_duration = time.time() - t_train_start

    # Load the best state for evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    # 5. Evaluation on the test set
    eval_metrics = evaluate_on_captures(
        model,
        test_captures,
        norm_stats,
        label=f"Exp_{exp_id}_Test",
        file_paths=test_file_paths,
        window_size=window_size,
    )

    mpjpe = eval_metrics.get("mpjpe", {})
    rmse = eval_metrics.get("rmse", {})
    disp = eval_metrics.get("displacement", {})

    # Save the checkpoint if applicable
    if checkpoints_dir is not None:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoints_dir / f"tcn_exp_{exp_id:03d}.pt"
        torch.save(
            {
                "exp_id": exp_id,
                "config": config,
                "model_state_dict": best_state,
                "norm_stats": norm_stats,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "test_mpjpe_mean": mpjpe.get("mean", 0.0),
            },
            ckpt_path,
        )

    # Compile the result record
    result_record = {
        "exp_id": exp_id,
        "lr": lr,
        "batch_size": batch_size,
        "window_size": window_size,
        "window_stride": window_stride,
        "n_test_subjects": len(test_subjects),
        "n_val_subjects": len(val_subjects),
        "n_train_subjects": len(train_subjects),
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_time_s": round(train_duration, 2),
        "test_mpjpe_mean_mm": round(mpjpe.get("mean", 0.0) * 1000.0, 2),
        "test_mpjpe_knee_r_mm": round(mpjpe.get("knee_r", 0.0) * 1000.0, 2),
        "test_mpjpe_knee_l_mm": round(mpjpe.get("knee_l", 0.0) * 1000.0, 2),
        "test_mpjpe_ankle_r_mm": round(mpjpe.get("ankle_r", 0.0) * 1000.0, 2),
        "test_mpjpe_ankle_l_mm": round(mpjpe.get("ankle_l", 0.0) * 1000.0, 2),
        "test_mpjpe_hip_r_mm": round(mpjpe.get("hip_r", 0.0) * 1000.0, 2),
        "test_mpjpe_hip_l_mm": round(mpjpe.get("hip_l", 0.0) * 1000.0, 2),
        "test_rmse_mean_mm": round(rmse.get("mean", 0.0) * 1000.0, 2),
        "test_ate_mean_cm": round(disp.get("ate_mean", 0.0) * 100.0, 2),
        "test_ate_x_cm": round(disp.get("ate_x", 0.0) * 100.0, 2),
        "test_ate_z_cm": round(disp.get("ate_z", 0.0) * 100.0, 2),
        "test_epe_mean_cm": round(disp.get("epe_mean", 0.0) * 100.0, 2),
        "test_subjects": ",".join(test_subjects),
        "val_subjects": ",".join(val_subjects),
    }

    return result_record


# ═══════════════════════════════════════════════════════════════════════
#  STATISTICAL ANALYSIS AND PLOT GENERATION
# ═══════════════════════════════════════════════════════════════════════


def generate_analysis_plots(csv_path: Path, output_dir: Path):
    """
    Generate analysis plots from the results CSV file:
      1. Interaction heatmaps between hyperparameters
      2. Individual trends (marginal effects)
      3. Summary table with the top 10 models
    """
    if not csv_path.exists():
        print(f"  [ERROR] CSV file not found at {csv_path}")
        return

    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in row:
                try:
                    row[k] = float(row[k])
                except (ValueError, TypeError):
                    pass
            records.append(row)

    if len(records) == 0:
        print("  [ERROR] The CSV file is empty.")
        return

    plots_dir = output_dir / "plots"
    heatmaps_dir = output_dir / "heatmaps"
    plots_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("═" * W)
    print(f"  🎨 GENERATING ANALYSIS AND PLOTS ({len(records)} EXPERIMENTS)")
    print("═" * W)

    # 1. One-dimensional trend plots
    params_to_analyze = [
        ("lr", "Learning Rate (lr)", True),
        ("batch_size", "Batch Size", False),
        ("window_size", "Window Size (frames)", False),
        ("window_stride", "Window Stride (frames)", False),
        ("n_test_subjects", "No. of test subjects", False),
        ("n_val_subjects", "No. of validation subjects", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (param_key, param_label, is_log) in enumerate(params_to_analyze):
        ax = axes[idx]
        grouped = defaultdict(list)
        for r in records:
            if param_key in r and "test_mpjpe_mean_mm" in r:
                grouped[r[param_key]].append(r["test_mpjpe_mean_mm"])

        sorted_keys = sorted(grouped.keys())
        means = [np.mean(grouped[k]) for k in sorted_keys]
        stds = [np.std(grouped[k]) for k in sorted_keys]
        mins = [np.min(grouped[k]) for k in sorted_keys]

        x_vals = np.array(sorted_keys)
        ax.plot(
            x_vals, means, "o-", color="royalblue", linewidth=2.2, label="Mean MPJPE"
        )
        ax.fill_between(
            x_vals,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            color="royalblue",
            alpha=0.18,
        )
        ax.plot(
            x_vals, mins, "s--", color="crimson", linewidth=1.5, label="Best MPJPE"
        )

        if is_log:
            ax.set_xscale("log")
        ax.set_title(f"Impact of {param_label}", fontweight="bold", fontsize=11)
        ax.set_xlabel(param_label)
        ax.set_ylabel("MPJPE en Test (mm)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle(
        "Marginal hyperparameter analysis — GaitMultiTaskNetTCN",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(plots_dir / "hyperparam_trends.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {plots_dir / 'hyperparam_trends.png'}")

    # 2. Heatmap: LR vs Batch Size
    plot_heatmap_2d(
        records,
        x_key="batch_size",
        y_key="lr",
        metric_key="test_mpjpe_mean_mm",
        title="Heatmap: Batch Size vs Learning Rate (MPJPE mm)",
        save_path=heatmaps_dir / "heatmap_lr_vs_batch_size.png",
    )

    # 3. Heatmap: Window Size vs Window Stride
    plot_heatmap_2d(
        records,
        x_key="window_stride",
        y_key="window_size",
        metric_key="test_mpjpe_mean_mm",
        title="Heatmap: Window Size vs Stride (MPJPE mm)",
        save_path=heatmaps_dir / "heatmap_window_size_vs_stride.png",
    )

    # 4. Top-10 summary table
    generate_summary_table_image(records, output_dir / "summary_table.png")


def plot_heatmap_2d(
    records: list[dict],
    x_key: str,
    y_key: str,
    metric_key: str,
    title: str,
    save_path: Path,
):
    """Plot a 2D heatmap of the interaction between two hyperparameters."""
    x_vals = sorted(
        list(set(r[x_key] for r in records if x_key in r and metric_key in r))
    )
    y_vals = sorted(
        list(set(r[y_key] for r in records if y_key in r and metric_key in r))
    )

    if len(x_vals) < 2 or len(y_vals) < 2:
        return

    matrix = np.full((len(y_vals), len(x_vals)), np.nan)

    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            matching = [
                r[metric_key]
                for r in records
                if r.get(x_key) == x and r.get(y_key) == y and metric_key in r
            ]
            if len(matching) > 0:
                matrix[i, j] = np.mean(matching)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="viridis_r", aspect="auto")

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels(x_vals)
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels(y_vals)

    # Annotate the cell values
    for i in range(len(y_vals)):
        for j in range(len(x_vals)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val > np.nanmean(matrix) else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight="bold",
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean MPJPE (mm)", rotation=270, labelpad=15)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {save_path}")


def generate_summary_table_image(records: list[dict], save_path: Path):
    """Produce an image with the formatted table of the top 10 configurations."""
    sorted_records = sorted(
        records, key=lambda r: r.get("test_mpjpe_mean_mm", float("inf"))
    )[:10]
    if len(sorted_records) == 0:
        return

    headers = [
        "Rank",
        "Exp ID",
        "LR",
        "Batch",
        "Win Size",
        "Stride",
        "N Test",
        "N Val",
        "Val Loss",
        "MPJPE (mm)",
        "ATE (cm)",
    ]
    rows = []
    for rank, r in enumerate(sorted_records, 1):
        rows.append(
            [
                f"#{rank}",
                f"{int(r.get('exp_id', 0)):02d}",
                f"{r.get('lr', 0):.1e}",
                f"{int(r.get('batch_size', 0))}",
                f"{int(r.get('window_size', 0))}",
                f"{int(r.get('window_stride', 0))}",
                f"{int(r.get('n_test_subjects', 0))}",
                f"{int(r.get('n_val_subjects', 0))}",
                f"{r.get('best_val_loss', 0):.4f}",
                f"{r.get('test_mpjpe_mean_mm', 0):.1f}",
                f"{r.get('test_ate_mean_cm', 0):.1f}",
            ]
        )

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    # Header and cell styles
    for col_idx in range(len(headers)):
        table[(0, col_idx)].set_facecolor("#2b5c8f")
        table[(0, col_idx)].set_text_props(color="white", fontweight="bold")

    for row_idx in range(1, len(rows) + 1):
        bg_color = "#f2f7fc" if row_idx % 2 == 0 else "white"
        if row_idx == 1:
            bg_color = "#d4edda"  # Highlight #1 in light green
        for col_idx in range(len(headers)):
            table[(row_idx, col_idx)].set_facecolor(bg_color)

    plt.title(
        "Top 10 configurations — TCN Lite experiment matrix",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {save_path}")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN EXECUTION (CLI)
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Experiment matrix for GaitMultiTaskNetTCN (TCN Lite)"
    )
    parser.add_argument(
        "--dataset", type=str, default=str(DEFAULT_DATASET_PATH), help="Path to the dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Results directory",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["targeted", "full", "random"],
        default="targeted",
        help="Grid mode",
    )
    parser.add_argument(
        "--n-trials", type=int, default=20, help="No. of trials if mode=random"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Maximum epochs per experiment"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early-stopping patience"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip combinations already computed in the CSV",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only generate plots from the existing CSV",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    csv_path = output_dir / "experiment_matrix_tcn.csv"

    if args.analyze_only:
        generate_analysis_plots(csv_path, output_dir)
        return

    # 1. Generate the combinations to evaluate
    if args.mode == "targeted":
        combinations = generate_targeted_grid(GRID_HYPERPARAMS, BASELINE_CONFIG)
    elif args.mode == "full":
        combinations = generate_full_grid(GRID_HYPERPARAMS)
    elif args.mode == "random":
        combinations = generate_random_grid(
            GRID_HYPERPARAMS, n_trials=args.n_trials, seed=args.seed
        )

    print()
    print("═" * W)
    print("  🚀 EXPERIMENT MATRIX: MULTI-TASK TCN LITE NETWORK")
    print("═" * W)
    print(f"  Mode:              {args.mode.upper()}")
    print(f"  Total runs:        {len(combinations)}")
    print(f"  Max epochs/exp:    {args.epochs} (patience: {args.patience})")
    print(f"  Device:            {DEVICE}")
    print(f"  Output:            {output_dir.resolve()}")
    print("═" * W)

    # 2. Load the dataset into memory once
    subjects_data, file_paths_by_subject, available_subjects = load_and_cache_dataset(
        Path(args.dataset)
    )

    # 3. Read previous experiments if resuming
    existing_exp_keys = set()
    csv_exists = csv_path.exists()
    fieldnames = [
        "exp_id",
        "lr",
        "batch_size",
        "window_size",
        "window_stride",
        "n_test_subjects",
        "n_val_subjects",
        "n_train_subjects",
        "train_windows",
        "val_windows",
        "best_epoch",
        "best_val_loss",
        "train_time_s",
        "test_mpjpe_mean_mm",
        "test_mpjpe_knee_r_mm",
        "test_mpjpe_knee_l_mm",
        "test_mpjpe_ankle_r_mm",
        "test_mpjpe_ankle_l_mm",
        "test_mpjpe_hip_r_mm",
        "test_mpjpe_hip_l_mm",
        "test_rmse_mean_mm",
        "test_ate_mean_cm",
        "test_ate_x_cm",
        "test_ate_z_cm",
        "test_epe_mean_cm",
        "test_subjects",
        "val_subjects",
    ]

    if csv_exists and args.skip_existing:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (
                    float(row["lr"]),
                    int(float(row["batch_size"])),
                    int(float(row["window_size"])),
                    int(float(row["window_stride"])),
                    int(float(row["n_test_subjects"])),
                    int(float(row["n_val_subjects"])),
                )
                existing_exp_keys.add(key)
        print(
            f"  ℹ️ Resuming: {len(existing_exp_keys)} experiments already completed."
        )

    csv_file = open(csv_path, "a" if csv_exists else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not csv_exists:
        writer.writeheader()
        csv_file.flush()

    best_global_mpjpe = float("inf")
    best_global_config = None
    t_matrix_start = time.time()

    # 4. Experiment loop
    for exp_idx, combo in enumerate(combinations, 1):
        combo_key = (
            combo["lr"],
            combo["batch_size"],
            combo["window_size"],
            combo["window_stride"],
            combo["n_test_subjects"],
            combo["n_val_subjects"],
        )
        if args.skip_existing and combo_key in existing_exp_keys:
            print(
                f"  ⏭️ [{exp_idx:02d}/{len(combinations):02d}] Skipping already processed combo: {combo}"
            )
            continue

        print()
        print("─" * W)
        print(f"  🧪 EXPERIMENT [{exp_idx:02d}/{len(combinations):02d}]")
        print(
            f"     LR={combo['lr']:.1e} │ Batch={combo['batch_size']} │ Window={combo['window_size']} │ Stride={combo['window_stride']} │ Test={combo['n_test_subjects']} │ Val={combo['n_val_subjects']}"
        )
        print("─" * W)

        t_exp_0 = time.time()
        record = run_single_experiment(
            exp_id=exp_idx,
            config=combo,
            subjects_data=subjects_data,
            file_paths_by_subject=file_paths_by_subject,
            available_subjects=available_subjects,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
            checkpoints_dir=checkpoints_dir,
        )

        if record is not None:
            writer.writerow(record)
            csv_file.flush()

            mpjpe_val = record["test_mpjpe_mean_mm"]
            ate_val = record["test_ate_mean_cm"]
            print(
                f"     ★ Result: MPJPE = {mpjpe_val:.1f} mm │ ATE = {ate_val:.1f} cm │ Epoch {record['best_epoch']} │ {time.time() - t_exp_0:.1f}s"
            )

            if mpjpe_val < best_global_mpjpe:
                best_global_mpjpe = mpjpe_val
                best_global_config = combo
                print(
                    f"     🏆 NEW BEST GLOBAL MODEL! MPJPE = {best_global_mpjpe:.1f} mm"
                )

    csv_file.close()
    total_time = time.time() - t_matrix_start

    print()
    print("═" * W)
    print(f"  ✅ EXPERIMENT MATRIX COMPLETED IN {total_time / 60:.2f} MINUTES")
    if best_global_config:
        print(
            f"  🏆 Best configuration found: MPJPE = {best_global_mpjpe:.1f} mm"
        )
        for k, v in best_global_config.items():
            print(f"     • {k:20s}: {v}")
    print("═" * W)

    # 5. Generate plots and report
    generate_analysis_plots(csv_path, output_dir)


if __name__ == "__main__":
    main()
