"""
Experiment matrix for hyperparameter optimization
==============================================================

This script evaluates the impact of 9 hyperparameters on the gait
reconstruction error:

  1. Cut-off frequency of the low-pass filter (Hz)
  2. Butterworth filter order
  3. PCA percentile (motion threshold for the yaw correction)
  4. Madgwick filter gain
  5. Clamp of the hip Slerp weight (slerp_clip_min)
  6. Calibration samples at rest
  7. Fusion filter (Madgwick vs EKF)
  8. Height threshold for the pivot switch
  9. Angular-velocity threshold for the swing override
"""

import itertools
import csv
import sys
import os
import time
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
from pathlib import Path

from process_all import process_files, split_char

DATA_SPLIT_PATH = Path(__file__).parent / "data_split.json"


def get_files_from_split(
    dataset_path: str, subjects: list[int]
) -> list[tuple[str, str]]:
    """Build the list of (mcd_file, aruco_file) tuples for the given subjects."""
    files = []
    for s in sorted(subjects):
        for i in range(4):
            mcd_file = f"{dataset_path}{split_char}{s}{split_char}mcd-{i}.csv"
            aruco_file = f"{dataset_path}{split_char}{s}{split_char}arUcos-{i}.csv"
            if not Path(mcd_file).exists() or not Path(aruco_file).exists():
                continue
            files.append((mcd_file, aruco_file))
    return files


# ─────────────────────────────────────────────────────────────
# Parameter grid (full Cartesian product: 20,736 combinations)
# ─────────────────────────────────────────────────────────────
CUTOFF_FREQS = [7, 9, 11, 13]  # Hz
FILTER_ORDERS = [5, 6, 7]  # Butterworth order
PCA_PERCENTILES = [5, 10, 20, 30]  # Percentile for the PCA threshold
MADGWICK_GAINS = [0.01, 0.02, 0.033]  # Madgwick filter gain
SLERP_CLIP_MINS = [
    0.3,
    0.5,
]  # Minimum hip Slerp weight (slerp_clip_max = 1.0 - min)
CALIBRATION_SAMPLES_LIST = [60, 70, 80, 100]  # Calibration samples
FUSION_FILTERS = ["madgwick", "ekf"]  # Fusion filter type
PIVOT_HEIGHT_THRESHOLDS = [
    0.005,
    0.01,
    0.015,
]  # Height threshold for the pivot switch (m)
SWING_VELOCITY_THRESHOLDS = [0.3, 0.5, 0.7]  # Angular-velocity threshold (rad/s)

# CUTOFF_FREQS = [5, 7]  # Hz
# FILTER_ORDERS = [4, 5, 6]  # Butterworth order
# PCA_PERCENTILES = [5, 10]  # Percentile for the PCA threshold
# MADGWICK_GAINS = [0.01, 0.02]  # Madgwick filter gain
# SLERP_CLIP_MINS = [0.3, 0.5]  # Minimum hip Slerp weight (slerp_clip_max = 1.0 - min)
# CALIBRATION_SAMPLES_LIST = [60, 70]  # Calibration samples
# FUSION_FILTERS = ["madgwick"]  # Fusion filter type
# PIVOT_HEIGHT_THRESHOLDS = [
#     0.005,
#     0.01,
# ]  # Height threshold for the pivot switch (m)
# SWING_VELOCITY_THRESHOLDS = [0.3, 0.5]  # Angular-velocity threshold (rad/s)

OUTPUT_DIR = Path("Results/experiment_matrix")

# Default values for individual plots/comparisons
DEFAULT_VALS = {
    "cutoff_freq": 9,
    "filter_order": 6,
    "pca_percentile": 20,
    "madgwick_gain": 0.02,
    "slerp_clip_min": 0.5,
    "calibration_samples": 100,
    "fusion_filter": "madgwick",
    "pivot_height_threshold": 0.005,
    "swing_velocity_threshold": 0.3,
}


def process_single_file(
    file_tuple,
    cutoff_freq,
    filter_order,
    pca_percentile,
    madgwick_gain,
    slerp_clip_min,
    calibration_samples,
    fusion_filter,
    pivot_height_threshold,
    swing_velocity_threshold,
):
    """Process a single file pair with the given parameters."""
    try:
        result = process_files(
            file_tuple,
            use_3d=False,
            cutoff_freq=cutoff_freq,
            filter_order=filter_order,
            pca_percentile=pca_percentile,
            madgwick_gain=madgwick_gain,
            slerp_clip_min=slerp_clip_min,
            slerp_clip_max=1.0 - slerp_clip_min,
            calibration_samples=calibration_samples,
            fusion_filter=fusion_filter,
            pivot_height_threshold=pivot_height_threshold,
            swing_velocity_threshold=swing_velocity_threshold,
        )
        if result is None:
            return None
        gen_error = result[0]
        homogeneous_data = result[4]
        # Mean Euclidean error per point (average of the 5 joints)
        mean_per_point = np.mean(
            [homogeneous_data[key]["euclidean_mean"] for key in homogeneous_data.keys()]
        )
        return {
            "gen_mean": np.mean(gen_error),
            "mean_per_point": mean_per_point,
        }
    except Exception as e:
        print(f"  [ERROR] {file_tuple[0]}: {e}", file=sys.stderr)
        return None


def evaluate_combination(
    target_files,
    cutoff_freq,
    filter_order,
    pca_percentile,
    madgwick_gain,
    slerp_clip_min,
    calibration_samples,
    fusion_filter,
    pivot_height_threshold,
    swing_velocity_threshold,
):
    """Evaluate a hyperparameter combination over the whole dataset."""
    worker = partial(
        process_single_file,
        cutoff_freq=cutoff_freq,
        filter_order=filter_order,
        pca_percentile=pca_percentile,
        madgwick_gain=madgwick_gain,
        slerp_clip_min=slerp_clip_min,
        calibration_samples=calibration_samples,
        fusion_filter=fusion_filter,
        pivot_height_threshold=pivot_height_threshold,
        swing_velocity_threshold=swing_velocity_threshold,
    )

    with Pool() as pool:
        results = pool.map(worker, target_files)

    valid = [r for r in results if r is not None]
    if len(valid) == 0:
        return None

    gen_means = np.array([r["gen_mean"] for r in valid])
    per_point_means = np.array([r["mean_per_point"] for r in valid])

    return {
        "cutoff_freq": cutoff_freq,
        "filter_order": filter_order,
        "pca_percentile": pca_percentile,
        "madgwick_gain": madgwick_gain,
        "slerp_clip_min": slerp_clip_min,
        "calibration_samples": calibration_samples,
        "fusion_filter": fusion_filter,
        "pivot_height_threshold": pivot_height_threshold,
        "swing_velocity_threshold": swing_velocity_threshold,
        "mean_error": float(np.mean(gen_means)),
        "std_error": float(np.std(gen_means)),
        "min_error": float(np.min(gen_means)),
        "max_error": float(np.max(gen_means)),
        "mean_per_point": float(np.mean(per_point_means)),
        "std_per_point": float(np.std(per_point_means)),
        "n_valid": len(valid),
        "n_total": len(target_files),
    }


def save_csv(results, filepath):
    """Save the results to a CSV file."""
    if not results:
        return
    fieldnames = results[0].keys()
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(results)
    print(f"  CSV saved: {filepath}")


def plot_heatmaps_by_pca(results, output_dir):
    """Plot heatmaps (cutoff_freq vs order) for each PCA percentile, keeping the other defaults."""
    # Keep the rows where the other parameters have their default value (or the closest one)
    for pca in PCA_PERCENTILES:
        subset = [
            r
            for r in results
            if r["pca_percentile"] == pca
            and abs(r["madgwick_gain"] - DEFAULT_VALS["madgwick_gain"]) < 1e-4
            and r["fusion_filter"] == DEFAULT_VALS["fusion_filter"]
            and abs(
                r["pivot_height_threshold"] - DEFAULT_VALS["pivot_height_threshold"]
            )
            < 1e-4
            and abs(
                r["swing_velocity_threshold"] - DEFAULT_VALS["swing_velocity_threshold"]
            )
            < 1e-4
        ]
        if not subset:
            continue

        freqs = sorted(set(r["cutoff_freq"] for r in subset))
        orders = sorted(set(r["filter_order"] for r in subset))
        matrix = np.full((len(orders), len(freqs)), np.nan)

        for r in subset:
            i = orders.index(r["filter_order"])
            j = freqs.index(r["cutoff_freq"])
            matrix[i, j] = r["mean_error"]

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", origin="lower")
        ax.set_xticks(range(len(freqs)))
        ax.set_xticklabels([f"{f} Hz" for f in freqs])
        ax.set_yticks(range(len(orders)))
        ax.set_yticklabels([f"Order {o}" for o in orders])
        ax.set_xlabel("Cut-off frequency (Hz)")
        ax.set_ylabel("Filter order")
        ax.set_title(f"Mean error (m) — PCA={pca}% (filters at defaults)")

        for i in range(len(orders)):
            for j in range(len(freqs)):
                if not np.isnan(matrix[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.4f}",
                        ha="center",
                        va="center",
                        color=(
                            "white" if matrix[i, j] > np.nanmedian(matrix) else "black"
                        ),
                        fontsize=9,
                        fontweight="bold",
                    )

        plt.colorbar(im, ax=ax, label="Mean Euclidean error (m)")
        plt.tight_layout()
        fig.savefig(output_dir / f"heatmap_pca_{pca}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Heatmap saved: heatmap_pca_{pca}.png")


def plot_heatmaps_by_order(results, output_dir):
    """Plot heatmaps (cutoff_freq vs PCA_percentile) for each filter order, at defaults."""
    for order in FILTER_ORDERS:
        subset = [
            r
            for r in results
            if r["filter_order"] == order
            and abs(r["madgwick_gain"] - DEFAULT_VALS["madgwick_gain"]) < 1e-4
            and r["fusion_filter"] == DEFAULT_VALS["fusion_filter"]
            and abs(
                r["pivot_height_threshold"] - DEFAULT_VALS["pivot_height_threshold"]
            )
            < 1e-4
            and abs(
                r["swing_velocity_threshold"] - DEFAULT_VALS["swing_velocity_threshold"]
            )
            < 1e-4
        ]
        if not subset:
            continue

        freqs = sorted(set(r["cutoff_freq"] for r in subset))
        pcas = sorted(set(r["pca_percentile"] for r in subset))
        matrix = np.full((len(pcas), len(freqs)), np.nan)

        for r in subset:
            i = pcas.index(r["pca_percentile"])
            j = freqs.index(r["cutoff_freq"])
            matrix[i, j] = r["mean_error"]

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", origin="lower")
        ax.set_xticks(range(len(freqs)))
        ax.set_xticklabels([f"{f} Hz" for f in freqs])
        ax.set_yticks(range(len(pcas)))
        ax.set_yticklabels([f"{p}%" for p in pcas])
        ax.set_xlabel("Cut-off frequency (Hz)")
        ax.set_ylabel("PCA percentile")
        ax.set_title(f"Mean error (m) — filter order {order} (defaults)")

        for i in range(len(pcas)):
            for j in range(len(freqs)):
                if not np.isnan(matrix[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.4f}",
                        ha="center",
                        va="center",
                        color=(
                            "white" if matrix[i, j] > np.nanmedian(matrix) else "black"
                        ),
                        fontsize=9,
                        fontweight="bold",
                    )

        plt.colorbar(im, ax=ax, label="Mean Euclidean error (m)")
        plt.tight_layout()
        fig.savefig(
            output_dir / f"heatmap_order_{order}.png", dpi=200, bbox_inches="tight"
        )
        plt.close(fig)
        print(f"  Heatmap saved: heatmap_order_{order}.png")


def plot_trend_lines(results, output_dir):
    """Plot trends of the error vs each individual parameter."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # 1. Error vs cut-off frequency
    freqs = sorted(set(r["cutoff_freq"] for r in results))
    avg_by_freq = [
        np.mean([r["mean_error"] for r in results if r["cutoff_freq"] == f])
        for f in freqs
    ]
    axes[0].plot(freqs, avg_by_freq, "o-b", linewidth=2)
    axes[0].set_xlabel("Cut-off frequency (Hz)")
    axes[0].set_ylabel("Mean error (m)")
    axes[0].set_title("Cut-off frequency", fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(freqs)

    # 2. Error vs filter order
    orders = sorted(set(r["filter_order"] for r in results))
    avg_by_order = [
        np.mean([r["mean_error"] for r in results if r["filter_order"] == o])
        for o in orders
    ]
    axes[1].plot(orders, avg_by_order, "o-g", linewidth=2)
    axes[1].set_xlabel("Filter order")
    axes[1].set_title("Filter order", fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(orders)

    # 3. Error vs PCA percentile
    pcas = sorted(set(r["pca_percentile"] for r in results))
    avg_by_pca = [
        np.mean([r["mean_error"] for r in results if r["pca_percentile"] == p])
        for p in pcas
    ]
    axes[2].plot(pcas, avg_by_pca, "o-r", linewidth=2)
    axes[2].set_xlabel("PCA percentile")
    axes[2].set_title("PCA percentile", fontweight="bold")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(pcas)

    # 4. Error vs Madgwick gain (Madgwick rows only)
    gains = sorted(
        set(r["madgwick_gain"] for r in results if r["fusion_filter"] == "madgwick")
    )
    if gains:
        avg_by_gain = [
            np.mean(
                [
                    r["mean_error"]
                    for r in results
                    if r["madgwick_gain"] == g and r["fusion_filter"] == "madgwick"
                ]
            )
            for g in gains
        ]
        axes[3].plot(gains, avg_by_gain, "o-c", linewidth=2)
        axes[3].set_xlabel("Madgwick gain")
        axes[3].set_title("Madgwick gain (filter = madgwick)", fontweight="bold")
        axes[3].grid(True, alpha=0.3)
        axes[3].set_xticks(gains)

    # 5. Error vs fusion filter
    filters = sorted(set(r["fusion_filter"] for r in results))
    avg_by_filter = [
        np.mean([r["mean_error"] for r in results if r["fusion_filter"] == f])
        for f in filters
    ]
    axes[4].bar(
        filters, avg_by_filter, color=["#4472C4", "#ED7D31"], alpha=0.8, width=0.4
    )
    axes[4].set_ylabel("Mean error (m)")
    axes[4].set_title("Fusion filter")
    axes[4].grid(True, alpha=0.3)

    # 6. Error vs pivot height threshold
    pivots = sorted(set(r["pivot_height_threshold"] for r in results))
    avg_by_pivot = [
        np.mean([r["mean_error"] for r in results if r["pivot_height_threshold"] == p])
        for p in pivots
    ]
    axes[5].plot(pivots, avg_by_pivot, "o-m", linewidth=2)
    axes[5].set_xlabel("Pivot height threshold (m)")
    axes[5].set_title("Pivot height threshold")
    axes[5].grid(True, alpha=0.3)
    axes[5].set_xticks(pivots)

    plt.suptitle(
        "Mean global trends per hyperparameter",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_dir / "trend_lines.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Trends saved: trend_lines.png")


def print_top_n(results, n=5):
    """Print the top-N best combinations."""
    sorted_results = sorted(results, key=lambda r: r["mean_error"])
    print(f"\n{'='*110}")
    print(f"  Top-{n} best hyperparameter combinations")
    print(f"{'='*110}")
    print(
        f"  {'#':>3}  {'Cutoff':>6}  {'Order':>5}  {'PCA%':>4}  {'Gain':>6}  {'Filter':>8}  {'Pivot':>6}  {'Swing':>5}  {'Mean error':>11}  {'Std':>8}"
    )
    print(
        f"  {'-'*3}  {'-'*6}  {'-'*5}  {'-'*4}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*5}  {'-'*11}  {'-'*8}"
    )
    for i, r in enumerate(sorted_results[:n]):
        print(
            f"  {i+1:>3}  {r['cutoff_freq']:>6}  {r['filter_order']:>5}  "
            f"{r['pca_percentile']:>4}  {r['madgwick_gain']:>6.3f}  "
            f"{r['fusion_filter']:>8}  {r['pivot_height_threshold']:>6.3f}  "
            f"{r['swing_velocity_threshold']:>5.2f}  {r['mean_error']:>11.6f}  "
            f"{r['std_error']:>8.6f}"
        )
    print(f"{'='*110}")

    # Also the 5 worst
    print(f"\n  Bottom-{n} worst combinations")
    print(
        f"  {'-'*3}  {'-'*6}  {'-'*5}  {'-'*4}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*5}  {'-'*11}  {'-'*8}"
    )
    for i, r in enumerate(sorted_results[-n:]):
        print(
            f"  {i+1:>3}  {r['cutoff_freq']:>6}  {r['filter_order']:>5}  "
            f"{r['pca_percentile']:>4}  {r['madgwick_gain']:>6.3f}  "
            f"{r['fusion_filter']:>8}  {r['pivot_height_threshold']:>6.3f}  "
            f"{r['swing_velocity_threshold']:>5.2f}  {r['mean_error']:>11.6f}  "
            f"{r['std_error']:>8.6f}"
        )
    print(f"{'='*110}\n")


def plot_summary_table(results, output_dir, n=10):
    """Produce an image with the summary table of the best combinations."""
    sorted_results = sorted(results, key=lambda r: r["mean_error"])[:n]

    fig, ax = plt.subplots(figsize=(14, max(3, 0.4 * n + 1.5)))
    ax.axis("off")

    headers = [
        "#",
        "Cutoff (Hz)",
        "Order",
        "PCA %",
        "Gain",
        "Filter",
        "Pivot (m)",
        "Swing (rad/s)",
        "Mean error",
        "Std",
    ]
    cell_data = []
    for i, r in enumerate(sorted_results):
        cell_data.append(
            [
                str(i + 1),
                str(r["cutoff_freq"]),
                str(r["filter_order"]),
                str(r["pca_percentile"]),
                f"{r['madgwick_gain']:.3f}",
                str(r["fusion_filter"]),
                f"{r['pivot_height_threshold']:.3f}",
                f"{r['swing_velocity_threshold']:.2f}",
                f"{r['mean_error']:.6f}",
                f"{r['std_error']:.6f}",
            ]
        )

    table = ax.table(
        cellText=cell_data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    # Colour the headers
    for j, header in enumerate(headers):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Colour alternate rows
    for i in range(len(cell_data)):
        color = "#D6E4F0" if i % 2 == 0 else "white"
        for j in range(len(headers)):
            table[i + 1, j].set_facecolor(color)

    # Highlight the best one
    for j in range(len(headers)):
        table[1, j].set_facecolor("#92D050")

    ax.set_title(
        f"Top-{n} best combinations of the 9 hyperparameters",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    fig.savefig(output_dir / "summary_table.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Summary table saved: summary_table.png")


def format_time(seconds):
    """Format a number of seconds as h/m/s."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    elif minutes > 0:
        return f"{minutes}m {secs:02d}s"
    else:
        return f"{secs}s"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hyperparameter experiment matrix for GaitTracking"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="DataSet",
        help="Dataset folder containing one sub-folder per subject (default: DataSet/ in the repository root)",
    )
    args = parser.parse_args()

    # ─────────────────────────────────────────────────────────
    # 1. Load data_split.json and build the file lists
    # ─────────────────────────────────────────────────────────
    if not DATA_SPLIT_PATH.exists():
        print(f"[ERROR] {DATA_SPLIT_PATH} not found")
        sys.exit(1)

    with open(DATA_SPLIT_PATH) as f:
        data_split = json.load(f)

    train_subjects = data_split["train_subjects"]
    val_subjects = data_split["val_subjects"]
    test_subjects = data_split["test_subjects"]

    train_files = get_files_from_split(args.dataset, train_subjects)
    val_files = get_files_from_split(args.dataset, val_subjects)
    test_files = get_files_from_split(args.dataset, test_subjects)

    print(f"\nTrain: {len(train_subjects)} subjects → {len(train_files)} files")
    print(f"Val:   {len(val_subjects)} subjects → {len(val_files)} files")
    print(f"Test:  {len(test_subjects)} subjects → {len(test_files)} files")

    if len(train_files) == 0:
        print("[ERROR] No training files found.")
        sys.exit(1)

    # ─────────────────────────────────────────────────────────
    # 2. Generate all the combinations
    # ─────────────────────────────────────────────────────────
    combinations = list(
        itertools.product(
            CUTOFF_FREQS,
            FILTER_ORDERS,
            PCA_PERCENTILES,
            MADGWICK_GAINS,
            SLERP_CLIP_MINS,
            CALIBRATION_SAMPLES_LIST,
            FUSION_FILTERS,
            PIVOT_HEIGHT_THRESHOLDS,
            SWING_VELOCITY_THRESHOLDS,
        )
    )
    total = len(combinations)
    print(f"Total combinations to evaluate: {total}")
    print()

    # ─────────────────────────────────────────────────────────
    # 3. Create the output directory
    # ─────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────
    # 4. Run the experiments (with incremental CSV saving)
    # ─────────────────────────────────────────────────────────
    all_results = []
    start_time = time.time()
    cumulative_times = []  # List of (index, cumulative_time) for the regression

    csv_path = OUTPUT_DIR / "experiment_matrix.csv"
    csv_fieldnames = [
        "cutoff_freq",
        "filter_order",
        "pca_percentile",
        "madgwick_gain",
        "slerp_clip_min",
        "calibration_samples",
        "fusion_filter",
        "pivot_height_threshold",
        "swing_velocity_threshold",
        "mean_error",
        "std_error",
        "min_error",
        "max_error",
        "mean_per_point",
        "std_per_point",
        "n_valid",
        "n_total",
    ]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames, delimiter=";")
    csv_writer.writeheader()
    csv_file.flush()
    print(f"  CSV incremental: {csv_path}")

    for idx, (
        cutoff,
        order,
        pca,
        gain,
        slerp_min,
        cal_samples,
        fusion_f,
        pivot_t,
        swing_t,
    ) in enumerate(combinations, 1):
        t0 = time.time()
        print(
            f"[{idx}/{total}] cutoff={cutoff}Hz, order={order}, pca={pca}%, gain={gain}, filter={fusion_f}, pivot={pivot_t}m, swing={swing_t}rad/s  ...",
            end="",
            flush=True,
        )

        result = evaluate_combination(
            train_files,
            cutoff,
            order,
            pca,
            gain,
            slerp_min,
            cal_samples,
            fusion_f,
            pivot_t,
            swing_t,
        )

        elapsed = time.time() - t0
        cumulative_elapsed = time.time() - start_time
        cumulative_times.append((idx, cumulative_elapsed))

        # Estimate the ETA with a linear regression over the cumulative times
        if len(cumulative_times) >= 3:
            # Linear regression: cumulative_time = m * index + b
            xs = np.array([ct[0] for ct in cumulative_times])
            ys = np.array([ct[1] for ct in cumulative_times])
            m, b = np.polyfit(xs, ys, 1)
            predicted_total_time = m * total + b
            eta_seconds = max(0.0, predicted_total_time - cumulative_elapsed)
        else:
            # Fallback: simple average when there are few data points
            avg_per_experiment = cumulative_elapsed / idx
            eta_seconds = avg_per_experiment * (total - idx)
        eta_str = format_time(eta_seconds)

        if result is not None:
            all_results.append(result)
            csv_writer.writerow(result)
            csv_file.flush()
            print(
                f"  error={result['mean_error']:.6f}  ({elapsed:.1f}s | ETA: {eta_str})  [{result['n_valid']}/{result['n_total']} valid]"
            )
        else:
            print(f"  FAILED ({elapsed:.1f}s | ETA: {eta_str})")

    csv_file.close()

    total_time = time.time() - start_time
    print(f"\n{'─'*60}")
    print(f"Experiments completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Valid results: {len(all_results)}/{total}")
    print(f"{'─'*60}")

    if len(all_results) == 0:
        print("No valid results were obtained.")
        sys.exit(1)

    # ─────────────────────────────────────────────────────────
    # 6. Generate visualizations
    # ─────────────────────────────────────────────────────────
    print("\nGenerating visualizations...")
    plot_heatmaps_by_pca(all_results, OUTPUT_DIR)
    plot_heatmaps_by_order(all_results, OUTPUT_DIR)
    plot_trend_lines(all_results, OUTPUT_DIR)
    plot_summary_table(all_results, OUTPUT_DIR, n=10)

    # ─────────────────────────────────────────────────────────
    # 7. Print the summary (train)
    # ─────────────────────────────────────────────────────────
    print_top_n(all_results, n=5)

    # Best combination on train
    best = min(all_results, key=lambda r: r["mean_error"])
    print(f"\n✓ BEST COMBINATION (train):")
    print(f"  Cut-off frequency:          {best['cutoff_freq']} Hz")
    print(f"  Filter order:               {best['filter_order']}")
    print(f"  PCA percentile:             {best['pca_percentile']}%")
    print(f"  Madgwick gain:              {best['madgwick_gain']}")
    print(f"  Fusion filter:              {best['fusion_filter']}")
    print(f"  Pivot height threshold:     {best['pivot_height_threshold']} m")
    print(f"  Swing velocity threshold:   {best['swing_velocity_threshold']} rad/s")
    print(f"  Mean error (train):         {best['mean_error']:.6f} m")

    # ─────────────────────────────────────────────────────────
    # 8. Validation with the best combination (val split)
    # ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"VALIDATION — {len(val_subjects)} subjects: {val_subjects}")
    print(f"  Files found: {len(val_files)}")

    if len(val_files) > 0:
        val_result = evaluate_combination(
            val_files,
            best["cutoff_freq"],
            best["filter_order"],
            best["pca_percentile"],
            best["madgwick_gain"],
            best["slerp_clip_min"],
            best["calibration_samples"],
            best["fusion_filter"],
            best["pivot_height_threshold"],
            best["swing_velocity_threshold"],
        )
        if val_result is not None:
            print(f"  Mean error (val):           {val_result['mean_error']:.6f} m")
            print(f"  Std (val):                  {val_result['std_error']:.6f} m")
            print(
                f"  Valid:                      {val_result['n_valid']}/{val_result['n_total']}"
            )
            save_csv([val_result], OUTPUT_DIR / "val_result.csv")
        else:
            print("  [ERROR] Validation failed.")
    else:
        print("  [WARN] No validation files found, skipping.")

    # ─────────────────────────────────────────────────────────
    # 9. Final test with the best combination (test split)
    # ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"FINAL TEST — {len(test_subjects)} subjects: {test_subjects}")
    print(f"  Files found: {len(test_files)}")

    if len(test_files) > 0:
        test_result = evaluate_combination(
            test_files,
            best["cutoff_freq"],
            best["filter_order"],
            best["pca_percentile"],
            best["madgwick_gain"],
            best["slerp_clip_min"],
            best["calibration_samples"],
            best["fusion_filter"],
            best["pivot_height_threshold"],
            best["swing_velocity_threshold"],
        )
        if test_result is not None:
            print(f"  Mean error (test):          {test_result['mean_error']:.6f} m")
            print(f"  Std (test):                 {test_result['std_error']:.6f} m")
            print(
                f"  Valid:                      {test_result['n_valid']}/{test_result['n_total']}"
            )
            save_csv([test_result], OUTPUT_DIR / "test_result.csv")
        else:
            print("  [ERROR] Test failed.")
    else:
        print("  [WARN] No test files found, skipping.")

    print(f"\n{'─'*60}")
    print(f"Results saved to: {OUTPUT_DIR.resolve()}")
