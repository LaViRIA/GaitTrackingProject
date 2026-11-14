"""
Analysis of the experiment-matrix results
=======================================================

Reads the CSV produced by experiment_matrix.py and produces:
  1. Console report (top-N best/worst combinations)
  2. Heatmaps for pairs of hyperparameters
  3. Trend plots (error vs each hyperparameter)
  4. Box plots per hyperparameter
  5. Correlation matrix between parameters and error
  6. Summary table (image)
  7. Sensitivity analysis (variance explained per parameter)

Usage:
  python analyze_results.py                          # uses the default CSV
  python analyze_results.py path/to/file.csv         # specific CSV
"""

import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
DEFAULT_CSV = Path("Results/experiment_matrix/experiment_matrix.csv")
OUTPUT_DIR = Path("Results/analysis")

# Hyperparameter and metric columns
HYPERPARAM_COLS = [
    "cutoff_freq",
    "filter_order",
    "pca_percentile",
    "madgwick_gain",
    "slerp_clip_min",
    "calibration_samples",
    "fusion_filter",
    "pivot_height_threshold",
    "swing_velocity_threshold",
]

METRIC_COLS = [
    "mean_error",
    "std_error",
    "min_error",
    "max_error",
    "mean_per_point",
    "std_per_point",
    "n_valid",
    "n_total",
]


# ─────────────────────────────────────────────────────────────
# 1. Leer CSV
# ─────────────────────────────────────────────────────────────
def read_csv(filepath: Path) -> list[dict]:
    """Read the CSV with ';' delimiter and convert the numeric types."""
    results = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            parsed = {}
            for key, val in row.items():
                key = key.strip()
                val = val.strip()
                # Try to convert to a number
                if key == "fusion_filter":
                    parsed[key] = val
                else:
                    try:
                        parsed[key] = int(val)
                    except ValueError:
                        try:
                            parsed[key] = float(val)
                        except ValueError:
                            parsed[key] = val
            results.append(parsed)
    return results


# ─────────────────────────────────────────────────────────────
# 2. Console report
# ─────────────────────────────────────────────────────────────
def print_summary(results: list[dict]):
    """Print general statistics of the CSV."""
    errors = np.array([r["mean_error"] for r in results])
    print(f"\n{'═' * 70}")
    print(f"  DATASET SUMMARY")
    print(f"{'═' * 70}")
    print(f"  Total combinations:        {len(results)}")
    print(f"  Global mean error:         {errors.mean():.6f} m")
    print(f"  Median error:              {np.median(errors):.6f} m")
    print(f"  Error std. deviation:      {errors.std():.6f} m")
    print(f"  Minimum error:             {errors.min():.6f} m")
    print(f"  Maximum error:             {errors.max():.6f} m")
    print(f"  Range:                     {errors.max() - errors.min():.6f} m")
    print(f"{'═' * 70}")

    # Unique values per hyperparameter
    print(f"\n  Unique values per hyperparameter:")
    print(f"  {'─' * 50}")
    for col in HYPERPARAM_COLS:
        unique_vals = sorted(set(str(r[col]) for r in results))
        print(f"  {col:30s} → {', '.join(unique_vals)}")
    print(f"  {'─' * 50}")


def print_top_n(results: list[dict], n: int = 10):
    """Print the top-N best and worst combinations."""
    sorted_results = sorted(results, key=lambda r: r["mean_error"])

    print(f"\n{'═' * 120}")
    print(f"  ✅ Top-{n} BEST hyperparameter combinations")
    print(f"{'═' * 120}")
    _print_table_header()
    for i, r in enumerate(sorted_results[:n]):
        _print_table_row(i + 1, r)
    print(f"{'═' * 120}")

    print(f"\n  ❌ Bottom-{n} WORST combinations")
    print(f"{'─' * 120}")
    _print_table_header()
    for i, r in enumerate(sorted_results[-n:]):
        _print_table_row(i + 1, r)
    print(f"{'═' * 120}\n")


def _print_table_header():
    print(
        f"  {'#':>3}  {'Cutoff':>6}  {'Order':>5}  {'PCA%':>4}  {'Gain':>6}  "
        f"{'Slerp':>5}  {'CalSamp':>7}  {'Filter':>8}  {'Pivot':>6}  "
        f"{'Swing':>5}  {'Mean error':>11}  {'Std':>8}  {'PerPoint':>10}"
    )
    print(
        f"  {'─' * 3}  {'─' * 6}  {'─' * 5}  {'─' * 4}  {'─' * 6}  "
        f"{'─' * 5}  {'─' * 7}  {'─' * 8}  {'─' * 6}  "
        f"{'─' * 5}  {'─' * 11}  {'─' * 8}  {'─' * 10}"
    )


def _print_table_row(idx, r):
    print(
        f"  {idx:>3}  {r['cutoff_freq']:>6}  {r['filter_order']:>5}  "
        f"{r['pca_percentile']:>4}  {r['madgwick_gain']:>6.3f}  "
        f"{r['slerp_clip_min']:>5.2f}  {r['calibration_samples']:>7}  "
        f"{r['fusion_filter']:>8}  {r['pivot_height_threshold']:>6.3f}  "
        f"{r['swing_velocity_threshold']:>5.2f}  {r['mean_error']:>11.6f}  "
        f"{r['std_error']:>8.6f}  {r['mean_per_point']:>10.6f}"
    )


# ─────────────────────────────────────────────────────────────
# 3. Trend plots
# ─────────────────────────────────────────────────────────────
def plot_trend_lines(results: list[dict], output_dir: Path):
    """Mean error vs each individual hyperparameter."""
    numeric_params = [p for p in HYPERPARAM_COLS if p != "fusion_filter"]
    n_params = len(numeric_params) + 1  # +1 for fusion_filter (categorical)
    cols = 3
    rows = (n_params + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    axes = axes.flatten()

    colors = plt.cm.Set2(np.linspace(0, 1, n_params))

    for idx, param in enumerate(numeric_params):
        values = sorted(set(r[param] for r in results))
        means = []
        stds = []
        for v in values:
            errs = [r["mean_error"] for r in results if r[param] == v]
            means.append(np.mean(errs))
            stds.append(np.std(errs))

        means = np.array(means)
        stds = np.array(stds)

        axes[idx].plot(values, means, "o-", color=colors[idx], linewidth=2, markersize=7)
        axes[idx].fill_between(
            values, means - stds, means + stds, alpha=0.2, color=colors[idx]
        )
        axes[idx].set_xlabel(param, fontsize=10)
        axes[idx].set_ylabel("Mean error (m)", fontsize=9)
        axes[idx].set_title(param.replace("_", " ").title(), fontsize=11, fontweight="bold")
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xticks(values)

    # Fusion filter (categorical)
    idx = len(numeric_params)
    filters = sorted(set(r["fusion_filter"] for r in results))
    filter_means = [np.mean([r["mean_error"] for r in results if r["fusion_filter"] == f]) for f in filters]
    filter_stds = [np.std([r["mean_error"] for r in results if r["fusion_filter"] == f]) for f in filters]
    bar_colors = ["#4472C4", "#ED7D31"][:len(filters)]
    bars = axes[idx].bar(filters, filter_means, yerr=filter_stds, color=bar_colors, alpha=0.85, width=0.4, capsize=5)
    axes[idx].set_ylabel("Mean error (m)", fontsize=9)
    axes[idx].set_title("Fusion filter", fontsize=11, fontweight="bold")
    axes[idx].grid(True, alpha=0.3, axis="y")
    for bar, mean in zip(bars, filter_means):
        axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                      f"{mean:.5f}", ha="center", va="bottom", fontsize=8)

    # Hide the unused axes
    for i in range(idx + 1, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(
        "Trends: mean error vs each hyperparameter",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    fig.savefig(output_dir / "trend_lines.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Trends saved: {output_dir / 'trend_lines.png'}")


# ─────────────────────────────────────────────────────────────
# 4. Box plots per hyperparameter
# ─────────────────────────────────────────────────────────────
def plot_box_plots(results: list[dict], output_dir: Path):
    """Box plot of the error distribution for each value of each hyperparameter."""
    params_to_plot = HYPERPARAM_COLS
    n_params = len(params_to_plot)
    cols = 3
    rows = (n_params + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    axes = axes.flatten()

    for idx, param in enumerate(params_to_plot):
        values = sorted(set(str(r[param]) for r in results))
        data_groups = []
        for v in values:
            errs = [r["mean_error"] for r in results if str(r[param]) == v]
            data_groups.append(errs)

        bp = axes[idx].boxplot(
            data_groups, labels=values, patch_artist=True,
            medianprops=dict(color="red", linewidth=2),
            flierprops=dict(marker=".", markersize=3, alpha=0.4)
        )
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(values)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        axes[idx].set_xlabel(param, fontsize=9)
        axes[idx].set_ylabel("Mean error (m)", fontsize=9)
        axes[idx].set_title(param.replace("_", " ").title(), fontsize=11, fontweight="bold")
        axes[idx].grid(True, alpha=0.3, axis="y")

    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(
        "Error distribution per value of each hyperparameter",
        fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    fig.savefig(output_dir / "box_plots.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Box plots saved: {output_dir / 'box_plots.png'}")


# ─────────────────────────────────────────────────────────────
# 5. Heatmaps for pairs of hyperparameters
# ─────────────────────────────────────────────────────────────
def plot_pairwise_heatmaps(results: list[dict], output_dir: Path):
    """Plot heatmaps for key pairs of hyperparameters."""
    pairs = [
        ("cutoff_freq", "filter_order"),
        ("cutoff_freq", "pca_percentile"),
        ("cutoff_freq", "madgwick_gain"),
        ("filter_order", "pca_percentile"),
        ("pivot_height_threshold", "swing_velocity_threshold"),
        ("calibration_samples", "pca_percentile"),
    ]

    for param_x, param_y in pairs:
        x_vals = sorted(set(r[param_x] for r in results))
        y_vals = sorted(set(r[param_y] for r in results))

        matrix = np.full((len(y_vals), len(x_vals)), np.nan)

        for r in results:
            try:
                xi = x_vals.index(r[param_x])
                yi = y_vals.index(r[param_y])
            except ValueError:
                continue
            # Average if there are several values
            if np.isnan(matrix[yi, xi]):
                matrix[yi, xi] = r["mean_error"]
            else:
                matrix[yi, xi] = (matrix[yi, xi] + r["mean_error"]) / 2

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", origin="lower")

        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels([f"{v}" for v in x_vals], fontsize=9)
        ax.set_yticks(range(len(y_vals)))
        ax.set_yticklabels([f"{v}" for v in y_vals], fontsize=9)
        ax.set_xlabel(param_x.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel(param_y.replace("_", " ").title(), fontsize=11)
        ax.set_title(
            f"Mean error: {param_x} vs {param_y}",
            fontsize=12, fontweight="bold"
        )

        # Annotate the cell values
        for i in range(len(y_vals)):
            for j in range(len(x_vals)):
                if not np.isnan(matrix[i, j]):
                    text_color = "white" if matrix[i, j] > np.nanmedian(matrix) else "black"
                    ax.text(
                        j, i, f"{matrix[i, j]:.4f}",
                        ha="center", va="center",
                        color=text_color, fontsize=8, fontweight="bold"
                    )

        plt.colorbar(im, ax=ax, label="Mean Euclidean error (m)")
        plt.tight_layout()
        fname = f"heatmap_{param_x}_vs_{param_y}.png"
        fig.savefig(output_dir / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Heatmap saved: {fname}")


# ─────────────────────────────────────────────────────────────
# 6. Sensitivity analysis
# ─────────────────────────────────────────────────────────────
def sensitivity_analysis(results: list[dict], output_dir: Path):
    """
    Compute the 'importance' of each hyperparameter by measuring how much
    the mean error changes when each parameter is varied individually.
    """
    importance = {}
    for param in HYPERPARAM_COLS:
        values = sorted(set(str(r[param]) for r in results))
        group_means = []
        for v in values:
            errs = [r["mean_error"] for r in results if str(r[param]) == v]
            group_means.append(np.mean(errs))
        # Range of the mean error across the parameter values
        importance[param] = max(group_means) - min(group_means)

    # Sort by importance
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'═' * 60}")
    print(f"  SENSITIVITY ANALYSIS (range of the mean error)")
    print(f"{'═' * 60}")
    for i, (param, imp) in enumerate(sorted_importance):
        bar = "█" * int(imp / max(importance.values()) * 30)
        print(f"  {i + 1:>2}. {param:35s} {imp:.6f} m  {bar}")
    print(f"{'═' * 60}\n")

    # Bar chart
    params = [p.replace("_", "\n") for p, _ in sorted_importance]
    values = [v for _, v in sorted_importance]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(params)))
    bars = ax.barh(params[::-1], values[::-1], color=colors[::-1], edgecolor="gray", alpha=0.9)

    for bar, val in zip(bars, values[::-1]):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.6f} m",
            va="center", fontsize=9
        )

    ax.set_xlabel("Range of the mean error (m)", fontsize=11)
    ax.set_title(
        "Sensitivity: impact of each hyperparameter on the error",
        fontsize=13, fontweight="bold"
    )
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(output_dir / "sensitivity_analysis.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Sensitivity saved: {output_dir / 'sensitivity_analysis.png'}")

    return sorted_importance


# ─────────────────────────────────────────────────────────────
# 7. Visual summary table
# ─────────────────────────────────────────────────────────────
def plot_summary_table(results: list[dict], output_dir: Path, n: int = 15):
    """Produce an image with the summary table of the best combinations."""
    sorted_results = sorted(results, key=lambda r: r["mean_error"])[:n]

    fig, ax = plt.subplots(figsize=(16, max(3, 0.4 * n + 2)))
    ax.axis("off")

    headers = [
        "#", "Cutoff\n(Hz)", "Order", "PCA\n%", "Gain", "Slerp\nMin",
        "Cal.\nSamples", "Filter", "Pivot\n(m)", "Swing\n(rad/s)",
        "Mean\nerror", "Std", "Per\nPoint"
    ]

    cell_data = []
    for i, r in enumerate(sorted_results):
        cell_data.append([
            str(i + 1),
            str(r["cutoff_freq"]),
            str(r["filter_order"]),
            str(r["pca_percentile"]),
            f"{r['madgwick_gain']:.3f}",
            f"{r['slerp_clip_min']:.2f}",
            str(r["calibration_samples"]),
            str(r["fusion_filter"]),
            f"{r['pivot_height_threshold']:.3f}",
            f"{r['swing_velocity_threshold']:.2f}",
            f"{r['mean_error']:.6f}",
            f"{r['std_error']:.6f}",
            f"{r['mean_per_point']:.6f}",
        ])

    table = ax.table(
        cellText=cell_data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.5)

    # Colour the headers
    for j in range(len(headers)):
        table[0, j].set_facecolor("#2E4057")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Colour alternate rows + highlight the best one
    for i in range(len(cell_data)):
        if i == 0:
            color = "#92D050"
        elif i % 2 == 0:
            color = "#D6E4F0"
        else:
            color = "white"
        for j in range(len(headers)):
            table[i + 1, j].set_facecolor(color)

    ax.set_title(
        f"Top-{n} best hyperparameter combinations",
        fontsize=14, fontweight="bold", pad=20
    )
    plt.tight_layout()
    fig.savefig(output_dir / "summary_table.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Summary table saved: {output_dir / 'summary_table.png'}")


# ─────────────────────────────────────────────────────────────
# 8. Histogram of the error distribution
# ─────────────────────────────────────────────────────────────
def plot_error_distribution(results: list[dict], output_dir: Path):
    """Histogram of the mean-error distribution."""
    errors = [r["mean_error"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    n_bins = min(60, len(set(errors)))
    counts, bins, patches = ax.hist(
        errors, bins=n_bins, edgecolor="gray", alpha=0.8, color="#4472C4"
    )

    # Colour gradient
    cm = plt.cm.RdYlGn_r
    norm = plt.Normalize(min(errors), max(errors))
    for patch, val in zip(patches, bins):
        patch.set_facecolor(cm(norm(val)))

    # Reference lines
    ax.axvline(np.mean(errors), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(errors):.5f}")
    ax.axvline(np.median(errors), color="orange", linestyle="--", linewidth=2, label=f"Median: {np.median(errors):.5f}")

    best = min(errors)
    ax.axvline(best, color="green", linestyle="-", linewidth=2, label=f"Best: {best:.5f}")

    ax.set_xlabel("Mean error (m)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Mean-error distribution over all combinations", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(output_dir / "error_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Distribution saved: {output_dir / 'error_distribution.png'}")


# ─────────────────────────────────────────────────────────────
# 9. Detailed best combination
# ─────────────────────────────────────────────────────────────
def print_best_combination(results: list[dict]):
    """Print the best combination with highlighted formatting."""
    best = min(results, key=lambda r: r["mean_error"])

    print(f"\n{'═' * 60}")
    print(f"  🏆 BEST COMBINATION FOUND")
    print(f"{'═' * 60}")
    print(f"  Cut-off frequency:          {best['cutoff_freq']} Hz")
    print(f"  Filter order:               {best['filter_order']}")
    print(f"  PCA percentile:             {best['pca_percentile']}%")
    print(f"  Madgwick gain:              {best['madgwick_gain']}")
    print(f"  Slerp clip min:             {best['slerp_clip_min']}")
    print(f"  Calibration samples:        {best['calibration_samples']}")
    print(f"  Fusion filter:              {best['fusion_filter']}")
    print(f"  Pivot height threshold:     {best['pivot_height_threshold']} m")
    print(f"  Swing velocity threshold:   {best['swing_velocity_threshold']} rad/s")
    print(f"  {'─' * 40}")
    print(f"  Mean error:                 {best['mean_error']:.6f} m")
    print(f"  Std. deviation:             {best['std_error']:.6f} m")
    print(f"  Minimum error:              {best['min_error']:.6f} m")
    print(f"  Maximum error:              {best['max_error']:.6f} m")
    print(f"  Mean error per point:       {best['mean_per_point']:.6f} m")
    print(f"  Valid files:                {best['n_valid']}/{best['n_total']}")
    print(f"{'═' * 60}\n")

    return best


# ─────────────────────────────────────────────────────────────
# 10. Madgwick vs EKF comparison
# ─────────────────────────────────────────────────────────────
def plot_filter_comparison(results: list[dict], output_dir: Path):
    """Compare Madgwick vs EKF with the other parameters held equal."""
    # Group by configuration (without fusion_filter)
    groups = defaultdict(dict)
    for r in results:
        key = tuple(
            r[p] for p in HYPERPARAM_COLS if p != "fusion_filter"
        )
        groups[key][r["fusion_filter"]] = r["mean_error"]

    # Only pairs where both filters exist
    madgwick_errs = []
    ekf_errs = []
    for key, filters in groups.items():
        if "madgwick" in filters and "ekf" in filters:
            madgwick_errs.append(filters["madgwick"])
            ekf_errs.append(filters["ekf"])

    if not madgwick_errs:
        print("  ⚠ No comparable Madgwick vs EKF pairs")
        return

    madgwick_errs = np.array(madgwick_errs)
    ekf_errs = np.array(ekf_errs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    max_val = max(madgwick_errs.max(), ekf_errs.max()) * 1.05
    min_val = min(madgwick_errs.min(), ekf_errs.min()) * 0.95
    axes[0].scatter(madgwick_errs, ekf_errs, alpha=0.4, s=15, c="#4472C4", edgecolors="none")
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="y = x")
    axes[0].set_xlabel("Error Madgwick (m)", fontsize=11)
    axes[0].set_ylabel("Error EKF (m)", fontsize=11)
    axes[0].set_title("Madgwick vs EKF (same config.)", fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect("equal")

    # Histogram of the differences
    diffs = ekf_errs - madgwick_errs  # positive = Madgwick better
    axes[1].hist(diffs, bins=40, edgecolor="gray", alpha=0.8, color="#ED7D31")
    axes[1].axvline(0, color="black", linestyle="-", linewidth=1.5)
    axes[1].axvline(np.mean(diffs), color="red", linestyle="--", linewidth=2,
                    label=f"Mean: {np.mean(diffs):.5f}")
    axes[1].set_xlabel("Difference (EKF - Madgwick)", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].set_title("Distribution of the differences", fontsize=12, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    n_madgwick_wins = np.sum(diffs > 0)
    n_ekf_wins = np.sum(diffs < 0)
    n_ties = np.sum(diffs == 0)
    fig.suptitle(
        f"Filter comparison: Madgwick wins {n_madgwick_wins}x, "
        f"EKF wins {n_ekf_wins}x, ties {n_ties}x",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig.savefig(output_dir / "filter_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Filter comparison: {output_dir / 'filter_comparison.png'}")


# ─────────────────────────────────────────────────────────────
# 11. Detailed analysis of the differences per fusion filter
# ─────────────────────────────────────────────────────────────
def analyze_filter_differences(results: list[dict], output_dir: Path):
    """
    Thorough analysis of the differences between fusion filters.
    Includes:
      - Descriptive statistics per filter
      - Statistical significance test (Mann-Whitney U)
      - Paired comparison per hyperparameter
      - Multi-panel visualization
    """
    from scipy import stats as scipy_stats

    filters = sorted(set(r["fusion_filter"] for r in results))
    if len(filters) < 2:
        print("  ⚠ There is only one fusion-filter type; nothing to compare.")
        return

    # ── Descriptive statistics per filter ──
    filter_data = {}
    for f in filters:
        errs = np.array([r["mean_error"] for r in results if r["fusion_filter"] == f])
        stds = np.array([r["std_error"] for r in results if r["fusion_filter"] == f])
        per_point = np.array([r["mean_per_point"] for r in results if r["fusion_filter"] == f])
        filter_data[f] = {
            "errors": errs,
            "stds": stds,
            "per_point": per_point,
            "n": len(errs),
            "mean": np.mean(errs),
            "median": np.median(errs),
            "std": np.std(errs),
            "min": np.min(errs),
            "max": np.max(errs),
            "q25": np.percentile(errs, 25),
            "q75": np.percentile(errs, 75),
            "mean_per_point": np.mean(per_point),
        }

    # ── Console report ──
    print(f"\n{'═' * 80}")
    print(f"  📊 DETAILED ANALYSIS: DIFFERENCES PER FUSION FILTER")
    print(f"{'═' * 80}")

    print(f"\n  {'Metric':<30}", end="")
    for f in filters:
        print(f"  {f.upper():>12}", end="")
    print(f"  {'Δ (diff)':>12}")
    print(f"  {'─' * 30}", end="")
    for _ in filters:
        print(f"  {'─' * 12}", end="")
    print(f"  {'─' * 12}")

    metrics_to_show = [
        ("N combinations", "n", "{:d}"),
        ("Mean error (m)", "mean", "{:.6f}"),
        ("Median error (m)", "median", "{:.6f}"),
        ("Std. deviation (m)", "std", "{:.6f}"),
        ("Minimum error (m)", "min", "{:.6f}"),
        ("Maximum error (m)", "max", "{:.6f}"),
        ("25th percentile (m)", "q25", "{:.6f}"),
        ("75th percentile (m)", "q75", "{:.6f}"),
        ("Interquartile range (m)", None, None),
        ("Mean error per point (m)", "mean_per_point", "{:.6f}"),
    ]

    for label, key, fmt in metrics_to_show:
        print(f"  {label:<30}", end="")
        vals = []
        for f in filters:
            if key is None:  # IQR
                v = filter_data[f]["q75"] - filter_data[f]["q25"]
                print(f"  {v:>12.6f}", end="")
                vals.append(v)
            elif fmt == "{:d}":
                v = filter_data[f][key]
                print(f"  {v:>12d}", end="")
                vals.append(v)
            else:
                v = filter_data[f][key]
                print(f"  {v:>12.6f}", end="")
                vals.append(v)
        # Difference (first filter - second filter)
        diff = vals[0] - vals[1] if len(vals) == 2 else 0
        if fmt == "{:d}":
            print(f"  {diff:>+12d}")
        else:
            print(f"  {diff:>+12.6f}")

    # ── Statistical significance test ──
    if len(filters) == 2:
        errs_a = filter_data[filters[0]]["errors"]
        errs_b = filter_data[filters[1]]["errors"]
        u_stat, p_value = scipy_stats.mannwhitneyu(errs_a, errs_b, alternative="two-sided")

        print(f"\n  {'─' * 60}")
        print(f"  Significance test (Mann-Whitney U):")
        print(f"    U statistic:    {u_stat:.2f}")
        print(f"    p-valor:        {p_value:.2e}")
        if p_value < 0.001:
            print(f"    Result:         *** VERY significant difference (p < 0.001)")
        elif p_value < 0.01:
            print(f"    Result:         ** Significant difference (p < 0.01)")
        elif p_value < 0.05:
            print(f"    Result:         * Significant difference (p < 0.05)")
        else:
            print(f"    Result:         Not significant (p ≥ 0.05)")

        # ── Paired comparison: which wins with the same configuration ──
        groups = defaultdict(dict)
        for r in results:
            key = tuple(r[p] for p in HYPERPARAM_COLS if p != "fusion_filter")
            groups[key][r["fusion_filter"]] = r["mean_error"]

        wins = {f: 0 for f in filters}
        ties = 0
        total_pairs = 0
        win_margins = {f: [] for f in filters}

        for key, fmap in groups.items():
            if all(f in fmap for f in filters):
                total_pairs += 1
                diff = fmap[filters[0]] - fmap[filters[1]]
                if abs(diff) < 1e-10:
                    ties += 1
                elif diff < 0:
                    wins[filters[0]] += 1
                    win_margins[filters[0]].append(abs(diff))
                else:
                    wins[filters[1]] += 1
                    win_margins[filters[1]].append(abs(diff))

        print(f"\n  Paired comparison (same configuration, {total_pairs} pairs):")
        for f in filters:
            pct = (wins[f] / total_pairs * 100) if total_pairs > 0 else 0
            avg_margin = np.mean(win_margins[f]) if win_margins[f] else 0
            print(f"    {f.upper():>10} wins: {wins[f]:>5}x ({pct:>5.1f}%)  "
                  f"mean margin: {avg_margin:.6f} m")
        if ties > 0:
            print(f"    {'Ties':>10}:     {ties:>5}x ({ties/total_pairs*100:>5.1f}%)")

    # ── Analysis per hyperparameter: which filter is better at each value ──
    numeric_params = [p for p in HYPERPARAM_COLS if p not in ("fusion_filter",)]

    print(f"\n  {'─' * 60}")
    print(f"  Mean error per filter for each hyperparameter value:")
    print(f"  {'─' * 60}")

    param_comparison = {}  # For the plot
    for param in numeric_params:
        values = sorted(set(str(r[param]) for r in results))
        print(f"\n  📌 {param}:")
        param_comparison[param] = {"values": values, "filter_means": {}}
        for f in filters:
            param_comparison[param]["filter_means"][f] = []

        for v in values:
            print(f"     {v:>10}: ", end="")
            filter_means_for_val = {}
            for f in filters:
                errs = [r["mean_error"] for r in results
                        if str(r[param]) == v and r["fusion_filter"] == f]
                m = np.mean(errs) if errs else float("nan")
                filter_means_for_val[f] = m
                param_comparison[param]["filter_means"][f].append(m)
                print(f"  {f}={m:.6f}", end="")

            # Mark the winner
            valid = {k: v for k, v in filter_means_for_val.items() if not np.isnan(v)}
            if len(valid) >= 2:
                best_f = min(valid, key=valid.get)
                diff_val = max(valid.values()) - min(valid.values())
                print(f"   → {best_f} (Δ={diff_val:.6f})")
            else:
                print()

    print(f"{'═' * 80}\n")

    # ── Multi-panel visualization ──
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    filter_colors = {}
    color_palette = ["#4472C4", "#ED7D31", "#70AD47", "#FFC000"]
    for i, f in enumerate(filters):
        filter_colors[f] = color_palette[i % len(color_palette)]

    # Panel 1: comparative violin plot
    ax1 = fig.add_subplot(gs[0, 0])
    violin_data = [filter_data[f]["errors"] for f in filters]
    parts = ax1.violinplot(violin_data, positions=range(len(filters)),
                           showmeans=True, showmedians=True, showextrema=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(filter_colors[filters[i]])
        pc.set_alpha(0.7)
    parts["cmeans"].set_color("red")
    parts["cmedians"].set_color("black")
    ax1.set_xticks(range(len(filters)))
    ax1.set_xticklabels([f.upper() for f in filters], fontsize=11, fontweight="bold")
    ax1.set_ylabel("Mean error (m)", fontsize=10)
    ax1.set_title("Error distribution per filter (violin)", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")
    # Annotate mean and median
    for i, f in enumerate(filters):
        ax1.annotate(
            f"μ={filter_data[f]['mean']:.5f}\nMd={filter_data[f]['median']:.5f}",
            xy=(i, filter_data[f]["mean"]),
            xytext=(i + 0.3, filter_data[f]["mean"]),
            fontsize=8, ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

    # Panel 2: comparative box plot with notch
    ax2 = fig.add_subplot(gs[0, 1])
    bp = ax2.boxplot(
        violin_data, labels=[f.upper() for f in filters],
        patch_artist=True, notch=True,
        medianprops=dict(color="red", linewidth=2),
        flierprops=dict(marker=".", markersize=3, alpha=0.3),
        widths=0.5
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(filter_colors[filters[i]])
        patch.set_alpha(0.7)
    ax2.set_ylabel("Mean error (m)", fontsize=10)
    ax2.set_title("Box-plot comparison with notch (95% CI of the median)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: bars grouped by hyperparameter (4 most sensitive)
    ax3 = fig.add_subplot(gs[1, :])
    # Select the 4 hyperparameters with the largest difference between filters
    param_diffs = {}
    for param in numeric_params:
        total_diff = 0
        for v_idx in range(len(param_comparison[param]["values"])):
            vals = [param_comparison[param]["filter_means"][f][v_idx] for f in filters]
            valid_vals = [x for x in vals if not np.isnan(x)]
            if len(valid_vals) >= 2:
                total_diff += max(valid_vals) - min(valid_vals)
        param_diffs[param] = total_diff
    top_params = sorted(param_diffs.keys(), key=lambda p: param_diffs[p], reverse=True)[:4]

    # Build the grouped bars
    all_labels = []
    all_positions = []
    group_centers = []
    pos = 0
    bar_width = 0.35
    n_filters = len(filters)

    for pi, param in enumerate(top_params):
        values = param_comparison[param]["values"]
        group_start = pos
        for vi, v in enumerate(values):
            for fi, f in enumerate(filters):
                mean_val = param_comparison[param]["filter_means"][f][vi]
                offset = (fi - (n_filters - 1) / 2) * bar_width
                bar = ax3.bar(pos + offset, mean_val, bar_width * 0.9,
                              color=filter_colors[f], alpha=0.8,
                              edgecolor="gray", linewidth=0.5,
                              label=f.upper() if pi == 0 and vi == 0 else "")
            all_labels.append(f"{v}")
            all_positions.append(pos)
            pos += 1
        group_centers.append((group_start + pos - 1) / 2)
        pos += 0.8  # Space between groups

    ax3.set_xticks(all_positions)
    ax3.set_xticklabels(all_labels, fontsize=8, rotation=45, ha="right")
    ax3.set_ylabel("Mean error (m)", fontsize=10)
    ax3.set_title("Per-filter comparison on the most sensitive hyperparameters", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.legend(fontsize=10, loc="upper right")

    # Group labels (hyperparameter name)
    y_min = ax3.get_ylim()[0]
    for center, param in zip(group_centers, top_params):
        ax3.annotate(
            param.replace("_", " ").title(),
            xy=(center, y_min),
            xytext=(center, y_min - (ax3.get_ylim()[1] - y_min) * 0.08),
            fontsize=9, fontweight="bold", ha="center", va="top",
            annotation_clip=False,
            color="#333333"
        )

    # Panel 4: summary metrics table
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis("off")
    table_headers = ["Metric"] + [f.upper() for f in filters] + ["Δ"]
    table_data = []
    metrics_table = [
        ("Mean error (m)", "mean"),
        ("Median (m)", "median"),
        ("Std. deviation (m)", "std"),
        ("Min. error (m)", "min"),
        ("Max. error (m)", "max"),
        ("IQR (m)", None),
        ("Error/point (m)", "mean_per_point"),
    ]
    for label, key in metrics_table:
        row = [label]
        vals = []
        for f in filters:
            if key is None:
                v = filter_data[f]["q75"] - filter_data[f]["q25"]
            else:
                v = filter_data[f][key]
            row.append(f"{v:.6f}")
            vals.append(v)
        diff = vals[0] - vals[1] if len(vals) == 2 else 0
        row.append(f"{diff:+.6f}")
        table_data.append(row)

    tbl = ax4.table(
        cellText=table_data, colLabels=table_headers,
        loc="center", cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.6)
    for j in range(len(table_headers)):
        tbl[0, j].set_facecolor("#2E4057")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(len(table_data)):
        bg = "#E8EEF4" if i % 2 == 0 else "white"
        for j in range(len(table_headers)):
            tbl[i + 1, j].set_facecolor(bg)
        # Colour the Δ column according to its sign
        diff_val = float(table_data[i][-1])
        if diff_val < -1e-7:
            tbl[i + 1, len(table_headers) - 1].set_facecolor("#C6EFCE")
        elif diff_val > 1e-7:
            tbl[i + 1, len(table_headers) - 1].set_facecolor("#FFC7CE")
    ax4.set_title("Summary of metrics per filter", fontsize=12, fontweight="bold", pad=15)

    # Panel 5: overlaid histograms
    ax5 = fig.add_subplot(gs[2, 1])
    for f in filters:
        ax5.hist(
            filter_data[f]["errors"],
            bins=40, alpha=0.55,
            color=filter_colors[f],
            label=f"{f.upper()} (n={filter_data[f]['n']})",
            edgecolor="gray", linewidth=0.5
        )
        ax5.axvline(
            filter_data[f]["mean"], color=filter_colors[f],
            linestyle="--", linewidth=2, alpha=0.9,
            label=f"μ {f.upper()} = {filter_data[f]['mean']:.5f}"
        )
    ax5.set_xlabel("Mean error (m)", fontsize=10)
    ax5.set_ylabel("Frequency", fontsize=10)
    ax5.set_title("Overlaid distributions per filter", fontsize=12, fontweight="bold")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Detailed analysis: differences between fusion filters",
        fontsize=16, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    fig.savefig(output_dir / "filter_fusion_differences.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Filter difference analysis: {output_dir / 'filter_fusion_differences.png'}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Determine the CSV file
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        csv_path = DEFAULT_CSV

    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        print(f"   Usage: python {Path(__file__).name} [path/to/file.csv]")
        sys.exit(1)

    print(f"\n📂 Reading: {csv_path}")

    # 1. Read the data
    results = read_csv(csv_path)
    print(f"   → {len(results)} combinations loaded")

    # 2. Create the output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Console reports
    print_summary(results)
    print_top_n(results, n=10)
    best = print_best_combination(results)

    # 4. Generate visualizations
    print("\n🎨 Generating visualizations...")
    plot_trend_lines(results, OUTPUT_DIR)
    plot_box_plots(results, OUTPUT_DIR)
    plot_pairwise_heatmaps(results, OUTPUT_DIR)
    plot_error_distribution(results, OUTPUT_DIR)
    plot_filter_comparison(results, OUTPUT_DIR)
    sensitivity_analysis(results, OUTPUT_DIR)
    plot_summary_table(results, OUTPUT_DIR, n=15)

    # 5. Detailed analysis of the differences per fusion filter
    print("\n🔬 Analysing the differences between fusion filters...")
    analyze_filter_differences(results, OUTPUT_DIR)

    print(f"\n✅ Analysis complete. Results saved to: {OUTPUT_DIR.resolve()}")
