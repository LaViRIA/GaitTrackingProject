"""
Matriz de Experimentación para Optimización de Hiperparámetros
==============================================================

Este script evalúa el impacto de 9 hiperparámetros en el error de
reconstrucción de la marcha:

  1. Frecuencia de corte del filtro pasa bajas (Hz)
  2. Orden del filtro Butterworth
  3. Percentil PCA (umbral de movimiento para corrección de yaw)
  4. Ganancia del filtro Madgwick
  5. Clamp del peso Slerp de la cadera (slerp_clip_min)
  6. Muestras de calibración en reposo
  7. Filtro de fusión (Madgwick vs EKF)
  8. Umbral de altura para cambio de pivote
  9. Umbral de velocidad angular para swing override
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
    """Construye la lista de tuplas (mcd_file, aruco_file) para los sujetos dados."""
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
# Grilla de parámetros (Targeted Grid: 432 combinaciones)
# ─────────────────────────────────────────────────────────────
CUTOFF_FREQS = [7, 9, 11, 13]  # Hz
FILTER_ORDERS = [5, 6, 7]  # Orden del Butterworth
PCA_PERCENTILES = [5, 10, 20, 30]  # Percentil para umbral PCA
MADGWICK_GAINS = [0.01, 0.02, 0.033]  # Ganancia del filtro Madgwick
SLERP_CLIP_MINS = [
    0.3,
    0.5,
]  # Peso Slerp de cadera mínimo (slerp_clip_max = 1.0 - min)
CALIBRATION_SAMPLES_LIST = [60, 70, 80, 100]  # Muestras de calibración
FUSION_FILTERS = ["madgwick", "ekf"]  # Tipo de filtro de fusión
PIVOT_HEIGHT_THRESHOLDS = [
    0.005,
    0.01,
    0.015,
]  # Umbral de altura para cambio de pivote (m)
SWING_VELOCITY_THRESHOLDS = [0.3, 0.5, 0.7]  # Umbral de velocidad angular (rad/s)

# CUTOFF_FREQS = [5, 7]  # Hz
# FILTER_ORDERS = [4, 5, 6]  # Orden del Butterworth
# PCA_PERCENTILES = [5, 10]  # Percentil para umbral PCA
# MADGWICK_GAINS = [0.01, 0.02]  # Ganancia del filtro Madgwick
# SLERP_CLIP_MINS = [0.3, 0.5]  # Peso Slerp de cadera mínimo (slerp_clip_max = 1.0 - min)
# CALIBRATION_SAMPLES_LIST = [60, 70]  # Muestras de calibración
# FUSION_FILTERS = ["madgwick"]  # Tipo de filtro de fusión
# PIVOT_HEIGHT_THRESHOLDS = [
#     0.005,
#     0.01,
# ]  # Umbral de altura para cambio de pivote (m)
# SWING_VELOCITY_THRESHOLDS = [0.3, 0.5]  # Umbral de velocidad angular (rad/s)

OUTPUT_DIR = Path("Results/experiment_matrix")

# Valores por defecto para graficar/comparar de forma individual
DEFAULT_VALS = {
    "cutoff_freq": 9,
    "filter_order": 6,
    "pca_percentile": 20,
    "madgwick_gain": 0.02,
    "slerp_clip_min": 0.3,
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
    """Procesa un solo par de archivos con los parámetros dados."""
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
        # Error euclidiano medio por punto (promedio de las 5 articulaciones)
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
    """Evalúa una combinación de hiperparámetros sobre todo el dataset."""
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
    """Guarda los resultados en un archivo CSV."""
    if not results:
        return
    fieldnames = results[0].keys()
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(results)
    print(f"  CSV guardado: {filepath}")


def plot_heatmaps_by_pca(results, output_dir):
    """Genera heatmaps (freq_corte vs orden) para cada percentil PCA, manteniendo otros defaults."""
    # Filtramos donde otros parámetros tengan su valor por defecto (o el más cercano)
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
        ax.set_yticklabels([f"Orden {o}" for o in orders])
        ax.set_xlabel("Frecuencia de Corte (Hz)")
        ax.set_ylabel("Orden del Filtro")
        ax.set_title(f"Error Medio (m) — PCA={pca}% (Filtros en defaults)")

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

        plt.colorbar(im, ax=ax, label="Error Euclidiano Medio (m)")
        plt.tight_layout()
        fig.savefig(output_dir / f"heatmap_pca_{pca}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Heatmap guardado: heatmap_pca_{pca}.png")


def plot_heatmaps_by_order(results, output_dir):
    """Genera heatmaps (freq_corte vs percentil_PCA) para cada orden del filtro, en defaults."""
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
        ax.set_xlabel("Frecuencia de Corte (Hz)")
        ax.set_ylabel("Percentil PCA")
        ax.set_title(f"Error Medio (m) — Orden del Filtro {order} (Defaults)")

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

        plt.colorbar(im, ax=ax, label="Error Euclidiano Medio (m)")
        plt.tight_layout()
        fig.savefig(
            output_dir / f"heatmap_order_{order}.png", dpi=200, bbox_inches="tight"
        )
        plt.close(fig)
        print(f"  Heatmap guardado: heatmap_order_{order}.png")


def plot_trend_lines(results, output_dir):
    """Genera gráficas de tendencia del error vs cada parámetro individual."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # 1. Error vs Frecuencia de corte
    freqs = sorted(set(r["cutoff_freq"] for r in results))
    avg_by_freq = [
        np.mean([r["mean_error"] for r in results if r["cutoff_freq"] == f])
        for f in freqs
    ]
    axes[0].plot(freqs, avg_by_freq, "o-b", linewidth=2)
    axes[0].set_xlabel("Frecuencia de Corte (Hz)")
    axes[0].set_ylabel("Error Medio (m)")
    axes[0].set_title("Frecuencia de Corte", fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(freqs)

    # 2. Error vs Orden del filtro
    orders = sorted(set(r["filter_order"] for r in results))
    avg_by_order = [
        np.mean([r["mean_error"] for r in results if r["filter_order"] == o])
        for o in orders
    ]
    axes[1].plot(orders, avg_by_order, "o-g", linewidth=2)
    axes[1].set_xlabel("Orden del Filtro")
    axes[1].set_title("Orden del Filtro", fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(orders)

    # 3. Error vs Percentil PCA
    pcas = sorted(set(r["pca_percentile"] for r in results))
    avg_by_pca = [
        np.mean([r["mean_error"] for r in results if r["pca_percentile"] == p])
        for p in pcas
    ]
    axes[2].plot(pcas, avg_by_pca, "o-r", linewidth=2)
    axes[2].set_xlabel("Percentil PCA")
    axes[2].set_title("Percentil PCA", fontweight="bold")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(pcas)

    # 4. Error vs Ganancia Madgwick (solo filtrando madgwick)
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
        axes[3].set_xlabel("Ganancia Madgwick")
        axes[3].set_title("Ganancia Madgwick (filtro = madgwick)", fontweight="bold")
        axes[3].grid(True, alpha=0.3)
        axes[3].set_xticks(gains)

    # 5. Error vs Filtro de Fusión
    filters = sorted(set(r["fusion_filter"] for r in results))
    avg_by_filter = [
        np.mean([r["mean_error"] for r in results if r["fusion_filter"] == f])
        for f in filters
    ]
    axes[4].bar(
        filters, avg_by_filter, color=["#4472C4", "#ED7D31"], alpha=0.8, width=0.4
    )
    axes[4].set_ylabel("Error Medio (m)")
    axes[4].set_title("Filtro de Fusión")
    axes[4].grid(True, alpha=0.3)

    # 6. Error vs Umbral de Altura Pivote
    pivots = sorted(set(r["pivot_height_threshold"] for r in results))
    avg_by_pivot = [
        np.mean([r["mean_error"] for r in results if r["pivot_height_threshold"] == p])
        for p in pivots
    ]
    axes[5].plot(pivots, avg_by_pivot, "o-m", linewidth=2)
    axes[5].set_xlabel("Umbral de Altura Pivote (m)")
    axes[5].set_title("Umbral Altura Pivote")
    axes[5].grid(True, alpha=0.3)
    axes[5].set_xticks(pivots)

    plt.suptitle(
        "Tendencias Globales Promedio por Hiperparámetro",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_dir / "trend_lines.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Tendencias guardadas: trend_lines.png")


def print_top_n(results, n=5):
    """Imprime las Top-N mejores combinaciones."""
    sorted_results = sorted(results, key=lambda r: r["mean_error"])
    print(f"\n{'='*110}")
    print(f"  Top-{n} Mejores Combinaciones de Hiperparámetros")
    print(f"{'='*110}")
    print(
        f"  {'#':>3}  {'Cutoff':>6}  {'Order':>5}  {'PCA%':>4}  {'Gain':>6}  {'Filter':>8}  {'Pivot':>6}  {'Swing':>5}  {'Error Medio':>11}  {'Std':>8}"
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

    # También las peores 5
    print(f"\n  Bottom-{n} Peores Combinaciones")
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
    """Genera una imagen con la tabla resumen de las mejores combinaciones."""
    sorted_results = sorted(results, key=lambda r: r["mean_error"])[:n]

    fig, ax = plt.subplots(figsize=(14, max(3, 0.4 * n + 1.5)))
    ax.axis("off")

    headers = [
        "#",
        "Cutoff (Hz)",
        "Orden",
        "PCA %",
        "Gain",
        "Filter",
        "Pivot (m)",
        "Swing (rad/s)",
        "Error Medio",
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

    # Colorear encabezados
    for j, header in enumerate(headers):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Colorear filas alternas
    for i in range(len(cell_data)):
        color = "#D6E4F0" if i % 2 == 0 else "white"
        for j in range(len(headers)):
            table[i + 1, j].set_facecolor(color)

    # Resaltar la mejor
    for j in range(len(headers)):
        table[1, j].set_facecolor("#92D050")

    ax.set_title(
        f"Top-{n} Mejores Combinaciones de los 9 Hiperparámetros",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    fig.savefig(output_dir / "summary_table.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Tabla resumen guardada: summary_table.png")


def format_time(seconds):
    """Formatea una cantidad de segundos en h/m/s."""
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
        description="Matriz de experimentación de hiperparámetros para GaitTracking"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Ruta a la carpeta raíz del dataset (ej: /media/user/data/dataset)",
    )
    args = parser.parse_args()

    # ─────────────────────────────────────────────────────────
    # 1. Cargar data_split.json y construir listas de archivos
    # ─────────────────────────────────────────────────────────
    if not DATA_SPLIT_PATH.exists():
        print(f"[ERROR] No se encontró {DATA_SPLIT_PATH}")
        sys.exit(1)

    with open(DATA_SPLIT_PATH) as f:
        data_split = json.load(f)

    train_subjects = data_split["train_subjects"]
    val_subjects = data_split["val_subjects"]
    test_subjects = data_split["test_subjects"]

    train_files = get_files_from_split(args.dataset, train_subjects)
    val_files = get_files_from_split(args.dataset, val_subjects)
    test_files = get_files_from_split(args.dataset, test_subjects)

    print(f"\nTrain: {len(train_subjects)} sujetos → {len(train_files)} archivos")
    print(f"Val:   {len(val_subjects)} sujetos → {len(val_files)} archivos")
    print(f"Test:  {len(test_subjects)} sujetos → {len(test_files)} archivos")

    if len(train_files) == 0:
        print("[ERROR] No se encontraron archivos de entrenamiento.")
        sys.exit(1)

    # ─────────────────────────────────────────────────────────
    # 2. Generar todas las combinaciones
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
    print(f"Total de combinaciones a evaluar: {total}")
    print()

    # ─────────────────────────────────────────────────────────
    # 3. Crear directorio de salida
    # ─────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────
    # 4. Ejecutar experimentos (con guardado incremental de CSV)
    # ─────────────────────────────────────────────────────────
    all_results = []
    start_time = time.time()
    cumulative_times = []  # Lista de (índice, tiempo_acumulado) para regresión

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

        # Estimar ETA usando regresión lineal sobre tiempos acumulados
        if len(cumulative_times) >= 3:
            # Regresión lineal: tiempo_acumulado = m * índice + b
            xs = np.array([ct[0] for ct in cumulative_times])
            ys = np.array([ct[1] for ct in cumulative_times])
            m, b = np.polyfit(xs, ys, 1)
            predicted_total_time = m * total + b
            eta_seconds = max(0.0, predicted_total_time - cumulative_elapsed)
        else:
            # Fallback: promedio simple cuando hay pocos datos
            avg_per_experiment = cumulative_elapsed / idx
            eta_seconds = avg_per_experiment * (total - idx)
        eta_str = format_time(eta_seconds)

        if result is not None:
            all_results.append(result)
            csv_writer.writerow(result)
            csv_file.flush()
            print(
                f"  error={result['mean_error']:.6f}  ({elapsed:.1f}s | ETA: {eta_str})  [{result['n_valid']}/{result['n_total']} válidos]"
            )
        else:
            print(f"  FALLÓ ({elapsed:.1f}s | ETA: {eta_str})")

    csv_file.close()

    total_time = time.time() - start_time
    print(f"\n{'─'*60}")
    print(f"Experimentos completados en {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Resultados válidos: {len(all_results)}/{total}")
    print(f"{'─'*60}")

    if len(all_results) == 0:
        print("No se obtuvieron resultados válidos.")
        sys.exit(1)

    # ─────────────────────────────────────────────────────────
    # 6. Generar visualizaciones
    # ─────────────────────────────────────────────────────────
    print("\nGenerando visualizaciones...")
    plot_heatmaps_by_pca(all_results, OUTPUT_DIR)
    plot_heatmaps_by_order(all_results, OUTPUT_DIR)
    plot_trend_lines(all_results, OUTPUT_DIR)
    plot_summary_table(all_results, OUTPUT_DIR, n=10)

    # ─────────────────────────────────────────────────────────
    # 7. Imprimir resumen (train)
    # ─────────────────────────────────────────────────────────
    print_top_n(all_results, n=5)

    # Mejor combinación según train
    best = min(all_results, key=lambda r: r["mean_error"])
    print(f"\n✓ MEJOR COMBINACIÓN (train):")
    print(f"  Frecuencia de corte:        {best['cutoff_freq']} Hz")
    print(f"  Orden del filtro:           {best['filter_order']}")
    print(f"  Percentil PCA:              {best['pca_percentile']}%")
    print(f"  Ganancia Madgwick:          {best['madgwick_gain']}")
    print(f"  Filtro de fusión:           {best['fusion_filter']}")
    print(f"  Umbral de altura pivote:    {best['pivot_height_threshold']} m")
    print(f"  Umbral velocidad swing:     {best['swing_velocity_threshold']} rad/s")
    print(f"  Error medio (train):        {best['mean_error']:.6f} m")

    # ─────────────────────────────────────────────────────────
    # 8. Validación con la mejor combinación (val split)
    # ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"VALIDACIÓN — {len(val_subjects)} sujetos: {val_subjects}")
    print(f"  Archivos encontrados: {len(val_files)}")

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
            print(f"  Error medio (val):          {val_result['mean_error']:.6f} m")
            print(f"  Std (val):                  {val_result['std_error']:.6f} m")
            print(
                f"  Válidos:                    {val_result['n_valid']}/{val_result['n_total']}"
            )
            save_csv([val_result], OUTPUT_DIR / "val_result.csv")
        else:
            print("  [ERROR] Validación falló.")
    else:
        print("  [WARN] No se encontraron archivos de validación, se omite.")

    # ─────────────────────────────────────────────────────────
    # 9. Test final con la mejor combinación (test split)
    # ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"TEST FINAL — {len(test_subjects)} sujetos: {test_subjects}")
    print(f"  Archivos encontrados: {len(test_files)}")

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
            print(f"  Error medio (test):         {test_result['mean_error']:.6f} m")
            print(f"  Std (test):                 {test_result['std_error']:.6f} m")
            print(
                f"  Válidos:                    {test_result['n_valid']}/{test_result['n_total']}"
            )
            save_csv([test_result], OUTPUT_DIR / "test_result.csv")
        else:
            print("  [ERROR] Test falló.")
    else:
        print("  [WARN] No se encontraron archivos de test, se omite.")

    print(f"\n{'─'*60}")
    print(f"Resultados guardados en: {OUTPUT_DIR.resolve()}")
