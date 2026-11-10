"""
generate_manuscript_figures.py
==============================
Generates publication-quality figures for the 'Resultados y Discusión' /
'Results and Discussion' section of the manuscript (MDPI Sensors / IEEE style).

Supports both Spanish ('es') and English ('en') language rendering.
Caches evaluated trial data in 'v2_manuscript/figures/eval_cache.pkl' for
instant regeneration of plots.

Figures produced in v2_manuscript/figures/:
  1. fig2_temporal_error_comparison_{lang}.png / .pdf
     Unified 3-panel figure: Best, Representative (Median), and Worst case error curves.
  2. fig3_statistical_error_distribution_{lang}.png / .pdf
     (a) Boxplots of MPJPE across joints (Analytical vs TCN Lite across all 20 test captures).
     (b) Empirical Cumulative Distribution Function (ECDF) of global 3D position error.
  3. fig4_trajectory_and_drift_analysis_{lang}.png / .pdf
     (a) Ground-plane (X-Z) pelvic trajectory for representative capture.
     (b) Cumulative drift (ATE) as a function of walked distance.
  4. fig5_biomechanical_agreement_{lang}.png / .pdf
     (a) Knee flexion-extension range of motion across gait cycle.
     (b) Bland-Altman agreement plot across X axis for TCN Lite vs Ground Truth.
  5. fig6_3d_joint_trajectories_{lang}.png / .pdf
     3D joint trajectories (ArUco GT vs TCN Lite) and intermediate gait pose reconstruction.
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from multiprocessing import Pool
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import torch

# Ensure writable matplotlib config directory
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_paper"
Path("/tmp/matplotlib_paper").mkdir(parents=True, exist_ok=True)

# Publication styling: Clean, modern, font sizes matching MDPI Sensors template
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "figure.dpi": 300,
        "lines.linewidth": 1.8,
        "grid.alpha": 0.35,
        "grid.linestyle": ":",
    }
)

# Color palette: distinct, accessible, professional
COLOR_GT = "#2CA02C"  # Ground truth green
COLOR_ANA = "#E66101"  # Analytical warm amber / orange
COLOR_TCN = "#5E3C99"  # TCN Lite deep purple / violet
COLOR_TTL = "#B279A2"  # TTL Lite mauve
COLOR_TRANS = "#F1A340"  # Transformer sand
COLOR_STATIC = "#4393C3"  # Static PoseNet blue

DATASET_PATH = Path("/home/badajoz/Documents/DataSet")
OUTPUT_DIR = Path("v2_manuscript/figures")
CACHE_FILE = OUTPUT_DIR / "eval_cache.pkl"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import project utilities
import evaluate_full_neural_vs_analytical as eval_module
from models.nn_static_pivot import (
    discover_captures,
    prepare_static_capture,
    JOINT_NAMES,
    N_JOINTS,
    N_COORDS,
)
from models.nn_tcn_lite import (
    GaitMultiTaskNetTCN,
    predict_full_sequence_multitask as predict_tcn,
)
from process_all import (
    process_files,
    compute_displacement_velocity,
    detect_stride_lengths,
)
from scipy.signal import find_peaks


def run_pipeline_evaluation(use_cache: bool = True):
    """Runs inference across the 5 test subjects (20 captures) or loads from cache."""
    if use_cache and CACHE_FILE.exists():
        print(f"  ✓ Loading evaluation cache from: {CACHE_FILE}")
        with open(CACHE_FILE, "rb") as f:
            data = pickle.load(f)
        return data["trials_data"], data["all_ana"], data["all_tcn"], data["all_static"]

    split_path = Path("data_split.json")
    with open(split_path, "r") as f:
        split = json.load(f)
    test_subjs = [str(s) for s in split["test_subjects"]]
    print(f"Loading test set for subjects: {test_subjs}")

    captures_info = discover_captures(DATASET_PATH)
    subjects_data = {}
    file_paths_by_subject = {}

    for mcd_path, aruco_path, subj in captures_info:
        subj_str = str(subj)
        if subj_str not in test_subjs:
            continue
        cap = prepare_static_capture(mcd_path, aruco_path)
        if cap is not None:
            if subj_str not in subjects_data:
                subjects_data[subj_str] = []
                file_paths_by_subject[subj_str] = []
            subjects_data[subj_str].append(cap)
            file_paths_by_subject[subj_str].append((mcd_path, aruco_path))

    # Pre-compute analytical pipeline in parallel
    all_test_files = []
    for subj in test_subjs:
        all_test_files.extend(file_paths_by_subject[subj])

    print(f"Executing Analytical Pipeline on {len(all_test_files)} test files...")
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

    # Load neural models
    print("Loading TCN Lite model...")
    tcn_ckpt_path = Path("nn-results/tcn_lite/checkpoints/best_tcn_lite.pt")
    ckpt_tcn = torch.load(tcn_ckpt_path, map_location=DEVICE, weights_only=False)
    state_tcn = (
        ckpt_tcn["model_state_dict"] if "model_state_dict" in ckpt_tcn else ckpt_tcn
    )

    # norm_stats del propio checkpoint; el .json compartido puede venir de otro entrenamiento
    if ckpt_tcn.get("norm_stats") is not None:
        norm_stats_tcn = dict(ckpt_tcn["norm_stats"])
    else:
        with open(tcn_ckpt_path.parent / "norm_stats.json", "r") as f:
            norm_stats_tcn = json.load(f)
    for k in ["mean", "std"]:
        norm_stats_tcn[k] = np.asarray(norm_stats_tcn[k], dtype=np.float32)

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
    tcn_d = (ckpt_tcn.get("config") or {}).get(
        "tcn_dilations", [2**i for i in range(num_tcn)]
    )
    window_size_tcn = eval_module.resolve_window_size(None, ckpt_tcn, "TCN Lite")
    head_h_tcn = (
        state_tcn["pose_head.1.weight"].shape[0]
        if "pose_head.1.weight" in state_tcn
        else 48
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

    # Load Static models
    pivot_model, norm_stats_pivot = eval_module.load_pivot_model(
        eval_module.PIVOT_CKPT_PATH
    )
    pose_model, norm_stats_pose = eval_module.load_pose_model(
        eval_module.POSE_CKPT_PATH
    )

    trials_data = []
    all_ana_errors_concat = []
    all_tcn_errors_concat = []
    all_static_errors_concat = []

    BONE_NORM = 0.45

    for subj in test_subjs:
        for cap, (mcd_path, aruco_path) in zip(
            subjects_data[subj], file_paths_by_subject[subj]
        ):
            cap_len = len(cap["gt_relative"])
            gt_global = eval_module.get_ground_truth_global_trajectory(
                mcd_path, aruco_path, cap
            )
            if gt_global is None or len(gt_global) != cap_len:
                continue

            # Analytical trajectory
            first_valid = cap.get("first_valid", 0)
            ana_res = ana_results_dict.get((mcd_path, aruco_path))
            if ana_res is None:
                continue
            _, _, rec, _, _, _, _ = ana_res
            if len(rec["hip_r"]) < (first_valid + cap_len):
                continue

            hip_center = (rec["hip_r"] + rec["hip_l"]) / 2.0
            ana_hip = hip_center[first_valid : first_valid + cap_len]
            gt_hip_all = (gt_global[:, 0:3] + gt_global[:, 3:6]) / 2.0
            ana_offset = (
                (gt_hip_all[0] - ana_hip[0]) if len(ana_hip) > 0 else np.zeros(3)
            )
            ana_global = np.zeros((cap_len, N_JOINTS * N_COORDS))
            for j_idx, j_name in enumerate(JOINT_NAMES):
                ana_global[:, j_idx * 3 : (j_idx + 1) * 3] = (
                    rec[j_name][first_valid : first_valid + cap_len] + ana_offset
                )

            # TCN Lite trajectory
            anthro_4d = np.array(
                [
                    cap["femur_length"] / BONE_NORM,
                    cap["tibia_length"] / BONE_NORM,
                    cap["hip_width"] / BONE_NORM,
                    1.0 if cap["direction"] == "R" else 0.0,
                ],
                dtype=np.float32,
            )
            gt_pelvis_0 = np.expand_dims(
                (gt_global[0, 0:3] + gt_global[0, 3:6]) / 2.0, axis=0
            )

            sensor_raw = cap["sensor_data"].astype(np.float32)
            pred_pose_tcn, _, pred_hip_global_tcn, valid_tcn = predict_tcn(
                tcn_model,
                sensor_raw,
                anthro_4d,
                norm_stats_tcn,
                window_size=window_size_tcn,
                initial_hip=gt_pelvis_0,
            )
            tcn_global = np.zeros_like(pred_pose_tcn)
            for j in range(N_JOINTS):
                tcn_global[:, j * 3 : (j + 1) * 3] = (
                    pred_pose_tcn[:, j * 3 : (j + 1) * 3] + pred_hip_global_tcn
                )

            # Static Network trajectory
            sensor_norm_pivot = (
                sensor_raw - norm_stats_pivot["mean"]
            ) / norm_stats_pivot["std"]
            sensor_norm_pose = (sensor_raw - norm_stats_pose["mean"]) / norm_stats_pose[
                "std"
            ]
            anthro_broadcast = np.tile(anthro_4d, (cap_len, 1))

            x_pivot = np.concatenate([sensor_norm_pivot, anthro_broadcast], axis=1)
            x_pivot_tensor = torch.from_numpy(x_pivot).to(DEVICE)
            with torch.no_grad():
                pivot_probs = (
                    torch.sigmoid(pivot_model(x_pivot_tensor)).cpu().numpy().flatten()
                )
            pred_pivot_flag = (pivot_probs >= 0.5).astype(np.float32).reshape(-1, 1)

            x_pose_nn = np.concatenate(
                [sensor_norm_pose, anthro_broadcast, pred_pivot_flag], axis=1
            )
            x_pose_nn_tensor = torch.from_numpy(x_pose_nn).to(DEVICE)
            with torch.no_grad():
                pred_rel_pose_nn = pose_model(x_pose_nn_tensor).cpu().numpy()

            init_pivot_is_right = pred_pivot_flag[0, 0] == 1.0
            init_pivot_idx = 4 if init_pivot_is_right else 5
            init_pivot_pos = gt_global[0, init_pivot_idx * 3 : (init_pivot_idx + 1) * 3]
            static_global = eval_module.accumulate_global_trajectory(
                pred_rel_pose_nn, pred_pivot_flag.flatten(), init_pivot_pos
            )

            # Common valid mask
            valid_mask = valid_tcn & (np.arange(cap_len) < len(tcn_global))
            if np.sum(valid_mask) < 20:
                continue

            time_s = np.arange(np.sum(valid_mask)) * 0.01  # 100 Hz

            gt_v = gt_global[valid_mask].reshape(-1, 6, 3)
            ana_v = ana_global[valid_mask].reshape(-1, 6, 3)
            tcn_v = tcn_global[valid_mask].reshape(-1, 6, 3)
            static_v = static_global[valid_mask].reshape(-1, 6, 3)

            # Frame-by-frame Euclidean error across joints [N_frames]
            err_ana_joints = np.linalg.norm(ana_v - gt_v, axis=2)  # [N, 6]
            err_tcn_joints = np.linalg.norm(tcn_v - gt_v, axis=2)  # [N, 6]
            err_static_joints = np.linalg.norm(static_v - gt_v, axis=2)

            err_ana_frame = np.mean(err_ana_joints, axis=1)
            err_tcn_frame = np.mean(err_tcn_joints, axis=1)
            err_static_frame = np.mean(err_static_joints, axis=1)

            all_ana_errors_concat.extend(err_ana_frame)
            all_tcn_errors_concat.extend(err_tcn_frame)
            all_static_errors_concat.extend(err_static_frame)

            # Pelvic trajectory (center of hips)
            gt_hip_pos = (gt_v[:, 0, :] + gt_v[:, 1, :]) / 2.0
            ana_hip_pos = (ana_v[:, 0, :] + ana_v[:, 1, :]) / 2.0
            tcn_hip_pos = (tcn_v[:, 0, :] + tcn_v[:, 1, :]) / 2.0
            static_hip_pos = (static_v[:, 0, :] + static_v[:, 1, :]) / 2.0

            trial_info = {
                "id": f"S{subj}_{Path(mcd_path).stem}",
                "subject": subj,
                "trial": Path(mcd_path).stem,
                "time_s": time_s,
                "err_ana_frame": err_ana_frame,
                "err_tcn_frame": err_tcn_frame,
                "err_static_frame": err_static_frame,
                "err_ana_joints": err_ana_joints,
                "err_tcn_joints": err_tcn_joints,
                "mean_ana": float(np.mean(err_ana_frame)),
                "mean_tcn": float(np.mean(err_tcn_frame)),
                "mean_static": float(np.mean(err_static_frame)),
                "p95_ana": float(np.percentile(err_ana_frame, 95)),
                "p95_tcn": float(np.percentile(err_tcn_frame, 95)),
                "gt_hip": gt_hip_pos,
                "ana_hip": ana_hip_pos,
                "tcn_hip": tcn_hip_pos,
                "static_hip": static_hip_pos,
                "gt_v": gt_v,
                "ana_v": ana_v,
                "tcn_v": tcn_v,
            }
            trials_data.append(trial_info)

    all_ana_arr = np.array(all_ana_errors_concat)
    all_tcn_arr = np.array(all_tcn_errors_concat)
    all_static_arr = np.array(all_static_errors_concat)

    # Save to cache
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(
            {
                "trials_data": trials_data,
                "all_ana": all_ana_arr,
                "all_tcn": all_tcn_arr,
                "all_static": all_static_arr,
            },
            f,
        )
    print(f"  ✓ Cached evaluation data to: {CACHE_FILE}")

    return trials_data, all_ana_arr, all_tcn_arr, all_static_arr


def plot_fig2_temporal_error(trials_data, lang: str = "en"):
    """
    Figure 2: Unified 3-panel error over time comparison (Best, Representative, Worst cases).
    """
    sorted_trials = sorted(trials_data, key=lambda x: x["mean_ana"])
    best_trial = sorted_trials[0]
    median_trial = sorted_trials[len(sorted_trials) // 2]
    worst_trial = sorted_trials[-1]

    if lang == "es":
        cases = [
            ("Mejor caso", best_trial, "a"),
            ("Caso representativo (mediana)", median_trial, "b"),
            ("Peor caso", worst_trial, "c"),
        ]
        label_ana = "Método Analítico"
        label_tcn = "TCN Lite (propuesto)"
        label_excess = "Exceso error analítico"
        xlabel = "Tiempo (s)"
        ylabel = "Error de Posición Global MPJPE (m)"
        subj_lbl = "Sujeto"
        trial_lbl = "Ensayo"
        ana_tag = "Analítico"
    else:
        cases = [
            ("Best Case", best_trial, "a"),
            ("Representative Case (Median)", median_trial, "b"),
            ("Worst Case", worst_trial, "c"),
        ]
        label_ana = "Analytical Pipeline"
        label_tcn = "TCN Lite (Proposed)"
        label_excess = "Analytical excess error"
        xlabel = "Time (s)"
        ylabel = "Global MPJPE Position Error (m)"
        subj_lbl = "Subject"
        trial_lbl = "Trial"
        ana_tag = "Analytical"

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharey=True)

    for ax, (title, tr, panel_label) in zip(axes, cases):
        t = tr["time_s"]
        e_ana = tr["err_ana_frame"]
        e_tcn = tr["err_tcn_frame"]

        ax.plot(
            t,
            e_ana,
            label=label_ana,
            color=COLOR_ANA,
            linestyle="--",
            linewidth=1.9,
            alpha=0.95,
        )
        ax.plot(
            t,
            e_tcn,
            label=label_tcn,
            color=COLOR_TCN,
            linestyle="-",
            linewidth=2.2,
            alpha=0.95,
        )

        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.fill_between(
            t,
            e_tcn,
            e_ana,
            where=(e_ana >= e_tcn),
            color=COLOR_ANA,
            alpha=0.15,
            label=label_excess,
        )

        subj_clean = tr["subject"]
        trial_clean = tr["trial"].replace("mcd-", f"{trial_lbl} ")
        ax.set_title(
            f"({panel_label}) {title}\n({subj_lbl} {subj_clean}, {trial_clean})",
            fontsize=10.5,
            pad=8,
        )
        ax.set_xlabel(xlabel)

        stat_text = (
            f"{ana_tag}: $\\mu={tr['mean_ana']*100:.1f}$ cm\n"
            f"TCN Lite:   $\\mu={tr['mean_tcn']*100:.1f}$ cm\n"
            f"$\\Delta$: {((tr['mean_tcn']-tr['mean_ana'])/tr['mean_ana'])*100:+.1f}%"
        )
        ax.text(
            0.05,
            0.93,
            stat_text,
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="#cccccc",
                alpha=0.9,
            ),
            family="monospace",
        )

        ax.set_xlim(0, max(t))
        ax.set_ylim(-0.01, 0.55)

    axes[0].set_ylabel(ylabel, fontweight="bold")
    axes[0].legend(loc="upper right", framealpha=0.9, fontsize=8)

    plt.tight_layout()
    suffix = f"_{lang}" if lang != "es" else ""
    out_png = OUTPUT_DIR / f"fig2_temporal_error_comparison{suffix}.png"
    out_pdf = OUTPUT_DIR / f"fig2_temporal_error_comparison{suffix}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved Figure 2 [{lang}]: {out_png}")


def plot_fig3_statistical_distribution(
    trials_data, all_ana, all_tcn, all_static, lang: str = "en"
):
    """
    Figure 3: Cross-subject statistical distribution across holdout trials:
      (a) Boxplots of MPJPE by joint (Analytical vs TCN Lite).
      (b) Empirical Cumulative Distribution Function (ECDF) of global error.
    """
    fig, (ax_box, ax_ecdf) = plt.subplots(
        1, 2, figsize=(14, 5.0), gridspec_kw={"width_ratios": [1.25, 1.0]}
    )

    p95_tcn = float(np.percentile(all_tcn, 95))
    p95_ana = float(np.percentile(all_ana, 95))

    if lang == "es":
        joint_names = [
            "Cadera D",
            "Cadera I",
            "Rodilla D",
            "Rodilla I",
            "Tobillo D",
            "Tobillo I",
            "General",
        ]
        ylabel_box = "MPJPE por Articulación (m)"
        title_box = "(a) Dispersión del Error por Articulación (20 Ensayos de Test)"
        lbl_ana_box = "Método Analítico (línea base)"
        lbl_tcn_box = "TCN Lite (propuesto)"
        lbl_mean_box = "Media"

        title_ecdf = "(b) Función de Distribución Acumulada (ECDF, >7000 tramas)"
        xlabel_ecdf = "Error Global de Posición 3D (m)"
        ylabel_ecdf = "Probabilidad Acumulada $P(X \\leq x)$"
        lbl_tcn_ecdf = "TCN Lite (propuesto)"
        lbl_ana_ecdf = "Método Analítico"
        lbl_static_ecdf = "Red Estática (P+P)"
        p95_tcn_text = f"P95 TCN\n{p95_tcn*100:.1f} cm".replace(".", ",")
        p95_ana_text = f"P95 Analítico\n{p95_ana*100:.1f} cm".replace(".", ",")
    else:
        joint_names = [
            "Right Hip",
            "Left Hip",
            "Right Knee",
            "Left Knee",
            "Right Ankle",
            "Left Ankle",
            "Overall",
        ]
        ylabel_box = "MPJPE by Joint (m)"
        title_box = "(a) Error Dispersion by Joint (20 Held-Out Test Trials)"
        lbl_ana_box = "Analytical Pipeline (Baseline)"
        lbl_tcn_box = "TCN Lite (Proposed)"
        lbl_mean_box = "Mean"

        title_ecdf = (
            "(b) Empirical Cumulative Distribution Function (ECDF, >7,000 frames)"
        )
        xlabel_ecdf = "Global 3D Position Error (m)"
        ylabel_ecdf = "Cumulative Probability $P(X \\leq x)$"
        lbl_tcn_ecdf = "TCN Lite (Proposed)"
        lbl_ana_ecdf = "Analytical Pipeline"
        lbl_static_ecdf = "Static Network (P+P)"
        p95_tcn_text = f"P95 TCN\n{p95_tcn*100:.1f} cm"
        p95_ana_text = f"P95 Analytical\n{p95_ana*100:.1f} cm"

    ana_per_joint = [[] for _ in range(7)]
    tcn_per_joint = [[] for _ in range(7)]

    for tr in trials_data:
        for j in range(6):
            ana_per_joint[j].append(np.mean(tr["err_ana_joints"][:, j]))
            tcn_per_joint[j].append(np.mean(tr["err_tcn_joints"][:, j]))
        ana_per_joint[6].append(tr["mean_ana"])
        tcn_per_joint[6].append(tr["mean_tcn"])

    pos_ana = np.arange(7) * 2.0 - 0.35
    pos_tcn = np.arange(7) * 2.0 + 0.35

    ax_box.boxplot(
        ana_per_joint,
        positions=pos_ana,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(
            marker="o", markerfacecolor="black", markeredgecolor="black", markersize=4
        ),
        boxprops=dict(facecolor=COLOR_ANA, color="#8c3b00", alpha=0.8),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="#8c3b00", linewidth=1.2),
        capprops=dict(color="#8c3b00", linewidth=1.2),
        flierprops=dict(marker="x", markeredgecolor=COLOR_ANA, markersize=5),
    )

    ax_box.boxplot(
        tcn_per_joint,
        positions=pos_tcn,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(
            marker="o", markerfacecolor="black", markeredgecolor="black", markersize=4
        ),
        boxprops=dict(facecolor=COLOR_TCN, color="#371e61", alpha=0.8),
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="#371e61", linewidth=1.2),
        capprops=dict(color="#371e61", linewidth=1.2),
        flierprops=dict(
            marker="o",
            markerfacecolor=COLOR_TCN,
            markeredgecolor="#371e61",
            markersize=4,
        ),
    )

    ax_box.set_xticks(np.arange(7) * 2.0)
    ax_box.set_xticklabels(joint_names, rotation=20, ha="right", fontweight="bold")
    ax_box.set_ylabel(ylabel_box, fontweight="bold")
    ax_box.set_title(title_box, pad=8)
    ax_box.set_ylim(0, 0.28)

    ax_box.plot([], [], color=COLOR_ANA, linewidth=6, label=lbl_ana_box)
    ax_box.plot([], [], color=COLOR_TCN, linewidth=6, label=lbl_tcn_box)
    ax_box.plot(
        [],
        [],
        marker="o",
        color="black",
        linestyle="None",
        markersize=4,
        label=lbl_mean_box,
    )
    ax_box.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    # ---------------- (b) ECDF ----------------
    def compute_ecdf(arr):
        sorted_arr = np.sort(arr)
        p = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
        return sorted_arr, p

    ana_x, ana_y = compute_ecdf(all_ana)
    tcn_x, tcn_y = compute_ecdf(all_tcn)
    static_x, static_y = compute_ecdf(all_static)

    ax_ecdf.plot(tcn_x, tcn_y, label=lbl_tcn_ecdf, color=COLOR_TCN, linewidth=2.4)
    ax_ecdf.plot(
        ana_x, ana_y, label=lbl_ana_ecdf, color=COLOR_ANA, linewidth=2.0, linestyle="--"
    )
    ax_ecdf.plot(
        static_x,
        static_y,
        label=lbl_static_ecdf,
        color=COLOR_STATIC,
        linewidth=1.6,
        linestyle=":",
    )

    p95_tcn = np.percentile(all_tcn, 95)
    p95_ana = np.percentile(all_ana, 95)

    ax_ecdf.axhline(0.95, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_ecdf.axvline(p95_tcn, color=COLOR_TCN, linestyle=":", linewidth=1.2, alpha=0.7)
    ax_ecdf.axvline(p95_ana, color=COLOR_ANA, linestyle=":", linewidth=1.2, alpha=0.7)

    ax_ecdf.text(
        p95_tcn + 0.005,
        0.50,
        p95_tcn_text,
        color=COLOR_TCN,
        fontsize=8.5,
        fontweight="bold",
    )
    ax_ecdf.text(
        p95_ana + 0.005,
        0.40,
        p95_ana_text,
        color=COLOR_ANA,
        fontsize=8.5,
        fontweight="bold",
    )

    ax_ecdf.set_xlim(0, 0.45)
    ax_ecdf.set_ylim(-0.01, 1.02)
    ax_ecdf.set_xlabel(xlabel_ecdf, fontweight="bold")
    ax_ecdf.set_ylabel(ylabel_ecdf, fontweight="bold")
    ax_ecdf.set_title(title_ecdf, pad=8)
    ax_ecdf.tick_params(axis="both", which="major", pad=6)
    ax_ecdf.legend(loc="lower right", framealpha=0.9, fontsize=8.5)

    plt.tight_layout()
    suffix = f"_{lang}" if lang != "es" else ""
    out_png = OUTPUT_DIR / f"fig3_statistical_error_distribution{suffix}.png"
    out_pdf = OUTPUT_DIR / f"fig3_statistical_error_distribution{suffix}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved Figure 3 [{lang}]: {out_png}")


def plot_fig4_trajectory_and_drift(trials_data, lang: str = "en"):
    """
    Figure 4: Spatial pelvic trajectory reconstruction and drift progression over distance.
    """
    sorted_trials = sorted(trials_data, key=lambda x: x["mean_ana"])
    median_trial = sorted_trials[len(sorted_trials) // 2]

    if lang == "es":
        title_traj = (
            f"(a) Reconstrucción de la Trayectoria Pélvica ({median_trial['id']})"
        )
        xlabel_traj = "Coordenada Longitudinal $X$ (m)"
        ylabel_traj = "Coordenada Vertical $Z$ (m)"
        lbl_gt = "Verdad de Campo (ArUco)"
        lbl_tcn = "TCN Lite (propuesto)"
        lbl_ana = "Método Analítico"
        lbl_static = "Red Estática (P+P)"
        lbl_start = "Inicio"
        lbl_end = "Fin"

        title_drift = "(b) Acumulación de Deriva en Función de la Distancia"
        xlabel_drift = "Distancia Recorrida Acumulada (m)"
        ylabel_drift = "Error de Trayectoria Absoluto ATE (m)"
        end_prefix = "Fin"
    else:
        title_traj = f"(a) Pelvic Trajectory Reconstruction ({median_trial['id']})"
        xlabel_traj = "Longitudinal Position $X$ (m)"
        ylabel_traj = "Vertical Position $Z$ (m)"
        lbl_gt = "Optical Ground Truth (ArUco)"
        lbl_tcn = "TCN Lite (Proposed)"
        lbl_ana = "Analytical Pipeline"
        lbl_static = "Static Network (P+P)"
        lbl_start = "Start"
        lbl_end = "End"

        title_drift = "(b) Odometry Drift Accumulation vs. Walked Distance"
        xlabel_drift = "Cumulative Walked Distance (m)"
        ylabel_drift = "Absolute Trajectory Error ATE (m)"
        end_prefix = "End"

    fig, (ax_traj, ax_drift) = plt.subplots(
        1, 2, figsize=(14, 5.0), gridspec_kw={"width_ratios": [1.1, 1.0]}
    )

    gt_hip = median_trial["gt_hip"]
    ana_hip = median_trial["ana_hip"]
    tcn_hip = median_trial["tcn_hip"]
    static_hip = median_trial["static_hip"]

    # ---------------- (a) Top-Down 2D Trajectory (X vs Z) ----------------
    ax_traj.plot(
        gt_hip[:, 0],
        gt_hip[:, 2],
        label=lbl_gt,
        color=COLOR_GT,
        linewidth=2.8,
        alpha=0.9,
    )
    ax_traj.plot(
        tcn_hip[:, 0],
        tcn_hip[:, 2],
        label=lbl_tcn,
        color=COLOR_TCN,
        linewidth=2.2,
        linestyle="-.",
    )
    ax_traj.plot(
        ana_hip[:, 0],
        ana_hip[:, 2],
        label=lbl_ana,
        color=COLOR_ANA,
        linewidth=2.0,
        linestyle="--",
    )
    ax_traj.plot(
        static_hip[:, 0],
        static_hip[:, 2],
        label=lbl_static,
        color=COLOR_STATIC,
        linewidth=1.6,
        linestyle=":",
    )

    ax_traj.scatter(
        [gt_hip[0, 0]],
        [gt_hip[0, 2]],
        color="black",
        s=50,
        marker="o",
        zorder=5,
        label=lbl_start,
    )
    ax_traj.scatter(
        [gt_hip[-1, 0]],
        [gt_hip[-1, 2]],
        color="red",
        s=50,
        marker="s",
        zorder=5,
        label=lbl_end,
    )

    ax_traj.set_xlabel(xlabel_traj, fontweight="bold")
    ax_traj.set_ylabel(ylabel_traj, fontweight="bold")
    ax_traj.set_title(title_traj, pad=8)
    ax_traj.legend(loc="best", framealpha=0.9, fontsize=8.5)

    # ---------------- (b) Drift (ATE) vs Walked Distance ----------------
    step_dists = np.linalg.norm(np.diff(gt_hip, axis=0), axis=1)
    cum_dist = np.concatenate([[0], np.cumsum(step_dists)])

    drift_ana = np.linalg.norm(ana_hip - gt_hip, axis=1)
    drift_tcn = np.linalg.norm(tcn_hip - gt_hip, axis=1)
    drift_static = np.linalg.norm(static_hip - gt_hip, axis=1)

    ax_drift.plot(cum_dist, drift_tcn, label=lbl_tcn, color=COLOR_TCN, linewidth=2.2)
    ax_drift.plot(
        cum_dist,
        drift_ana,
        label=lbl_ana,
        color=COLOR_ANA,
        linewidth=2.0,
        linestyle="--",
    )
    ax_drift.plot(
        cum_dist,
        drift_static,
        label=lbl_static,
        color=COLOR_STATIC,
        linewidth=1.6,
        linestyle=":",
    )

    final_dist = cum_dist[-1]
    ax_drift.annotate(
        f"{end_prefix}: {drift_tcn[-1]*100:.1f} cm",
        xy=(final_dist, drift_tcn[-1]),
        xytext=(final_dist - 0.7, drift_tcn[-1] + 0.05),
        arrowprops=dict(arrowstyle="->", color=COLOR_TCN, lw=1.2),
        fontweight="bold",
        color=COLOR_TCN,
        fontsize=8.5,
    )
    ax_drift.annotate(
        f"{end_prefix}: {drift_ana[-1]*100:.1f} cm",
        xy=(final_dist, drift_ana[-1]),
        xytext=(final_dist - 0.7, drift_ana[-1] + 0.06),
        arrowprops=dict(arrowstyle="->", color=COLOR_ANA, lw=1.2),
        fontweight="bold",
        color=COLOR_ANA,
        fontsize=8.5,
    )

    ax_drift.set_xlabel(xlabel_drift, fontweight="bold")
    ax_drift.set_ylabel(ylabel_drift, fontweight="bold")
    ax_drift.set_title(title_drift, pad=8)
    ax_drift.set_ylim(-0.01, max(np.max(drift_ana), np.max(drift_tcn)) * 1.35)
    ax_drift.legend(loc="upper left", framealpha=0.9, fontsize=8.5)

    plt.tight_layout()
    suffix = f"_{lang}" if lang != "es" else ""
    out_png = OUTPUT_DIR / f"fig4_trajectory_and_drift_analysis{suffix}.png"
    out_pdf = OUTPUT_DIR / f"fig4_trajectory_and_drift_analysis{suffix}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved Figure 4 [{lang}]: {out_png}")


def plot_fig5_biomechanical_and_bland_altman(trials_data, lang: str = "en"):
    """
    Figure 5: Biomechanical ROM and Bland-Altman agreement.
    """
    fig, (ax_rom, ax_ba) = plt.subplots(1, 2, figsize=(14, 5.0))

    sorted_trials = sorted(trials_data, key=lambda x: x["mean_ana"])
    tr = sorted_trials[len(sorted_trials) // 2]
    gt_v = tr["gt_v"]
    tcn_v = tr["tcn_v"]
    ana_v = tr["ana_v"]
    t = tr["time_s"]

    if lang == "es":
        title_rom = (
            "(a) Flexión/Extensión de Rodilla Derecha ($r_{{\\text{{TCN}}}} = {r:.3f}$)"
        )
        xlabel_rom = "Tiempo (s)"
        ylabel_rom = "Ángulo de Flexión de Rodilla (°)"
        lbl_gt = "Verdad de Campo (ArUco)"
        lbl_tcn = "TCN Lite (propuesto)"
        lbl_ana = "Método Analítico"

        title_ba = "(b) Concordancia de Bland-Altman: TCN Lite vs. ArUco (Eje $X$)"
        xlabel_ba = "Media de Posición $(M_1 + M_2)/2$ (m)"
        ylabel_ba = "Diferencia $(M_{\\text{TCN}} - M_{\\text{GT}})$ (m)"
        bias_lbl = "Sesgo"
    else:
        title_rom = "(a) Right Knee Flexion/Extension ($r_{{\\text{{TCN}}}} = {r:.3f}$)"
        xlabel_rom = "Time (s)"
        ylabel_rom = "Knee Flexion Angle (°)"
        lbl_gt = "Optical Ground Truth (ArUco)"
        lbl_tcn = "TCN Lite (Proposed)"
        lbl_ana = "Analytical Pipeline"

        title_ba = "(b) Bland-Altman Agreement: TCN Lite vs. ArUco (Forward Axis $X$)"
        xlabel_ba = "Mean Position $(M_1 + M_2)/2$ (m)"
        ylabel_ba = "Difference $(M_{\\text{TCN}} - M_{\\text{GT}})$ (m)"
        bias_lbl = "Bias"

    def compute_knee_angle(v_joints, side="r"):
        hip_idx = 0 if side == "r" else 1
        knee_idx = 2 if side == "r" else 3
        ank_idx = 4 if side == "r" else 5

        u = v_joints[:, hip_idx, :] - v_joints[:, knee_idx, :]
        v = v_joints[:, ank_idx, :] - v_joints[:, knee_idx, :]

        u_norm = np.linalg.norm(u, axis=1, keepdims=True) + 1e-6
        v_norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-6

        dot = np.sum((u / u_norm) * (v / v_norm), axis=1)
        dot = np.clip(dot, -1.0, 1.0)
        angle_rad = np.arccos(dot)
        angle_deg = 180.0 - np.degrees(angle_rad)
        return angle_deg

    knee_gt = compute_knee_angle(gt_v, "r")
    knee_tcn = compute_knee_angle(tcn_v, "r")
    knee_ana = compute_knee_angle(ana_v, "r")

    n_pts = min(len(t), 200)
    t_crop = t[:n_pts]

    ax_rom.plot(t_crop, knee_gt[:n_pts], label=lbl_gt, color=COLOR_GT, linewidth=2.4)
    ax_rom.plot(t_crop, knee_tcn[:n_pts], label=lbl_tcn, color=COLOR_TCN, linewidth=2.0)
    ax_rom.plot(
        t_crop,
        knee_ana[:n_pts],
        label=lbl_ana,
        color=COLOR_ANA,
        linewidth=1.8,
        linestyle="--",
    )

    r_tcn = np.corrcoef(knee_gt[:n_pts], knee_tcn[:n_pts])[0, 1]

    ax_rom.set_title(title_rom.format(r=r_tcn), pad=8)
    ax_rom.set_xlabel(xlabel_rom, fontweight="bold")
    ax_rom.set_ylabel(ylabel_rom, fontweight="bold")
    ax_rom.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    # --- (b) Bland-Altman for TCN Lite vs Ground Truth ---
    all_gt_coords = []
    all_tcn_coords = []
    for tr_i in trials_data:
        all_gt_coords.append(tr_i["gt_v"].reshape(-1, 3))
        all_tcn_coords.append(tr_i["tcn_v"].reshape(-1, 3))

    gt_coords_cat = np.concatenate(all_gt_coords, axis=0)
    tcn_coords_cat = np.concatenate(all_tcn_coords, axis=0)

    diff_x = tcn_coords_cat[:, 0] - gt_coords_cat[:, 0]
    mean_x = (tcn_coords_cat[:, 0] + gt_coords_cat[:, 0]) / 2.0

    bias = np.mean(diff_x)
    std_diff = np.std(diff_x)
    loa_upper = bias + 1.96 * std_diff
    loa_lower = bias - 1.96 * std_diff

    rng = np.random.default_rng(42)
    idx_sub = rng.choice(len(diff_x), size=min(len(diff_x), 3000), replace=False)
    ax_ba.scatter(
        mean_x[idx_sub],
        diff_x[idx_sub],
        alpha=0.15,
        s=6,
        color=COLOR_TCN,
        edgecolors="none",
    )

    ax_ba.axhline(
        bias,
        color="black",
        linestyle="-",
        linewidth=1.5,
        label=f"{bias_lbl}: {bias*100:+.2f} cm",
    )
    ax_ba.axhline(
        loa_upper,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"+1.96σ: {loa_upper*100:+.2f} cm",
    )
    ax_ba.axhline(
        loa_lower,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"−1.96σ: {loa_lower*100:+.2f} cm",
    )

    ax_ba.set_title(title_ba, pad=8)
    ax_ba.set_xlabel(xlabel_ba, fontweight="bold")
    ax_ba.set_ylabel(ylabel_ba, fontweight="bold")
    ax_ba.set_ylim(-0.35, 0.35)
    ax_ba.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    plt.tight_layout()
    suffix = f"_{lang}" if lang != "es" else ""
    out_png = OUTPUT_DIR / f"fig5_biomechanical_agreement{suffix}.png"
    out_pdf = OUTPUT_DIR / f"fig5_biomechanical_agreement{suffix}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved Figure 5 [{lang}]: {out_png}")


def plot_fig6_3d_joint_trajectories(trials_data, lang: str = "en"):
    """
    Figure 6: 3D joint trajectories and intermediate gait pose reconstruction.
      Global 3D perspective showing continuous trajectories for all 6 joints
      (Optical ArUco Ground Truth vs TCN Lite) across the walking trial,
      along with the lower-limb kinematic skeleton (pelvis, femurs, tibias)
      at an intermediate gait pose (mid-stride).
    """
    sorted_trials = sorted(trials_data, key=lambda x: x["mean_ana"])
    median_trial = sorted_trials[len(sorted_trials) // 2]

    gt = median_trial["gt_v"]  # [N, 6, 3]
    tcn = median_trial["tcn_v"]  # [N, 6, 3]
    time_s = median_trial["time_s"]
    N = len(time_s)

    # Intermediate frame (mid-stride)
    f_inter = N // 2
    t_inter = time_s[f_inter]

    # Kinematic connectivity of lower limb (pairs of joint indices)
    # 0: hip_r, 1: hip_l, 2: knee_r, 3: knee_l, 4: ankle_r, 5: ankle_l
    bones = [
        (0, 1),  # Pelvis: hip_r - hip_l
        (0, 2),  # Right Femur: hip_r - knee_r
        (2, 4),  # Right Tibia: knee_r - ankle_r
        (1, 3),  # Left Femur: hip_l - knee_l
        (3, 5),  # Left Tibia: knee_l - ankle_l
    ]

    # Distinct color per joint (Right side: warm tones, Left side: cool tones)
    joint_colors = [
        "#D62728",  # Hip R (Crimson)
        "#1F77B4",  # Hip L (Dark Blue)
        "#FF7F0E",  # Knee R (Orange)
        "#17BECF",  # Knee L (Teal)
        "#D4AC0D",  # Ankle R (Gold)
        "#9467BD",  # Ankle L (Purple)
    ]

    if lang == "es":
        title = (
            f"Trayectorias Articulares 3D y Pose Intermedia ({median_trial['id']})\n"
            f"Línea continua / Círculos: ArUco (Ground Truth) | Línea discontinua / Cuadrados: TCN Lite | $t = {t_inter:.2f}$ s"
        )
        xlabel = "Eje Longitudinal $X$ (m)"
        ylabel = "Eje Lateral $Y$ (m)"
        zlabel = "Eje Vertical $Z$ (m)"
        lbl_gt_traj = "Trayectoria ArUco (GT)"
        lbl_tcn_traj = "Trayectoria TCN Lite"
        lbl_gt_skel = f"Esqueleto ArUco ($t = {t_inter:.2f}$ s)"
        lbl_tcn_skel = f"Esqueleto TCN Lite ($t = {t_inter:.2f}$ s)"
        joint_names = [
            "Cadera Der.",
            "Cadera Izq.",
            "Rodilla Der.",
            "Rodilla Izq.",
            "Tobillo Der.",
            "Tobillo Izq.",
        ]
    else:
        title = (
            f"3D Joint Trajectories and Intermediate Pose ({median_trial['id']})\n"
            f"Solid line / Circles: ArUco (Ground Truth) | Dashed line / Squares: TCN Lite (Proposed) | $t = {t_inter:.2f}$ s"
        )
        xlabel = "Longitudinal Axis $X$ (m)"
        ylabel = "Lateral Axis $Y$ (m)"
        zlabel = "Vertical Axis $Z$ (m)"
        lbl_gt_traj = "ArUco Trajectory (GT)"
        lbl_tcn_traj = "TCN Lite Trajectory"
        lbl_gt_skel = f"ArUco Skeleton ($t = {t_inter:.2f}$ s)"
        lbl_tcn_skel = f"TCN Lite Skeleton ($t = {t_inter:.2f}$ s)"
        joint_names = [
            "Right Hip",
            "Left Hip",
            "Right Knee",
            "Left Knee",
            "Right Ankle",
            "Left Ankle",
        ]

    fig = plt.figure(figsize=(11.0, 7.0), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # 1. Trajectories
    for j in range(6):
        ax.plot(
            gt[:, j, 0],
            gt[:, j, 1],
            gt[:, j, 2],
            color=joint_colors[j],
            linestyle="-",
            linewidth=1.7,
            alpha=0.55,
        )
        ax.plot(
            tcn[:, j, 0],
            tcn[:, j, 1],
            tcn[:, j, 2],
            color=joint_colors[j],
            linestyle="--",
            linewidth=1.5,
            alpha=0.55,
        )

    # 2. Subtle floor ground trails (shadow projection for ankles)
    z_floor = float(np.min(gt[:, :, 2])) - 0.05
    for j in [4, 5]:
        ax.plot(
            gt[:, j, 0],
            gt[:, j, 1],
            np.full(N, z_floor),
            color="gray",
            linestyle=":",
            linewidth=0.8,
            alpha=0.30,
        )

    # 3. Intermediate skeleton bones
    for j1, j2 in bones:
        ax.plot(
            [gt[f_inter, j1, 0], gt[f_inter, j2, 0]],
            [gt[f_inter, j1, 1], gt[f_inter, j2, 1]],
            [gt[f_inter, j1, 2], gt[f_inter, j2, 2]],
            color=COLOR_GT,
            linewidth=3.6,
            solid_capstyle="round",
        )
        ax.plot(
            [tcn[f_inter, j1, 0], tcn[f_inter, j2, 0]],
            [tcn[f_inter, j1, 1], tcn[f_inter, j2, 1]],
            [tcn[f_inter, j1, 2], tcn[f_inter, j2, 2]],
            color=COLOR_TCN,
            linewidth=3.0,
            linestyle="--",
            solid_capstyle="round",
        )

    # 4. Residual error displacement vector connecting GT and TCN
    for j in range(6):
        ax.plot(
            [gt[f_inter, j, 0], tcn[f_inter, j, 0]],
            [gt[f_inter, j, 1], tcn[f_inter, j, 1]],
            [gt[f_inter, j, 2], tcn[f_inter, j, 2]],
            color="black",
            linestyle=":",
            linewidth=1.4,
            alpha=0.85,
        )

    # 5. Joint markers plotted LAST ("hasta arriba") with ax.plot so no line covers them
    for j in range(6):
        ax.plot(
            [gt[f_inter, j, 0]],
            [gt[f_inter, j, 1]],
            [gt[f_inter, j, 2]],
            marker="o",
            markersize=9.5,
            color=joint_colors[j],
            markeredgecolor="black",
            markeredgewidth=1.4,
            linestyle="None",
        )
        ax.plot(
            [tcn[f_inter, j, 0]],
            [tcn[f_inter, j, 1]],
            [tcn[f_inter, j, 2]],
            marker="s",
            markersize=9.0,
            color=joint_colors[j],
            markeredgecolor="black",
            markeredgewidth=1.4,
            linestyle="None",
        )

    ax.view_init(elev=20, azim=-62)
    ax.set_box_aspect((2.8, 1.0, 1.0))
    ax.set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2])
    ax.set_xlabel(xlabel, fontweight="bold", labelpad=6)
    ax.set_ylabel(ylabel, fontweight="bold", labelpad=6)
    ax.set_zlabel(zlabel, fontweight="bold", labelpad=6)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)

    # ---------------- Bottom Legend ----------------
    legend_elements = [
        Line2D(
            [0], [0], color="black", linestyle="-", linewidth=2.0, label=lbl_gt_traj
        ),
        Line2D(
            [0], [0], color="black", linestyle="--", linewidth=2.0, label=lbl_tcn_traj
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_GT,
            linewidth=3.0,
            marker="o",
            markeredgecolor="black",
            markersize=7.5,
            label=lbl_gt_skel,
        ),
        Line2D(
            [0],
            [0],
            color=COLOR_TCN,
            linewidth=2.8,
            linestyle="--",
            marker="s",
            markeredgecolor="black",
            markersize=7.0,
            label=lbl_tcn_skel,
        ),
    ]
    for j in range(6):
        legend_elements.append(
            Line2D([0], [0], color=joint_colors[j], linewidth=2.2, label=joint_names[j])
        )

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=5,
        fontsize=8.8,
        frameon=True,
        framealpha=0.95,
    )

    plt.subplots_adjust(left=0.03, right=0.97, bottom=0.14, top=0.92)
    suffix = f"_{lang}" if lang != "es" else ""
    out_png = OUTPUT_DIR / f"fig6_3d_joint_trajectories{suffix}.png"
    out_pdf = OUTPUT_DIR / f"fig6_3d_joint_trajectories{suffix}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved Figure 6 [{lang}]: {out_png}")


def compute_biomechanical_metrics(trials_data):
    """
    Computes and prints the biomechanical metrics reported in Section 3.3
    (Biomechanical Metrics and Clinical Concordance) directly from the
    evaluated test-set trials. All values are reproducible from the same
    global-frame joint positions used to generate Figure 5, so the numbers
    printed here match the manuscript exactly.

    Metrics computed:
      - Right-knee flexion peak (swing-phase) for GT / TCN Lite / Analytical.
      - Pearson correlation r of the knee-angle waveform (TCN Lite vs GT).
      - Stride length (m) from heel-strike detection, GT and TCN Lite.
      - Cadence (steps/min) from combined heel strikes of both feet.
      - Pelvic translational velocity (m/s), GT / TCN Lite / Analytical.
      - Bland-Altman bias and 95% Limits of Agreement on the forward axis (X).
    """

    def knee_angle_series(v_joints, side="r"):
        hip_idx = 0 if side == "r" else 1
        knee_idx = 2 if side == "r" else 3
        ank_idx = 4 if side == "r" else 5
        u = v_joints[:, hip_idx, :] - v_joints[:, knee_idx, :]
        v = v_joints[:, ank_idx, :] - v_joints[:, knee_idx, :]
        u_norm = np.linalg.norm(u, axis=1, keepdims=True) + 1e-6
        v_norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
        dot = np.sum((u / u_norm) * (v / v_norm), axis=1)
        dot = np.clip(dot, -1.0, 1.0)
        return 180.0 - np.degrees(np.arccos(dot))

    def extract_strides_and_cadence(v_joints, time_s):
        """
        Detecta zancadas completas bilaterales a partir de heel strikes consecutivos
        del mismo pie (mínimos locales en Z). Filtra periodos fisiológicos de zancada
        (0.8s a 2.2s) y distancias válidas (0.5m a 2.2m).
        """
        strides = []
        cadences = []
        for ank_idx in (4, 5):  # 4=ankle_r, 5=ankle_l
            z = v_joints[:, ank_idx, 2]
            peaks, _ = find_peaks(-z, distance=70, prominence=0.008)
            if len(peaks) >= 2:
                for i in range(len(peaks) - 1):
                    dt = time_s[peaks[i + 1]] - time_s[peaks[i]]
                    if 0.8 <= dt <= 2.2:
                        d_xy = np.linalg.norm(
                            v_joints[peaks[i + 1], ank_idx, :2]
                            - v_joints[peaks[i], ank_idx, :2]
                        )
                        if 0.5 <= d_xy <= 2.2:
                            strides.append(d_xy)
                            cadences.append(120.0 / dt)
        mean_cad = float(np.mean(cadences)) if len(cadences) > 0 else np.nan
        return strides, mean_cad

    knee_peak = {"gt": [], "tcn": [], "ana": []}
    knee_r = []
    stride_gt = []
    stride_tcn = []
    cadence = {"gt": [], "tcn": [], "ana": []}
    vel_inst = {"gt": [], "tcn": [], "ana": []}
    vel_fwd = {"gt": [], "tcn": [], "ana": []}

    for tr in trials_data:
        gt_v, tcn_v, ana_v = tr["gt_v"], tr["tcn_v"], tr["ana_v"]
        t_s = tr["time_s"]
        t_ms = t_s * 1000.0

        # --- Right-knee flexion peak and waveform correlation ---
        kg = knee_angle_series(gt_v, "r")
        kt = knee_angle_series(tcn_v, "r")
        ka = knee_angle_series(ana_v, "r")
        knee_peak["gt"].append(float(np.max(kg)))
        knee_peak["tcn"].append(float(np.max(kt)))
        knee_peak["ana"].append(float(np.max(ka)))
        if np.std(kg) > 1e-6 and np.std(kt) > 1e-6:
            knee_r.append(float(np.corrcoef(kg, kt)[0, 1]))

        # --- Stride length & Cadence (bilateral, consecutive same-foot heel strikes) ---
        s_gt, c_gt = extract_strides_and_cadence(gt_v, t_s)
        s_tcn, c_tcn = extract_strides_and_cadence(tcn_v, t_s)
        _, c_ana = extract_strides_and_cadence(ana_v, t_s)

        stride_gt.extend(s_gt)
        stride_tcn.extend(s_tcn)
        if not np.isnan(c_gt):
            cadence["gt"].append(c_gt)
        if not np.isnan(c_tcn):
            cadence["tcn"].append(c_tcn)
        if not np.isnan(c_ana):
            cadence["ana"].append(c_ana)

        # --- Pelvic translational velocity (m/s) ---
        dur = t_s[-1] - t_s[0]
        for key, hip in (
            ("gt", tr["gt_hip"]),
            ("tcn", tr["tcn_hip"]),
            ("ana", tr["ana_hip"]),
        ):
            v = compute_displacement_velocity(hip, t_ms, trim_ms=1000)
            vel_inst[key].append(float(v["mean_velocity"]))
            if dur > 0:
                vel_fwd[key].append(float(abs(hip[-1, 0] - hip[0, 0]) / dur))

    # --- Bland-Altman on the forward axis (X), all joints, all trials ---
    all_gt_x = []
    all_tcn_x = []
    for tr in trials_data:
        all_gt_x.append(tr["gt_v"].reshape(-1, 3)[:, 0])
        all_tcn_x.append(tr["tcn_v"].reshape(-1, 3)[:, 0])
    gt_x = np.concatenate(all_gt_x)
    tcn_x = np.concatenate(all_tcn_x)
    diff_x = tcn_x - gt_x
    bias = float(np.mean(diff_x))
    std_diff = float(np.std(diff_x))
    loa_upper = bias + 1.96 * std_diff
    loa_lower = bias - 1.96 * std_diff

    def ms(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return "N/A"
        return f"{np.mean(arr):.3f} ± {np.std(arr):.3f}"

    rel_stride_err = (
        abs(np.mean(stride_tcn) - np.mean(stride_gt)) / np.mean(stride_gt) * 100.0
        if len(stride_gt) > 0 and len(stride_tcn) > 0
        else 0.0
    )

    print("\n" + "=" * 70)
    print("  📊 BIOMECHANICAL METRICS (Section 3.3) — held-out test set")
    print(f"     {len(trials_data)} trials | 5 subjects | global 3D frame")
    print("=" * 70)
    print("  Right-knee swing-phase flexion peak:")
    print(f"    GT (ArUco):        {ms(knee_peak['gt'])} °")
    print(f"    TCN Lite:          {ms(knee_peak['tcn'])} °")
    print(f"    Analytical:        {ms(knee_peak['ana'])} °")
    print(
        f"    Waveform corr r (TCN vs GT): {np.mean(knee_r):.3f} ± {np.std(knee_r):.3f}"
    )
    print(
        f"  Stride length (m) [consecutive same-foot strikes, rel error = {rel_stride_err:.2f}%]:"
    )
    print(f"    GT (ArUco):        {ms(stride_gt)} (N={len(stride_gt)})")
    print(f"    TCN Lite:          {ms(stride_tcn)} (N={len(stride_tcn)})")
    print("  Cadence (steps/min) [from fundamental stride duration]:")
    print(f"    GT (ArUco):        {ms(cadence['gt'])}")
    print(f"    TCN Lite:          {ms(cadence['tcn'])}")
    print(f"    Analytical:        {ms(cadence['ana'])}")
    print("  Pelvic translational velocity (m/s):")
    print(
        f"    Instantaneous (1s trim): GT={ms(vel_inst['gt'])}, TCN={ms(vel_inst['tcn'])}, Ana={ms(vel_inst['ana'])}"
    )
    print(
        f"    Forward net speed (ΔX/Δt): GT={ms(vel_fwd['gt'])}, TCN={ms(vel_fwd['tcn'])}, Ana={ms(vel_fwd['ana'])}"
    )
    print("  Bland-Altman, forward axis X (all joints):")
    print(f"    Bias:              {bias*100:+.2f} cm")
    print(f"    95% LoA:           [{loa_lower*100:+.2f}, {loa_upper*100:+.2f}] cm")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Publication Figure Generator (MDPI Sensors / IEEE style)"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="all",
        choices=["all", "en", "es"],
        help="Language to render figure text (default: 'all' generates both English and Spanish)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force recomputation of full pipeline instead of using cache",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("  🎨 GENERADOR DE GRÁFICAS DE ERROR PARA EL ARTÍCULO (v2_manuscript)")
    print("  MDPI Sensors Format | Multilingual Figures (EN / ES)")
    print("=" * 80)

    trials_data, all_ana, all_tcn, all_static = run_pipeline_evaluation(
        use_cache=not args.no_cache
    )

    langs = ["es", "en"] if args.lang == "all" else [args.lang]

    for lang in langs:
        print(f"\nGenerando figuras para publicación en idioma: [{lang.upper()}]...")
        plot_fig2_temporal_error(trials_data, lang=lang)
        plot_fig3_statistical_distribution(
            trials_data, all_ana, all_tcn, all_static, lang=lang
        )
        plot_fig4_trajectory_and_drift(trials_data, lang=lang)
        plot_fig5_biomechanical_and_bland_altman(trials_data, lang=lang)
        plot_fig6_3d_joint_trajectories(trials_data, lang=lang)

    # Print the Section 3.3 biomechanical metrics (once, language-independent)
    compute_biomechanical_metrics(trials_data)

    print("\n" + "=" * 80)
    print(f"  ✓ Todas las figuras generadas con éxito en: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
