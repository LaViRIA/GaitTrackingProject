"""
Visualizador Interactivo de Reconstrucción de Marcha en 3D
============================================================

Permite revisar interactivamente las 4 capturas de cada sujeto en tiempo real
con Matplotlib 3D (cuadrícula 2x2), comparando simultáneamente:
  - 🔵 Red Neuronal (CNN + BiLSTM + Attention)
  - 🔴 Marcadores ArUco (Ground Truth)
  - 🟢 Pipeline Analítico (Filtro Madgwick / EKF) [opcional]

Controles:
  - Botones GUI: Anterior / Siguiente Sujeto, Reproducir / Pausa, Cambiar Vista
  - Teclado:
      [Espacio] : Reproducir / Pausar animación
      [N]       : Siguiente sujeto
      [P]       : Sujeto anterior
      [V]       : Alternar vista (Global vs Local cadera)

Uso:
    python visualize_captures.py                      # Revisa 4 capturas por sujeto en 3D
    python visualize_captures.py --subject 30         # Inicia directamente en el sujeto 30
    python visualize_captures.py --compare-analytical # Incluye el pipeline analítico en la animación

Autor: GaitTracking Project
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np

import matplotlib

# Forzar backend interactivo si el backend actual es no interactivo (Agg)
if matplotlib.get_backend().lower() == "agg":
    for gui_backend in ["TkAgg", "QtAgg", "Qt5Agg", "GTK4Agg", "WXAgg"]:
        try:
            matplotlib.use(gui_backend, force=True)
            break
        except Exception:
            continue

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation

import torch

# Importar utilidades del proyecto
from process_all import process_files
from nn_tcn_lite import (
    GaitMultiTaskNetTCN as GaitMultiTaskNet,
    GaitDataset,
    predict_full_sequence_multitask,
    prepare_capture,
    discover_captures,
    DATASET_PATH,
    CHECKPOINT_DIR,
    INPUT_DIM,
    OUTPUT_DIM,
    N_JOINTS,
    N_COORDS,
    JOINT_NAMES,
    DEVICE,
)


class GaitAnimationViewer:
    """Visor interactivo en 3D para animación de las 4 capturas por sujeto simultáneamente."""

    def __init__(
        self,
        captures_info: list[tuple[str, str, str]],
        model: GaitMultiTaskNet,
        norm_stats: dict,
        compare_analytical: bool = False,
        start_subject_id: str | None = None,
    ):
        self.model = model
        self.norm_stats = norm_stats
        self.compare_analytical = compare_analytical

        # Agrupar capturas por sujeto
        self.subjects_dict: dict[str, list[tuple[str, str, str]]] = {}
        for mcd_path, aruco_path, subject_id in captures_info:
            if subject_id not in self.subjects_dict:
                self.subjects_dict[subject_id] = []
            self.subjects_dict[subject_id].append((mcd_path, aruco_path, subject_id))

        self.sorted_subjects = sorted(self.subjects_dict.keys(), key=lambda x: int(x))

        self.current_subject_idx = 0
        if start_subject_id and start_subject_id in self.sorted_subjects:
            self.current_subject_idx = self.sorted_subjects.index(start_subject_id)

        self.is_playing = True
        self.global_view = True
        self.frame_idx = 0
        self.step_subsample = 2
        self.max_frames = 1

        # Crear figura con 4 subplots 3D (2x2)
        self.fig = plt.figure(figsize=(15, 9))
        self.fig.canvas.manager.set_window_title(
            "Visor 3D de marcha — 4 Capturas por Sujeto"
        )
        self.fig.subplots_adjust(
            bottom=0.12, top=0.91, left=0.04, right=0.96, hspace=0.22, wspace=0.15
        )

        self.axes = []
        self.subplot_data = []

        for i in range(4):
            ax = self.fig.add_subplot(2, 2, i + 1, projection="3d")
            self.axes.append(ax)

            (line_nn,) = ax.plot(
                [],
                [],
                [],
                "o-",
                label="Red Neuronal (NN)",
                color="#1F77B4",
                linewidth=2.5,
                markersize=5,
            )
            (line_gt,) = ax.plot(
                [],
                [],
                [],
                "o-",
                label="ArUco (GT)",
                color="#D62728",
                linewidth=2.5,
                markersize=5,
                alpha=0.7,
            )
            if self.compare_analytical:
                (line_ana,) = ax.plot(
                    [],
                    [],
                    [],
                    "o--",
                    label="Pipeline Analítico",
                    color="#2CA02C",
                    linewidth=2.0,
                    markersize=4,
                )
            else:
                line_ana = None

            time_text = ax.text2D(
                0.03, 0.94, "", transform=ax.transAxes, fontsize=9, fontweight="bold"
            )
            metrics_text = ax.text2D(
                0.03,
                0.84,
                "",
                transform=ax.transAxes,
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            )

            self.subplot_data.append(
                {
                    "ax": ax,
                    "line_nn": line_nn,
                    "line_gt": line_gt,
                    "line_ana": line_ana,
                    "time_text": time_text,
                    "metrics_text": metrics_text,
                    "cap_data": None,
                    "nn_pred": None,
                    "nn_hip_global": None,
                    "analytical_pred": None,
                    "analytical_hip_global": None,
                    "valid_indices": [],
                }
            )

        # Controles GUI (Botones inferiores)
        ax_btn_prev = plt.axes([0.15, 0.02, 0.15, 0.045])
        ax_btn_play = plt.axes([0.33, 0.02, 0.15, 0.045])
        ax_btn_next = plt.axes([0.51, 0.02, 0.15, 0.045])
        ax_btn_view = plt.axes([0.69, 0.02, 0.16, 0.045])

        self.btn_prev = Button(
            ax_btn_prev, "◀ Anterior Sujeto (P)", color="#E0E0E0", hovercolor="#BDBDBD"
        )
        self.btn_play = Button(
            ax_btn_play, "⏸ Pausa (Espacio)", color="#E0E0E0", hovercolor="#BDBDBD"
        )
        self.btn_next = Button(
            ax_btn_next, "Siguiente Sujeto (N) ▶", color="#E0E0E0", hovercolor="#BDBDBD"
        )
        self.btn_view = Button(
            ax_btn_view, "Vista: Global (V)", color="#E0E0E0", hovercolor="#BDBDBD"
        )

        self.btn_prev.on_clicked(self.on_prev)
        self.btn_play.on_clicked(self.toggle_play)
        self.btn_next.on_clicked(self.on_next)
        self.btn_view.on_clicked(self.toggle_view)

        # Eventos de teclado
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        # Cargar sujeto inicial
        self.load_subject(self.current_subject_idx)

        # Configurar animación
        self.anim = FuncAnimation(
            self.fig, self.update_frame, interval=30, blit=False, cache_frame_data=False
        )

    def load_subject(self, sub_idx: int):
        """Carga y visualiza las 4 capturas del sujeto indicado."""
        if len(self.sorted_subjects) == 0:
            return

        sub_idx = sub_idx % len(self.sorted_subjects)
        self.current_subject_idx = sub_idx
        subject_id = self.sorted_subjects[sub_idx]
        caps = self.subjects_dict[subject_id]

        print(f"\n  📂 Cargando Sujeto {subject_id} ({len(caps)} capturas)...")
        self.fig.suptitle(
            f"Sujeto {subject_id} — (Sujeto {sub_idx + 1}/{len(self.sorted_subjects)})",
            fontsize=14,
            fontweight="bold",
        )

        valid_frame_counts = []

        for i in range(4):
            sp = self.subplot_data[i]
            ax = sp["ax"]

            if i < len(caps):
                mcd_path, aruco_path, _ = caps[i]
                file_name = Path(mcd_path).name
                cap = prepare_capture(mcd_path, aruco_path)

                if cap is not None:
                    anthropometric = np.array(
                        [
                            cap["femur_length"] / GaitDataset.BONE_NORM,
                            cap["tibia_length"] / GaitDataset.BONE_NORM,
                            cap["hip_width"] / GaitDataset.BONE_NORM,
                            1.0 if cap["direction"] == "R" else 0.0,
                        ],
                        dtype=np.float32,
                    )

                    nn_pred_pose, _, nn_hip_global, nn_valid = (
                        predict_full_sequence_multitask(
                            self.model,
                            cap["sensor_data"],
                            anthropometric,
                            self.norm_stats,
                            initial_hip=cap["gt_hip"],
                        )
                    )

                    # Inferencia Pipeline Analítico
                    analytical_pred = None
                    analytical_hip_global = None
                    if self.compare_analytical:
                        try:
                            res = process_files((mcd_path, aruco_path), use_3d=False)
                            if res is not None:
                                rec = res[2]
                                hip_center = (rec["hip_r"] + rec["hip_l"]) / 2.0
                                n_samples = len(rec["hip_r"])
                                ana_matrix = np.zeros((n_samples, N_JOINTS * N_COORDS))
                                for j_idx, j_name in enumerate(JOINT_NAMES):
                                    rel = rec[j_name] - hip_center
                                    ana_matrix[:, j_idx * 3 : (j_idx + 1) * 3] = rel

                                first_valid = cap.get("first_valid", 0)
                                cap_len = len(cap["sensor_data"])
                                analytical_pred = ana_matrix[
                                    first_valid : first_valid + cap_len
                                ]

                                ana_hip = hip_center[
                                    first_valid : first_valid + cap_len
                                ]
                                if len(ana_hip) > 0 and len(cap["gt_hip"]) > 0:
                                    ana_offset = cap["gt_hip"][0] - ana_hip[0]
                                    analytical_hip_global = ana_hip + ana_offset
                                else:
                                    analytical_hip_global = ana_hip
                        except Exception as e:
                            print(
                                f"  [INFO] Analítico no disponible para {file_name}: {e}"
                            )

                    mask = cap["gt_mask"] & nn_valid
                    valid_indices = np.where(mask)[0][:: self.step_subsample]
                    valid_frame_counts.append(len(valid_indices))

                    gt_valid = cap["gt_relative"][mask]
                    nn_valid_pred = nn_pred_pose[mask]
                    err_nn = np.mean(
                        np.linalg.norm(
                            nn_valid_pred.reshape(-1, N_JOINTS, N_COORDS)
                            - gt_valid.reshape(-1, N_JOINTS, N_COORDS),
                            axis=-1,
                        )
                    )

                    metrics_str = (
                        f"Dir: {cap['direction']} │ Error NN: {err_nn*100:.2f} cm"
                    )
                    if analytical_pred is not None and len(analytical_pred) == len(
                        cap["sensor_data"]
                    ):
                        ana_valid = analytical_pred[mask]
                        if len(ana_valid) == len(gt_valid):
                            err_ana = np.mean(
                                np.linalg.norm(
                                    ana_valid.reshape(-1, N_JOINTS, N_COORDS)
                                    - gt_valid.reshape(-1, N_JOINTS, N_COORDS),
                                    axis=-1,
                                )
                            )
                            metrics_str += f"\nError Analítico: {err_ana*100:.2f} cm"

                    sp["cap_data"] = cap
                    sp["nn_pred"] = nn_pred_pose
                    sp["nn_hip_global"] = nn_hip_global
                    sp["analytical_pred"] = analytical_pred
                    sp["analytical_hip_global"] = analytical_hip_global
                    sp["valid_indices"] = valid_indices

                    sp["metrics_text"].set_text(metrics_str)
                    ax.set_title(
                        f"{file_name} (Dir: {cap['direction']})",
                        fontsize=11,
                        fontweight="bold",
                    )
                    ax.legend(loc="upper right", fontsize=8)
                    ax.grid(True, alpha=0.3)
                    ax.set_xlabel("X (m)", labelpad=5, fontsize=8)
                    ax.set_ylabel("Y (m)", labelpad=5, fontsize=8)
                    ax.set_zlabel("Z (m)", labelpad=5, fontsize=8)
                    continue

            # Celda sin captura
            sp["cap_data"] = None
            sp["valid_indices"] = []
            sp["line_nn"].set_data_3d([], [], [])
            sp["line_gt"].set_data_3d([], [], [])
            if sp["line_ana"]:
                sp["line_ana"].set_data_3d([], [], [])
            sp["metrics_text"].set_text("")
            ax.set_title(f"Captura {i+1} (No disponible)", fontsize=10, color="gray")

        self.max_frames = max(valid_frame_counts) if len(valid_frame_counts) > 0 else 1
        self.frame_idx = 0

    def update_frame(self, _):
        """Actualiza el cuadro actual de animación en los 4 subplots simultáneamente."""
        if not self.is_playing:
            return

        for i in range(4):
            sp = self.subplot_data[i]
            if sp["cap_data"] is None or len(sp["valid_indices"]) == 0:
                continue

            v_indices = sp["valid_indices"]
            frame_pos = v_indices[self.frame_idx % len(v_indices)]
            cap = sp["cap_data"]

            # Posición cadera
            gt_hip = cap["gt_hip"][frame_pos]
            gt_hip_center = gt_hip if self.global_view else np.zeros(3)
            nn_hip = sp["nn_hip_global"][frame_pos] if self.global_view else np.zeros(3)

            # Red Neuronal
            pred_j = sp["nn_pred"][frame_pos].reshape(N_JOINTS, N_COORDS)
            nn_hip_r = pred_j[0] + nn_hip
            nn_hip_l = pred_j[1] + nn_hip
            nn_knee_r = pred_j[2] + nn_hip
            nn_knee_l = pred_j[3] + nn_hip
            nn_ankle_r = pred_j[4] + nn_hip
            nn_ankle_l = pred_j[5] + nn_hip

            nn_skel = np.array(
                [nn_ankle_r, nn_knee_r, nn_hip_r, nn_hip_l, nn_knee_l, nn_ankle_l]
            )
            sp["line_nn"].set_data_3d(nn_skel[:, 0], nn_skel[:, 1], nn_skel[:, 2])

            # Ground Truth
            gt_j = cap["gt_relative"][frame_pos].reshape(N_JOINTS, N_COORDS)
            gt_hip_r = gt_j[0] + gt_hip_center
            gt_hip_l = gt_j[1] + gt_hip_center
            gt_knee_r = gt_j[2] + gt_hip_center
            gt_knee_l = gt_j[3] + gt_hip_center
            gt_ankle_r = gt_j[4] + gt_hip_center
            gt_ankle_l = gt_j[5] + gt_hip_center

            gt_skel = np.array(
                [gt_ankle_r, gt_knee_r, gt_hip_r, gt_hip_l, gt_knee_l, gt_ankle_l]
            )
            sp["line_gt"].set_data_3d(gt_skel[:, 0], gt_skel[:, 1], gt_skel[:, 2])

            # Pipeline Analítico
            if (
                self.compare_analytical
                and sp["analytical_pred"] is not None
                and sp["line_ana"] is not None
            ):
                ana_j = sp["analytical_pred"][frame_pos].reshape(N_JOINTS, N_COORDS)
                ana_hip = (
                    sp["analytical_hip_global"][frame_pos]
                    if (self.global_view and sp["analytical_hip_global"] is not None)
                    else np.zeros(3)
                )
                ana_hip_r = ana_j[0] + ana_hip
                ana_hip_l = ana_j[1] + ana_hip
                ana_knee_r = ana_j[2] + ana_hip
                ana_knee_l = ana_j[3] + ana_hip
                ana_ankle_r = ana_j[4] + ana_hip
                ana_ankle_l = ana_j[5] + ana_hip

                ana_skel = np.array(
                    [
                        ana_ankle_r,
                        ana_knee_r,
                        ana_hip_r,
                        ana_hip_l,
                        ana_knee_l,
                        ana_ankle_l,
                    ]
                )
                sp["line_ana"].set_data_3d(
                    ana_skel[:, 0], ana_skel[:, 1], ana_skel[:, 2]
                )

            # Ajustar vista
            ax = sp["ax"]
            if self.global_view:
                hx = nn_hip[0]
                ax.set_xlim(hx - 0.6, hx + 0.6)
                ax.set_ylim(-0.5, 0.5)
                ax.set_zlim(-0.1, 1.1)
            else:
                ax.set_xlim(-0.6, 0.6)
                ax.set_ylim(-0.5, 0.5)
                ax.set_zlim(-0.8, 0.3)

            t_sec = (cap["timestamps"][frame_pos] - cap["timestamps"][0]) / 1000.0
            sp["time_text"].set_text(f"t: {t_sec:.2f} s")

        self.frame_idx = (self.frame_idx + 1) % self.max_frames

    def toggle_play(self, event=None):
        self.is_playing = not self.is_playing
        self.btn_play.label.set_text(
            "▶ Reproducir" if not self.is_playing else "⏸ Pausa (Espacio)"
        )
        self.fig.canvas.draw_idle()

    def toggle_view(self, event=None):
        self.global_view = not self.global_view
        self.btn_view.label.set_text(
            "Vista: Global (V)" if self.global_view else "Vista: Local Cadera (V)"
        )
        self.fig.canvas.draw_idle()

    def on_prev(self, event=None):
        self.load_subject(self.current_subject_idx - 1)
        self.fig.canvas.draw_idle()

    def on_next(self, event=None):
        self.load_subject(self.current_subject_idx + 1)
        self.fig.canvas.draw_idle()

    def on_key_press(self, event):
        if event.key == " ":
            self.toggle_play()
        elif event.key in ["n", "N"]:
            self.on_next()
        elif event.key in ["p", "P"]:
            self.on_prev()
        elif event.key in ["v", "V"]:
            self.toggle_view()


def main():
    parser = argparse.ArgumentParser(
        description="Visor 3D interactivo para 4 capturas por sujeto"
    )
    parser.add_argument(
        "--dataset", type=str, default=str(DATASET_PATH), help="Ruta al dataset"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Ruta al checkpoint del modelo"
    )
    parser.add_argument(
        "--subject", type=str, default=None, help="Número de sujeto para iniciar"
    )
    parser.add_argument(
        "--compare-analytical",
        action="store_true",
        help="Incluir reconstrucción del pipeline analítico en la animación",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    default_ckpt = (
        CHECKPOINT_DIR / "best_tcn_lite.pt"
        if (CHECKPOINT_DIR / "best_tcn_lite.pt").exists()
        else CHECKPOINT_DIR / "best_multitask_model.pt"
    )
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_ckpt
    norm_stats_path = CHECKPOINT_DIR / "norm_stats.json"

    print()
    print("═" * 72)
    print("  🖥️  VISOR INTERACTIVO 3D: 4 CAPTURAS POR SUJETO (MULTITAREA UNIFICADA)")
    print("═" * 72)
    print(f"  Dataset:          {dataset_path}")
    print(f"  Device:           {DEVICE}")
    print("═" * 72)

    # 1. Verificar checkpoint y norm_stats
    if not checkpoint_path.exists():
        print(f"  [ERROR] No se encontró el modelo guardado en {checkpoint_path}")
        print("  Ejecute primero 'python nn_tcn_lite.py' para entrenar el modelo.")
        sys.exit(1)

    if not norm_stats_path.exists():
        print(
            f"  [ERROR] No se encontraron estadísticas de normalización en {norm_stats_path}"
        )
        sys.exit(1)

    with open(norm_stats_path, "r") as f:
        norm_stats_raw = json.load(f)
        norm_stats = {
            "mean": np.array(norm_stats_raw["mean"], dtype=np.float32),
            "std": np.array(norm_stats_raw["std"], dtype=np.float32),
        }

    # 2. Cargar modelo
    print("\n  🧠 Cargando red neuronal multitarea unificada...")
    model = GaitMultiTaskNet(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"  ✓ GaitMultiTaskNet cargado exitosamente (época {checkpoint['epoch']}, val_loss: {checkpoint['val_loss']:.6f})"
    )

    # 3. Descubrir capturas
    print("\n  📂 Buscando capturas disponibles...")
    captures_info = discover_captures(dataset_path)
    print(f"  Encontradas {len(captures_info)} capturas procesables.")

    if len(captures_info) == 0:
        print("  [ERROR] No se encontraron pares MCD/ArUco.")
        sys.exit(1)

    print(f"\n  ▶️ Iniciando ventana interactiva Matplotlib...")
    print("     Controles de teclado:")
    print("       [Espacio] : Pausa / Reproducir")
    print("       [N]       : Siguiente sujeto")
    print("       [P]       : Sujeto anterior")
    print("       [V]       : Alternar vista (Global / Local)")
    print()

    # Iniciar visor
    viewer = GaitAnimationViewer(
        captures_info=captures_info,
        model=model,
        norm_stats=norm_stats,
        compare_analytical=args.compare_analytical,
        start_subject_id=args.subject,
    )

    plt.show()
    return viewer


if __name__ == "__main__":
    global_viewer = main()
