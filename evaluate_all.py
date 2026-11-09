import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

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
)


def evaluate_all_captures():
    print("Iniciando evaluación masiva de todas las capturas...")

    # Cargar estadísticas de normalización
    stats_path = CHECKPOINT_DIR / "norm_stats.json"
    if not stats_path.exists():
        print(f"Error: No se encontró el archivo de normalización en {stats_path}")
        return

    with open(stats_path, "r") as f:
        norm_stats_raw = json.load(f)
        norm_stats = {
            "mean": np.array(norm_stats_raw["mean"], dtype=np.float32),
            "std": np.array(norm_stats_raw["std"], dtype=np.float32),
        }

    # Inicializar modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GaitMultiTaskNet(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM).to(device)

    ckpt_path = (
        CHECKPOINT_DIR / "best_tcn_lite.pt"
        if (CHECKPOINT_DIR / "best_tcn_lite.pt").exists()
        else CHECKPOINT_DIR / "best_multitask_model.pt"
    )
    if not ckpt_path.exists():
        print(f"Error: No se encontró el modelo entrenado en {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Descubrir todas las capturas
    captures_info = discover_captures(DATASET_PATH)
    print(f"Se encontraron {len(captures_info)} capturas en total.")

    # Listas para acumular métricas
    results = []

    nn_mpjpe_list, nn_mae_list, nn_rmse_list = [], [], []
    ana_mpjpe_list, ana_mae_list, ana_rmse_list = [], [], []

    for idx, (mcd_path, aruco_path, subject_id) in enumerate(captures_info):
        file_name = Path(mcd_path).name
        label = f"{subject_id}/{file_name}"
        print(f"Procesando [{idx+1}/{len(captures_info)}]: {label}")

        cap = prepare_capture(mcd_path, aruco_path, crop=True)
        if cap is None:
            continue

        # ── 1. Predicción NN Multitarea ──
        anthropometric = np.array(
            [
                cap["femur_length"] / GaitDataset.BONE_NORM,
                cap["tibia_length"] / GaitDataset.BONE_NORM,
                cap["hip_width"] / GaitDataset.BONE_NORM,
                1.0 if cap["direction"] == "R" else 0.0,
            ],
            dtype=np.float32,
        )

        t0_nn = time.time()
        nn_pred_pose, nn_pred_disp, nn_hip_global, nn_valid = (
            predict_full_sequence_multitask(
                model,
                cap["sensor_data"],
                anthropometric,
                norm_stats,
                initial_hip=cap["gt_hip"],
            )
        )
        t1_nn = time.time()
        time_nn = t1_nn - t0_nn

        # ── 2. Reconstrucción Analítica ──
        try:
            t0_ana = time.time()
            analytical_result = process_files((mcd_path, aruco_path), use_3d=False)
            t1_ana = time.time()
            time_ana = t1_ana - t0_ana
            if analytical_result is None:
                continue
        except Exception as e:
            print(f"Error en Analítico para {label}: {e}")
            continue

        _, _, reconstruction, _, _, _, _ = analytical_result

        # Convertir a relativo a cadera
        hip_center = (reconstruction["hip_r"] + reconstruction["hip_l"]) / 2.0
        analytical_relative = {}
        for j_name in JOINT_NAMES:
            analytical_relative[j_name] = reconstruction[j_name] - hip_center

        # ── 3. Calcular Métricas ──
        mask = cap["gt_mask"] & nn_valid
        if not np.any(mask):
            continue

        gt_valid = cap["gt_relative"][mask].reshape(-1, N_JOINTS, N_COORDS)
        nn_valid_pred = nn_pred_pose[mask].reshape(-1, N_JOINTS, N_COORDS)

        first_valid = cap.get("first_valid", 0)
        cap_len = len(cap["sensor_data"])

        # Sincronizar analítico
        ana_rel_arr = np.zeros_like(nn_pred_pose)
        for j_idx, j_name in enumerate(JOINT_NAMES):
            analytic_slice = analytical_relative[j_name][
                first_valid : first_valid + cap_len
            ]
            if len(analytic_slice) == cap_len:
                ana_rel_arr[:, j_idx * 3 : (j_idx + 1) * 3] = analytic_slice

        ana_valid_pred = ana_rel_arr[mask].reshape(-1, N_JOINTS, N_COORDS)

        # Calcular errores por frame (tiempo)
        err_nn_t = np.mean(np.linalg.norm(nn_valid_pred - gt_valid, axis=-1), axis=1)
        err_ana_t = np.mean(np.linalg.norm(ana_valid_pred - gt_valid, axis=-1), axis=1)
        t_arr = cap["timestamps"][mask]

        # Calcular errores globales escalares
        err_nn_mpjpe = np.mean(err_nn_t)
        err_nn_mae = np.mean(np.abs(nn_valid_pred - gt_valid))
        err_nn_rmse = np.sqrt(np.mean((nn_valid_pred - gt_valid) ** 2))

        err_ana_mpjpe = np.mean(err_ana_t)
        err_ana_mae = np.mean(np.abs(ana_valid_pred - gt_valid))
        err_ana_rmse = np.sqrt(np.mean((ana_valid_pred - gt_valid) ** 2))

        # Guardar en listas
        nn_mpjpe_list.append(err_nn_mpjpe)
        nn_mae_list.append(err_nn_mae)
        nn_rmse_list.append(err_nn_rmse)

        ana_mpjpe_list.append(err_ana_mpjpe)
        ana_mae_list.append(err_ana_mae)
        ana_rmse_list.append(err_ana_rmse)

        results.append(
            {
                "label": label,
                "t": t_arr,
                "err_nn_t": err_nn_t,
                "err_ana_t": err_ana_t,
                "nn_mpjpe": err_nn_mpjpe * 100,
                "nn_mae": err_nn_mae * 100,
                "nn_rmse": err_nn_rmse * 100,
                "ana_mpjpe": err_ana_mpjpe * 100,
                "ana_mae": err_ana_mae * 100,
                "ana_rmse": err_ana_rmse * 100,
                "nn_time": time_nn,
                "ana_time": time_ana,
            }
        )

    # ── 4. Mostrar Reporte ──
    print("\n" + "=" * 80)
    print("  RESULTADOS DE EVALUACIÓN MASIVA (TODAS LAS CAPTURAS)")
    print("=" * 80)

    print(
        f"{'Captura':<15} | {'NN MPJPE':<9} | {'NN MAE':<9} | {'NN RMSE':<9} | {'Ana MPJPE':<9} | {'Ana MAE':<9} | {'Ana RMSE':<9}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['label']:<15} | {r['nn_mpjpe']:>6.2f} cm | {r['nn_mae']:>6.2f} cm | {r['nn_rmse']:>6.2f} cm | "
            f"{r['ana_mpjpe']:>6.2f} cm | {r['ana_mae']:>6.2f} cm | {r['ana_rmse']:>6.2f} cm"
        )

    if len(results) == 0:
        print("  [AVISO] No hay resultados válidos para mostrar.")
        return

    print("-" * 80)
    print(
        f"{'PROMEDIO':<15} | {np.mean(nn_mpjpe_list)*100:>6.2f} cm | {np.mean(nn_mae_list)*100:>6.2f} cm | {np.mean(nn_rmse_list)*100:>6.2f} cm | "
        f"{np.mean(ana_mpjpe_list)*100:>6.2f} cm | {np.mean(ana_mae_list)*100:>6.2f} cm | {np.mean(ana_rmse_list)*100:>6.2f} cm"
    )
    print("=" * 80)

    nn_wins = sum(1 for r in results if r["nn_mpjpe"] < r["ana_mpjpe"])
    print(
        f"Red Neuronal gana en: {nn_wins} de {len(results)} capturas ({(nn_wins/len(results))*100:.1f}%)"
    )

    total_nn_time = sum(r["nn_time"] for r in results)
    total_ana_time = sum(r["ana_time"] for r in results)
    avg_nn_time = total_nn_time / len(results)
    avg_ana_time = total_ana_time / len(results)

    print("-" * 80)
    print("  RENDIMIENTO Y TIEMPOS DE EJECUCIÓN")
    print("-" * 80)
    print(
        f"Tiempo total Red Neuronal:    {total_nn_time:>6.2f} s  (Promedio por captura: {avg_nn_time*1000:>6.2f} ms)"
    )
    print(
        f"Tiempo total Analítico:       {total_ana_time:>6.2f} s  (Promedio por captura: {avg_ana_time*1000:>6.2f} ms)"
    )
    if avg_nn_time < avg_ana_time:
        print(
            f"La Red Neuronal es {avg_ana_time / max(avg_nn_time, 1e-6):.1f}x más rápida."
        )
    else:
        print(
            f"El método Analítico es {avg_nn_time / max(avg_ana_time, 1e-6):.1f}x más rápido."
        )

    # ── 5. Generar Gráficas de Error (Estilo 'All Errors') ──
    print("\nGenerando gráficas de error temporal...")

    fig, (ax_nn, ax_ana) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for r in results:
        # t is in ms, we can plot in ms or s
        t = r["t"]
        # err_nn_t and err_ana_t are in meters
        ax_nn.plot(t, r["err_nn_t"], alpha=0.5, linewidth=1.0)
        ax_ana.plot(t, r["err_ana_t"], alpha=0.5, linewidth=1.0)

    ax_nn.set_title("Red Neuronal (Multitarea) - Todos los Errores")
    ax_nn.set_xlabel("Tiempo (ms)")
    ax_nn.set_ylabel("Error Euclídeo MPJPE (m)")
    ax_nn.grid(True, alpha=0.3)
    ax_nn.set_ylim(0, 0.8)  # Mismo límite que process_all

    ax_ana.set_title("Pipeline Analítico - Todos los Errores")
    ax_ana.set_xlabel("Tiempo (ms)")
    ax_ana.grid(True, alpha=0.3)

    plt.tight_layout()
    out_img = CHECKPOINT_DIR / "all_errors_comparison.png"
    plt.savefig(out_img, dpi=300, bbox_inches="tight")
    print(f"Gráfica guardada exitosamente en: {out_img}")


if __name__ == "__main__":
    evaluate_all_captures()
