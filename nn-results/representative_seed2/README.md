# Representative retrained networks (seed 2)

The five networks of one multi-seed retraining run (`SEED=2 ./train_all.sh` on an NVIDIA RTX 4060), used for **Figures 2–6** of the manuscript and the per-trial analyses of its Results section.

The run was chosen because its TCN Lite global MPJPE (0.0945 m) is the closest to the mean of the 15 retrained runs (0.0941 m; 5 seeds × 3 GPUs). The released checkpoints in `nn-results/*/checkpoints/` are a favourable outlier (TCN Lite 0.0655 m), so the figures do not use them.

The layout mirrors `nn-results/`: `reproducibility/figures.sh` copies these files over the released checkpoints in a scratch copy of the repository and runs `generate_manuscript_figures.py`.

| File | SHA-256 |
|---|---|
| `pivot_classifier/checkpoints/best_pivot_classifier.pt` | `42a756536508921aecfe5e5390795c431a3c9c057332e2cd59cad6ccfa94cdaf` |
| `static_pivot/checkpoints/best_static_model.pt` | `49f99c1498714920118cdf3ab8294c23a07a3c7f7aa11d5d896fcdeb24463a9f` |
| `tcn_lite/checkpoints/best_tcn_lite.pt` | `408de1651bce35ece3de587f1fe9afa19c3a0f7c50b56db909739d4539e96789` |
| `ttl_lite/checkpoints/best_ttl_lite.pt` | `968b6bbe2162318e98363d08e2b4afd3d11ebda3ebb27be802b455715b8aa124` |
| `transformer_multitask/checkpoints/best_multitask_model_trans.pt` | `35d6bbe5dfe2d0f3013577b5307b0f159c62a0174fda78afb6fe2a9619ff313f` |

Evaluation of this run (`evaluate_full_neural_vs_analytical.py`, Slerp clip 0.5, P95 trimming):

| Method | Global MPJPE | Rel. MPJPE |
|---|---|---|
| Static (PivotNet + PoseNet) | 0.2641 m | 0.0320 m |
| Oracle | 0.2963 m | 0.0322 m |
| TCN Lite | 0.0945 m | 0.0400 m |
| TTL Lite | 0.0973 m | 0.0431 m |
| Transformer | 0.0831 m | 0.0456 m |
| Analytical | 0.1274 m | 0.0587 m |

PivotNet in-pipeline accuracy: 90.03 %.
