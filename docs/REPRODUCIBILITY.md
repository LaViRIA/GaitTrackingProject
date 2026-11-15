# Training reproducibility

This document records the configuration used to train the five networks of the paper and the commands to retrain them. Retraining does not give bit-identical checkpoints, and for TCN Lite it gives a less accurate network than the committed one on average; see [Determinism](#determinism) and [Origin of each configuration](#origin-of-each-configuration).

All commands are run from the repository root.

To retrain with several seeds, evaluate each one and aggregate mean ± SD without touching the committed checkpoints, see [`reproducibility/README.md`](../reproducibility/README.md).

## Common conditions

| Item | Value |
|---|---|
| Seed | `42` (`--seed 42`) for all five networks |
| Subject split | `data_split.json` (20 train / 5 val / 5 test) |
| Dataset | `DataSet/` in the repository root (see the [README](../README.md#dataset)); override with `--dataset` or `DATASET` |
| Signal filter (networks) | Butterworth low-pass, order 6, 7 Hz |

`data_split.json`:

- Train: 25, 1, 27, 29, 12, 30, 10, 23, 9, 28, 19, 22, 5, 11, 17, 4, 13, 26, 7, 6
- Validation: 8, 3, 2, 16, 20
- Test: 24, 18, 14, 21, 15

## Train everything

```bash
./train_all.sh
```

To train only some of the networks:

```bash
./train_all.sh tcn ttl
```

The logs are written to `nn-results/train_logs/<net>.log`. The script overwrites the checkpoints in `nn-results/<net>/checkpoints/`, so make a backup first.

Optional environment variables: `PYTHON` (default `.venv/bin/python`), `SEED`, `SPLIT`, `DATASET`, `LOG_DIR`.

## Command for each network

### 1. Stance-foot classifier (`GaitStaticPivotNet`)

```bash
python -m models.nn_pivot_classifier --epochs 100 --batch-size 1024 --lr 1e-3 --patience 25 --seed 42 --data-split data_split.json
```

- Hidden layers [16, 8, 8], dropout 0.15. Per-frame input (no windows).
- AdamW (fused on GPU), weight decay 1e-4, ReduceLROnPlateau (factor 0.5, patience 5), BCE loss.
- Checkpoint: `nn-results/pivot_classifier/checkpoints/best_pivot_classifier.pt`

### 2. Pose relative to the stance foot (`GaitStaticNet`)

```bash
python -m models.nn_static_pivot --epochs 300 --batch-size 128 --lr 1e-3 --patience 30 --seed 42 --data-split data_split.json
```

- Hidden layers [64, 64, 32], dropout [0.15, 0.15, 0.1]. Per-frame input.
- AdamW (fused on GPU), weight decay 1e-4, ReduceLROnPlateau (factor 0.5, patience 5).
- Checkpoint: `nn-results/static_pivot/checkpoints/best_static_model.pt`

### 3. TCN Lite (`GaitMultiTaskNetTCN`, proposed model)

```bash
python -m models.nn_tcn_lite --window-size 25 --window-stride 2 --batch-size 12 --lr 5e-4 --epochs 500 --patience 60 --seed 42 --data-split data_split.json
```

- Channels [8, 16, 32, 32], kernels [7, 7, 5, 5], dilations [1, 2, 4, 8], 48-neuron heads, dropout 0.2 (14,729 parameters).
- AdamW, weight decay 1e-5, CosineAnnealingWarmRestarts (T_0 = 30, T_mult = 2), mixed precision on GPU.
- Checkpoint: `nn-results/tcn_lite/checkpoints/best_tcn_lite.pt`

### 4. TTL Lite (`GaitMultiTaskNetTTLLite`)

```bash
python -m models.nn_ttl_lite --window-size 25 --window-stride 3 --batch-size 24 --lr 5e-4 --epochs 300 --patience 60 --seed 42 --data-split data_split.json
```

- TCN [16, 16, 32] (kernels [7, 7, 5], dilations [1, 2, 4]), BiLSTM 32 × 1 layer, attention with 2 heads, dropout 0.2.
- AdamW, weight decay 1e-5, CosineAnnealingWarmRestarts (T_0 = 30, T_mult = 2), mixed precision on GPU.
- Checkpoint: `nn-results/ttl_lite/checkpoints/best_ttl_lite.pt`

### 5. Multi-task Transformer (`GaitMultiTaskNet`)

```bash
python -m models.nn_transformer_multitask --window-size 25 --window-stride 3 --batch-size 24 --lr 5e-4 --epochs 300 --patience 60 --seed 42 --data-split data_split.json
```

- CNN [4, 8, 16], BiLSTM 12 × 3 layers, attention with 12 heads, dropout 0.2.
- AdamW, weight decay 1e-5, CosineAnnealingWarmRestarts (T_0 = 30, T_mult = 2), mixed precision on GPU.
- Checkpoint: `nn-results/transformer_multitask/checkpoints/best_multitask_model_trans.pt`

## What each checkpoint stores

Besides the weights, each `.pt` stores:

- `norm_stats`: mean and standard deviation of the z-score normalization, computed on the training set only.
- `config`: the actual training hyperparameters (seed, batch, lr, window, stride, subjects of each set, architecture).
- `epoch` and `val_loss` of the best model.

The evaluation scripts read the inference window and the normalization from the checkpoint itself, so they do not depend on the loose `norm_stats.json` files.

## Determinism

At start-up each script seeds `random`, `numpy` and `torch`, and PyTorch seeds the `DataLoader` workers from that seed, so the batch order and the data augmentation follow the seed.

Training is **not** guaranteed to be bit-reproducible:

- The code does not enable `torch.use_deterministic_algorithms(True)`, so cuDNN may pick non-deterministic kernels.
- The sequence networks train with mixed precision (AMP) on GPU.
- Kernel selection depends on the GPU model and on the PyTorch/CUDA/cuDNN versions.

In practice the outcome is stable across machines but not identical. Retraining TCN Lite with the same seed on an RTX 4060 and an RTX 5070 (PyTorch 2.15, CUDA 13.2), and on the RTX 5070 with PyTorch 2.14.1 (CUDA 13.0), stopped at the same best epoch to within one epoch, and the global MPJPE agreed to within 1–4 mm. The choice of seed matters more: across seeds the global MPJPE varies by about ±5 mm. Results should therefore be reported as mean ± SD over several seeds ([`reproducibility/README.md`](../reproducibility/README.md)).

Changing the architecture also changes the batch order, even with the same seed: the model initialization consumes random numbers before the batches are created.

## Origin of each configuration

- **TCN Lite, TTL Lite and Transformer:** the committed checkpoints were trained before `train_all.sh` existed. Their weights are unchanged since commit c27fa43; commit 197b721 only added the `config` block that records these commands. Retraining with these commands does **not** reproduce them bit for bit:
  - TTL Lite and the Transformer: the retrained networks reach the accuracy of the committed checkpoints (global MPJPE within one SD of the multi-seed mean).
  - TCN Lite: with seed 42, retraining stops at best epoch 81–82 (committed checkpoint: epoch 350) with a global MPJPE of 0.096–0.098 m (committed: 0.0656 m). Over seeds 42, 1, 2, 3, 4 on two GPUs, the global MPJPE is 0.094 ± 0.005 m (n = 10). The committed TCN Lite checkpoint is therefore better than what the current code and commands produce on average.
- **Pivot and static:** the original checkpoints (September 2026) came from a version of the code that is not in git and could not be reproduced. The current configuration was chosen with a batch × lr grid search, selecting by `val_loss` (never by test) with seed 42:

| Network | Batch | lr | Best `val_loss` |
|---|---|---|---|
| Static | 64 / **128** / 256 | **1e-3** / 3e-4 | **0.00080** (128, 1e-3; with 300 epochs, 0.00075 at epoch 201) |
| Pivot | 256 / **1024** | **1e-3** / 5e-3 | **0.3448** (1024, 1e-3, epoch 20) |

The original checkpoints were kept in a local backup (`backups/`, not part of the repository).

## Current results (retraining of 2026-11-12)

Full evaluation in `nn-results/full_neural_vs_analytical/results-v4.txt`; the previous one, with the original pivot and static networks, is in `results-v3.txt`. TCN Lite, TTL Lite, Transformer and the analytical pipeline do not change. Only the rows of the static baseline change:

| Metric (test, P95 filter) | v3 (original) | v4 (current) |
|---|---|---|
| Pivot accuracy in the pipeline | 88.25 % | 89.64 % |
| Relative MPJPE, Static (P+P) | 0.0444 m | 0.0338 m |
| Global MPJPE, Static (P+P) | 0.2669 m | 0.2670 m |
| Global MAE / RMSE, Static (P+P) | 0.0917 / 0.1470 m | 0.0927 / 0.1493 m |
| Global MPJPE, Oracle | 0.3012 m | 0.2938 m |
| Global MAE / RMSE, Oracle | 0.0981 / 0.1594 m | 0.1002 / 0.1633 m |
| P95 of the per-frame error, Static (figures) | ~65 cm | 63.0 cm |

Consequence for the paper: the static network now has the **best pelvis-relative MPJPE** (0.034 m vs 0.036 m for TCN Lite), but it is still much worse in the global frame (0.27 m vs 0.066 m). `v2_manuscript/template.tex` and the figures in `v2_manuscript/figures/` are already updated with these values.

## Analytical pipeline: Slerp clip 0.5 (2026-11-15)

A full re-run of the 20,736-configuration grid (`experiment_matrix.py`, training split) found the optimum at the previously documented values except `slerp_clip_min` = 0.5 (training centroid error 0.1780 m, versus 0.1804 m with 0.3; validation 0.1670 m, test 0.1756 m). Because `slerp_clip_max` = 1 − `slerp_clip_min`, this fixes the hip Slerp weight at 0.5: the pelvis is the Slerp midpoint of the two femurs. All scripts now use 0.5/0.5. The networks are unaffected; the analytical pipeline changes from 0.1286 to 0.1274 m global MPJPE (relative MPJPE 0.0586 → 0.0587 m). Full evaluation with the committed checkpoints on an RTX 5070: `nn-results/full_neural_vs_analytical/results-v5.txt` (results-v4.txt used 0.3).


## Evaluation

```bash
python evaluate_full_neural_vs_analytical.py
python generate_manuscript_figures.py
```

The evaluation uses the same test subjects of `data_split.json` and the window stored in each checkpoint (25 samples).
