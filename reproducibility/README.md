# Reproducing the results

These scripts reproduce the paper's evaluation, a multi-seed retraining of the five networks, and the timing benchmark. They need only a shell and the dataset.

Every run happens inside its own copy of the repository under `runs/`. The working tree and the committed checkpoints are never modified.

## Requirements

- Linux, Python 3.12 managed by [uv](https://docs.astral.sh/uv/). Install uv with `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- An NVIDIA GPU is recommended for training. Evaluation and timing also run on CPU.
- The dataset (`DataSet_GaitFromIMU.zip`), unzipped into `DataSet/` in the repository root, so that the subject folders are `DataSet/1/`, …, `DataSet/30/`:

  ```bash
  unzip DataSet_GaitFromIMU.zip
  mv DataSet_Final/DataSet ./DataSet
  ```

  To keep it elsewhere, set `DATASET` to the folder that holds the subject sub-folders (`export DATASET=/path/to/DataSet`).

## Steps

Run everything from the repository root.

```bash
reproducibility/setup_env.sh            # 1. create .venv (torch build chosen for the local CUDA driver)
reproducibility/evaluate_committed.sh   # 2. evaluate the committed checkpoints      (~1 min)
reproducibility/multiseed.sh            # 3. retrain + evaluate, seeds 42 1 2 3 4    (~40 min per seed)
.venv/bin/python reproducibility/aggregate_seeds.py runs/seed_* \
    --committed runs/committed/eval.log --json runs/aggregate.json
reproducibility/timing.sh               # 4. execution-time benchmark, run twice     (several minutes)
```

| Step | Output | What it does |
|---|---|---|
| 2 | `runs/committed/eval.log` | `evaluate_full_neural_vs_analytical.py` on the five test subjects of `data_split.json`, using the checkpoints in `nn-results/`. Compare with `nn-results/full_neural_vs_analytical/results-v4.txt`. |
| 3 | `runs/seed_<s>/` | For each seed: `train_all.sh` (all five networks, `SEED=<s>`), then the same evaluation. The training logs go to `runs/seed_<s>/nn-results/train_logs/`. Finished seeds are skipped, so an interrupted run can be resumed by running the command again. To choose the seeds, pass them as arguments: `reproducibility/multiseed.sh 5 6 7`. |
| 3b | stdout, `runs/aggregate.json` | Per-seed values and mean ± SD of the P95 MPJPE, MAE and RMSE (global and pelvis-relative); the improvement over the analytical pipeline; the best network for each seed; and a paired TCN Lite vs. Transformer test. |
| 4 | `runs/timing/timing_<host>_run{1,2}.{json,log}` | `benchmark_timing.py`: streaming latency per 25-sample window (GPU, all CPU threads, one CPU thread), offline throughput, and analytical-pipeline cost per frame. The JSON also records the hardware and software versions and the checkpoint hashes. |

Optional environment variables:

- `PYTHON`: interpreter to use (default `.venv/bin/python`).
- `RUNS_DIR`: output folder (default `runs/`).

Runs from several machines can be pooled by passing all their `seed_*` folders to `aggregate_seeds.py`.

## Reference results

These are the values we obtained on two machines. Both used torch 2.15.0+cu132 and the same five seeds.

**Committed checkpoints (step 2), global P95 MPJPE:**

| | TCN Lite | TTL Lite | Transformer | Analytical |
|---|---|---|---|---|
| RTX 5070 | 0.0655 m | 0.0931 m | 0.0804 m | 0.1286 m |
| RTX 4060 | 0.0656 m | 0.0931 m | 0.0804 m | 0.1286 m |

**Retrained networks (step 3), global P95 MPJPE, mean ± SD (10 runs: 5 seeds × 2 machines):**

| TCN Lite | TTL Lite | Transformer | Analytical |
|---|---|---|---|
| 0.094 ± 0.005 m | 0.095 ± 0.009 m | 0.080 ± 0.005 m (8 runs) | 0.1286 m |

- **Training is not bit-reproducible.** The code does not enable `torch.use_deterministic_algorithms`, and it trains with mixed precision on GPU. The same seed on two different GPUs gives slightly different weights.
- **Expect the same spread, not the same digits.** The seed changes global MPJPE by about ±5 mm. The GPU model changes it much less.
- **One Transformer run diverges.** With seed 3, the Transformer drifts in displacement on both machines (global MPJPE ≈ 1.7–1.8 m; validation loss ≈ 0.24, versus ≈ 0.003 for the other seeds).
- **Diverged runs are excluded automatically.** `aggregate_seeds.py` lists any run above `--diverged-above` (0.5 m by default) and leaves it out of that network's mean and SD.

**Timing (step 4), streaming step with batch 1, median [p99]:**

| | RTX 5070 / Ryzen 7 9700X | RTX 4060 / Ryzen 5 7600 |
|---|---|---|
| TCN Lite, GPU | 0.41 [0.43] ms | 0.62 [0.66] ms |
| TCN Lite, 1 CPU thread | 0.19 [0.20] ms | 0.29 [0.31] ms |
| Analytical pipeline, per frame | 0.24 ms | 0.32 ms |

Timings depend on the hardware. Compare the two runs on your own machine against each other, not against this table.

## Other entry points

- `python process_all.py`: the analytical pipeline over the whole dataset.
- `python generate_manuscript_figures.py`: the paper figures. It writes to `v2_manuscript/figures/`; `--dataset` overrides the dataset location.
