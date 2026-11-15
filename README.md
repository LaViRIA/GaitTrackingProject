# GaitTrackingProject

3D reconstruction of lower-limb gait from 4 IMUs, comparing an analytical kinematic pipeline against several neural networks, with ArUco markers as ground truth.

## Structure

```
process_all.py                         Analytical pipeline (Madgwick/EKF + PCA + forward kinematics); base library
evaluate_full_neural_vs_analytical.py  Final comparison: analytical vs all the networks
experiment_matrix.py                   Hyperparameter grid search of the analytical pipeline
experiment_matrix_tcn.py               Hyperparameter grid search of TCN Lite
analyze_results.py                     Analysis/plots of the experiment_matrix.py CSV
generate_manuscript_figures.py         Manuscript figures (v2_manuscript/figures/)
generate_split.py, data_split.json     Fixed train/val/test subject split
benchmark_timing.py                    Execution-time benchmark (latency per window, analytical pipeline)
train_all.sh                           Retrains the 5 networks (see docs/REPRODUCIBILITY.md)
reproducibility/                       Scripts for reviewers: evaluation, multi-seed retraining, timing

models/                                Neural networks (training + classes used in the evaluation)
  nn_tcn_lite.py                       Non-causal dilated TCN (proposed model)
  nn_ttl_lite.py                       TCN + Transformer hybrid
  nn_transformer_multitask.py          Conv1D + BiLSTM + Transformer
  nn_pivot_classifier.py               Stance-foot classifier
  nn_static_pivot.py                   Pose relative to the pivot foot

nn-results/                            Network checkpoints and results
v2_manuscript/                         Manuscript (template.tex/.md/.pdf, sections, figures)
docs/                                  Code guide, training reproducibility, figure-change notes
```

`DataSet/` (the dataset, see above) and `Results/` (outputs of the analytical pipeline and of the experiment matrices) are not under version control.

## Dataset

The scripts expect the dataset in a folder called `DataSet/` in the repository root, with one sub-folder per subject (`DataSet/1/`, …, `DataSet/30/`), each holding the `mcd-<n>.csv` (IMU) and `arUcos-<n>.csv` (ArUco ground truth) files of every trial. The dataset is distributed as `DataSet_GaitFromIMU.zip`; the subject folders are inside `DataSet_Final/DataSet/`:

```bash
unzip DataSet_GaitFromIMU.zip
mv DataSet_Final/DataSet ./DataSet      # run from the repository root
```

`DataSet/` is in `.gitignore`. To keep the dataset elsewhere, pass `--dataset /path/to/DataSet` to any script (or set `DATASET=/path/to/DataSet` for the shell scripts).

## Usage

All commands are run from the repository root.

```bash
pip install -r requirements.txt

python process_all.py                            # analytical pipeline
python experiment_matrix.py                       # analytical grid search
python analyze_results.py                         # grid-search analysis
python -m models.nn_tcn_lite                      # train a model (same for the others in models/)
python experiment_matrix_tcn.py                   # TCN Lite grid search
python evaluate_full_neural_vs_analytical.py      # final comparison
python generate_manuscript_figures.py             # manuscript figures
```

To reproduce the results of the paper (evaluation, retraining with several seeds, and timing), see [reproducibility/README.md](reproducibility/README.md).

See [docs/CODE_GUIDE.md](docs/CODE_GUIDE.md) for details on each script.
