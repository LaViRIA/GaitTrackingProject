#!/usr/bin/env bash
# Reentrena todas las redes del artículo con la configuración documentada en
# docs/REPRODUCIBILIDAD.md (misma semilla y misma división de sujetos).
#
# Uso (desde la raíz del repositorio):
#   ./train_all.sh                 # entrena las 5 redes en orden
#   ./train_all.sh tcn ttl         # solo las redes indicadas
#
# Redes: pivot, static, tcn, ttl, trans
# ⚠ Sobrescribe los checkpoints de nn-results/<red>/checkpoints/. Haz backup antes.

set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
SEED="${SEED:-42}"
SPLIT="${SPLIT:-data_split.json}"
DATASET="${DATASET:-$HOME/Documents/DataSet}"
LOG_DIR="${LOG_DIR:-nn-results/train_logs}"

COMMON=(--seed "$SEED" --data-split "$SPLIT" --dataset "$DATASET")

declare -A CMD
CMD[pivot]="models.nn_pivot_classifier --epochs 100 --batch-size 1024 --lr 1e-3 --patience 25"
CMD[static]="models.nn_static_pivot --epochs 300 --batch-size 128 --lr 1e-3 --patience 30"
CMD[tcn]="models.nn_tcn_lite --window-size 25 --window-stride 2 --batch-size 12 --lr 5e-4 --epochs 500 --patience 60"
CMD[ttl]="models.nn_ttl_lite --window-size 25 --window-stride 3 --batch-size 24 --lr 5e-4 --epochs 300 --patience 60"
CMD[trans]="models.nn_transformer_multitask --window-size 25 --window-stride 3 --batch-size 24 --lr 5e-4 --epochs 300 --patience 60"

NETS=("$@")
[ ${#NETS[@]} -eq 0 ] && NETS=(pivot static tcn ttl trans)

mkdir -p "$LOG_DIR"
for net in "${NETS[@]}"; do
    [ -n "${CMD[$net]:-}" ] || { echo "Red desconocida: $net (usa: pivot static tcn ttl trans)"; exit 1; }
    echo "▶ $net: $PYTHON -m ${CMD[$net]} ${COMMON[*]}"
    # shellcheck disable=SC2086
    "$PYTHON" -m ${CMD[$net]} "${COMMON[@]}" > "$LOG_DIR/$net.log" 2>&1
    echo "  ✓ terminado (log: $LOG_DIR/$net.log)"
done
