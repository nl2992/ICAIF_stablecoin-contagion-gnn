#!/usr/bin/env bash
# Reproduce every GNN result and figure end-to-end.
# Requires the shared venv at ../.venv (numpy<2, torch CPU, torch_geometric, xgboost).
# The OpenMP guard is REQUIRED: xgboost + torch in one process segfault without it.
set -euo pipefail
cd "$(dirname "$0")"
PY="../.venv/bin/python"
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1

echo "[1/7] build real dataset (Binance + Coinbase 1-min)"
$PY scripts/build_real_dataset.py

echo "[2/7] model ladder + LOCO, all horizons + primary horizon"
$PY eval/run_benchmark.py --all-horizons
$PY eval/run_benchmark.py --horizon 1440

echo "[3/7] multi-seed robustness + lift/robust verdict + graph ablation"
$PY eval/robustness_multiseed.py
$PY eval/polish_results.py
$PY eval/ablation_graph.py

echo "[4/7] interpretability + hub export (GAT, 24h) for every analysable episode"
for ep in USDC_SVB UST_Terra USDT_May2022 DAI_FTX BUSD_winddown; do
  $PY interpret/run_interpret.py --episode "$ep" --horizon 1440 --kind gat || true
done

echo "[5/7] calibration-target export for the ABM"
$PY scripts/export_calibration.py

echo "[6/7] figures"
$PY scripts/generate_figures.py

echo "[7/7] tests"
$PY -m pytest tests/ -q -p no:warnings

echo "Done. Results in results/ and exports/."
