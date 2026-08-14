#!/usr/bin/env bash
set -e

SMILES_DIR="$1"
PRED_DIR="$2"
PROCS="$3"
MOLS_TO_DOCK="${4:-all_mol}"
CONDA_ENV="$5"     # e.g. deep-docking
ITER_DIR="$6"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# make conda available in this non-interactive bash
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
else
    echo "conda not found on PATH. Make sure conda is installed and on PATH."
    exit 1
fi

OUT_DIR="${ITER_DIR%/}/final_extraction"
mkdir -p "$OUT_DIR"

if [ "$MOLS_TO_DOCK" = "all_mol" ]; then
  python "$SCRIPT_DIR/final_extraction.py" \
    -smile_dir "$SMILES_DIR" \
    -prediction_dir "$PRED_DIR" \
    -processors "$PROCS" \
    -output_dir "$OUT_DIR"
else
  python "$SCRIPT_DIR/final_extraction.py" \
    -smile_dir "$SMILES_DIR" \
    -prediction_dir "$PRED_DIR" \
    -processors "$PROCS" \
    -mols_to_dock "$MOLS_TO_DOCK" \
    -output_dir "$OUT_DIR"
fi
