#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src/data:$PROJECT_ROOT/src/features:$PROJECT_ROOT/src/models:$PROJECT_ROOT/src/evaluation:$PROJECT_ROOT/src/interpretation:$PROJECT_ROOT/src/external_validation:${PYTHONPATH:-}"
if [[ -n "${PAPER_PYTHON:-}" ]]; then
  PYTHON_BIN="$PAPER_PYTHON"
elif [[ -x "$PROJECT_ROOT/../.venv_pathmethnet/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/../.venv_pathmethnet/bin/python"
else
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

"$PYTHON_BIN" "$PROJECT_ROOT/src/models/q1_train_ml_baselines.py" \
  --matrix-npy "$PROJECT_ROOT/data/tcga_450k/X_candidate.npy" \
  --splits-dir "$PROJECT_ROOT/data/tcga_450k" \
  --out-dir "$PROJECT_ROOT/results/internal_cv/model_artifacts/q1_candidate_baseline_rerun" \
  --models sgd_logloss,logreg,elasticnet \
  --feature-methods variance,effect_size \
  --top-k 100,500,1000 \
  --bootstrap 100 \
  --n-jobs 16 \
  --resume
