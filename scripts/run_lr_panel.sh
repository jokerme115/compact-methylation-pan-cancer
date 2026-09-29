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

"$PYTHON_BIN" "$PROJECT_ROOT/src/models/q1_lr_cpg_panel.py" \
  --matrix-npy "$PROJECT_ROOT/data/tcga_450k/X_candidate.npy" \
  --probe-ids "$PROJECT_ROOT/data/tcga_450k/probe_ids.tsv" \
  --splits-dir "$PROJECT_ROOT/data/tcga_450k" \
  --annotation-csv-gz "$PROJECT_ROOT/data/annotations/GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz" \
  --pathway-tsv "$PROJECT_ROOT/data/annotations/reactome_pathway_gene.tsv" \
  --out-dir "$PROJECT_ROOT/results/internal_cv/lr_cpg_panel_rerun" \
  --feature-method variance \
  --panel-sizes 10,20,50,100,200,500,1000 \
  --interpret-panel-size 100 \
  --class-top-n 20 \
  --n-jobs 8 \
  --resume
