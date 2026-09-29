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

export PATH="/opt/dtk-24.04.2/bin:/opt/dtk/bin:/opt/dtk-26.04/bin:${PATH:-}"
export LD_LIBRARY_PATH="/opt/dtk-24.04.2/lib:/opt/dtk-24.04.2/llvm/lib:/opt/dtk-24.04.2/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk-26.04/lib:/opt/dtk-26.04/.hyhal/rocm_smi/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

"$PYTHON_BIN" "$PROJECT_ROOT/src/models/q1_train_pathmethnet.py" \
  --matrix-npy "$PROJECT_ROOT/data/tcga_450k/X_candidate.npy" \
  --probe-ids "$PROJECT_ROOT/data/tcga_450k/probe_ids.tsv" \
  --splits-dir "$PROJECT_ROOT/data/tcga_450k" \
  --probe-annotation-tsv "$PROJECT_ROOT/data/annotations/illumina450k_probe_gene.tsv" \
  --pathway-tsv "$PROJECT_ROOT/data/annotations/reactome_pathway_gene.tsv" \
  --out-dir "$PROJECT_ROOT/results/internal_cv/model_artifacts/q1_pathmethnet_v2_rerun" \
  --feature-method variance \
  --top-k 1000 \
  --epochs 100 \
  --batch-size 128 \
  --hidden-dim 64 \
  --dropout 0.25 \
  --patience 15 \
  --device cuda \
  --resume
