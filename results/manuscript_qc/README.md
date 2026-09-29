# Locked consensus-panel manifest QC

This directory contains a descriptive annotation audit of the locked 500- and
1,000-CpG consensus panels. It does not refit a model or exclude any probe.

Inputs:

- `server_results_20260728/results/internal_cv/lr_cpg_panel_frozen_20260721/consensus_locked_panels.tsv`
- `data/annotations/GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz`

Outputs:

- `consensus_panel_probe_qc_summary.tsv`: panel-level counts.
- `consensus_panel_probe_qc_detail.tsv`: probe-level chromosome and Illumina-manifest SNP annotations.
- `run_manifest.json`: input hashes and scope statement.

Interpretation boundary: `Probe_SNPs` and `Probe_SNPs_10` are manifest
annotations, not evidence that a probe biased the classifier. A curated,
versioned cross-reactive or multi-mapping list was not part of the locked input
set, so that status is reported as not assessed rather than inferred.
