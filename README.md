# Compact DNA methylation panels for patient-level pan-cancer lineage classification

This repository contains analysis code and selected non-restricted reproducibility materials for the manuscript by Tao He, Hao Zhang, Haixia Long, Fei Zhou and Xia Yu. The analysis assets were assembled from the local study repository at commit `8e3eeb4`; the author metadata was updated on 2026-09-30.

## Contents

| Directory | Contents |
| --- | --- |
| `src/` | Data preparation, patient grouping, feature selection, model fitting, evaluation and external cohort code |
| `scripts/` | Analysis, statistics, sensitivity analysis and manuscript figure scripts |
| `data/tcga_450k/` | Locked sample and fold manifests, probe identifiers and input metadata; no methylation matrix |
| `frozen_models/` | 500- and 1,000-CpG locked panel model specification, including class order, preprocessing parameters, coefficients and intercepts |
| `server_results_20260728/results/internal_cv/lr_cpg_panel_frozen_20260721/` | Frozen panel lists and compact patient-level performance summaries |
| `results/` | Selected manuscript tables, statistics and descriptive probe annotations |
| `environment/` | Recorded Python version and broad dependency list; package versions are not fully locked |

The frozen model specification is derived from the GSE56044 evaluation manifest. Local server paths were removed; its model parameters and panel order were retained. The same locked development models were applied to the independent 450K cohorts without refitting.

## Data and reproduction

The TCGA methylation profiles must be obtained from the NCI Genomic Data Commons. External datasets are available from GEO under GSE56044, GSE53051, GSE48684, GSE105260, GSE121377, GSE133556, GSE136380, GSE144487, GSE148766 and GSE164269. Source matrices, normal-tissue profiles and other raw data are not redistributed here. Use the locked patient/fold manifest when reproducing patient-level estimates; do not create a new random split.

The `file_path` fields in the two sample manifests were normalized to `data/raw_tcga_450k/<file_id>/<file_name>` for this public package. They preserve row order and sample identifiers; the source data are absent and `exists_local` is set to `0`. Supply your own local raw-data root when reconstructing the candidate matrix.

The scripts were developed for Python 3.12.6. `environment/requirements.txt` records the top-level dependencies but does not lock versions. Several figure and sensitivity scripts require intermediate result files beyond this compact code release. This repository is therefore a transparent code and parameter archive, not a one-command reproduction package. The manuscript and its Supplementary Information define the evaluation endpoints and statistical interpretation.

The 500-CpG panel is a compact candidate and the 1,000-CpG panel is a reference. The code and parameters are for retrospective research use; clinical assay validation and tumour-versus-normal discrimination remain open tasks.

## Citation and reuse

Please cite the accompanying manuscript. No reuse license has yet been assigned; contact the authors before redistributing or adapting the code.
