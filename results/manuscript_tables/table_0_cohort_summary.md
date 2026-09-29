| Cohort | Role | Samples | Classes / labels | Evaluation | Use in manuscript |
| --- | --- | --- | --- | --- | --- |
| TCGA candidate manifest | Source audit | 9812 | 33 TCGA classes; 747 normal rows | Normal rows excluded before formal training | Provenance audit only |
| TCGA 450k | Model development and internal validation | 9065 | 33 TCGA cancer classes | Patient-level stratified 5-fold x 3 repeated CV | Primary training and internal validation cohort |
| GSE56044 | Formal external validation | 106 | TCGA-LUAD: 83; TCGA-LUSC: 23 | Exact LUAD/LUSC | independent lung cancer cohort |
| GSE48684 | Formal external validation | 64 | TCGA-COADREAD: 64 | COAD/READ lineage | independent colorectal cancer cohort |
| GSE53051 | Formal external validation | 102 | TCGA-THCA: 38; TCGA-COADREAD: 23; TCGA-PAAD: 18; TCGA-BRCA: 14; TCGA-LUNG: 9 | Multi-tissue lineage | multi-tissue external cohort |
| GSE105260 official | Formal external validation | 35 | TCGA-KIRC: 35 | Exact KIRC from official beta | official minfi/SWAN processed ccRCC cohort |
