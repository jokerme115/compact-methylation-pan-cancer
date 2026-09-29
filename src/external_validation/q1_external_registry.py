from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_ROWS = [
    {
        "source": "GEO",
        "accession": "to_curate",
        "platform": "Illumina HumanMethylation450",
        "priority": "P0",
        "intended_use": "TCGA-trained external validation",
        "required_fields": "probe_id,beta_value,cancer_label,sample_id,platform",
        "status": "candidate_search_required",
        "notes": "Prioritize multi-cancer tissue-of-origin cohorts or multiple single-cancer 450k cohorts.",
    },
    {
        "source": "ICGC",
        "accession": "to_curate",
        "platform": "450k or compatible common probes",
        "priority": "P1",
        "intended_use": "reverse validation and platform robustness",
        "required_fields": "probe_id,beta_value,cancer_label,sample_id,platform",
        "status": "candidate_search_required",
        "notes": "Use only cohorts with labels mappable to TCGA project_id or curated core labels.",
    },
    {
        "source": "ArrayExpress/EWAS DataHub/CMAtlas",
        "accession": "to_curate",
        "platform": "450k/EPIC",
        "priority": "P1",
        "intended_use": "cross-platform validation",
        "required_fields": "probe_id,beta_value,cancer_label,sample_id,platform",
        "status": "candidate_search_required",
        "notes": "Treat EPIC as common-probe validation unless harmonized raw processing is available.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an external validation registry template.")
    parser.add_argument("--out-tsv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(DEFAULT_ROWS).to_csv(out_path, sep="\t", index=False)
    print(f"Saved external registry template: {out_path.resolve()}")


if __name__ == "__main__":
    main()
