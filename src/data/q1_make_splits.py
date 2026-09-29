from __future__ import annotations

import argparse
from pathlib import Path

from q1_split_utils import write_split_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create locked repeated patient-level CV splits for Q1 experiments.")
    parser.add_argument("--samples-tsv", required=True, help="Original samples_used.tsv aligned to the matrix rows.")
    parser.add_argument("--out-dir", required=True, help="Output directory for locked split artifacts.")
    parser.add_argument("--label-col", default="project_id")
    parser.add_argument("--tissue-type", default="tumor")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_split_artifacts(
        samples_tsv=Path(args.samples_tsv),
        out_dir=Path(args.out_dir),
        label_col=args.label_col,
        tissue_type=args.tissue_type,
        n_splits=args.n_splits,
        repeats=args.repeats,
        seed=args.seed,
    )
    print(f"Saved Q1 split artifacts to: {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
