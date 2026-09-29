from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare full TCGA 450k multiclass dataset metadata without training.")
    parser.add_argument("--matrix-npy", required=True, help="Path to X_topk.npy")
    parser.add_argument("--samples-tsv", required=True, help="Path to samples_used.tsv")
    parser.add_argument("--out-dir", required=True, help="Directory to store preparation artifacts")
    parser.add_argument(
        "--label-col",
        default="project_id",
        help="Label column for downstream task. Defaults to project_id.",
    )
    parser.add_argument(
        "--tissue-type",
        default="tumor",
        help="Tissue type subset to keep. Use empty string to keep all rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = np.load(args.matrix_npy, mmap_mode="r")
    sample_df = pd.read_csv(args.samples_tsv, sep="\t")

    if len(sample_df) != x.shape[0]:
        raise ValueError(
            f"Row mismatch between matrix ({x.shape[0]}) and samples table ({len(sample_df)})."
        )
    if args.label_col not in sample_df.columns:
        raise ValueError(f"Label column not found: {args.label_col}")

    filtered_df = sample_df.copy()
    if args.tissue_type.strip():
        filtered_df = filtered_df[
            filtered_df["tissue_type"].fillna("").str.lower() == args.tissue_type.strip().lower()
        ].copy()

    if filtered_df.empty:
        raise ValueError("No samples left after filtering.")

    label_counts = filtered_df[args.label_col].value_counts().sort_index()
    label_mapping = pd.DataFrame(
        {
            "label_index": np.arange(len(label_counts), dtype=int),
            "label_name": label_counts.index.astype(str),
            "sample_count": label_counts.to_numpy(dtype=int),
        }
    )
    label_mapping.to_csv(out_dir / "label_mapping.tsv", sep="\t", index=False)

    filtered_df.reset_index(drop=True).to_csv(out_dir / "samples_for_training.tsv", sep="\t", index=False)

    summary = {
        "matrix_path": str(Path(args.matrix_npy).resolve()),
        "samples_path": str(Path(args.samples_tsv).resolve()),
        "label_column": args.label_col,
        "tissue_type_filter": args.tissue_type,
        "n_matrix_rows": int(x.shape[0]),
        "n_matrix_features": int(x.shape[1]),
        "n_samples_after_filter": int(len(filtered_df)),
        "n_classes": int(len(label_counts)),
        "class_distribution": {str(k): int(v) for k, v in label_counts.to_dict().items()},
    }
    (out_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {out_dir / 'label_mapping.tsv'}")
    print(f"Saved: {out_dir / 'samples_for_training.tsv'}")
    print(f"Saved: {out_dir / 'dataset_summary.json'}")


if __name__ == "__main__":
    main()
