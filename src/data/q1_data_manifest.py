from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a Q1 dataset manifest for local/remote reproducibility.")
    parser.add_argument("--matrix-npy", required=True)
    parser.add_argument("--samples-tsv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--selected-probes-npy", default="")
    parser.add_argument("--local-raw-root", default="")
    parser.add_argument("--version-name", default="v_top_target_q1")
    parser.add_argument("--label-col", default="project_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x = np.load(args.matrix_npy, mmap_mode="r")
    samples = pd.read_csv(args.samples_tsv, sep="\t")
    if len(samples) != x.shape[0]:
        raise ValueError(f"Matrix/sample mismatch: matrix={x.shape[0]}, samples={len(samples)}")

    probes: list[str] | None = None
    if args.selected_probes_npy:
        probes_raw = np.load(args.selected_probes_npy, allow_pickle=True)
        probes = [str(item) for item in probes_raw.tolist()]
        pd.DataFrame({"probe_id": probes}).to_csv(out_dir / "probe_ids.tsv", sep="\t", index=False)

    platform_counts = samples.get("platform", pd.Series(["unknown"] * len(samples))).value_counts().to_dict()
    tissue_counts = samples.get("tissue_type", pd.Series(["unknown"] * len(samples))).value_counts().to_dict()
    label_counts = samples[args.label_col].value_counts().sort_index().to_dict() if args.label_col in samples.columns else {}
    manifest = {
        "version_name": args.version_name,
        "local_raw_root": args.local_raw_root,
        "remote_matrix_npy": str(Path(args.matrix_npy).resolve()),
        "remote_samples_tsv": str(Path(args.samples_tsv).resolve()),
        "selected_probes_npy": str(Path(args.selected_probes_npy).resolve()) if args.selected_probes_npy else None,
        "matrix_shape": [int(x.shape[0]), int(x.shape[1])],
        "n_probe_ids": len(probes) if probes is not None else None,
        "label_col": args.label_col,
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "platform_counts": {str(k): int(v) for k, v in platform_counts.items()},
        "tissue_counts": {str(k): int(v) for k, v in tissue_counts.items()},
        "q1_constraints": [
            "Raw TCGA files are local-only; remote experiments must use synchronized matrices and metadata.",
            "Main paper results must use fold-internal feature selection to avoid leakage.",
            "External validation datasets must be versioned as separate manifests before model evaluation.",
        ],
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
