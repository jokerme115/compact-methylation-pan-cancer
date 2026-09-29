from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_SAMPLE_COLUMNS = {"sample_id", "label", "platform"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a curated external methylation cohort contract.")
    parser.add_argument("--external-dir", required=True)
    parser.add_argument("--train-probe-ids", required=True, help="Training probe IDs TSV/NPY for alignment audit.")
    parser.add_argument("--out-json", default="")
    return parser.parse_args()


def load_probe_ids(path: Path) -> list[str]:
    if path.suffix.lower() == ".npy":
        return [str(x) for x in np.load(path, allow_pickle=True).tolist()]
    df = pd.read_csv(path, sep="\t")
    if "probe_id" not in df.columns:
        raise ValueError(f"{path} must contain column probe_id")
    return df["probe_id"].astype(str).tolist()


def main() -> None:
    args = parse_args()
    external_dir = Path(args.external_dir)
    matrix_path = external_dir / "X_external.npy"
    samples_path = external_dir / "samples_external.tsv"
    probes_path = external_dir / "probe_ids.tsv"
    manifest_path = external_dir / "dataset_manifest.json"

    missing = [str(p) for p in (matrix_path, samples_path, probes_path, manifest_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing external contract files: {missing}")

    x = np.load(matrix_path, mmap_mode="r")
    samples = pd.read_csv(samples_path, sep="\t")
    external_probes = load_probe_ids(probes_path)
    train_probes = load_probe_ids(Path(args.train_probe_ids))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    missing_cols = sorted(REQUIRED_SAMPLE_COLUMNS - set(samples.columns))
    if missing_cols:
        raise ValueError(f"samples_external.tsv missing columns: {missing_cols}")
    if x.ndim != 2:
        raise ValueError("X_external.npy must be a 2D matrix")
    if x.shape[0] != len(samples):
        raise ValueError(f"sample count mismatch: matrix rows={x.shape[0]}, samples={len(samples)}")
    if x.shape[1] != len(external_probes):
        raise ValueError(f"probe count mismatch: matrix cols={x.shape[1]}, probe_ids={len(external_probes)}")

    train_probe_set = set(train_probes)
    overlap = sum(1 for probe in external_probes if probe in train_probe_set)
    aligned_prefix = external_probes[: min(len(external_probes), len(train_probes))] == train_probes[: min(len(external_probes), len(train_probes))]
    report = {
        "external_dir": str(external_dir.resolve()),
        "matrix_shape": [int(x.shape[0]), int(x.shape[1])],
        "n_labels": int(samples["label"].nunique()),
        "label_counts": {str(k): int(v) for k, v in samples["label"].value_counts().sort_index().to_dict().items()},
        "platform_counts": {str(k): int(v) for k, v in samples["platform"].value_counts().sort_index().to_dict().items()},
        "training_probe_count": len(train_probes),
        "external_probe_count": len(external_probes),
        "probe_overlap_count": int(overlap),
        "probe_overlap_ratio_external": float(overlap / max(len(external_probes), 1)),
        "aligned_prefix": bool(aligned_prefix),
        "manifest": manifest,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
