#!/usr/bin/env python3
"""Extract the formal tumor-only cohort and the union of stable panel probes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--out-matrix", required=True)
    parser.add_argument("--out-probes", required=True)
    parser.add_argument("--chunk-rows", type=int, default=128)
    args = parser.parse_args()

    matrix = np.load(args.matrix, mmap_mode="r")
    samples = pd.read_csv(args.samples, sep="\t")
    if "tissue_type" not in samples or "project_id" not in samples:
        raise ValueError("samples manifest must contain tissue_type and project_id")
    tumor_mask = samples["tissue_type"].astype(str).str.lower().eq("tumor").to_numpy()
    if int(tumor_mask.sum()) != 9065:
        raise ValueError(f"Expected 9065 tumor rows, found {int(tumor_mask.sum())}")
    if matrix.shape[0] != len(samples):
        raise ValueError(f"Matrix rows ({matrix.shape[0]}) do not match samples ({len(samples)})")

    probe_ids = pd.read_csv(args.out_probes, sep="\t") if Path(args.out_probes).exists() else None
    panel = pd.read_csv(args.panel, sep="\t")
    panel_sizes = [100, 200, 500, 1000]
    selected = []
    for size in panel_sizes:
        selected.extend(panel.loc[panel["panel_size"].eq(size), "probe_id"].astype(str).tolist())
    selected = list(dict.fromkeys(selected))
    all_probe_ids = pd.read_csv(Path(args.matrix).with_name("probe_ids.tsv"), sep="\t")["probe_id"].astype(str)
    probe_to_idx = {probe: i for i, probe in enumerate(all_probe_ids)}
    missing = [probe for probe in selected if probe not in probe_to_idx]
    if missing:
        raise ValueError(f"{len(missing)} panel probes are absent from the matrix")
    col_idx = np.asarray([probe_to_idx[probe] for probe in selected], dtype=np.int64)

    out_matrix = Path(args.out_matrix)
    out_matrix.parent.mkdir(parents=True, exist_ok=True)
    output = np.lib.format.open_memmap(out_matrix, mode="w+", dtype=np.float32, shape=(int(tumor_mask.sum()), len(selected)))
    tumor_rows = np.flatnonzero(tumor_mask)
    for start in range(0, len(tumor_rows), args.chunk_rows):
        rows = tumor_rows[start : start + args.chunk_rows]
        output[start : start + len(rows)] = np.asarray(matrix[rows, :][:, col_idx], dtype=np.float32)
        if start == 0 or start % (args.chunk_rows * 10) == 0:
            print(f"extracted {start + len(rows)}/{len(tumor_rows)} rows", flush=True)
    output.flush()
    pd.DataFrame({"probe_id": selected}).to_csv(args.out_probes, sep="\t", index=False)
    locked = samples.loc[tumor_mask].reset_index(drop=True)
    locked.to_csv(out_matrix.with_name("samples_locked.tsv"), sep="\t", index=False)
    print(f"wrote {out_matrix} with shape {output.shape}")


if __name__ == "__main__":
    main()
