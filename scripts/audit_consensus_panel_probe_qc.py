#!/usr/bin/env python3
"""Summarise manifest-level QC flags for the locked 500/1,000-CpG panels.

This is a descriptive annotation audit. It does not refit a model, exclude
probes, or infer cross-reactive/multi-mapping status without a versioned list.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_panels(path: Path) -> dict[int, list[dict[str, str]]]:
    panels: dict[int, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            size = int(row["panel_size"])
            if size in {500, 1000}:
                panels[size].append(row)
    for size in (500, 1000):
        rows = sorted(panels[size], key=lambda row: int(row["panel_rank"]))
        if len(rows) != size:
            raise ValueError(f"Expected {size} rows for panel {size}, found {len(rows)}")
        if len({row['probe_id'] for row in rows}) != size:
            raise ValueError(f"Duplicate probe IDs in panel {size}")
        panels[size] = rows
    return dict(panels)


def read_manifest_rows(path: Path, wanted: set[str]) -> dict[str, dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for line in handle:
            if line.startswith("IlmnID,"):
                fieldnames = next(csv.reader([line]))
                break
        else:
            raise ValueError("Illumina manifest assay header was not found")

        rows: dict[str, dict[str, str]] = {}
        for row in csv.DictReader(handle, fieldnames=fieldnames):
            probe_id = (row.get("IlmnID") or "").strip()
            if probe_id in wanted:
                rows[probe_id] = row
    missing = wanted.difference(rows)
    if missing:
        preview = ", ".join(sorted(missing)[:10])
        raise ValueError(f"Manifest annotations missing for {len(missing)} probes: {preview}")
    return rows


def normalise_chromosome(value: str) -> str:
    chromosome = value.strip().upper()
    return chromosome[3:] if chromosome.startswith("CHR") else chromosome


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panels",
        type=Path,
        default=Path(
            "server_results_20260728/results/internal_cv/"
            "lr_cpg_panel_frozen_20260721/consensus_locked_panels.tsv"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/annotations/"
            "GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/manuscript_qc"),
    )
    args = parser.parse_args()

    panels = read_panels(args.panels)
    wanted = {row["probe_id"] for rows in panels.values() for row in rows}
    annotations = read_manifest_rows(args.manifest, wanted)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    detail_path = args.out_dir / "consensus_panel_probe_qc_detail.tsv"
    detail_fields = [
        "panel_size",
        "panel_rank",
        "probe_id",
        "chromosome",
        "is_chr_x",
        "is_chr_y",
        "probe_snps",
        "probe_snps_10",
        "manifest_any_snp_flag",
    ]
    summary_rows: list[dict[str, object]] = []

    with detail_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields, delimiter="\t")
        writer.writeheader()
        for size in (500, 1000):
            chr_x = chr_y = probe_snps = probe_snps_10 = any_snp = 0
            for panel_row in panels[size]:
                annotation = annotations[panel_row["probe_id"]]
                chromosome = normalise_chromosome(annotation.get("CHR", ""))
                snps = (annotation.get("Probe_SNPs") or "").strip()
                snps_10 = (annotation.get("Probe_SNPs_10") or "").strip()
                is_x = chromosome == "X"
                is_y = chromosome == "Y"
                has_snp = bool(snps or snps_10)
                chr_x += int(is_x)
                chr_y += int(is_y)
                probe_snps += int(bool(snps))
                probe_snps_10 += int(bool(snps_10))
                any_snp += int(has_snp)
                writer.writerow(
                    {
                        "panel_size": size,
                        "panel_rank": panel_row["panel_rank"],
                        "probe_id": panel_row["probe_id"],
                        "chromosome": chromosome,
                        "is_chr_x": int(is_x),
                        "is_chr_y": int(is_y),
                        "probe_snps": snps,
                        "probe_snps_10": snps_10,
                        "manifest_any_snp_flag": int(has_snp),
                    }
                )
            summary_rows.append(
                {
                    "panel_size": size,
                    "total_probes": size,
                    "manifest_matched": size,
                    "chr_x": chr_x,
                    "chr_y": chr_y,
                    "sex_chromosome_total": chr_x + chr_y,
                    "sex_chromosome_fraction": f"{(chr_x + chr_y) / size:.6f}",
                    "probe_snps_nonempty": probe_snps,
                    "probe_snps_10_nonempty": probe_snps_10,
                    "manifest_any_snp_flag": any_snp,
                    "manifest_any_snp_fraction": f"{any_snp / size:.6f}",
                    "cross_reactive_multimapping_status": "not_assessed_no_versioned_list",
                }
            )

    summary_path = args.out_dir / "consensus_panel_probe_qc_summary.tsv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)

    run_manifest = {
        "analysis": "locked consensus-panel manifest QC",
        "scope": "descriptive annotation only; no model refitting or probe exclusion",
        "inputs": {
            "panels": str(args.panels),
            "panels_sha256": sha256(args.panels),
            "illumina_manifest": str(args.manifest),
            "illumina_manifest_sha256": sha256(args.manifest),
        },
        "outputs": [str(summary_path), str(detail_path)],
        "cross_reactive_multimapping_status": (
            "not assessed because no authoritative versioned list was supplied "
            "with the locked analysis inputs"
        ),
    }
    manifest_path = args.out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    print(summary_path)
    print(detail_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
