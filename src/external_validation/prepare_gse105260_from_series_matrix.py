from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GSE105260 official GEO series-matrix beta values.")
    parser.add_argument("--series-matrix", required=True, help="GSE105260_series_matrix.txt.gz")
    parser.add_argument("--train-probe-ids", required=True, help="TCGA candidate probe_ids.tsv")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def load_train_probe_ids(path: Path) -> list[str]:
    df = pd.read_csv(path, sep="\t")
    if "probe_id" not in df.columns:
        raise ValueError(f"{path} must contain probe_id")
    return df["probe_id"].astype(str).tolist()


def parse_metadata(path: Path) -> tuple[list[str], list[str]]:
    sample_ids: list[str] = []
    titles: list[str] = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [item.strip().strip('"') for item in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_title"):
                titles = [item.strip().strip('"') for item in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!series_matrix_table_begin"):
                break
    if not sample_ids or not titles or len(sample_ids) != len(titles):
        raise ValueError("Could not parse complete Sample_geo_accession/Sample_title metadata from series matrix")
    return sample_ids, titles


def classify_title(title: str) -> tuple[str, str, str, str]:
    text = title.lower()
    if "normal renal" in text:
        return "", "", "0", "normal_renal"
    if "primary ccrcc" in text:
        return "TCGA-KIRC", "TCGA-KIRC", "1", "primary_ccRCC"
    if "metastasis ccrcc" in text or "metastatic ccrcc" in text:
        return "TCGA-KIRC", "TCGA-KIRC", "1", "metastatic_ccRCC"
    return "", "", "0", "unknown"


def read_series_beta(path: Path, train_probe_ids: list[str]) -> pd.DataFrame:
    train_probe_set = set(train_probe_ids)
    rows = []
    header: list[str] | None = None
    in_table = False
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if not in_table:
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            parts = [item.strip().strip('"') for item in line.rstrip("\n").split("\t")]
            if not parts:
                continue
            if parts[0] == "ID_REF":
                header = ["probe_id", *parts[1:]]
                continue
            if header is None:
                raise ValueError("Series matrix data encountered before ID_REF header")
            probe = parts[0]
            if probe in train_probe_set:
                rows.append(parts)
    if header is None:
        raise ValueError("No ID_REF header found in series matrix table")
    if not rows:
        raise ValueError("No overlapping probes found in series matrix")
    beta = pd.DataFrame(rows, columns=header)
    beta["probe_id"] = beta["probe_id"].astype(str)
    beta = beta.drop_duplicates("probe_id", keep="first").set_index("probe_id")
    beta = beta.apply(pd.to_numeric, errors="coerce")
    return beta


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    series_path = Path(args.series_matrix)
    train_probe_ids = load_train_probe_ids(Path(args.train_probe_ids))
    sample_ids, titles = parse_metadata(series_path)
    beta = read_series_beta(series_path, train_probe_ids)
    beta = beta[[sample for sample in sample_ids if sample in beta.columns]]

    sample_rows = []
    for sample_id, title in zip(sample_ids, titles):
        label, accepted, tumor_flag, sample_group = classify_title(title)
        if not label:
            continue
        sample_rows.append(
            {
                "sample_id": sample_id,
                "label": label,
                "accepted_labels": accepted,
                "histology": "ccRCC",
                "sample_group": sample_group,
                "geo_title": title,
                "tumor_flag": int(tumor_flag),
            }
        )
    samples = pd.DataFrame(sample_rows)
    if samples.empty:
        raise ValueError("No tumor samples retained from GSE105260 metadata")

    ordered_probe_ids = [probe for probe in train_probe_ids if probe in beta.index]
    x_df = beta.loc[ordered_probe_ids, samples["sample_id"].tolist()].T
    x = x_df.to_numpy(dtype=np.float32, copy=True)

    samples.insert(0, "external_index", np.arange(len(samples), dtype=int))
    samples["platform"] = "GPL13534"
    samples["cohort"] = "GSE105260"
    samples = samples[
        [
            "external_index",
            "sample_id",
            "label",
            "accepted_labels",
            "histology",
            "sample_group",
            "geo_title",
            "tumor_flag",
            "platform",
            "cohort",
        ]
    ]

    np.save(out_dir / "X_external.npy", x)
    samples.to_csv(out_dir / "samples_external.tsv", sep="\t", index=False)
    pd.DataFrame({"probe_id": ordered_probe_ids}).to_csv(out_dir / "probe_ids.tsv", sep="\t", index=False)

    manifest = {
        "cohort": "GSE105260",
        "source": "official_geo_series_matrix",
        "series_matrix": str(series_path.resolve()),
        "n_series_samples": len(sample_ids),
        "n_kept_samples": int(samples.shape[0]),
        "label_counts": samples["label"].value_counts().to_dict(),
        "sample_group_counts": samples["sample_group"].value_counts().to_dict(),
        "evaluation_mode": "kirc_exact_from_official_processed_beta",
        "n_train_probes": len(train_probe_ids),
        "n_external_probes": len(ordered_probe_ids),
        "probe_overlap_ratio_train": float(len(ordered_probe_ids) / max(len(train_probe_ids), 1)),
        "matrix_shape": list(x.shape),
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"event": "done", **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
