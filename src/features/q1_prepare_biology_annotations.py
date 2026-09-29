from __future__ import annotations

import argparse
import gzip
import json
import zipfile
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare probe-gene and pathway-gene tables for PathMethNet.")
    parser.add_argument("--gpl13534-csv-gz", required=True)
    parser.add_argument("--reactome-gmt-zip", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def assay_header_line(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for i, line in enumerate(handle):
            if line.startswith("IlmnID,"):
                return i
    raise ValueError(f"Could not find GPL13534 assay header in {path}")


def prepare_probe_gene(path: Path, out_path: Path) -> int:
    header = assay_header_line(path)
    usecols = ["IlmnID", "UCSC_RefGene_Name"]
    df = pd.read_csv(path, compression="gzip", skiprows=header, usecols=usecols, dtype=str)
    df = df.rename(columns={"IlmnID": "probe_id", "UCSC_RefGene_Name": "gene_symbol"})
    rows: list[dict[str, str]] = []
    for row in df.itertuples(index=False):
        probe = str(row.probe_id).strip()
        genes = str(row.gene_symbol).strip()
        if not probe or not probe.startswith("cg") or not genes or genes.lower() == "nan":
            continue
        for gene in genes.split(";"):
            gene = gene.strip()
            if gene and gene.lower() != "nan":
                rows.append({"probe_id": probe, "gene_symbol": gene})
    out = pd.DataFrame(rows).drop_duplicates()
    out.to_csv(out_path, sep="\t", index=False)
    return int(len(out))


def prepare_reactome(path: Path, out_path: Path) -> int:
    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".gmt")]
        if not names:
            raise ValueError(f"No GMT file found in {path}")
        with archive.open(names[0]) as handle:
            for raw in handle:
                parts = raw.decode("utf-8", errors="replace").rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                pathway = parts[0].strip()
                for gene in parts[2:]:
                    gene = gene.strip()
                    if gene:
                        rows.append({"pathway": pathway, "gene_symbol": gene})
    out = pd.DataFrame(rows).drop_duplicates()
    out.to_csv(out_path, sep="\t", index=False)
    return int(len(out))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_gene = out_dir / "illumina450k_probe_gene.tsv"
    reactome = out_dir / "reactome_pathway_gene.tsv"
    summary = {
        "probe_gene_rows": prepare_probe_gene(Path(args.gpl13534_csv_gz), probe_gene),
        "pathway_gene_rows": prepare_reactome(Path(args.reactome_gmt_zip), reactome),
        "probe_gene_tsv": str(probe_gene.resolve()),
        "pathway_gene_tsv": str(reactome.resolve()),
    }
    (out_dir / "biology_annotation_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
