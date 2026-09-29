from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build padded CpG-gene-pathway index maps for the Q1 hierarchical model.")
    parser.add_argument("--probe-ids", required=True, help="TSV with column probe_id, or NPY selected_probe_ids.")
    parser.add_argument("--probe-annotation-tsv", required=True, help="Columns: probe_id,gene_symbol.")
    parser.add_argument("--pathway-tsv", required=True, help="Columns: pathway,gene_symbol.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-cpg-per-gene", type=int, default=64)
    parser.add_argument("--max-genes-per-pathway", type=int, default=128)
    parser.add_argument("--min-cpg-per-gene", type=int, default=1)
    parser.add_argument("--min-genes-per-pathway", type=int, default=3)
    return parser.parse_args()


def load_probe_ids(path: Path) -> list[str]:
    if path.suffix.lower() == ".npy":
        return [str(x) for x in np.load(path, allow_pickle=True).tolist()]
    df = pd.read_csv(path, sep="\t")
    if "probe_id" not in df.columns:
        raise ValueError("probe id TSV must contain column: probe_id")
    return df["probe_id"].astype(str).tolist()


def pad_rows(rows: list[list[int]], width: int) -> np.ndarray:
    out = np.full((len(rows), width), -1, dtype=np.int64)
    for i, row in enumerate(rows):
        clipped = row[:width]
        out[i, : len(clipped)] = clipped
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_ids = load_probe_ids(Path(args.probe_ids))
    probe_to_idx = {probe: i for i, probe in enumerate(probe_ids)}
    annot = pd.read_csv(args.probe_annotation_tsv, sep="\t")
    pathways = pd.read_csv(args.pathway_tsv, sep="\t")
    for col in ("probe_id", "gene_symbol"):
        if col not in annot.columns:
            raise ValueError(f"probe annotation missing column: {col}")
    for col in ("pathway", "gene_symbol"):
        if col not in pathways.columns:
            raise ValueError(f"pathway TSV missing column: {col}")

    gene_to_cpg: dict[str, list[int]] = defaultdict(list)
    for row in annot.itertuples(index=False):
        probe = str(getattr(row, "probe_id"))
        gene = str(getattr(row, "gene_symbol")).strip()
        if not gene or gene.lower() == "nan" or probe not in probe_to_idx:
            continue
        gene_to_cpg[gene].append(probe_to_idx[probe])
    genes = sorted(g for g, idxs in gene_to_cpg.items() if len(idxs) >= args.min_cpg_per_gene)
    gene_to_idx = {gene: i for i, gene in enumerate(genes)}
    gene_rows = [gene_to_cpg[gene] for gene in genes]

    pathway_to_genes: dict[str, list[int]] = defaultdict(list)
    for row in pathways.itertuples(index=False):
        pathway = str(getattr(row, "pathway")).strip()
        gene = str(getattr(row, "gene_symbol")).strip()
        if pathway and gene in gene_to_idx:
            pathway_to_genes[pathway].append(gene_to_idx[gene])
    pathway_names = sorted(p for p, idxs in pathway_to_genes.items() if len(set(idxs)) >= args.min_genes_per_pathway)
    pathway_rows = [sorted(set(pathway_to_genes[pathway])) for pathway in pathway_names]

    gene_cpg_index = pad_rows(gene_rows, args.max_cpg_per_gene)
    pathway_gene_index = pad_rows(pathway_rows, args.max_genes_per_pathway)
    np.save(out_dir / "gene_cpg_index.npy", gene_cpg_index)
    np.save(out_dir / "pathway_gene_index.npy", pathway_gene_index)
    pd.DataFrame({"gene_index": range(len(genes)), "gene_symbol": genes}).to_csv(out_dir / "gene_mapping.tsv", sep="\t", index=False)
    pd.DataFrame({"pathway_index": range(len(pathway_names)), "pathway": pathway_names}).to_csv(out_dir / "pathway_mapping.tsv", sep="\t", index=False)
    summary = {
        "n_input_probes": len(probe_ids),
        "n_genes": len(genes),
        "n_pathways": len(pathway_names),
        "max_cpg_per_gene": args.max_cpg_per_gene,
        "max_genes_per_pathway": args.max_genes_per_pathway,
    }
    (out_dir / "index_map_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
