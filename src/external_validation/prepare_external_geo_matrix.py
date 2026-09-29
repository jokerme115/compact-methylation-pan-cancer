from __future__ import annotations

import argparse
import gzip
import html
import json
import re
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HISTOLOGY_TO_LABEL = {
    "AC": "TCGA-LUAD",
    "ADENO": "TCGA-LUAD",
    "ADENOCARCINOMA": "TCGA-LUAD",
    "SQCC": "TCGA-LUSC",
    "SQUAMOUS": "TCGA-LUSC",
    "SQUAMOUS CELL CARCINOMA": "TCGA-LUSC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare GEO 450k external beta matrix for TCGA-trained validation.")
    parser.add_argument("--cohort", required=True, choices=["GSE56044", "GSE48684", "GSE53051", "GSE69914", "GSE105260"])
    parser.add_argument("--raw-beta", help="GSE56044 methylation_raw.txt.gz")
    parser.add_argument("--sample-annotation", help="GSE56044 GEO_Annotations.xls.gz")
    parser.add_argument("--signal-files", nargs="*", help="GSE48684 signal intensity matrix files")
    parser.add_argument("--methylated-signals", help="GSE53051 methylated signal matrix")
    parser.add_argument("--unmethylated-signals", help="GSE53051 unmethylated signal matrix")
    parser.add_argument("--raw-tar", help="GSE69914/GSE105260 RAW tar file")
    parser.add_argument("--geo-html", help="Downloaded GEO accession HTML page for sample titles")
    parser.add_argument("--train-probe-ids", required=True, help="TCGA candidate probe_ids.tsv")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def normalize_sample_from_beta_column(col: str) -> str:
    sample = re.sub(r"\.AVG_Beta(\.\d+)?$", "", col)
    return sample.strip()


def normalize_histology(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return HISTOLOGY_TO_LABEL.get(text.upper(), "")


def read_gse56044_annotation(path: Path) -> pd.DataFrame:
    try:
        with gzip.open(path, "rb") as handle:
            xls = pd.ExcelFile(handle)
            sheet_name = "DiscoveryCohortAnnotations"
            if sheet_name not in xls.sheet_names:
                sheet_name = xls.sheet_names[-1]
            annot = pd.read_excel(xls, sheet_name=sheet_name)
    except ImportError:
        return pd.DataFrame(columns=["sample_id", "histology", "tumor_flag", "label", "accepted_labels"])
    annot.columns = [str(col).strip() for col in annot.columns]
    lower_map = {col.lower(): col for col in annot.columns}

    sample_col = lower_map.get("sampleid") or lower_map.get("sample id")
    histology_col = lower_map.get("histology")
    tumor_col = lower_map.get("tumor")
    if sample_col is None or histology_col is None:
        raise ValueError(f"Could not find SampleID/Histology columns in {path}; columns={annot.columns.tolist()}")

    out = pd.DataFrame(
        {
            "sample_id": annot[sample_col].astype(str).str.strip(),
            "histology": annot[histology_col].astype(str).str.strip(),
        }
    )
    out["tumor_flag"] = 1
    if tumor_col is not None:
        out["tumor_flag"] = pd.to_numeric(annot[tumor_col], errors="coerce").fillna(1).astype(int)
    out["label"] = out["histology"].map(normalize_histology)
    out["accepted_labels"] = out["label"]
    out = out[out["sample_id"].ne("")].drop_duplicates("sample_id", keep="first")
    return out


def load_train_probe_ids(path: Path) -> list[str]:
    df = pd.read_csv(path, sep="\t")
    if "probe_id" not in df.columns:
        raise ValueError(f"{path} must contain probe_id column")
    return df["probe_id"].astype(str).tolist()


def parse_geo_sample_titles(path: Path) -> pd.DataFrame:
    text = path.read_text(errors="ignore")
    rows = []
    pattern = re.compile(
        r">(?P<gsm>GSM\d+)</a></td>\s*<td[^>]*>(?P<title>.*?)</td>",
        flags=re.DOTALL,
    )
    for match in pattern.finditer(text):
        title = re.sub(r"<.*?>", "", match.group("title"))
        title = html.unescape(title).strip()
        if title:
            rows.append({"gsm": match.group("gsm"), "title": title})
    return pd.DataFrame(rows).drop_duplicates("gsm", keep="first")


def classify_title(title: str) -> tuple[str, str, str]:
    text = title.lower()
    if "normal" in text or "adenoma" in text or "hyperplastic" in text or "ipmn" in text:
        return "", "", "0"
    if "breast" in text and ("cancer" in text or "dcis" in text):
        return "TCGA-BRCA", "TCGA-BRCA", "1"
    if "colon" in text and ("cancer" in text or "met_" in text or "met " in text):
        return "TCGA-COADREAD", "TCGA-COAD;TCGA-READ", "1"
    if "crc" in text and "normal" not in text:
        return "TCGA-COADREAD", "TCGA-COAD;TCGA-READ", "1"
    if "lung" in text and "cancer" in text:
        return "TCGA-LUNG", "TCGA-LUAD;TCGA-LUSC", "1"
    if "pancreas" in text and ("cancer" in text or "lcc" in text):
        return "TCGA-PAAD", "TCGA-PAAD", "1"
    if "thyroid" in text and any(token in text for token in ["ptc", "fc", "fvptc", "hc"]):
        return "TCGA-THCA", "TCGA-THCA", "1"
    if "ccrcc" in text:
        return "TCGA-KIRC", "TCGA-KIRC", "1"
    return "", "", ""


def write_prepared_external(
    *,
    cohort: str,
    out_dir: Path,
    beta: pd.DataFrame,
    samples: pd.DataFrame,
    train_probe_ids: list[str],
    platform: str,
    sources: dict[str, Any],
    evaluation_mode: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    beta.index = beta.index.astype(str)
    beta = beta[~beta.index.duplicated(keep="first")]
    ordered_probe_ids = [probe for probe in train_probe_ids if probe in beta.index]
    if not ordered_probe_ids:
        raise ValueError(f"{cohort}: no overlap with TCGA training probes")

    samples = samples.copy()
    samples = samples[samples["sample_id"].isin(beta.columns)].copy()
    samples = samples[samples["label"].fillna("").ne("")].copy()
    if samples.empty:
        raise ValueError(f"{cohort}: no labeled tumor samples retained")
    samples = samples.drop_duplicates("sample_id", keep="first")
    x_df = beta.loc[ordered_probe_ids, samples["sample_id"].tolist()].T
    x = x_df.to_numpy(dtype=np.float32, copy=True)

    samples.insert(0, "external_index", np.arange(len(samples), dtype=int))
    samples["cohort"] = cohort
    samples["platform"] = platform
    samples["accepted_labels"] = samples["accepted_labels"].fillna(samples["label"])
    samples = samples[
        ["external_index", "sample_id", "label", "accepted_labels", "histology", "tumor_flag", "platform", "cohort"]
    ]

    np.save(out_dir / "X_external.npy", x)
    samples.to_csv(out_dir / "samples_external.tsv", sep="\t", index=False)
    pd.DataFrame({"probe_id": ordered_probe_ids}).to_csv(out_dir / "probe_ids.tsv", sep="\t", index=False)

    manifest = {
        "cohort": cohort,
        **sources,
        "n_unique_beta_samples": int(beta.shape[1]),
        "n_kept_samples": int(samples.shape[0]),
        "label_counts": samples["label"].value_counts().to_dict(),
        "evaluation_mode": evaluation_mode,
        "n_train_probes": len(train_probe_ids),
        "n_external_probes": len(ordered_probe_ids),
        "probe_overlap_ratio_train": float(len(ordered_probe_ids) / max(len(train_probe_ids), 1)),
        "matrix_shape": list(x.shape),
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"event": "done", **manifest}, ensure_ascii=False))


def prepare_gse56044(args: argparse.Namespace) -> None:
    if not args.raw_beta or not args.sample_annotation:
        raise ValueError("GSE56044 requires --raw-beta and --sample-annotation")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_probe_ids = load_train_probe_ids(Path(args.train_probe_ids))
    train_probe_set = set(train_probe_ids)
    annot = read_gse56044_annotation(Path(args.sample_annotation))

    header = pd.read_csv(args.raw_beta, sep="\t", nrows=0)
    beta_cols = [col for col in header.columns if re.search(r"\.AVG_Beta(\.\d+)?$", str(col))]
    if not beta_cols:
        raise ValueError("No .AVG_Beta columns found in raw beta file")
    usecols = ["TargetID", *beta_cols]
    beta = pd.read_csv(args.raw_beta, sep="\t", usecols=usecols, low_memory=False)
    beta = beta.rename(columns={"TargetID": "probe_id"})
    beta["probe_id"] = beta["probe_id"].astype(str)
    beta = beta[beta["probe_id"].isin(train_probe_set)].copy()
    beta = beta.drop_duplicates("probe_id", keep="first")
    beta = beta.set_index("probe_id")

    rename = {col: normalize_sample_from_beta_column(col) for col in beta.columns}
    beta = beta.rename(columns=rename)
    beta = beta.apply(pd.to_numeric, errors="coerce")
    beta = beta.T.groupby(level=0).mean().T

    external_sample_ids = list(beta.columns)
    samples = pd.DataFrame({"sample_id": external_sample_ids})
    samples = samples.merge(annot, on="sample_id", how="left")
    samples["is_normal_by_id"] = samples["sample_id"].str.endswith("_N")
    samples["label"] = samples["label"].fillna("")
    samples["histology"] = samples["histology"].fillna("")
    samples["tumor_flag"] = samples["tumor_flag"].fillna(1).astype(int)
    if samples["label"].ne("").any():
        keep = samples["label"].ne("") & samples["tumor_flag"].eq(1) & ~samples["is_normal_by_id"]
    else:
        samples["label"] = "TCGA-LUNG"
        samples["accepted_labels"] = "TCGA-LUAD;TCGA-LUSC"
        samples["histology"] = "lung_carcinoma_unspecified"
        keep = ~samples["is_normal_by_id"]
    samples = samples[keep].copy()
    if samples.empty:
        raise ValueError("No external tumor samples retained")

    ordered_probe_ids = [probe for probe in train_probe_ids if probe in beta.index]
    x_df = beta.loc[ordered_probe_ids, samples["sample_id"].tolist()].T
    x = x_df.to_numpy(dtype=np.float32, copy=True)

    samples.insert(0, "external_index", np.arange(len(samples), dtype=int))
    samples["cohort"] = args.cohort
    samples["platform"] = "GPL13534"
    samples["accepted_labels"] = samples["accepted_labels"].fillna(samples["label"])
    samples = samples[
        ["external_index", "sample_id", "label", "accepted_labels", "histology", "tumor_flag", "platform", "cohort"]
    ]

    np.save(out_dir / "X_external.npy", x)
    samples.to_csv(out_dir / "samples_external.tsv", sep="\t", index=False)
    pd.DataFrame({"probe_id": ordered_probe_ids}).to_csv(out_dir / "probe_ids.tsv", sep="\t", index=False)

    manifest = {
        "cohort": args.cohort,
        "raw_beta": str(Path(args.raw_beta).resolve()),
        "sample_annotation": str(Path(args.sample_annotation).resolve()),
        "n_raw_beta_columns": len(beta_cols),
        "n_unique_beta_samples": len(external_sample_ids),
        "n_kept_samples": int(samples.shape[0]),
        "label_counts": samples["label"].value_counts().to_dict(),
        "evaluation_mode": "exact_histology" if set(samples["label"]) <= {"TCGA-LUAD", "TCGA-LUSC"} else "lung_lineage",
        "n_train_probes": len(train_probe_ids),
        "n_external_probes": len(ordered_probe_ids),
        "probe_overlap_ratio_train": float(len(ordered_probe_ids) / max(len(train_probe_ids), 1)),
        "matrix_shape": list(x.shape),
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"event": "done", **manifest}, ensure_ascii=False))


def prepare_gse48684(args: argparse.Namespace) -> None:
    if not args.signal_files:
        raise ValueError("GSE48684 requires --signal-files")
    train_probe_ids = load_train_probe_ids(Path(args.train_probe_ids))
    train_probe_set = set(train_probe_ids)
    frames = []
    all_sample_ids: list[str] = []
    for path in [Path(p) for p in args.signal_files]:
        header = pd.read_csv(path, sep="\t", comment="#", nrows=0)
        sample_ids = []
        for col in header.columns:
            if not str(col).endswith(".Signal_A"):
                continue
            sample = re.sub(r"\.Signal_A$", "", str(col)).strip()
            if f"{sample}.Signal_B" in header.columns:
                sample_ids.append(sample)
        all_sample_ids.extend(sample_ids)
        usecols = ["TargetID", *[f"{sample}.Signal_A" for sample in sample_ids], *[f"{sample}.Signal_B" for sample in sample_ids]]
        df = pd.read_csv(path, sep="\t", comment="#", usecols=usecols, low_memory=False)
        df = df.rename(columns={"TargetID": "probe_id"})
        df["probe_id"] = df["probe_id"].astype(str)
        df = df[df["probe_id"].isin(train_probe_set)].drop_duplicates("probe_id", keep="first").set_index("probe_id")
        beta_cols = {}
        for sample in sample_ids:
            unmethylated = pd.to_numeric(df[f"{sample}.Signal_A"], errors="coerce")
            methylated = pd.to_numeric(df[f"{sample}.Signal_B"], errors="coerce")
            beta_cols[sample] = methylated / (methylated + unmethylated + 100.0)
        frames.append(pd.DataFrame(beta_cols, index=df.index))
    beta = pd.concat(frames, axis=1)
    beta = beta.T.groupby(level=0).mean().T

    ordered_ids = list(beta.columns)
    rows = []
    for idx, sample in enumerate(ordered_ids, start=1):
        is_crc = 15 <= idx <= 28 or 74 <= idx <= 123
        if not is_crc:
            continue
        rows.append(
            {
                "sample_id": sample,
                "label": "TCGA-COADREAD",
                "accepted_labels": "TCGA-COAD;TCGA-READ",
                "histology": "colorectal_cancer",
                "tumor_flag": 1,
            }
        )
    write_prepared_external(
        cohort="GSE48684",
        out_dir=Path(args.out_dir),
        beta=beta,
        samples=pd.DataFrame(rows),
        train_probe_ids=train_probe_ids,
        platform="GPL13534",
        sources={"signal_files": [str(Path(p).resolve()) for p in args.signal_files], "crc_position_rule": "GSM1183453-3561 CRC samples by GEO order"},
        evaluation_mode="colorectal_lineage",
    )


def prepare_gse53051(args: argparse.Namespace) -> None:
    if not args.methylated_signals or not args.unmethylated_signals or not args.geo_html:
        raise ValueError("GSE53051 requires --methylated-signals, --unmethylated-signals, and --geo-html")
    train_probe_ids = load_train_probe_ids(Path(args.train_probe_ids))
    train_probe_set = set(train_probe_ids)
    methylated = pd.read_csv(args.methylated_signals, index_col=0, low_memory=False)
    unmethylated = pd.read_csv(args.unmethylated_signals, index_col=0, low_memory=False)
    methylated.index = methylated.index.astype(str)
    unmethylated.index = unmethylated.index.astype(str)
    common_probes = methylated.index.intersection(unmethylated.index).intersection(pd.Index(train_probe_ids))
    common_cols = [col for col in methylated.columns if col in unmethylated.columns]
    methylated = methylated.loc[common_probes, common_cols].apply(pd.to_numeric, errors="coerce")
    unmethylated = unmethylated.loc[common_probes, common_cols].apply(pd.to_numeric, errors="coerce")
    beta = methylated / (methylated + unmethylated + 100.0)

    titles = parse_geo_sample_titles(Path(args.geo_html))
    if len(titles) < len(common_cols):
        raise ValueError(f"GSE53051: only {len(titles)} GEO titles for {len(common_cols)} matrix columns")
    rows = []
    for sample_id, title in zip(common_cols, titles["title"].tolist()):
        label, accepted, tumor_flag = classify_title(title)
        if not label:
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "label": label,
                "accepted_labels": accepted,
                "histology": title,
                "tumor_flag": int(tumor_flag or 1),
            }
        )
    write_prepared_external(
        cohort="GSE53051",
        out_dir=Path(args.out_dir),
        beta=beta,
        samples=pd.DataFrame(rows),
        train_probe_ids=train_probe_ids,
        platform="GPL13534",
        sources={
            "methylated_signals": str(Path(args.methylated_signals).resolve()),
            "unmethylated_signals": str(Path(args.unmethylated_signals).resolve()),
            "geo_html": str(Path(args.geo_html).resolve()),
        },
        evaluation_mode="multi_tissue_lineage",
    )


def prepare_gse69914(args: argparse.Namespace) -> None:
    if not args.raw_tar:
        raise ValueError("GSE69914 requires --raw-tar")
    train_probe_ids = load_train_probe_ids(Path(args.train_probe_ids))
    train_probe_set = set(train_probe_ids)
    series = {}
    rows = []
    with tarfile.open(args.raw_tar) as tar:
        members = [m for m in tar.getmembers() if m.name.endswith("_Raw.txt.gz")]
        for member in members:
            sample_id = re.sub(r"^GSM\d+_", "", member.name)
            sample_id = re.sub(r"_Raw\.txt\.gz$", "", sample_id)
            handle = tar.extractfile(member)
            if handle is None:
                continue
            with gzip.open(handle, "rt") as gz:
                df = pd.read_csv(gz, sep="\t", usecols=["IlmnID", "Beta"])
            df["IlmnID"] = df["IlmnID"].astype(str)
            df = df[df["IlmnID"].isin(train_probe_set)].drop_duplicates("IlmnID", keep="first")
            series[sample_id] = pd.to_numeric(df.set_index("IlmnID")["Beta"], errors="coerce")
            rows.append(
                {
                    "sample_id": sample_id,
                    "label": "TCGA-BRCA",
                    "accepted_labels": "TCGA-BRCA",
                    "histology": "breast_cohort_unresolved_geo_metadata",
                    "tumor_flag": 1,
                }
            )
    if not series:
        raise ValueError("GSE69914: no raw beta samples found")
    beta = pd.DataFrame(series)
    write_prepared_external(
        cohort="GSE69914",
        out_dir=Path(args.out_dir),
        beta=beta,
        samples=pd.DataFrame(rows),
        train_probe_ids=train_probe_ids,
        platform="GPL16304",
        sources={"raw_tar": str(Path(args.raw_tar).resolve()), "metadata_warning": "GEO page reports incomplete metadata; all BCFD raw beta files are labeled BRCA for provisional prediction."},
        evaluation_mode="provisional_breast_cohort_prediction",
    )


def prepare_gse105260(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cohort": "GSE105260",
        "raw_tar": str(Path(args.raw_tar).resolve()) if args.raw_tar else "",
        "status": "blocked",
        "reason": "RAW tar contains IDAT files only; worker-0 has no Rscript/minfi and no local beta matrix was provided.",
        "required_next_input": "processed beta matrix from GEO sample table/series matrix or an IDAT preprocessing environment.",
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"event": "blocked", **manifest}, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if args.cohort == "GSE56044":
        prepare_gse56044(args)
    elif args.cohort == "GSE48684":
        prepare_gse48684(args)
    elif args.cohort == "GSE53051":
        prepare_gse53051(args)
    elif args.cohort == "GSE69914":
        prepare_gse69914(args)
    elif args.cohort == "GSE105260":
        prepare_gse105260(args)


if __name__ == "__main__":
    main()
