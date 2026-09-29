"""
GSE105260: process IDAT files directly using methylprep's low-level IdatDataset,
map Illumina addresses to cg probe IDs via annotation manifest, compute beta values,
and write prepared external matrix.

This bypasses methylprep's manifest validation which fails on GEO-supplied manifests.
"""
from __future__ import annotations

import gzip
import io
import json
import tarfile
import tempfile
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from methylprep.files import IdatDataset


ILLUMINA_ADDRESS_COLUMN = "Illumina_ID"
METHYLATED_CHANNEL = "Red"
UNMETHYLATED_CHANNEL = "Grn"

INFINIUM_I_ADDRESS = "AddressA_ID"
INFINIUM_II_ADDRESS = "AddressA_ID"

# Probe types based on Infinium design
INFINIUM_I = "I"
INFINIUM_II = "II"


def parse_manifest(manifest_path: str) -> pd.DataFrame:
    """Parse Illumina 450k manifest CSV (GEO format with [Heading]/[Assay] sections)."""
    rows = []
    in_assay = False
    with gzip.open(manifest_path, "rt") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("[Assay]"):
                in_assay = True
                continue
            if stripped.startswith("["):
                in_assay = False
                continue
            if not in_assay or not stripped:
                continue

            parts = stripped.split(",")
            if parts[0] == "IlmnID":
                continue
            if len(parts) < 7:
                continue

            ilmn_id = parts[0].strip()
            address_a = parts[2].strip()
            address_b = parts[4].strip() if len(parts) > 4 else ""
            design = parts[6].strip() if len(parts) > 6 else "II"
            color = parts[8].strip() if len(parts) > 8 else ""

            rows.append({
                "IlmnID": ilmn_id,
                "AddressA_ID": int(address_a) if address_a and address_a != "0" else 0,
                "AddressB_ID": int(address_b) if address_b and address_b != "0" else 0,
                "Infinium_Design_Type": design,
                "Color_Channel": color,
            })
    return pd.DataFrame(rows)


def _safe_addr(val: Any) -> int | None:
    """Convert manifest address (string/float) to int, returning None if invalid."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def build_address_to_probe(manifest: pd.DataFrame) -> dict[int, dict[str, Any]]:
    """Build mapping from Illumina bead address -> probe info.

    For Infinium I: address_a = methylated, address_b = unmethylated
    For Infinium II: address_a = both (methylated in Red, unmethylated in Grn)
    """
    mapping: dict[int, dict[str, Any]] = {}
    for _, row in manifest.iterrows():
        design = row["Infinium_Design_Type"]
        ilmn_id = row["IlmnID"]
        color = str(row.get("Color_Channel", ""))

        if design == "I":
            addr_a = _safe_addr(row["AddressA_ID"])
            addr_b = _safe_addr(row["AddressB_ID"])
            if addr_a is not None:
                mapping.setdefault(addr_a, {"probe_id": ilmn_id, "design": "I", "channel": "Red" if color == "R" else "Grn"})
            if addr_b is not None:
                mapping.setdefault(addr_b, {"probe_id": ilmn_id, "design": "I", "channel": "Grn" if color == "R" else "Red"})
        else:
            addr_a = _safe_addr(row["AddressA_ID"])
            if addr_a is not None:
                mapping.setdefault(addr_a, {"probe_id": ilmn_id, "design": "II"})
    return mapping


def process_idat_file(fileobj: io.BytesIO, channel: str) -> pd.Series:
    """Read IDAT file and return probe mean intensities as a Series indexed by Illumina address."""
    idat = IdatDataset(fileobj, channel)
    return idat.probe_means["mean_value"]


def read_idat_pair(grn_path: str, red_path: str) -> tuple[pd.Series, pd.Series]:
    """Read both Grn and Red IDAT files for a sample."""
    with gzip.open(grn_path, "rb") as f:
        grn_data = f.read()
    with gzip.open(red_path, "rb") as f:
        red_data = f.read()
    grn = process_idat_file(io.BytesIO(grn_data), "Grn")
    red = process_idat_file(io.BytesIO(red_data), "Red")
    return grn, red


def compute_beta_for_sample(
    grn_intensities: pd.Series,
    red_intensities: pd.Series,
    address_map: dict[int, dict[str, Any]],
) -> pd.Series:
    """Compute beta values for all probes in a sample.

    For Infinium II: methylated = Red, unmethylated = Grn at same address
    For Infinium I: methylated and unmethylated at different addresses
    """
    grn_dict = grn_intensities.to_dict()
    red_dict = red_intensities.to_dict()

    probe_m: dict[str, float] = {}
    probe_u: dict[str, float] = {}

    for addr, info in address_map.items():
        probe_id = info["probe_id"]
        design = info["design"]

        if design == "I":
            channel = info["channel"]
            if channel == "Red":
                intensity = red_dict.get(addr, np.nan)
                probe_m.setdefault(probe_id, 0.0)
                probe_m[probe_id] += intensity if not np.isnan(intensity) else 0
            else:  # Grn
                intensity = grn_dict.get(addr, np.nan)
                probe_u.setdefault(probe_id, 0.0)
                probe_u[probe_id] += intensity if not np.isnan(intensity) else 0
        else:  # Infinium II
            meth = red_dict.get(addr, np.nan)
            unmeth = grn_dict.get(addr, np.nan)
            if not np.isnan(meth):
                probe_m[probe_id] = meth
            if not np.isnan(unmeth):
                probe_u[probe_id] = unmeth

    common_probes = set(probe_m.keys()) & set(probe_u.keys())
    betas = {}
    for probe in common_probes:
        m = probe_m[probe]
        u = probe_u[probe]
        if m + u > 0:
            betas[probe] = m / (m + u + 100.0)
        else:
            betas[probe] = np.nan
    return pd.Series(betas)


def extract_idats_from_tar(tar_path: str, dest_dir: str) -> list[str]:
    """Extract IDAT files from GEO RAW tar to destination directory.

    Returns sorted list of unique sample IDs (GSM numbers).
    """
    samples_seen: set[str] = set()
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            name = member.name
            if not (name.endswith(".idat.gz") or name.endswith(".idat")):
                continue
            tar.extract(member, path=dest_dir)
            gsm = name.split("_")[0]
            samples_seen.add(gsm)
    return sorted(samples_seen)


def classify_gse105260_title(title: str) -> tuple[str, str, str]:
    """Classify GSE105260 sample title into TCGA label.

    Returns (label, accepted_labels, tumor_flag).
    """
    text = title.lower()
    if "normal" in text and "renal" in text:
        return "", "", "0"
    if "primary ccrcc" in text:
        return "TCGA-KIRC", "TCGA-KIRC", "1"
    if "metastasis ccrcc" in text or "metastatic ccrcc" in text:
        return "TCGA-KIRC", "TCGA-KIRC", "1"
    return "", "", ""


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Process GSE105260 IDATs to beta matrix")
    parser.add_argument("--raw-tar", required=True, help="GSE105260_RAW.tar path")
    parser.add_argument("--train-probe-ids", required=True, help="TCGA candidate probe_ids.tsv")
    parser.add_argument("--geo-html", help="Downloaded GEO HTML for sample title labeling")
    parser.add_argument("--manifest-csv", required=True, help="Illumina 450k manifest CSV (GPL13534)")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load training probe IDs
    train_probe_df = pd.read_csv(args.train_probe_ids, sep="\t")
    if "probe_id" not in train_probe_df.columns:
        raise ValueError("train_probe_ids must contain probe_id column")
    train_probes = set(train_probe_df["probe_id"].astype(str).tolist())
    print(json.dumps({"event": "train_probes_loaded", "count": len(train_probes)}))

    # Parse manifest
    manifest = parse_manifest(args.manifest_csv)
    print(json.dumps({"event": "manifest_parsed", "probes_in_manifest": len(manifest)}))

    # Filter manifest to training probes only
    manifest = manifest[manifest["IlmnID"].isin(train_probes)]
    print(json.dumps({"event": "manifest_filtered", "overlap_probes": len(manifest)}))
    if manifest.empty:
        raise ValueError("No overlap between manifest and training probes")

    # Build address -> probe mapping
    address_map = build_address_to_probe(manifest)
    print(json.dumps({"event": "address_map_built", "unique_addresses": len(address_map)}))

    # Extract IDATs
    tmp_dir = tempfile.mkdtemp(prefix="gse105260_idat_")
    try:
        sample_gsms = extract_idats_from_tar(args.raw_tar, tmp_dir)
        print(json.dumps({"event": "idats_extracted", "n_samples": len(sample_gsms)}))

        # Parse sample titles from GEO HTML for labeling
        labels: dict[str, tuple[str, str, str]] = {}
        if args.geo_html:
            html_path = Path(args.geo_html)
            if html_path.exists():
                text = html_path.read_text(errors="ignore")
                import re, html as html_mod
                pattern = re.compile(
                    r">(?P<gsm>GSM\d+)</a></td>\s*<td[^>]*>(?P<title>.*?)</td>",
                    flags=re.DOTALL,
                )
                for match in pattern.finditer(text):
                    title = re.sub(r"<.*?>", "", match.group("title"))
                    title = html_mod.unescape(title).strip()
                    gsm = match.group("gsm")
                    label, accepted, tumor_flag = classify_gse105260_title(title)
                    if label:
                        labels[gsm] = (label, accepted, tumor_flag)
                print(json.dumps({"event": "titles_parsed", "n_labeled": len(labels)}))

        # Process each sample
        sample_rows = []
        beta_frames: list[pd.Series] = []

        for gsm in sample_gsms:
            grn_files = sorted(Path(tmp_dir).glob(f"{gsm}_*_Grn.idat.gz"))
            red_files = sorted(Path(tmp_dir).glob(f"{gsm}_*_Red.idat.gz"))
            if not grn_files or not red_files:
                print(json.dumps({"event": "missing_idat", "gsm": gsm, "grn": len(grn_files), "red": len(red_files)}))
                continue

            grn_int, red_int = read_idat_pair(str(grn_files[0]), str(red_files[0]))
            beta_series = compute_beta_for_sample(grn_int, red_int, address_map)
            beta_series.name = gsm
            beta_frames.append(beta_series)

            label = labels.get(gsm, ("", "", "0"))
            sample_rows.append({
                "sample_id": gsm,
                "label": label[0],
                "accepted_labels": label[1],
                "histology": "ccRCC" if label[0] else "normal_kidney",
                "tumor_flag": int(label[2]),
            })

        if not beta_frames:
            raise ValueError("No samples processed")

        # Combine into beta matrix
        beta_df = pd.concat(beta_frames, axis=1)
        print(json.dumps({"event": "beta_matrix_computed", "shape": list(beta_df.shape)}))

        # Sort probes to match training probe order
        ordered_probes = [p for p in train_probe_df["probe_id"] if p in beta_df.index]
        beta_df = beta_df.loc[ordered_probes]

        # Subset to labeled tumor samples
        samples_df = pd.DataFrame(sample_rows)
        samples_df = samples_df[samples_df["label"].ne("")].copy()
        valid_samples = [s for s in samples_df["sample_id"] if s in beta_df.columns]
        beta_df = beta_df[valid_samples]
        samples_df = samples_df[samples_df["sample_id"].isin(valid_samples)].copy()
        samples_df = samples_df.drop_duplicates("sample_id", keep="first")

        # Write outputs (transpose to samples x probes, matching other cohorts)
        x = beta_df.to_numpy(dtype=np.float32, copy=True).T
        np.save(out_dir / "X_external.npy", x)

        samples_df.insert(0, "external_index", np.arange(len(samples_df), dtype=int))
        samples_df["cohort"] = "GSE105260"
        samples_df["platform"] = "GPL13534"
        samples_df = samples_df[["external_index", "sample_id", "label", "accepted_labels", "histology", "tumor_flag", "platform", "cohort"]]
        samples_df.to_csv(out_dir / "samples_external.tsv", sep="\t", index=False)

        pd.DataFrame({"probe_id": ordered_probes}).to_csv(out_dir / "probe_ids.tsv", sep="\t", index=False)

        manifest_out = {
            "cohort": "GSE105260",
            "raw_tar": str(Path(args.raw_tar).resolve()),
            "n_unique_beta_samples": int(beta_df.shape[1]),
            "n_kept_samples": int(samples_df.shape[0]),
            "label_counts": samples_df["label"].value_counts().to_dict(),
            "evaluation_mode": "kirc_lineage",
            "n_train_probes": len(train_probes),
            "n_external_probes": len(ordered_probes),
            "probe_overlap_ratio_train": float(len(ordered_probes) / len(train_probes)),
            "matrix_shape": list(x.shape),
        }
        (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest_out, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"event": "done", **manifest_out}, ensure_ascii=False))

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
