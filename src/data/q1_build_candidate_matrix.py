from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a leakage-safe candidate probe matrix from per-sample GDC methylation TXT files."
    )
    parser.add_argument("--sample-index", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-k", type=int, default=0, help="0 keeps every probe passing coverage filter.")
    parser.add_argument("--min-coverage-ratio", type=float, default=0.7)
    parser.add_argument("--platform", default="auto")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--skip-probe-check", action="store_true")
    parser.add_argument("--impute-global", action="store_true", help="Debug only; main-paper matrix should preserve NaN.")
    return parser.parse_args()


def progress(stage: str, current: int, total: int) -> None:
    if total <= 0:
        return
    width = 30
    ratio = min(max(current / total, 0.0), 1.0)
    bar = "#" * int(width * ratio) + "-" * (width - int(width * ratio))
    sys.stdout.write(f"\r[{stage}] [{bar}] {current}/{total} ({ratio * 100:5.1f}%)")
    if current >= total:
        sys.stdout.write("\n")
    sys.stdout.flush()


def read_probe_and_beta(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["probe_id", "beta"],
        na_values=["NA", "NaN", ""],
        dtype={"probe_id": str},
        low_memory=False,
    )
    return df["probe_id"].to_numpy(), df["beta"].to_numpy(dtype=np.float32, copy=False)


def read_beta_only(path: Path) -> np.ndarray:
    df = pd.read_csv(path, sep="\t", header=None, usecols=[1], names=["beta"], na_values=["NA", "NaN", ""], low_memory=False)
    return df["beta"].to_numpy(dtype=np.float32, copy=False)


def read_beta(path: Path, expected: np.ndarray | None, check_probe_order: bool) -> tuple[np.ndarray | None, np.ndarray]:
    if expected is None:
        return read_probe_and_beta(path)
    if check_probe_order:
        probes, values = read_probe_and_beta(path)
        if len(probes) != len(expected) or not np.array_equal(probes, expected):
            raise ValueError(f"Probe order mismatch: {path}")
        return None, values
    values = read_beta_only(path)
    if len(values) != len(expected):
        raise ValueError(f"Probe count mismatch: {path}")
    return None, values


def first_pass(file_paths: list[Path], skip_probe_check: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probe_ids: np.ndarray | None = None
    count: np.ndarray | None = None
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None
    for i, path in enumerate(file_paths, start=1):
        check = not skip_probe_check or i <= 2
        probes, values = read_beta(path, probe_ids, check)
        if probe_ids is None:
            assert probes is not None
            probe_ids = probes
            count = np.zeros(len(probe_ids), dtype=np.int32)
            mean = np.zeros(len(probe_ids), dtype=np.float64)
            m2 = np.zeros(len(probe_ids), dtype=np.float64)
        assert count is not None and mean is not None and m2 is not None
        valid = ~np.isnan(values)
        count[valid] += 1
        delta = values[valid] - mean[valid]
        mean[valid] += delta / count[valid]
        delta2 = values[valid] - mean[valid]
        m2[valid] += delta * delta2
        progress("first_pass ", i, len(file_paths))
    assert probe_ids is not None and count is not None and mean is not None and m2 is not None
    variance = np.zeros_like(mean)
    enough = count > 1
    variance[enough] = m2[enough] / (count[enough] - 1)
    return probe_ids, mean, variance, count


def second_pass(
    file_paths: list[Path],
    probe_ids: np.ndarray,
    selected_idx: np.ndarray,
    probe_mean: np.ndarray,
    out_path: Path,
    skip_probe_check: bool,
    impute_global: bool,
) -> None:
    x = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(len(file_paths), len(selected_idx)))
    selected_mean = probe_mean[selected_idx].astype(np.float32, copy=False)
    for i, path in enumerate(file_paths, start=1):
        check = not skip_probe_check or i <= 2
        _, values = read_beta(path, probe_ids, check)
        row = values[selected_idx].astype(np.float32, copy=True)
        if impute_global:
            missing = np.isnan(row)
            if missing.any():
                row[missing] = selected_mean[missing]
        x[i - 1] = row
        progress("second_pass", i, len(file_paths))
    x.flush()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_df = pd.read_csv(args.sample_index, sep="\t")
    sample_df = sample_df[sample_df["exists_local"] == 1].copy()
    if sample_df.empty:
        raise ValueError("No local files available in sample index.")

    platform_counts = sample_df["platform"].value_counts()
    platform = str(platform_counts.index[0]) if args.platform.lower() == "auto" else args.platform
    sample_df = sample_df[sample_df["platform"] == platform].copy()
    if args.max_samples > 0:
        sample_df = sample_df.head(args.max_samples).copy()
    file_paths = [Path(value) for value in sample_df["file_path"].tolist()]

    probe_ids, probe_mean, probe_var, probe_count = first_pass(file_paths, args.skip_probe_check)
    coverage = probe_count / len(file_paths)
    candidate_idx = np.where(coverage >= args.min_coverage_ratio)[0]
    if args.top_k > 0:
        k = min(args.top_k, len(candidate_idx))
        rel = np.argpartition(probe_var[candidate_idx], -k)[-k:]
        selected_idx = np.sort(candidate_idx[rel])
    else:
        selected_idx = candidate_idx

    second_pass(
        file_paths=file_paths,
        probe_ids=probe_ids,
        selected_idx=selected_idx,
        probe_mean=probe_mean,
        out_path=out_dir / "X_candidate.npy",
        skip_probe_check=args.skip_probe_check,
        impute_global=args.impute_global,
    )

    selected_probes = probe_ids[selected_idx]
    np.save(out_dir / "probe_ids.npy", selected_probes)
    np.save(out_dir / "probe_index.npy", selected_idx)
    np.save(out_dir / "probe_mean.npy", probe_mean[selected_idx].astype(np.float32))
    np.save(out_dir / "probe_variance.npy", probe_var[selected_idx].astype(np.float32))
    np.save(out_dir / "probe_coverage.npy", coverage[selected_idx].astype(np.float32))
    pd.DataFrame({"probe_id": selected_probes}).to_csv(out_dir / "probe_ids.tsv", sep="\t", index=False)
    sample_df.reset_index(drop=True).to_csv(out_dir / "samples_used.tsv", sep="\t", index=False)

    summary = {
        "sample_index": str(Path(args.sample_index).resolve()),
        "platform_selected": platform,
        "n_samples": int(len(sample_df)),
        "n_total_probes": int(len(probe_ids)),
        "n_candidate_probes": int(len(candidate_idx)),
        "n_selected_probes": int(len(selected_idx)),
        "top_k": int(args.top_k),
        "min_coverage_ratio": float(args.min_coverage_ratio),
        "preserved_nan": not args.impute_global,
        "matrix": "X_candidate.npy",
    }
    (out_dir / "candidate_matrix_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
