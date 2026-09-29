from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from q1_feature_selection import select_features
from q1_metrics import calibration_metrics, multiclass_metrics, per_class_metrics, uncertainty_frame
from q1_train_ml_baselines import (
    fold_pairs,
    labels_for_rows,
    prepare_fold_matrix_mapping,
    resolve_matrix_rows,
    subset_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PathMethNet on leakage-safe Q1 folds.")
    parser.add_argument("--matrix-npy", required=True)
    parser.add_argument(
        "--matrix-samples-tsv",
        default="",
        help="Optional source manifest defining the row order of --matrix-npy.",
    )
    parser.add_argument("--probe-ids", required=True, help="Candidate probe IDs as TSV column probe_id or NPY.")
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--probe-annotation-tsv", required=True, help="Columns: probe_id,gene_symbol.")
    parser.add_argument("--pathway-tsv", required=True, help="Columns: pathway,gene_symbol.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--feature-method", default="effect_size")
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--val-folds", type=int, default=5)
    parser.add_argument("--max-fold-runs", type=int, default=0)
    parser.add_argument("--run-start", type=int, default=1, help="1-based fold-run index to start from.")
    parser.add_argument("--run-end", type=int, default=0, help="1-based fold-run index to end at, inclusive. 0 means no limit.")
    parser.add_argument("--resume", action="store_true", help="Skip fold-runs already present in fold_metrics.tsv.")
    parser.add_argument("--log-every-epoch", type=int, default=0, help="Print validation progress every N epochs. 0 disables epoch logs.")
    parser.add_argument("--distill-teacher", choices=["none", "logreg"], default="none")
    parser.add_argument("--distill-alpha", type=float, default=0.0, help="Weight for KL distillation loss.")
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--teacher-max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--min-cpg-per-gene", type=int, default=1)
    parser.add_argument("--min-genes-per-pathway", type=int, default=3)
    parser.add_argument("--max-cpg-per-gene", type=int, default=64)
    parser.add_argument("--max-genes-per-pathway", type=int, default=128)
    parser.add_argument("--max-pathways", type=int, default=0, help="Keep the top pathways by mapped-gene count. 0 keeps all pathways.")
    parser.add_argument("--allow-flat-fallback", action="store_true", help="Debug only: use probe-level pseudo pathways if annotations do not map.")
    return parser.parse_args()


def load_probe_ids(path: Path) -> list[str]:
    if path.suffix.lower() == ".npy":
        return [str(x) for x in np.load(path, allow_pickle=True).tolist()]
    df = pd.read_csv(path, sep="\t")
    if "probe_id" not in df.columns:
        raise ValueError("probe ID TSV must contain column: probe_id")
    return df["probe_id"].astype(str).tolist()


def pad_rows(rows: list[list[int]], width: int) -> np.ndarray:
    out = np.full((len(rows), width), -1, dtype=np.int64)
    for i, row in enumerate(rows):
        clipped = row[:width]
        out[i, : len(clipped)] = clipped
    return out


def build_index_maps(
    selected_probe_ids: list[str],
    annotation_tsv: Path,
    pathway_tsv: Path,
    min_cpg_per_gene: int,
    min_genes_per_pathway: int,
    max_cpg_per_gene: int,
    max_genes_per_pathway: int,
    max_pathways: int,
    allow_flat_fallback: bool,
) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame, pd.DataFrame]:
    probe_to_idx = {probe: i for i, probe in enumerate(selected_probe_ids)}
    annot = pd.read_csv(annotation_tsv, sep="\t")
    pathways = pd.read_csv(pathway_tsv, sep="\t")
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
        if probe in probe_to_idx and gene and gene.lower() != "nan":
            gene_to_cpg[gene].append(probe_to_idx[probe])
    genes = sorted(gene for gene, idxs in gene_to_cpg.items() if len(idxs) >= min_cpg_per_gene)

    if not genes and allow_flat_fallback:
        genes = [f"probe_{i}" for i in range(len(selected_probe_ids))]
        gene_rows = [[i] for i in range(len(selected_probe_ids))]
        pathway_names = ["all_selected_probes"]
        pathway_rows = [list(range(len(genes)))]
    else:
        gene_to_idx = {gene: i for i, gene in enumerate(genes)}
        gene_rows = [gene_to_cpg[gene] for gene in genes]
        pathway_to_genes: dict[str, list[int]] = defaultdict(list)
        for row in pathways.itertuples(index=False):
            pathway = str(getattr(row, "pathway")).strip()
            gene = str(getattr(row, "gene_symbol")).strip()
            if pathway and gene in gene_to_idx:
                pathway_to_genes[pathway].append(gene_to_idx[gene])
        pathway_items = [
            (pathway, sorted(set(idxs)))
            for pathway, idxs in pathway_to_genes.items()
            if len(set(idxs)) >= min_genes_per_pathway
        ]
        pathway_items = sorted(pathway_items, key=lambda item: (-len(item[1]), item[0]))
        if max_pathways > 0:
            pathway_items = pathway_items[:max_pathways]
        pathway_names = [pathway for pathway, _ in pathway_items]
        pathway_rows = [idxs for _, idxs in pathway_items]

    if not genes or not pathway_names:
        raise ValueError("No valid gene/pathway index maps. Check annotation/pathway tables or use --allow-flat-fallback for smoke tests.")

    gene_map = pd.DataFrame({"gene_index": range(len(genes)), "gene_symbol": genes})
    pathway_map = pd.DataFrame({"pathway_index": range(len(pathway_names)), "pathway": pathway_names})
    return (
        torch.tensor(pad_rows(gene_rows, max_cpg_per_gene), dtype=torch.long),
        torch.tensor(pad_rows(pathway_rows, max_genes_per_pathway), dtype=torch.long),
        gene_map,
        pathway_map,
    )


def inner_train_val_rows(samples_df: pd.DataFrame, train_rows: np.ndarray, y_train: np.ndarray, n_splits: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    groups = samples_df.set_index("matrix_row").loc[train_rows, "q1_patient_id"].astype(str).to_numpy()
    use_groups = len(np.unique(groups)) < len(groups)
    if use_groups:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_idx, val_idx = next(splitter.split(np.zeros(len(y_train)), y_train, groups=groups))
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_idx, val_idx = next(splitter.split(np.zeros(len(y_train)), y_train))
    return train_rows[train_idx], train_rows[val_idx]


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    soft_targets: np.ndarray | None = None,
) -> DataLoader:
    tensors = [torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)]
    if soft_targets is not None:
        tensors.append(torch.tensor(soft_targets, dtype=torch.float32))
    ds = TensorDataset(*tensors)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def variance_select_features(x_train_raw: np.ndarray, top_k: int) -> np.ndarray:
    scores = np.nanvar(x_train_raw, axis=0)
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    if top_k >= scores.shape[0]:
        return np.argsort(scores)[::-1].astype(np.int64)
    selected = np.argpartition(scores, -top_k)[-top_k:]
    selected = selected[np.argsort(scores[selected])[::-1]]
    return selected.astype(np.int64, copy=False)


def median_impute_selected(
    x_train_raw: np.ndarray,
    x_val_raw: np.ndarray,
    x_test_raw: np.ndarray,
    selected_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train_selected = np.asarray(x_train_raw[:, selected_idx], dtype=np.float32)
    medians = np.nanmedian(x_train_selected, axis=0).astype(np.float32, copy=False)
    medians = np.nan_to_num(medians, nan=0.0)

    def fill(arr: np.ndarray) -> np.ndarray:
        out = np.asarray(arr[:, selected_idx], dtype=np.float32).copy()
        missing = np.isnan(out)
        if missing.any():
            out[missing] = np.take(medians, np.where(missing)[1])
        return out

    return fill(x_train_raw), fill(x_val_raw), fill(x_test_raw)


def train_logreg_teacher(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    max_iter: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    teacher = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        multi_class="auto",
        max_iter=max_iter,
        class_weight="balanced",
        n_jobs=1,
        random_state=seed,
    )
    teacher.fit(x_train, y_train)
    return (
        teacher.predict_proba(x_train).astype(np.float32),
        teacher.predict_proba(x_val).astype(np.float32),
        teacher.predict_proba(x_test).astype(np.float32),
    )


def distillation_kl_loss(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    eps = 1e-8
    teacher_probs = torch.clamp(teacher_probs, min=eps)
    teacher_probs = teacher_probs / teacher_probs.sum(dim=1, keepdim=True)
    teacher_logits = torch.log(teacher_probs)
    teacher_soft = torch.softmax(teacher_logits / temperature, dim=1)
    student_log_soft = F.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(student_log_soft, teacher_soft, reduction="batchmean") * (temperature**2)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, n_classes: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels: list[np.ndarray] = []
    probas: list[np.ndarray] = []
    losses: list[float] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)["class_logits"]
            losses.append(float(F.cross_entropy(logits, yb).item()))
            probas.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(yb.cpu().numpy())
    y_true = np.concatenate(labels)
    y_proba = np.concatenate(probas)
    y_pred = y_proba.argmax(axis=1)
    metrics = multiclass_metrics(y_true, y_pred, y_proba)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics, y_true, y_pred, y_proba


def write_outputs(
    out_dir: Path,
    fold_rows: list[dict[str, Any]],
    per_class_frames: list[pd.DataFrame],
    calibration_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
) -> None:
    metrics_df = pd.DataFrame(fold_rows)
    metrics_df.to_csv(out_dir / "fold_metrics.tsv", sep="\t", index=False)
    if not metrics_df.empty:
        numeric_cols = [c for c in metrics_df.columns if c not in {"run_id", "model", "feature_method"}]
        metrics_df.groupby(["model", "feature_method", "top_k"], dropna=False)[numeric_cols].agg(["mean", "std"]).to_csv(
            out_dir / "summary_metrics_by_setting.csv"
        )
    pd.DataFrame(history_rows).to_csv(out_dir / "training_history.tsv", sep="\t", index=False)
    if per_class_frames:
        pd.concat(per_class_frames, ignore_index=True).to_csv(out_dir / "per_class_metrics.tsv", sep="\t", index=False)
    if calibration_rows:
        pd.DataFrame(calibration_rows).to_csv(out_dir / "calibration_metrics.tsv", sep="\t", index=False)


def main() -> None:
    args = parse_args()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    from q1_hierarchical_model import CPGGenePathwayAttentionNet

    globals()["torch"] = torch
    globals()["nn"] = nn
    globals()["F"] = F
    globals()["DataLoader"] = DataLoader
    globals()["TensorDataset"] = TensorDataset
    globals()["CPGGenePathwayAttentionNet"] = CPGGenePathwayAttentionNet

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(exist_ok=True)
    (out_dir / "selected_features").mkdir(exist_ok=True)
    (out_dir / "index_maps").mkdir(exist_ok=True)

    splits_dir = Path(args.splits_dir)
    samples_df = pd.read_csv(splits_dir / "samples_locked.tsv", sep="\t")
    folds_df = pd.read_csv(splits_dir / "fold_assignments.tsv", sep="\t")
    split_summary = json.loads((splits_dir / "split_summary.json").read_text(encoding="utf-8"))
    label_classes = list(split_summary["labels"])
    n_classes = len(label_classes)
    if "q1_label_id" not in samples_df.columns:
        label_to_id = {label: idx for idx, label in enumerate(label_classes)}
        samples_df["q1_label_id"] = samples_df["project_id"].astype(str).map(label_to_id)
        if samples_df["q1_label_id"].isna().any():
            raise ValueError("samples manifest contains project_id values absent from split_summary labels")
        samples_df["q1_label_id"] = samples_df["q1_label_id"].astype(int)
    all_probe_ids = load_probe_ids(Path(args.probe_ids))
    x = np.load(args.matrix_npy, mmap_mode="r")
    locked_to_matrix_rows = resolve_matrix_rows(samples_df, args.matrix_samples_tsv, x.shape[0])
    samples_df, fold_to_matrix = prepare_fold_matrix_mapping(samples_df, folds_df, locked_to_matrix_rows)
    device = torch.device(args.device)

    fold_rows: list[dict[str, Any]] = []
    per_class_frames: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    completed_run_ids: set[str] = set()
    if args.resume and (out_dir / "fold_metrics.tsv").exists():
        previous_metrics = pd.read_csv(out_dir / "fold_metrics.tsv", sep="\t")
        if "run_id" in previous_metrics.columns:
            completed_run_ids = set(previous_metrics["run_id"].astype(str))
            fold_rows.extend(previous_metrics.to_dict("records"))
    if args.resume and (out_dir / "training_history.tsv").exists():
        history_rows.extend(pd.read_csv(out_dir / "training_history.tsv", sep="\t").to_dict("records"))
    if args.resume and (out_dir / "calibration_metrics.tsv").exists():
        calibration_rows.extend(pd.read_csv(out_dir / "calibration_metrics.tsv", sep="\t").to_dict("records"))
    if args.resume and (out_dir / "per_class_metrics.tsv").exists():
        per_class_frames.append(pd.read_csv(out_dir / "per_class_metrics.tsv", sep="\t"))
    run_count = 0

    for repeat, fold in fold_pairs(folds_df):
        run_count += 1
        if run_count < args.run_start:
            continue
        if args.run_end and run_count > args.run_end:
            break
        if args.max_fold_runs and run_count > args.max_fold_runs:
            break
        run_id = f"pathmethnet__{args.feature_method}__top{args.top_k}__r{repeat:02d}f{fold:02d}"
        if run_id in completed_run_ids:
            print(json.dumps({"run_id": run_id, "status": "skipped_existing"}, ensure_ascii=False), flush=True)
            continue
        start = time.time()

        outer_train_rows = subset_rows(folds_df, repeat, fold, "train")
        test_rows = subset_rows(folds_df, repeat, fold, "test")
        outer_y_train = labels_for_rows(samples_df, outer_train_rows)
        train_rows, val_rows = inner_train_val_rows(samples_df, outer_train_rows, outer_y_train, args.val_folds, args.seed + repeat * 100 + fold)

        y_train = labels_for_rows(samples_df, train_rows)
        y_val = labels_for_rows(samples_df, val_rows)
        y_test = labels_for_rows(samples_df, test_rows)
        x_train_raw = np.asarray(x[fold_to_matrix[train_rows]], dtype=np.float32)
        x_val_raw = np.asarray(x[fold_to_matrix[val_rows]], dtype=np.float32)
        x_test_raw = np.asarray(x[fold_to_matrix[test_rows]], dtype=np.float32)

        if args.feature_method == "variance":
            selected_idx = variance_select_features(x_train_raw, args.top_k)
            x_train_filled, x_val_filled, x_test_filled = median_impute_selected(
                x_train_raw, x_val_raw, x_test_raw, selected_idx
            )
        else:
            imputer = SimpleImputer(strategy="median")
            x_train_imputed = imputer.fit_transform(x_train_raw)
            selected_idx = select_features(x_train_imputed, y_train, method=args.feature_method, top_k=args.top_k, seed=args.seed)
            x_train_filled = x_train_imputed[:, selected_idx]
            x_val_filled = imputer.transform(x_val_raw)[:, selected_idx]
            x_test_filled = imputer.transform(x_test_raw)[:, selected_idx]
        np.save(out_dir / "selected_features" / f"{run_id}.npy", selected_idx)
        selected_probe_ids = [all_probe_ids[i] for i in selected_idx]

        scaler = StandardScaler(with_mean=True)
        x_train_selected = scaler.fit_transform(x_train_filled)
        x_val_selected = scaler.transform(x_val_filled)
        x_test_selected = scaler.transform(x_test_filled)

        train_teacher_proba: np.ndarray | None = None
        val_teacher_proba: np.ndarray | None = None
        test_teacher_proba: np.ndarray | None = None
        if args.distill_teacher == "logreg" and args.distill_alpha > 0:
            teacher_start = time.time()
            train_teacher_proba, val_teacher_proba, test_teacher_proba = train_logreg_teacher(
                x_train_selected,
                y_train,
                x_val_selected,
                x_test_selected,
                max_iter=args.teacher_max_iter,
                seed=args.seed + repeat * 100 + fold,
            )
            print(
                json.dumps(
                    {
                        "run_id": run_id,
                        "event": "teacher_fit",
                        "teacher": "logreg",
                        "seconds": float(time.time() - teacher_start),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        gene_idx, pathway_idx, gene_map, pathway_map = build_index_maps(
            selected_probe_ids=selected_probe_ids,
            annotation_tsv=Path(args.probe_annotation_tsv),
            pathway_tsv=Path(args.pathway_tsv),
            min_cpg_per_gene=args.min_cpg_per_gene,
            min_genes_per_pathway=args.min_genes_per_pathway,
            max_cpg_per_gene=args.max_cpg_per_gene,
            max_genes_per_pathway=args.max_genes_per_pathway,
            max_pathways=args.max_pathways,
            allow_flat_fallback=args.allow_flat_fallback,
        )
        map_dir = out_dir / "index_maps" / run_id
        map_dir.mkdir(parents=True, exist_ok=True)
        np.save(map_dir / "gene_cpg_index.npy", gene_idx.numpy())
        np.save(map_dir / "pathway_gene_index.npy", pathway_idx.numpy())
        gene_map.to_csv(map_dir / "gene_mapping.tsv", sep="\t", index=False)
        pathway_map.to_csv(map_dir / "pathway_mapping.tsv", sep="\t", index=False)
        pd.DataFrame({"probe_index": selected_idx, "probe_id": selected_probe_ids}).to_csv(
            map_dir / "selected_probe_ids.tsv", sep="\t", index=False
        )

        model = CPGGenePathwayAttentionNet(
            gene_cpg_index=gene_idx,
            pathway_gene_index=pathway_idx,
            num_classes=n_classes,
            num_features=len(selected_probe_ids),
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        ).to(device)
        class_counts = np.bincount(y_train, minlength=n_classes).astype(float)
        class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
        class_weights = class_weights / np.mean(class_weights[class_counts > 0])
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        train_loader = make_loader(x_train_selected, y_train, args.batch_size, shuffle=True, soft_targets=train_teacher_proba)
        val_loader = make_loader(x_val_selected, y_val, args.batch_size, shuffle=False)
        test_loader = make_loader(x_test_selected, y_test, args.batch_size, shuffle=False)

        best_state: dict[str, torch.Tensor] | None = None
        best_val_f1 = -1.0
        stale = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses: list[float] = []
            for batch in train_loader:
                xb = batch[0]
                yb = batch[1]
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)["class_logits"]
                loss = criterion(logits, yb)
                if len(batch) == 3 and args.distill_alpha > 0:
                    teacher_batch = batch[2].to(device)
                    loss = loss + args.distill_alpha * distillation_kl_loss(
                        logits, teacher_batch, args.distill_temperature
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.item()))
            val_metrics, _, _, _ = evaluate(model, val_loader, device, n_classes)
            history_rows.append(
                {
                    "run_id": run_id,
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)) if losses else float("nan"),
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                }
            )
            if args.log_every_epoch and (epoch == 1 or epoch % args.log_every_epoch == 0):
                print(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "epoch": epoch,
                            "train_loss": float(np.mean(losses)) if losses else float("nan"),
                            "val_loss": val_metrics["loss"],
                            "val_macro_f1": val_metrics["macro_f1"],
                            "best_val_macro_f1": max(best_val_f1, float(val_metrics["macro_f1"])),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if float(val_metrics["macro_f1"]) > best_val_f1:
                best_val_f1 = float(val_metrics["macro_f1"])
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= args.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        test_metrics, y_true, y_pred, y_proba = evaluate(model, test_loader, device, n_classes)
        row = {
            "run_id": run_id,
            "model": "pathmethnet",
            "feature_method": args.feature_method,
            "top_k": args.top_k,
            "repeat": repeat,
            "fold": fold,
            "n_train": int(len(train_rows)),
            "n_val": int(len(val_rows)),
            "n_test": int(len(test_rows)),
            "best_val_macro_f1": best_val_f1,
            "seconds": float(time.time() - start),
            **test_metrics,
        }
        if test_teacher_proba is not None:
            teacher_pred = test_teacher_proba.argmax(axis=1)
            teacher_metrics = multiclass_metrics(y_test, teacher_pred, test_teacher_proba)
            row["teacher_macro_f1"] = teacher_metrics["macro_f1"]
            row["teacher_accuracy"] = teacher_metrics["accuracy"]
            row["teacher_balanced_accuracy"] = teacher_metrics["balanced_accuracy"]
        fold_rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        pred_df = uncertainty_frame(y_true, y_proba, label_classes)
        pred_df.insert(0, "run_id", run_id)
        pred_df.to_csv(out_dir / "predictions" / f"{run_id}.tsv", sep="\t", index=False)
        np.savez_compressed(out_dir / "predictions" / f"{run_id}_proba.npz", y_true=y_true, y_pred=y_pred, y_proba=y_proba)

        class_df = per_class_metrics(y_true, y_pred, label_classes)
        class_df.insert(0, "run_id", run_id)
        per_class_frames.append(class_df)
        calibration_rows.append({"run_id": run_id, **calibration_metrics(y_true, y_proba)})

        torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "run_id": run_id}, out_dir / f"{run_id}.pt")
        write_outputs(out_dir, fold_rows, per_class_frames, calibration_rows, history_rows)

    write_outputs(out_dir, fold_rows, per_class_frames, calibration_rows, history_rows)

    manifest = {
        "matrix_npy": str(Path(args.matrix_npy).resolve()),
        "matrix_samples_tsv": str(Path(args.matrix_samples_tsv).resolve()) if args.matrix_samples_tsv else "",
        "probe_ids": str(Path(args.probe_ids).resolve()),
        "splits_dir": str(Path(args.splits_dir).resolve()),
        "feature_method": args.feature_method,
        "top_k": args.top_k,
        "model": "pathmethnet",
        "distill_teacher": args.distill_teacher,
        "distill_alpha": args.distill_alpha,
        "distill_temperature": args.distill_temperature,
        "outputs": ["fold_metrics.tsv", "summary_metrics_by_setting.csv", "training_history.tsv", "predictions/", "index_maps/"],
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
