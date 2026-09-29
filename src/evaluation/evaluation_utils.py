from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize


def _safe_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(numeric):
        return None
    return numeric


def _maybe_import_matplotlib() -> tuple[object | None, str | None]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt, None
    except ModuleNotFoundError as exc:
        return None, f"matplotlib not installed, skipped figure generation: {exc}"
    except Exception as exc:  # pragma: no cover - defensive fallback
        return None, f"failed to initialize matplotlib, skipped figure generation: {exc}"


def compute_summary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float | None]:
    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_proba, labels=np.arange(y_proba.shape[1]))),
    }

    for average in ("macro", "weighted", "micro"):
        precision, recall, f1_value, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=average,
            zero_division=0,
        )
        summary[f"{average}_precision"] = float(precision)
        summary[f"{average}_recall"] = float(recall)
        summary[f"{average}_f1"] = float(f1_value)

    num_classes = int(y_proba.shape[1])
    if num_classes > 3:
        summary["top3_accuracy"] = float(top_k_accuracy_score(y_true, y_proba, k=3, labels=np.arange(num_classes)))
    if num_classes > 5:
        summary["top5_accuracy"] = float(top_k_accuracy_score(y_true, y_proba, k=5, labels=np.arange(num_classes)))

    for multi_class in ("ovr", "ovo"):
        for average in ("macro", "weighted"):
            key = f"roc_auc_{multi_class}_{average}"
            try:
                summary[key] = float(roc_auc_score(y_true, y_proba, multi_class=multi_class, average=average))
            except ValueError:
                summary[key] = None

    return summary


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    label_classes: list[str],
) -> pd.DataFrame:
    labels = np.arange(len(label_classes))
    precision, recall, f1_value, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    confusion = confusion_matrix(y_true, y_pred, labels=labels)

    y_true_binarized = label_binarize(y_true, classes=labels)
    if y_true_binarized.ndim == 1:
        y_true_binarized = y_true_binarized.reshape(-1, 1)

    auc_values: list[float | None] = []
    for class_idx in labels:
        binary_true = y_true_binarized[:, class_idx]
        if binary_true.min() == binary_true.max():
            auc_values.append(None)
            continue
        auc_values.append(_safe_float(roc_auc_score(binary_true, y_proba[:, class_idx])))

    row_sums = confusion.sum(axis=1)
    correct_counts = np.diag(confusion)
    per_class_accuracy = np.divide(
        correct_counts,
        row_sums,
        out=np.zeros_like(correct_counts, dtype=float),
        where=row_sums != 0,
    )

    return pd.DataFrame(
        {
            "label": label_classes,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_value,
            "support": support.astype(int),
            "per_class_accuracy": per_class_accuracy,
            "roc_auc_ovr": auc_values,
        }
    )


def build_prediction_frame(
    sample_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    label_classes: list[str],
) -> pd.DataFrame:
    predictions_df = sample_df.reset_index(drop=True).copy()
    predictions_df["true_label"] = [label_classes[idx] for idx in y_true]
    predictions_df["pred_label"] = [label_classes[idx] for idx in y_pred]
    predictions_df["is_correct"] = y_true == y_pred
    predictions_df["pred_probability"] = y_proba.max(axis=1)
    predictions_df["true_label_probability"] = y_proba[np.arange(len(y_true)), y_true]

    top_indices = np.argsort(-y_proba, axis=1)
    top_limit = min(3, len(label_classes))
    for rank in range(top_limit):
        class_indices = top_indices[:, rank]
        predictions_df[f"top{rank + 1}_label"] = [label_classes[idx] for idx in class_indices]
        predictions_df[f"top{rank + 1}_probability"] = y_proba[np.arange(len(y_true)), class_indices]

    return predictions_df


def save_matrix_csv(matrix: np.ndarray, labels: list[str], output_path: Path) -> None:
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(output_path, encoding="utf-8-sig")


def plot_training_history(history_df: pd.DataFrame, output_path: Path) -> None:
    plt, warning = _maybe_import_matplotlib()
    if plt is None:
        raise RuntimeError(warning)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], color="#c84b31", linewidth=2)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)

    axes[1].plot(history_df["epoch"], history_df["train_accuracy"], color="#1f6f8b", linewidth=2)
    axes[1].set_title("Training Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _annotate_heatmap(ax: object, matrix: np.ndarray, is_normalized: bool) -> None:
    if matrix.shape[0] > 20:
        return
    threshold = float(matrix.max()) / 2.0 if matrix.size else 0.0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            text = f"{value:.2f}" if is_normalized else f"{int(value)}"
            color = "white" if value > threshold else "black"
            ax.text(col_idx, row_idx, text, ha="center", va="center", color=color, fontsize=8)


def plot_confusion_matrices(raw_matrix: np.ndarray, norm_matrix: np.ndarray, labels: list[str], output_path: Path) -> None:
    plt, warning = _maybe_import_matplotlib()
    if plt is None:
        raise RuntimeError(warning)

    side = max(7.5, 0.55 * len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(side * 2, side))

    im0 = axes[0].imshow(raw_matrix, cmap="Blues")
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].set_yticklabels(labels)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    _annotate_heatmap(axes[0], raw_matrix, is_normalized=False)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(norm_matrix, cmap="Oranges", vmin=0.0, vmax=1.0)
    axes[1].set_title("Normalized Confusion Matrix")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].set_yticklabels(labels)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    _annotate_heatmap(axes[1], norm_matrix, is_normalized=True)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_metrics(per_class_df: pd.DataFrame, output_path: Path) -> None:
    plt, warning = _maybe_import_matplotlib()
    if plt is None:
        raise RuntimeError(warning)

    plot_df = per_class_df.sort_values("f1_score", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(plot_df))
    height = max(5.5, 0.55 * len(plot_df))
    fig, axes = plt.subplots(1, 2, figsize=(15, height))

    bar_width = 0.22
    axes[0].barh(y_pos - bar_width, plot_df["precision"], height=bar_width, label="Precision", color="#1f77b4")
    axes[0].barh(y_pos, plot_df["recall"], height=bar_width, label="Recall", color="#ff7f0e")
    axes[0].barh(y_pos + bar_width, plot_df["f1_score"], height=bar_width, label="F1", color="#2ca02c")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(plot_df["label"])
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_xlabel("Score")
    axes[0].set_title("Per-class Precision / Recall / F1")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].legend(loc="lower right")

    axes[1].barh(y_pos, plot_df["support"], color="#6a4c93")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(plot_df["label"])
    axes[1].set_xlabel("Samples")
    axes[1].set_title("Per-class Support")
    axes[1].grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_auc(per_class_df: pd.DataFrame, output_path: Path) -> bool:
    if per_class_df["roc_auc_ovr"].notna().sum() == 0:
        return False

    plt, warning = _maybe_import_matplotlib()
    if plt is None:
        raise RuntimeError(warning)

    plot_df = per_class_df.dropna(subset=["roc_auc_ovr"]).sort_values("roc_auc_ovr", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(plot_df))
    height = max(5.0, 0.5 * len(plot_df))
    fig, ax = plt.subplots(figsize=(9.5, height))
    ax.barh(y_pos, plot_df["roc_auc_ovr"], color="#1982c4")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("ROC AUC")
    ax.set_title("Per-class One-vs-Rest ROC AUC")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def write_evaluation_summary(
    output_path: Path,
    model_name: str,
    label_column: str,
    summary_metrics: dict[str, float | None],
    plots: dict[str, str],
) -> None:
    lines = [
        "# Evaluation Summary",
        "",
        f"- Model: `{model_name}`",
        f"- Label column: `{label_column}`",
        "",
        "## Key Metrics",
        "",
    ]

    ordered_keys = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "mcc",
        "cohen_kappa",
        "log_loss",
        "top3_accuracy",
        "top5_accuracy",
        "roc_auc_ovr_macro",
        "roc_auc_ovr_weighted",
        "roc_auc_ovo_macro",
        "roc_auc_ovo_weighted",
    ]
    for key in ordered_keys:
        if key not in summary_metrics:
            continue
        value = summary_metrics[key]
        lines.append(f"- `{key}`: {value:.6f}" if value is not None else f"- `{key}`: N/A")

    lines.extend(["", "## Figures", ""])
    if plots:
        for name, relative_path in plots.items():
            lines.append(f"- `{name}`: `{relative_path}`")
    else:
        lines.append("- No figures generated.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_evaluation_bundle(
    out_dir: Path,
    model_name: str,
    label_column: str,
    label_classes: list[str],
    history_df: pd.DataFrame,
    test_sample_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, object]:
    summary_metrics = compute_summary_metrics(y_true, y_pred, y_proba)
    per_class_df = compute_per_class_metrics(y_true, y_pred, y_proba, label_classes)

    labels = np.arange(len(label_classes))
    raw_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    norm_matrix = raw_matrix.astype(float)
    row_sums = norm_matrix.sum(axis=1, keepdims=True)
    norm_matrix = np.divide(norm_matrix, row_sums, out=np.zeros_like(norm_matrix), where=row_sums != 0)

    predictions_df = build_prediction_frame(test_sample_df, y_true, y_pred, y_proba, label_classes)

    history_path = out_dir / "training_history.csv"
    per_class_path = out_dir / "per_class_metrics.csv"
    test_predictions_path = out_dir / "test_predictions.tsv"
    confusion_raw_path = out_dir / "confusion_matrix.csv"
    confusion_norm_path = out_dir / "confusion_matrix_normalized.csv"
    summary_path = out_dir / "evaluation_summary.md"
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")
    per_class_df.to_csv(per_class_path, index=False, encoding="utf-8-sig")
    predictions_df.to_csv(test_predictions_path, sep="\t", index=False, encoding="utf-8-sig")
    save_matrix_csv(raw_matrix, label_classes, confusion_raw_path)
    save_matrix_csv(norm_matrix, label_classes, confusion_norm_path)

    plots: dict[str, str] = {}
    warnings: list[str] = []
    plt, warning = _maybe_import_matplotlib()
    if plt is None:
        if warning is not None:
            warnings.append(warning)
    else:
        try:
            training_plot_path = figures_dir / "training_curves.png"
            plot_training_history(history_df, training_plot_path)
            plots["training_curves"] = str(training_plot_path.relative_to(out_dir))

            confusion_plot_path = figures_dir / "confusion_matrices.png"
            plot_confusion_matrices(raw_matrix, norm_matrix, label_classes, confusion_plot_path)
            plots["confusion_matrices"] = str(confusion_plot_path.relative_to(out_dir))

            per_class_plot_path = figures_dir / "per_class_metrics.png"
            plot_per_class_metrics(per_class_df, per_class_plot_path)
            plots["per_class_metrics"] = str(per_class_plot_path.relative_to(out_dir))

            per_class_auc_path = figures_dir / "per_class_auc.png"
            if plot_per_class_auc(per_class_df, per_class_auc_path):
                plots["per_class_auc"] = str(per_class_auc_path.relative_to(out_dir))
        except Exception as exc:  # pragma: no cover - defensive fallback
            warnings.append(f"figure generation failed: {exc}")

    write_evaluation_summary(summary_path, model_name, label_column, summary_metrics, plots)

    return {
        "summary_metrics": summary_metrics,
        "outputs": {
            "training_history_csv": str(history_path.relative_to(out_dir)),
            "per_class_metrics_csv": str(per_class_path.relative_to(out_dir)),
            "test_predictions_tsv": str(test_predictions_path.relative_to(out_dir)),
            "confusion_matrix_csv": str(confusion_raw_path.relative_to(out_dir)),
            "confusion_matrix_normalized_csv": str(confusion_norm_path.relative_to(out_dir)),
            "evaluation_summary_md": str(summary_path.relative_to(out_dir)),
            "figures": plots,
        },
        "warnings": warnings,
    }
