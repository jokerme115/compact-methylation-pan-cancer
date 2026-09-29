from __future__ import annotations

import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif


def effect_size_one_vs_rest(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Class-imbalance aware one-vs-rest effect-size score per feature."""
    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) < 2:
        return np.zeros(x_train.shape[1], dtype=float)

    scores = np.zeros(x_train.shape[1], dtype=float)
    eps = 1e-8
    for cls, cls_count in zip(classes, counts):
        pos = y_train == cls
        neg = ~pos
        if pos.sum() < 2 or neg.sum() < 2:
            continue
        pos_mean = np.nanmean(x_train[pos], axis=0)
        pos_std = np.nanstd(x_train[pos], axis=0)
        neg_classes = classes[classes != cls]
        neg_weights = []
        neg_means = []
        neg_vars = []
        for neg_cls in neg_classes:
            cls_mask = y_train == neg_cls
            n_cls = cls_mask.sum()
            if n_cls < 2:
                continue
            neg_weights.append(1.0 / max(n_cls, 1))
            neg_means.append(np.nanmean(x_train[cls_mask], axis=0))
            neg_vars.append(np.nanvar(x_train[cls_mask], axis=0))
        if not neg_weights:
            neg_mean = np.nanmean(x_train[neg], axis=0)
            neg_std = np.nanstd(x_train[neg], axis=0)
        else:
            weights = np.asarray(neg_weights, dtype=float)
            weights = weights / weights.sum()
            neg_mean = np.average(np.vstack(neg_means), axis=0, weights=weights)
            neg_std = np.sqrt(np.average(np.vstack(neg_vars), axis=0, weights=weights))
        cls_score = np.abs(pos_mean - neg_mean) / (pos_std + neg_std + eps)
        class_weight = 1.0 / max(float(cls_count), 1.0)
        scores = np.maximum(scores, cls_score * class_weight)
    return np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(float).max, neginf=-np.inf)


def select_features(
    x_train: np.ndarray,
    y_train: np.ndarray,
    method: str,
    top_k: int,
    seed: int = 42,
) -> np.ndarray:
    method_key = method.lower()
    n_features = x_train.shape[1]
    k = min(int(top_k), n_features)
    if k <= 0:
        raise ValueError("top_k must be positive")

    if method_key == "variance":
        scores = np.nanvar(x_train, axis=0)
    elif method_key in {"anova", "f_classif", "ftest"}:
        scores, _ = f_classif(x_train, y_train)
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(float).max, neginf=-np.inf)
    elif method_key in {"mutual_info", "mi"}:
        scores = mutual_info_classif(x_train, y_train, random_state=seed, discrete_features=False)
        scores = np.nan_to_num(scores, nan=-np.inf)
    elif method_key in {"effect_size", "ovr_effect_size", "one_vs_rest_effect_size"}:
        scores = effect_size_one_vs_rest(x_train, y_train)
    else:
        raise ValueError(f"Unsupported feature selection method: {method}")

    selected = np.argpartition(scores, -k)[-k:]
    selected = selected[np.argsort(scores[selected])[::-1]]
    return selected.astype(int)


def apply_feature_selection(x: np.ndarray, selected_idx: np.ndarray) -> np.ndarray:
    return x[:, selected_idx]
