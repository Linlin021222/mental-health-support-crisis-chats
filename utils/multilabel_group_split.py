"""Deterministic user-disjoint stratification for sparse multi-label targets.

``StratifiedGroupKFold`` can preserve one multiclass target, but Task 2 has 24
simultaneous and very imbalanced targets.  The helpers below first aggregate
posts by user, then apply the iterative stratification algorithm to augmented
user targets.  Besides label presence, the augmented targets encode users with
multiple posts for a label, risk-class presence and large user timelines.  No
user can occur in more than one fold.
"""
from __future__ import annotations

import numpy as np


def _iterative_assignment(targets: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """Assign binary multi-label rows using the Sechidis-style greedy rule."""
    targets = np.asarray(targets, dtype=np.int8)
    if targets.ndim != 2:
        raise ValueError("targets must be a two-dimensional binary matrix")
    n_rows = len(targets)
    if n_rows < n_splits:
        raise ValueError("number of groups must be at least n_splits")

    rng = np.random.default_rng(seed)
    desired_rows = np.full(n_splits, n_rows / n_splits, dtype=np.float64)
    desired_labels = np.repeat(
        (targets.sum(axis=0, dtype=np.float64) / n_splits)[None, :],
        n_splits, axis=0,
    )
    remaining = np.ones(n_rows, dtype=bool)
    assignment = np.full(n_rows, -1, dtype=np.int16)

    while remaining.any():
        remaining_counts = targets[remaining].sum(axis=0)
        positive_labels = np.flatnonzero(remaining_counts > 0)
        if not len(positive_labels):
            rows = np.flatnonzero(remaining)
            rng.shuffle(rows)
            for row in rows:
                best = np.flatnonzero(desired_rows == desired_rows.max())
                fold = int(rng.choice(best))
                assignment[row] = fold
                remaining[row] = False
                desired_rows[fold] -= 1.0
            break

        rare_count = remaining_counts[positive_labels].min()
        rare_labels = positive_labels[remaining_counts[positive_labels] == rare_count]
        label = int(rng.choice(rare_labels))
        rows = np.flatnonzero(remaining & (targets[:, label] > 0))
        rng.shuffle(rows)
        for row in rows:
            label_need = desired_labels[:, label]
            candidates = np.flatnonzero(label_need == label_need.max())
            row_need = desired_rows[candidates]
            candidates = candidates[row_need == row_need.max()]
            fold = int(rng.choice(candidates))
            assignment[row] = fold
            remaining[row] = False
            desired_rows[fold] -= 1.0
            desired_labels[fold] -= targets[row]

    return assignment


def multilabel_group_folds(
    factors: np.ndarray,
    groups: np.ndarray,
    risk: np.ndarray | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return post indices for factor-balanced, strictly group-disjoint folds."""
    factors = np.asarray(factors, dtype=np.int8)
    groups = np.asarray(groups).astype(str)
    if len(factors) != len(groups):
        raise ValueError("factors and groups must have the same number of rows")

    unique_groups, inverse = np.unique(groups, return_inverse=True)
    n_groups, n_labels = len(unique_groups), factors.shape[1]
    factor_counts = np.zeros((n_groups, n_labels), dtype=np.int16)
    group_sizes = np.bincount(inverse, minlength=n_groups)
    np.add.at(factor_counts, inverse, factors)

    # Presence preserves rare labels. Count thresholds also distribute users
    # who contribute many positive posts instead of balancing users alone.
    augmented = [factor_counts >= threshold for threshold in (1, 2, 4, 8)]
    if risk is not None:
        risk = np.asarray(risk, dtype=np.int64)
        n_risk = int(risk.max()) + 1
        risk_counts = np.zeros((n_groups, n_risk), dtype=np.int16)
        np.add.at(risk_counts, (inverse, risk), 1)
        augmented.extend([risk_counts >= 1, risk_counts >= 3])
    for threshold in np.unique(np.quantile(group_sizes, (0.25, 0.5, 0.75)).astype(int)):
        augmented.append((group_sizes >= max(1, int(threshold)))[:, None])

    group_targets = np.concatenate(augmented, axis=1).astype(np.int8)
    membership = _iterative_assignment(group_targets, n_splits, seed)
    row_membership = membership[inverse]
    indices = np.arange(len(groups))
    folds = []
    for fold in range(n_splits):
        valid = indices[row_membership == fold]
        train = indices[row_membership != fold]
        if set(groups[train]).intersection(groups[valid]):
            raise RuntimeError("group leakage detected in multilabel split")
        folds.append((train, valid))
    return folds


def split_audit(
    folds: list[tuple[np.ndarray, np.ndarray]], factors: np.ndarray, groups: np.ndarray
) -> dict:
    """Summarise post/user support and zero-positive labels in each fold."""
    factors = np.asarray(factors, dtype=np.int8)
    groups = np.asarray(groups).astype(str)
    rows = []
    for fold, (_, valid) in enumerate(folds):
        support = factors[valid].sum(axis=0).astype(int)
        rows.append({
            "fold": fold,
            "posts": int(len(valid)),
            "users": int(len(np.unique(groups[valid]))),
            "zero_positive_labels": int((support == 0).sum()),
            "factor_support": support.tolist(),
        })
    support_matrix = np.asarray([row["factor_support"] for row in rows], dtype=float)
    mean = support_matrix.mean(axis=0)
    coefficient_of_variation = np.divide(
        support_matrix.std(axis=0), mean,
        out=np.zeros_like(mean), where=mean > 0,
    )
    return {
        "folds": rows,
        "total_zero_positive_fold_labels": int((support_matrix == 0).sum()),
        "mean_factor_support_cv": float(coefficient_of_variation.mean()),
        "max_factor_support_cv": float(coefficient_of_variation.max()),
    }
