"""Unsupervised clustering of codas and evaluation against expert labels.

Strategy: cluster separately within each click count. Click count is a hard,
directly observable property, so there is no need to "discover" it; the
open question is how many rhythm/tempo groups exist *within* each count.

The number of clusters is chosen by BIC, never by looking at the labels.
Labels are used only afterwards, to evaluate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .data import features_for_click_count


@dataclass
class ClickGroupResult:
    n_clicks: int
    n_codas: int
    k_selected: int
    bic_curve: list[float] = field(default_factory=list)


def fit_gmm_bic(
    X: np.ndarray, k_max: int = 8, seed: int = 0, n_init: int = 3
) -> tuple[GaussianMixture, list[float]]:
    """Fit GMMs with k = 1..k_max and return the one with the lowest BIC."""
    k_max = max(1, min(k_max, len(X) // 10))  # need enough points per component
    models, bics = [], []
    for k in range(1, k_max + 1):
        m = GaussianMixture(k, covariance_type="full", random_state=seed, n_init=n_init)
        m.fit(X)
        models.append(m)
        bics.append(m.bic(X))
    return models[int(np.argmin(bics))], bics


def cluster_by_click_count(
    df: pd.DataFrame,
    kind: str = "rhythm+tempo",
    min_group_size: int = 30,
    k_max: int = 8,
    seed: int = 0,
    scale: bool = True,
) -> tuple[np.ndarray, list[ClickGroupResult]]:
    """Cluster codas within each click count using GMM + BIC.

    Groups smaller than `min_group_size` are kept as a single cluster
    (too few points to estimate a mixture reliably).

    Returns labels like "5_2" (click count _ cluster id) and a per-group summary.
    """
    labels = np.empty(len(df), dtype=object)
    summary: list[ClickGroupResult] = []
    for n in sorted(df["nClicks"].unique()):
        X, idx = features_for_click_count(df, n, kind)
        if len(idx) < min_group_size:
            labels[idx] = f"{n}_0"
            summary.append(ClickGroupResult(n, len(idx), 1))
            continue
        if scale:
            X = StandardScaler().fit_transform(X)
        model, bics = fit_gmm_bic(X, k_max=k_max, seed=seed)
        pred = model.predict(X)
        labels[idx] = [f"{n}_{c}" for c in pred]
        summary.append(ClickGroupResult(n, len(idx), model.n_components, bics))
    return labels, summary


def evaluate(true_labels, pred_labels) -> dict[str, float]:
    """Compare clusters with expert labels.

    homogeneity   1.0 if every cluster contains a single expert type
                  (splitting a type into several clusters is NOT penalised)
    completeness  1.0 if every expert type falls in a single cluster
                  (merging types is NOT penalised)
    NMI           harmonic mean of the two (= V-measure)
    ARI           pair-counting agreement, corrected for chance

    Reading them separately matters: high homogeneity + lower completeness
    means the algorithm finds groups *finer* than the expert labels.
    """
    return {
        "ARI": adjusted_rand_score(true_labels, pred_labels),
        "NMI": normalized_mutual_info_score(true_labels, pred_labels),
        "homogeneity": homogeneity_score(true_labels, pred_labels),
        "completeness": completeness_score(true_labels, pred_labels),
        "n_clusters": len(set(pred_labels)),
        "n_types": len(set(true_labels)),
    }


def seed_stability(df: pd.DataFrame, n_seeds: int = 5, **kwargs) -> np.ndarray:
    """ARI between the seed-0 clustering and clusterings with other seeds."""
    ref, _ = cluster_by_click_count(df, seed=0, **kwargs)
    return np.array(
        [
            adjusted_rand_score(ref, cluster_by_click_count(df, seed=s, **kwargs)[0])
            for s in range(1, n_seeds)
        ]
    )
