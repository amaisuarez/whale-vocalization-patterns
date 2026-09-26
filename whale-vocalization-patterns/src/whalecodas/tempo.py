"""Tempo analysis: are there discrete tempo classes within a coda type?

Tempo is modelled as log(duration), because duration varies multiplicatively
(a coda 20% slower is 20% slower whether it lasts 0.3 s or 1.3 s).

Three safeguards against false "classes":
1. BIC chooses the number of Gaussian components k.
2. Ashman's D checks that adjacent components are genuinely separated
   (D > 2) and not just several Gaussians fitting one skewed peak.
3. Robustness is checked with a *session* bootstrap: codas recorded in the
   same session (same social unit, same day) are not independent, so whole
   sessions are resampled rather than individual codas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def add_session(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `session` column = recording day + social unit.

    This is the closest available proxy for an independent observation.
    """
    return df.assign(session=df["Date"].dt.strftime("%Y-%m-%d") + "_" + df["Unit"])


def log_duration(df: pd.DataFrame) -> np.ndarray:
    return np.log(df["Duration"].to_numpy())[:, None]


def fit_tempo_mixture(
    x: np.ndarray, k_max: int = 4, min_per_component: int = 30, seed: int = 0, n_init: int = 5
) -> tuple[GaussianMixture, np.ndarray]:
    """Fit 1-D GMMs on log-duration, k = 1..k_max, and return the BIC-best.

    k_max is capped so each component has on average >= min_per_component codas.
    Returns (model, delta_bic) where delta_bic[k-1] = BIC(k) - BIC(best).
    """
    k_max = max(1, min(k_max, len(x) // min_per_component))
    models = [
        GaussianMixture(k, random_state=seed, n_init=n_init).fit(x) for k in range(1, k_max + 1)
    ]
    bics = np.array([m.bic(x) for m in models])
    return models[int(np.argmin(bics))], bics - bics.min()


def sorted_components(model: GaussianMixture) -> pd.DataFrame:
    """Components ordered by centre, in seconds."""
    mu = model.means_.ravel()
    var = model.covariances_.ravel()
    order = np.argsort(mu)
    return pd.DataFrame(
        {"centre_s": np.exp(mu[order]), "weight": model.weights_[order], "log_sd": np.sqrt(var[order])}
    )


def ashman_d(model: GaussianMixture) -> list[float]:
    """Ashman's D between adjacent components (D > 2 = clean separation)."""
    comp = sorted_components(model)
    mu, sd = np.log(comp["centre_s"].to_numpy()), comp["log_sd"].to_numpy()
    return [
        float(np.sqrt(2) * abs(mu[i + 1] - mu[i]) / np.sqrt(sd[i] ** 2 + sd[i + 1] ** 2))
        for i in range(len(mu) - 1)
    ]


def session_bootstrap_k(
    df: pd.DataFrame, n_boot: int = 100, k_max: int = 4, seed: int = 0
) -> np.ndarray:
    """Distribution of the BIC-selected k when whole sessions are resampled.

    `df` must contain one coda type and a `session` column (see add_session).
    """
    rng = np.random.default_rng(seed)
    groups = {s: np.log(g["Duration"].to_numpy()) for s, g in df.groupby("session")}
    sessions = np.array(list(groups))
    ks = []
    for _ in range(n_boot):
        pick = rng.choice(sessions, size=len(sessions), replace=True)
        x = np.concatenate([groups[s] for s in pick])[:, None]
        ks.append(fit_tempo_mixture(x, k_max=k_max, n_init=3)[0].n_components)
    return np.array(ks)


def tempo_scan(df: pd.DataFrame, min_codas: int = 30, n_boot: int = 100, seed: int = 0) -> pd.DataFrame:
    """Run the full tempo test on every coda type with at least `min_codas` codas."""
    df = df if "session" in df else add_session(df)
    rows = []
    for coda_type, g in df.groupby("CodaType"):
        if len(g) < min_codas:
            continue
        model, _ = fit_tempo_mixture(log_duration(g))
        d = ashman_d(model)
        ks = session_bootstrap_k(g, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "type": coda_type,
                "nClicks": int(g["nClicks"].iloc[0]),
                "codas": len(g),
                "sessions": g["session"].nunique(),
                "k_BIC": model.n_components,
                "centres_s": np.round(sorted_components(model)["centre_s"].to_numpy(), 3).tolist(),
                "min_ashman_D": round(min(d), 2) if d else np.nan,
                "P(k>=2) sessions": float(np.mean(ks >= 2)),
                "P(k=k_BIC) sessions": float(np.mean(ks == model.n_components)),
            }
        )
    return pd.DataFrame(rows).sort_values("codas", ascending=False).reset_index(drop=True)
