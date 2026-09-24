"""Loading, cleaning and feature extraction for the DSWP coda dataset.

A coda is a short sequence of clicks. In `DominicaCodas.csv` each row is one
coda, described by:
    nClicks   number of clicks
    Duration  time from first to last click (s)
    ICI1..9   inter-click intervals (s); unused slots are 0
    CodaType  expert label (e.g. "1+1+3", "5R1"); "*-NOISE" = unclassifiable
    Clan      vocal clan (EC1 or EC2)
    Unit      social unit (letter); IDN = individual whale ID (0 = unknown)

Two representations matter (Sharma et al. 2024):
    rhythm  ICIs divided by Duration -> the "shape" of the coda, speed-free
    tempo   Duration itself          -> how fast the coda is produced
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MAX_ICIS = 9
ICI_COLS = [f"ICI{i}" for i in range(1, MAX_ICIS + 1)]


def load_codas(path: str | Path | None = None) -> pd.DataFrame:
    """Load DominicaCodas.csv with light type cleaning. No rows are dropped."""
    path = Path(path) if path else RAW_DIR / "DominicaCodas.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    df = pd.read_csv(path)
    # Dates mix "/" and "-" separators; all are day-first (verified: the first
    # field exceeds 12 in ~3,900 rows, the second field never does).
    df["Date"] = pd.to_datetime(df["Date"].str.replace("-", "/"), format="%d/%m/%Y")
    df["IDN"] = df["IDN"].astype(str)
    df["is_noise"] = df["CodaType"].str.contains("NOISE")
    df["whale_known"] = ~df["IDN"].isin(["0", "9999"])
    return df


def filter_codas(
    df: pd.DataFrame,
    clan: str | None = "EC1",
    drop_noise: bool = True,
    min_clicks: int = 3,
) -> pd.DataFrame:
    """Standard analysis subset.

    Defaults: EC1 clan only (the clan analysed by Sharma et al.), no NOISE
    labels, and at least 3 clicks (1-2 click codas have no rhythm to speak of).
    """
    out = df
    if clan is not None:
        out = out[out["Clan"] == clan]
    if drop_noise:
        out = out[~out["is_noise"]]
    out = out[out["nClicks"] >= min_clicks]
    return out.reset_index(drop=True)


def ici_matrix(df: pd.DataFrame) -> np.ndarray:
    """Raw ICIs as an (n_codas, 9) array, zero-padded."""
    return df[ICI_COLS].to_numpy(dtype=float)


def rhythm_matrix(df: pd.DataFrame) -> np.ndarray:
    """ICIs normalised by coda duration (each row of used ICIs sums to 1)."""
    return ici_matrix(df) / df["Duration"].to_numpy()[:, None]


def features_for_click_count(
    df: pd.DataFrame, n_clicks: int, kind: str = "rhythm+tempo"
) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix for the codas with exactly `n_clicks` clicks.

    Codas are compared only within the same click count, so vectors have a
    fixed length (n_clicks - 1) and no zero-padding distorts distances.

    kind:
        "rhythm"        normalised ICIs only
        "rhythm+tempo"  normalised ICIs + log(Duration)
        "raw"           absolute ICIs (rhythm and tempo mixed together)

    Returns (X, idx) where idx are the positional row indices in `df`.
    Features are NOT scaled here; scaling is a modelling decision.
    """
    idx = np.flatnonzero(df["nClicks"].to_numpy() == n_clicks)
    k = n_clicks - 1
    if kind == "rhythm":
        X = rhythm_matrix(df)[idx, :k]
    elif kind == "rhythm+tempo":
        X = np.c_[rhythm_matrix(df)[idx, :k], np.log(df["Duration"].to_numpy()[idx])]
    elif kind == "raw":
        X = ici_matrix(df)[idx, :k]
    else:
        raise ValueError(f"unknown kind: {kind!r}")
    return X, idx
