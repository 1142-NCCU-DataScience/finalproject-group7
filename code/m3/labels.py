"""
M3 — Label generation.

Architecture §5.2: y_{i,t} = 1 if shortage_rate at t+60min > 0.8, else 0.

The lookahead is 6 ticks (10-minute granularity × 6 = 60 minutes). Per
station, the last 6 rows have no lookahead available and produce NaN
labels; M4 drops these before training.
"""

from __future__ import annotations

import pandas as pd

SHORTAGE_THRESHOLD = 0.8
LOOKAHEAD_TICKS = 6


def add_shortage_label(
    df: pd.DataFrame,
    *,
    threshold: float = SHORTAGE_THRESHOLD,
    lookahead_ticks: int = LOOKAHEAD_TICKS,
) -> pd.DataFrame:
    """Add a binary `y` column = (shortage_rate at t+lookahead > threshold).

    Parameters
    ----------
    df : DataFrame
        Must contain columns: sno, timestamp, shortage_rate.
        Should already be sorted by (sno, timestamp) — caller's responsibility,
        identical to the contract for `build_features`.
    threshold : float
        Shortage threshold (default 0.8 per §5.2).
    lookahead_ticks : int
        Number of ticks to look ahead. Default 6 = 60 minutes at 10-min granularity.

    Returns
    -------
    DataFrame
        Original frame plus columns:
            future_shortage_rate : shortage_rate at t + lookahead_ticks (NaN at tail)
            y                    : Int64 nullable, 1/0/NaN
    """
    out = df.copy()
    future = out.groupby("sno")["shortage_rate"].shift(-lookahead_ticks)
    out["future_shortage_rate"] = future
    # Use nullable Int64 so the tail-NaNs survive (plain int would coerce 1.0/0.0)
    out["y"] = (future > threshold).astype("boolean").astype("Int64")
    out.loc[future.isna(), "y"] = pd.NA
    return out
