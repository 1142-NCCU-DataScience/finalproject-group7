"""
M2 — Spatial analysis (k-NN weight matrix + Local Moran's I).

Implements §2 and §4.2.2 of the architecture document. Two public
entry points:

    compute_spatial_lag(snapshot_df, station_static) -> pd.Series
        Fast. Builds a k-NN(k=6) row-normalized weight matrix W on
        the snapshot's surviving stations and returns W @ shortage_rate.
        Used by the offline training-set assembler — no permutation
        significance test needed because only the continuous spatial
        lag enters the feature matrix (§2.6).

    compute_lisa(snapshot_df, station_static) -> pd.DataFrame
        Full LISA. Same W, plus 999 conditional permutation tests for
        Local Moran's I, mapped to HH/LL/LH/HL/NS quadrants. Used by
        the online producer to populate `moran_type` and `moran_p_value`
        in predictions/latest.json for the Shiny map.

Design choices
--------------
* W is rebuilt per snapshot because the active station set varies
  with `act` and `Quantity > 0` filtering. K-NN construction on
  ~1750 points is fast (~50 ms via libpysal), so caching across
  snapshots offers little benefit and risks subtle bugs when the
  station set changes.
* Coordinates are taken from `station_static` (data/youbike_station.csv).
  Snapshot rows whose sno has no entry there cause a fail-fast
  ValueError — there is no sensible fallback (a missing station has
  no neighbours).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from esda.moran import Moran_Local
from libpysal.weights import KNN

K_NEIGHBOURS = 6
LISA_PERMUTATIONS = 999
LISA_SIG_LEVEL = 0.05

# PySAL quadrant codes → architecture §2.4 labels
_QUADRANT_MAP = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}


def _build_knn_w(snapshot_df: pd.DataFrame, station_static: pd.DataFrame) -> KNN:
    """Build a row-normalized k-NN(k=6) weight matrix on the snapshot's stations."""
    merged = snapshot_df[["sno"]].merge(
        station_static[["sno", "latitude", "longitude"]],
        on="sno",
        how="left",
    )
    missing = merged["latitude"].isna()
    if missing.any():
        bad = sorted({str(s) for s in merged.loc[missing, "sno"].unique()})
        raise ValueError(
            f"station_static missing lat/lng for {len(bad)} sno(s) in this snapshot: "
            f"{bad[:5]}. Update data/youbike_station.csv."
        )
    coords = merged[["latitude", "longitude"]].to_numpy(dtype=float)
    w = KNN.from_array(coords, k=K_NEIGHBOURS)
    w.transform = "R"  # row-normalize → spatial lag is a weighted average
    return w


def compute_spatial_lag(
    snapshot_df: pd.DataFrame,
    station_static: pd.DataFrame,
) -> pd.Series:
    """W @ shortage_rate for the snapshot. Series aligned with snapshot_df.index."""
    w = _build_knn_w(snapshot_df, station_static)
    x = snapshot_df["shortage_rate"].to_numpy(dtype=float)
    lag = w.sparse @ x
    return pd.Series(lag, index=snapshot_df.index, name="spatial_lag_shortage")


def compute_lisa(
    snapshot_df: pd.DataFrame,
    station_static: pd.DataFrame,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Full Local Moran's I with quadrant + p-value.

    Returns
    -------
    DataFrame with one row per snapshot row (same length, same row order):
        sno                   : str
        spatial_lag_shortage  : float (continuous lag, for ML features)
        moran_type            : str  (HH/LL/LH/HL/NS, for map colouring)
        moran_p_value         : float (pseudo p-value from 999 permutations)
    Rows where p >= 0.05 are labelled "NS" (not significant) regardless
    of their quadrant.
    """
    w = _build_knn_w(snapshot_df, station_static)
    x = snapshot_df["shortage_rate"].to_numpy(dtype=float)

    moran = Moran_Local(x, w, permutations=LISA_PERMUTATIONS, seed=seed)
    spatial_lag = w.sparse @ x

    types = np.array([_QUADRANT_MAP.get(int(q), "NS") for q in moran.q], dtype=object)
    types[moran.p_sim >= LISA_SIG_LEVEL] = "NS"

    return pd.DataFrame(
        {
            "sno": snapshot_df["sno"].astype(str).to_numpy(),
            "spatial_lag_shortage": spatial_lag,
            "moran_type": types,
            "moran_p_value": moran.p_sim,
        },
        index=snapshot_df.index,
    )
