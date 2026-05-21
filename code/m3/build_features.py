"""
M3 Feature Engineering — shared offline + online feature builder.

Offline path  : build_features(df, station_static, spatial_lag)
Online path   : LagBuffer.build_online_features(snapshot_df, station_static, spatial_lag)
Both paths produce identical values for every FEATURE_COLUMNS entry.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

# Default station-static CSV (real data, 1758 Taipei YouBike 2.0 stations).
# File lives at code/features/build_features.py; data/ is at repo root.
STATION_STATIC_CSV = Path(__file__).resolve().parents[2] / "data" / "youbike_station.csv"

# ---------------------------------------------------------------------------
# Single source of truth — column order is locked for training / inference
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: list[str] = [
    # Realtime supply/demand (3) — provided by M1
    "shortage_rate",
    "available_bikes",
    "total_capacity",
    # Time lags in 10-minute multiples (4)
    "lag_10min",
    "lag_20min",
    "lag_30min",
    "lag_60min",
    # Rate-of-change (2)
    "delta_10min",
    "delta_30min",
    # Cyclical time features (6)
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_rush_hour",
    "is_weekend",
    # Spatial lag (1) — provided by M2
    "spatial_lag_shortage",
    # Static attribute (1)
    "distance_to_mrt",
]
assert len(FEATURE_COLUMNS) == 17

# Columns where NaN is NEVER acceptable in online inference.
# Lag/delta columns are intentionally excluded — they are NaN during the
# per-station 60-minute warm-up window. After warm-up they should be populated.
NON_NULLABLE_FEATURES: list[str] = [
    "shortage_rate",
    "available_bikes",
    "total_capacity",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_rush_hour",
    "is_weekend",
    "spatial_lag_shortage",
    "distance_to_mrt",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_RUSH_HOUR_START_AM = 7
_RUSH_HOUR_END_AM = 9    # inclusive start, exclusive end → [7, 9)
_RUSH_HOUR_START_PM = 17
_RUSH_HOUR_END_PM = 19   # [17, 19)


def _time_features(ts: pd.Series) -> pd.DataFrame:
    """Derive cyclical + categorical time features from a Timestamp series.

    Input series must already be in Asia/Taipei time (no tz conversion done here).
    """
    hour = ts.dt.hour
    dow = ts.dt.dayofweek  # Monday=0, Sunday=6

    hour_sin = np.sin(2 * math.pi * hour / 24)
    hour_cos = np.cos(2 * math.pi * hour / 24)
    dow_sin  = np.sin(2 * math.pi * dow / 7)
    dow_cos  = np.cos(2 * math.pi * dow / 7)

    is_weekend   = (dow >= 5).astype(int)
    is_weekday   = (dow < 5).astype(int)
    am_rush      = ((hour >= _RUSH_HOUR_START_AM) & (hour < _RUSH_HOUR_END_AM)).astype(int)
    pm_rush      = ((hour >= _RUSH_HOUR_START_PM) & (hour < _RUSH_HOUR_END_PM)).astype(int)
    is_rush_hour = (is_weekday & (am_rush | pm_rush)).astype(int)

    return pd.DataFrame(
        {
            "hour_sin":     hour_sin.values,
            "hour_cos":     hour_cos.values,
            "dow_sin":      dow_sin.values,
            "dow_cos":      dow_cos.values,
            "is_rush_hour": is_rush_hour.values,
            "is_weekend":   is_weekend.values,
        },
        index=ts.index,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    station_static: pd.DataFrame,
    spatial_lag: pd.Series | pd.DataFrame,
) -> pd.DataFrame:
    """Build the 17-dimensional feature matrix from raw snapshot data.

    Parameters
    ----------
    df:
        Must contain columns: sno, timestamp, shortage_rate,
        available_bikes, total_capacity.
        timestamp must be pd.Timestamp with Asia/Taipei timezone.
        Rows should be sorted by (sno, timestamp) for correct lag computation.
    station_static:
        DataFrame containing at least columns: sno, distance_to_mrt.
        In production this is `pd.read_csv("data/youbike_station.csv")`,
        which also carries latitude/longitude columns (ignored here).
    spatial_lag:
        Series or single-column DataFrame indexed/keyed on the DataFrame's
        index, providing spatial_lag_shortage values aligned to df's index.
        If DataFrame, must have a column named 'spatial_lag_shortage'.

    Returns
    -------
    pd.DataFrame with exactly FEATURE_COLUMNS in order.
    """
    df = df.copy()

    # --- Lag features (offline path: groupby shift, NaN at group boundaries) ---
    grp = df.groupby("sno")["shortage_rate"]
    df["lag_10min"] = grp.shift(1)
    df["lag_20min"] = grp.shift(2)
    df["lag_30min"] = grp.shift(3)
    df["lag_60min"] = grp.shift(6)

    # --- Rate-of-change ---
    df["delta_10min"] = df["shortage_rate"] - df["lag_10min"]
    df["delta_30min"] = df["shortage_rate"] - df["lag_30min"]

    # --- Time features ---
    time_df = _time_features(df["timestamp"])
    for col in time_df.columns:
        df[col] = time_df[col].values

    # --- Spatial lag (from M2) ---
    if isinstance(spatial_lag, pd.DataFrame):
        spatial_series = spatial_lag["spatial_lag_shortage"]
    else:
        spatial_series = spatial_lag
    df["spatial_lag_shortage"] = spatial_series.values if hasattr(spatial_series, "values") else spatial_series

    # --- Static attribute: distance_to_mrt (left join by sno) ---
    df = df.merge(
        station_static[["sno", "distance_to_mrt"]],
        on="sno",
        how="left",
    )

    # Fail-fast: distance_to_mrt is static; NaN here means station_static
    # is missing one or more snos. Catch this in M3 instead of letting it
    # silently propagate into the training set or the XGBoost predictor.
    nan_dist = df["distance_to_mrt"].isna()
    if nan_dist.any():
        bad_snos = sorted({str(s) for s in df.loc[nan_dist, "sno"].unique()})
        raise ValueError(
            f"distance_to_mrt is NaN for {int(nan_dist.sum()):,} rows across "
            f"{len(bad_snos)} sno(s). Sample: {bad_snos[:10]}. "
            f"station_static is missing these snos — "
            f"re-run scripts/add_distance_to_mrt.py or update data/youbike_station.csv."
        )

    # --- Select and reorder to locked column order ---
    result = df[FEATURE_COLUMNS].copy()

    assert result.columns.tolist() == FEATURE_COLUMNS, (
        f"Column order mismatch: {result.columns.tolist()}"
    )
    return result


def load_station_static(path: str | Path | None = None) -> pd.DataFrame:
    """Load the station-static table used as the source of `distance_to_mrt`.

    Defaults to `data/youbike_station.csv` at repo root, which carries
    columns: sno, latitude, longitude, distance_to_mrt.
    """
    return pd.read_csv(path or STATION_STATIC_CSV)


def validate_features(X: pd.DataFrame, *, strict: bool = False) -> None:
    """Validate that X contains exactly FEATURE_COLUMNS in order.

    Intended for use by M4 Predictor before inference.

    Parameters
    ----------
    X : DataFrame
        Feature matrix to validate.
    strict : bool, default False
        When True, additionally raise if any NON_NULLABLE_FEATURES column
        contains NaN. Use strict=True for online inference (every station
        must have a prediction); use strict=False for training-set assembly
        where lag warm-up + label lookahead NaNs are expected and dropped.

    Raises
    ------
    ValueError if columns are missing, extra, or out of order; or, when
    strict=True, if any non-nullable feature contains NaN.
    """
    actual = X.columns.tolist()
    if actual != FEATURE_COLUMNS:
        missing = set(FEATURE_COLUMNS) - set(actual)
        extra   = set(actual) - set(FEATURE_COLUMNS)
        wrong   = [
            f"pos {i}: expected {e!r}, got {a!r}"
            for i, (e, a) in enumerate(zip(FEATURE_COLUMNS, actual))
            if e != a
        ]
        parts = []
        if missing:
            parts.append(f"missing={missing}")
        if extra:
            parts.append(f"extra={extra}")
        if wrong:
            parts.append(f"order_mismatch={wrong}")
        raise ValueError("Feature validation failed: " + "; ".join(parts))

    if strict:
        na_per_col = X[NON_NULLABLE_FEATURES].isna().sum()
        bad = na_per_col[na_per_col > 0]
        if len(bad) > 0:
            details = ", ".join(f"{c}={int(n)}" for c, n in bad.items())
            raise ValueError(
                f"strict validation: NaN in non-nullable features [{details}]. "
                f"Lag/delta columns may be NaN during warm-up — those are not checked. "
                f"Other NaNs indicate an upstream bug (M1 cleaner, M2 spatial_lag, "
                f"or station_static)."
            )
