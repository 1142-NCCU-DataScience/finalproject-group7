"""
M3 Online Inference — per-station ring buffer for lag features.

LagBuffer maintains the last 6 shortage_rate readings (60 minutes) for
each station (sno). It produces the same lag/delta values as the offline
groupby-shift path in build_features.py, allowing byte-for-byte identical
feature vectors during live inference.

The buffer is pickle-serializable so the Kafka Producer can persist and
restore it across restarts without replaying history.
"""

from __future__ import annotations

import math
import pickle
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from m3.build_features import (
    FEATURE_COLUMNS,
    _time_features,
    validate_features,
)

_MAXLEN = 7  # current reading + 6 historical = lag_60min requires offset 6 → index n-7


class LagBuffer:
    """Per-station ring buffer for online lag feature computation.

    Usage (M1 integration)
    ----------------------
    buf = LagBuffer()

    # Each time a new snapshot arrives for a station:
    buf.append(sno, shortage_rate)
    lags = buf.get_lags(sno)  # {"lag_10min": ..., ...}

    # Build full feature vector for a single snapshot:
    X = buf.build_online_features(snapshot_df, station_static, spatial_lag)
    """

    def __init__(self) -> None:
        # sno -> deque of up to 6 floats, newest appended to the right
        self._buffers: dict[str, deque[float]] = {}

    # ------------------------------------------------------------------
    # Core buffer operations
    # ------------------------------------------------------------------

    def append(self, sno: str, shortage_rate: float) -> None:
        """Record a new shortage_rate observation for a station."""
        if sno not in self._buffers:
            self._buffers[sno] = deque(maxlen=_MAXLEN)
        self._buffers[sno].append(float(shortage_rate))

    def get_lags(self, sno: str) -> dict[str, float]:
        """Return lag values for a station.

        The buffer stores history oldest-left, newest-right.
        After append(), the current value is already in the buffer at
        position [-1]; lag_10min is the value one step before → [-2].

        If fewer than k+1 readings exist, the k-th lag is np.nan.
        """
        buf = list(self._buffers.get(sno, []))
        n = len(buf)

        def _get(offset: int) -> float:
            # offset=1 → one step before the most recent (lag_10min)
            idx = n - 1 - offset
            return float(buf[idx]) if idx >= 0 else np.nan

        return {
            "lag_10min": _get(1),
            "lag_20min": _get(2),
            "lag_30min": _get(3),
            "lag_60min": _get(6),
        }

    # ------------------------------------------------------------------
    # Full feature vector construction (mirrors build_features offline path)
    # ------------------------------------------------------------------

    def build_online_features(
        self,
        snapshot_df: pd.DataFrame,
        station_static: pd.DataFrame,
        spatial_lag: pd.Series | pd.DataFrame,
    ) -> pd.DataFrame:
        """Build a 17-column feature DataFrame for the current snapshot.

        Parameters
        ----------
        snapshot_df:
            One row per station for the current 10-minute tick.
            Required columns: sno, timestamp, shortage_rate,
            available_bikes, total_capacity.
            NOTE: This method does NOT call append() — the caller must
            call buf.append(sno, shortage_rate) before this method so
            that the current reading is already in the buffer.
        station_static:
            DataFrame with at least columns: sno, distance_to_mrt.
            In production this is `pd.read_csv("data/youbike_station.csv")`.
        spatial_lag:
            Series or DataFrame providing spatial_lag_shortage aligned
            to snapshot_df's index.

        Returns
        -------
        pd.DataFrame with exactly FEATURE_COLUMNS in order.
        """
        df = snapshot_df.copy().reset_index(drop=True)

        # --- Lag features from buffer ---
        lag_records = [self.get_lags(str(sno)) for sno in df["sno"]]
        lag_df = pd.DataFrame(lag_records, index=df.index)
        for col in ["lag_10min", "lag_20min", "lag_30min", "lag_60min"]:
            df[col] = lag_df[col]

        # --- Rate-of-change ---
        df["delta_10min"] = df["shortage_rate"] - df["lag_10min"]
        df["delta_30min"] = df["shortage_rate"] - df["lag_30min"]

        # --- Time features ---
        time_df = _time_features(df["timestamp"])
        for col in time_df.columns:
            df[col] = time_df[col].values

        # --- Spatial lag ---
        if isinstance(spatial_lag, pd.DataFrame):
            spatial_series = spatial_lag["spatial_lag_shortage"]
        else:
            spatial_series = spatial_lag
        df["spatial_lag_shortage"] = (
            spatial_series.values if hasattr(spatial_series, "values") else spatial_series
        )

        # --- Static attribute ---
        df = df.merge(
            station_static[["sno", "distance_to_mrt"]],
            on="sno",
            how="left",
        )

        result = df[FEATURE_COLUMNS].copy()
        validate_features(result)
        return result

    # ------------------------------------------------------------------
    # High-level online entry point (single tick)
    # ------------------------------------------------------------------

    def tick(
        self,
        snapshot_df: pd.DataFrame,
        spatial_lag: pd.Series | pd.DataFrame,
        station_static: pd.DataFrame,
        *,
        strict: bool = True,
    ) -> pd.DataFrame:
        """One online tick: append, build features, validate.

        Convenience wrapper for M1's main loop so the integration is a
        single call instead of three (append + build + validate).

        Example
        -------
            >>> snapshot = clean_snapshot(api_records)              # M1
            >>> spatial = m2.compute_lisa(snapshot, station_static) # M2
            >>> out = lag_buffer.tick(snapshot, spatial, station_static)
            >>> X = out[FEATURE_COLUMNS]
            >>> probs = predictor.predict_proba(X)                  # M4

        Parameters
        ----------
        snapshot_df : DataFrame
            Output of `src.m1.cleaner.clean_snapshot`. Required columns:
            sno, timestamp, shortage_rate, available_bikes, total_capacity.
        spatial_lag : Series or DataFrame
            M2 output. If a DataFrame, must contain a `spatial_lag_shortage`
            column; otherwise a Series of values aligned to snapshot_df rows.
        station_static : DataFrame
            Must contain at least sno and distance_to_mrt.
        strict : bool, default True
            Strict NaN validation on non-nullable features. Pass False
            during the per-station 60-minute warm-up if the caller wants
            to tolerate lag NaNs without raising (still rejects NaN in
            spatial_lag / distance_to_mrt / time features).

        Returns
        -------
        DataFrame with columns: [sno, timestamp] + FEATURE_COLUMNS,
        one row per snapshot row (same order).
        """
        for _, row in snapshot_df.iterrows():
            self.append(str(row["sno"]), float(row["shortage_rate"]))

        features = self.build_online_features(snapshot_df, station_static, spatial_lag)
        validate_features(features, strict=strict)

        features.insert(0, "timestamp", snapshot_df["timestamp"].values)
        features.insert(0, "sno", snapshot_df["sno"].astype(str).values)
        return features

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist buffer state to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self._buffers, f)

    @classmethod
    def load(cls, path: str | Path) -> "LagBuffer":
        """Restore buffer state from a pickle file."""
        instance = cls()
        with open(path, "rb") as f:
            instance._buffers = pickle.load(f)
        return instance

    def __repr__(self) -> str:
        return f"LagBuffer(stations={len(self._buffers)}, maxlen={_MAXLEN})"
