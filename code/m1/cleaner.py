"""
M1 — Cleaning + shortage_rate.

Architecture §3.4 — filter Quantity > 0 and act == '1', then compute
`shortage_rate = 1 - available_rent_bikes / Quantity`.

The cleaner is path-agnostic: it works on a single snapshot or on the
long-format history frame produced by `load_history`. Output columns
match what `features.build_features` expects: sno, timestamp,
shortage_rate, available_bikes, total_capacity.

If a caller passes a `stats` dict, the cleaner populates it with
per-stage drop / NaN / clip counts so the driver can produce a
forensic report instead of a single opaque "dropped N rows" line.
"""

from __future__ import annotations

import pandas as pd


# Output column names align with what M3 build_features consumes.
M3_INPUT_COLUMNS = [
    "sno",
    "timestamp",
    "shortage_rate",
    "available_bikes",
    "total_capacity",
]


def clean_snapshot(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "tick",
    stats: dict | None = None,
) -> pd.DataFrame:
    """Clean a single snapshot (or stacked snapshots) and compute shortage_rate.

    Parameters
    ----------
    df : DataFrame
        Must contain: sno, act, Quantity, available_rent_bikes, and the
        column named by `timestamp_col` (tz-aware Asia/Taipei).
    timestamp_col : str
        Source column for the canonical tick timestamp.
    stats : dict, optional
        If provided, populated in-place with diagnostic counts:
            input_rows, na_act, na_quantity, na_available,
            dropped_act_not_1, dropped_qty_invalid, kept,
            clip_negative_count.

    Returns
    -------
    DataFrame
        Columns: sno, timestamp, shortage_rate, available_bikes, total_capacity.
        Rows where `act != '1'` or `Quantity <= 0` (or NaN) are dropped.
    """
    if timestamp_col not in df.columns:
        raise KeyError(f"timestamp column {timestamp_col!r} not in DataFrame")

    qty_num = pd.to_numeric(df["Quantity"], errors="coerce")
    avail_num = pd.to_numeric(df["available_rent_bikes"], errors="coerce")

    # Diagnostic counts BEFORE filtering — caller decides whether to log.
    if stats is not None:
        stats["input_rows"] = int(len(df))
        stats["na_act"] = int(df["act"].isna().sum())
        stats["na_quantity"] = int(qty_num.isna().sum())
        stats["na_available"] = int(avail_num.isna().sum())
        stats["dropped_act_not_1"] = int((df["act"] != "1").sum())
        stats["dropped_qty_invalid"] = int(((qty_num <= 0) | qty_num.isna()).sum())

    keep_mask = (df["act"] == "1") & (qty_num > 0)
    out = df.loc[keep_mask, ["sno", timestamp_col, "Quantity", "available_rent_bikes"]].copy()

    out = out.rename(
        columns={
            timestamp_col: "timestamp",
            "Quantity": "total_capacity",
            "available_rent_bikes": "available_bikes",
        }
    )
    out["total_capacity"] = out["total_capacity"].astype("int64")
    out["available_bikes"] = out["available_bikes"].astype("int64")

    # Raw shortage may be < 0 if available > total_capacity (transient API quirk).
    # We clip to [0, 1] but record how often it happens so upstream quality is visible.
    raw_shortage = 1.0 - out["available_bikes"] / out["total_capacity"]
    if stats is not None:
        stats["clip_negative_count"] = int((raw_shortage < 0).sum())
        stats["kept"] = int(len(out))

    out["shortage_rate"] = raw_shortage.clip(lower=0.0, upper=1.0)

    out = out[M3_INPUT_COLUMNS]
    return out.reset_index(drop=True)


def clean_history(
    history_df: pd.DataFrame,
    *,
    stats: dict | None = None,
) -> pd.DataFrame:
    """Clean the full historical frame from `load_history` and sort by (sno, timestamp)."""
    cleaned = clean_snapshot(history_df, timestamp_col="tick", stats=stats)
    cleaned = cleaned.sort_values(["sno", "timestamp"]).reset_index(drop=True)
    return cleaned
