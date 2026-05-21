"""
Driver: assemble training_dataset.csv from data/youbike/*.json snapshots.

End-to-end pipeline (M3 SoT, §5.7.1):
    load_history → clean_history → spatial_lag (M2) → build_features
                 → add_shortage_label → drop NaN → CSV

Spatial lag
-----------
For training, only the continuous `spatial_lag_shortage = W @ x_t` is
needed (§2.6 — moran_type / p_value go to the publisher's JSON, not the
model). We call `m2.compute_spatial_lag` per snapshot; the 999-permutation
significance test is skipped here for speed and is only run in the online
producer.

Forensic logging
----------------
Each NaN-producing stage is instrumented. The full breakdown is printed
to stdout and persisted to `data/build_log.json` for M4 audit. Stations
that survive raw load but are filtered out entirely (typically `act='1'`
never holds) are written to `data/dropped_stations.csv`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Make sibling packages (features, m1, m3) importable when invoked as a plain script.
CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from m3.build_features import FEATURE_COLUMNS, build_features, load_station_static
from m1.cleaner import clean_history
from m1.load_history import load_history
from m2.lisa import compute_spatial_lag
from m3.labels import LOOKAHEAD_TICKS, add_shortage_label


def _spatial_lag_over_history(
    cleaned: pd.DataFrame, station_static: pd.DataFrame
) -> pd.Series:
    """Per-tick `W @ x_t` over the entire cleaned history.

    Iterates timestamp-by-timestamp because the surviving station set
    (and therefore the k-NN graph) changes between snapshots. ~50 ms per
    tick × thousands of ticks ≈ a few minutes for a full 27-day run.
    """
    parts: list[pd.Series] = []
    n_ticks = cleaned["timestamp"].nunique()
    for i, (ts, snap) in enumerate(cleaned.groupby("timestamp", sort=False), start=1):
        parts.append(compute_spatial_lag(snap, station_static))
        if i % 500 == 0 or i == n_ticks:
            print(f"[lisa]    tick {i:>5}/{n_ticks}")
    return pd.concat(parts).reindex(cleaned.index)


def build(
    history_dir: str | Path | None = None,
    limit: int | None = None,
    out_path: str | Path = "data/training_dataset.csv",
    log_path: str | Path = "data/build_log.json",
) -> pd.DataFrame:
    """Run the full offline pipeline and write training_dataset.csv.

    Returns the in-memory frame (after NaN drop) for caller inspection.
    """
    log: dict = {"started_at": datetime.now().isoformat(timespec="seconds")}

    print(f"[load] reading snapshots (limit={limit})...")
    raw = load_history(history_dir=history_dir, limit=limit)
    raw_snos = set(raw["sno"].astype(str).unique())
    log["raw"] = {
        "rows": int(len(raw)),
        "unique_snos": int(len(raw_snos)),
        "n_snapshot_files": int(raw["tick"].nunique()),
    }
    print(f"[load]   rows={len(raw):,}  unique_snos={len(raw_snos)}  files={raw['tick'].nunique()}")

    # --- Stage 1: cleaner -------------------------------------------------
    print("[clean] applying act='1' + Quantity>0 filter...")
    cleaner_stats: dict = {}
    cleaned = clean_history(raw, stats=cleaner_stats)
    log["clean"] = cleaner_stats
    print(f"[clean]   input_rows           = {cleaner_stats['input_rows']:,}")
    print(f"[clean]   NaN act              = {cleaner_stats['na_act']:,}")
    print(f"[clean]   NaN Quantity         = {cleaner_stats['na_quantity']:,}")
    print(f"[clean]   NaN available_bikes  = {cleaner_stats['na_available']:,}")
    print(f"[clean]   dropped act!=1       = {cleaner_stats['dropped_act_not_1']:,}")
    print(f"[clean]   dropped Quantity<=0  = {cleaner_stats['dropped_qty_invalid']:,}")
    print(f"[clean]   available>capacity (clipped to 0) = {cleaner_stats['clip_negative_count']:,}")
    print(f"[clean]   kept                 = {cleaner_stats['kept']:,}")

    # --- Stage 2: spatial lag (M2 k-NN, k=6) -----------------------------
    print("[lisa]  loading station_static + computing spatial_lag per tick...")
    station_static = load_station_static()
    station_static["sno"] = station_static["sno"].astype(str)
    spatial_lag = _spatial_lag_over_history(cleaned, station_static)

    # --- Stage 3: features -----------------------------------------------
    print("[m3]    running build_features...")

    # build_features fail-fasts on NaN distance_to_mrt (raises ValueError);
    # we only reach this line on the clean path.
    features = build_features(cleaned, station_static, spatial_lag)
    log["features"] = {
        "rows": int(len(features)),
        "distance_to_mrt_na_rows": 0,
    }

    # --- Stage 4: labels --------------------------------------------------
    print("[m3]    generating labels (t+60min lookahead > 0.8)...")
    labelled = add_shortage_label(cleaned)

    # --- Stage 5: assemble + classify drops -------------------------------
    full = features.copy()
    full["sno"] = cleaned["sno"].values
    full["timestamp"] = cleaned["timestamp"].values
    full["y"] = labelled["y"].values

    feature_nan = full[FEATURE_COLUMNS].isna().any(axis=1)
    y_nan = full["y"].isna()
    drop_mask = feature_nan | y_nan

    # Per-station counts (for expected warmup/lookahead attribution)
    surviving_snos = (
        cleaned.groupby("sno").size().rename("ticks").reset_index()
    )
    n_stations_with_data = int(len(surviving_snos))
    expected_warmup = int((surviving_snos["ticks"].clip(upper=6)).sum())
    expected_lookahead = int(
        (surviving_snos["ticks"].clip(upper=LOOKAHEAD_TICKS)).sum()
    )

    log["assemble"] = {
        "rows_before_drop": int(len(full)),
        "feature_nan_rows": int(feature_nan.sum()),
        "label_nan_rows": int(y_nan.sum()),
        "both_nan_rows": int((feature_nan & y_nan).sum()),
        "total_dropped": int(drop_mask.sum()),
        "expected_warmup_drops_per_lag_60min": expected_warmup,
        "expected_lookahead_drops": expected_lookahead,
        "stations_with_any_kept_row": n_stations_with_data,
    }
    print(f"[drop]  feature NaN rows   = {feature_nan.sum():,}")
    print(f"[drop]  label NaN rows     = {y_nan.sum():,}")
    print(f"[drop]  overlap (both NaN) = {(feature_nan & y_nan).sum():,}")
    print(f"[drop]  total dropped      = {drop_mask.sum():,}")
    print(
        f"[drop]  expected: warmup(lag_60min)={expected_warmup:,}  "
        f"lookahead={expected_lookahead:,}  "
        f"(union <= sum because of overlap on short stations)"
    )

    full = full.loc[~drop_mask].reset_index(drop=True)
    full = full[["sno", "timestamp"] + FEATURE_COLUMNS + ["y"]]

    # --- Stage 6: dropped-station audit -----------------------------------
    survived_snos = set(full["sno"].astype(str).unique())
    fully_dropped = sorted(raw_snos - survived_snos)
    log["fully_dropped_stations"] = {
        "count": len(fully_dropped),
        "snos": fully_dropped,
    }
    print(f"[audit] stations fully removed: {len(fully_dropped)}")

    if fully_dropped:
        # Enrich with sna/sarea/ar from the first raw snapshot for human review.
        first_snapshot = sorted(
            (Path(history_dir) if history_dir else REPO_ROOT / "data" / "youbike").glob(
                "youbike_*.json"
            )
        )[0]
        with open(first_snapshot, encoding="utf-8") as f:
            records = json.load(f)
        lookup = {str(r["sno"]): r for r in records}
        rows = []
        for sno in fully_dropped:
            r = lookup.get(sno, {})
            rows.append(
                {
                    "sno": sno,
                    "sna": r.get("sna", ""),
                    "sarea": r.get("sarea", ""),
                    "ar": r.get("ar", ""),
                    "Quantity": r.get("Quantity", ""),
                    "act_in_first_snapshot": r.get("act", ""),
                    "reason": "act!=1 throughout the loaded window",
                }
            )
        dropped_csv = REPO_ROOT / "data" / "dropped_stations.csv"
        pd.DataFrame(rows).to_csv(dropped_csv, index=False, encoding="utf-8-sig")
        print(f"[audit]   wrote {dropped_csv.relative_to(REPO_ROOT)}")

    # --- Stage 7: write CSV + log ----------------------------------------
    out_path = REPO_ROOT / out_path if not Path(out_path).is_absolute() else Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_path, index=False)

    log["output"] = {
        "rows": int(len(full)),
        "cols": int(len(full.columns)),
        "path": str(out_path),
        "size_mb": round(out_path.stat().st_size / (1024 * 1024), 1),
    }
    log["finished_at"] = datetime.now().isoformat(timespec="seconds")

    log_path = REPO_ROOT / log_path if not Path(log_path).is_absolute() else Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(
        f"[done]  wrote {out_path.relative_to(REPO_ROOT)}  "
        f"({len(full):,} rows, {len(full.columns)} cols, {log['output']['size_mb']} MB)"
    )
    print(f"[done]  audit log: {log_path.relative_to(REPO_ROOT)}")

    return full


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble training_dataset.csv from raw snapshots.")
    ap.add_argument("--history-dir", default=None, help="Override data/youbike directory.")
    ap.add_argument("--limit", type=int, default=None, help="Read at most N snapshots (smoke test).")
    ap.add_argument("--out", default="data/training_dataset.csv", help="Output CSV path.")
    ap.add_argument("--log", default="data/build_log.json", help="Audit log path.")
    args = ap.parse_args()

    build(history_dir=args.history_dir, limit=args.limit, out_path=args.out, log_path=args.log)


if __name__ == "__main__":
    main()
